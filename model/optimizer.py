import math
from functools import partial
from typing import Literal
import torch
from torch.optim.lr_scheduler import LambdaLR

# color codes for terminal output
MG = '\033[35m'     # magenta
X  = '\033[0m'      # reset


def _inverse_sqrt_factor(step: int, warmup_iters: int) -> float:
    t = step + 1
    if t <= warmup_iters:
        return t / warmup_iters
    return math.sqrt(warmup_iters / t)


def _cosine_warmup_factor(step: int, warmup_iters: int, cosine_steps: int) -> float:
    t = step + 1
    if warmup_iters and t <= warmup_iters:
        return t / warmup_iters
    progress = min(max(0, t - warmup_iters), cosine_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress / cosine_steps))


def _warmup_only_factor(step: int, warmup_iters: int) -> float:
    t = step + 1
    if t <= warmup_iters:
        return t / warmup_iters
    return 1.0



class AdamWarlock(torch.optim.AdamW):
    '''
    AdamW optimizer with integrated learning rate scheduling.

        extends torch.optim.AdamW to include built-in learning rate scheduling with optional warmup and cosine annealing decay. 
        encapsulates optimizer configuration and scheduling logic to simplify the training loop.

        adamW decouples weight decay from gradient-based updates
        providing better regularization than standard Adam
        weight decay is applied directly to weights rather than through gradients.

    Parameters:
    -----------
    params : iterable
        Iterable of model parameters to optimize or dicts defining parameter groups.

    lr : float, optional
        Base learning rate (default: 1e-4).
        This is the peak learning rate reached after warmup.

    betas : tuple[float, float], optional
        Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        beta1: momentum coefficient for gradient
        beta2: momentum coefficient for squared gradient

    eps : float, optional
        Term added to denominator for numerical stability (default: 1e-8).

    weight_decay : float, optional
        Weight decay coefficient for L2 regularization (default: 0.01).
        Applied directly to weights (decoupled from gradients).

    warmup_steps : int, optional
        Number of steps for linear learning rate warmup (default: 0 = no warmup).
        Learning rate increases linearly from ~0 to base_lr during warmup.
        Helps stabilize training in early stages.

    total_steps : int, optional
        Total number of training steps across all epochs (default: None).
        Retained for backward compatibility; inverse-sqrt decay does not require it.

    schedule : str, optional
        Learning-rate policy. Supported values:
            - 'inverse_sqrt' (default): Transformer-style warmup then 1/sqrt(t) decay
            - 'cosine'      : optional warmup followed by cosine annealing over total_steps
            - 'none'        : constant LR (except optional warmup)
        The legacy `use_cosine_decay` flag maps to 'cosine' when True.

    Attributes:
    -----------
    base_lr : float
        The base learning rate (used for warmup reference).

    warmup_steps : int
        Number of warmup steps configured.

    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        The underlying PyTorch learning rate scheduler.
        Can be LinearLR (warmup only) or LambdaLR (for inverse-sqrt/cosine/constant).

    Notes:
    ------
    - Call step() after optimizer.step() to update model weights
    - Call step_scheduler() after each training step to update learning rate
    - Use get_lr() to retrieve current learning rate for logging
    - state_dict() and load_state_dict() handle both optimizer and scheduler state

    '''

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        total_steps: int = None,
        schedule: Literal['inverse_sqrt', 'cosine', 'none'] = 'cosine'
    ):

        print(f'\n{MG}[ADAM WARLOCK INIT]{X}')

        # initialize parent AdamW optimizer with standard parameters
        super().__init__(params, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        print(f'  [base learning rate]: {lr}')
        print(f'  [weight decay]: {weight_decay}')
        print(f'  [betas]: {betas}')

        # store base learning rate for reference (used in warmup)
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.scheduler = None
        self.schedule = schedule

        valid_schedules = {'inverse_sqrt', 'cosine', 'none'}
        if self.schedule not in valid_schedules:
            raise ValueError(f"Unsupported schedule '{self.schedule}'. Choose from {valid_schedules}.")

        # - LEARNING RATE SCHEDULER SETUP - 

        match self.schedule:
            case 'inverse_sqrt':
                warmup_iters = max(1, warmup_steps)
                lambda_fn = partial(_inverse_sqrt_factor, warmup_iters = warmup_iters)

                print(f'  [scheduler]: inverse-square-root')
                print(f'  [warmup steps]: {warmup_iters}')
    
            case 'cosine':
                if total_steps is None:
                    raise ValueError('schedule = "cosine" requires total_steps to be set.')
                
                cosine_steps = max(1, total_steps - warmup_steps)
                warmup_iters = max(1, warmup_steps) if warmup_steps > 0 else 0
                lambda_fn = partial(_cosine_warmup_factor, warmup_iters = warmup_iters, cosine_steps = cosine_steps)

                print(f'  [scheduler]: cosine annealing (LambdaLR)')
                print(f'  [total steps]: {total_steps}')
                print(f'  [warmup steps]: {warmup_steps}')

            case 'none':
                if warmup_steps > 0:
                    warmup_iters = warmup_steps
                    lambda_fn = partial(_warmup_only_factor, warmup_iters = warmup_iters)

                    print(f'  [scheduler]: warmup only → constant')
                    print(f'  [warmup steps]: {warmup_iters}')
                else:
                    print(f'  [scheduler]: none (constant LR)')

            case _:
                raise ValueError(f"Unsupported schedule '{self.schedule}'. Choose from {valid_schedules}.")

        self.scheduler = LambdaLR(self, lr_lambda = lambda_fn)


    @property
    def lr(self) -> float:
        '''
        Get current learning rate.

        Returns:
        --------
        float
            Current learning rate from the first parameter group.
            All parameter groups typically share the same LR unless explicitly configured otherwise.

        Notes:
        ------
            useful for logging and debugging learning rate schedules.
        '''
        return self.param_groups[0]['lr']
