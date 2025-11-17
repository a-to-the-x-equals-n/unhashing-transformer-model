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


def _cosine_warmup_factor(step: int, warmup_iters: int, cosine_steps: int, eta_floor: float = 0.0) -> float:
    t = step + 1
    if warmup_iters and t <= warmup_iters:
        return t / warmup_iters
    progress = min(max(0, t - warmup_iters), cosine_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress / cosine_steps))
    return max(eta_floor, cosine)


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

    Parameters:
    -----------
    params : iterable
        Iterable of model parameters to optimize or dicts defining parameter groups.

    lr : float, optional
        Base learning rate (default: 1e-4).

    betas : tuple[float, float], optional
        Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).

    eps : float, optional
        Term added to denominator for numerical stability (default: 1e-8).

    weight_decay : float, optional
        Weight decay coefficient for L2 regularization (default: 0.01).

    warmup_steps : int, optional
        Number of steps for linear learning rate warmup (default: 0 = no warmup).

    total_steps : int, optional
        Total number of training steps across all epochs (default: None).

    schedule : str, optional
        Learning-rate policy.

    Attributes:
    -----------
    base_lr : float
        The base learning rate (used for warmup reference).

    warmup_steps : int
        Number of warmup steps configured.

    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        The underlying PyTorch learning rate scheduler.
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

        lambda_fn = None

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
                eta_floor = 5e-5 / self.base_lr
                lambda_fn = partial(_cosine_warmup_factor, warmup_iters = warmup_iters, cosine_steps = cosine_steps, eta_floor = eta_floor)

                print(f'  [scheduler]: cosine annealing (LambdaLR)')
                print(f'  [total steps]: {total_steps}')
                print(f'  [warmup steps]: {warmup_steps}')
                print(f'  [eta_min]: {eta_floor * self.base_lr:.2e}')

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

        if lambda_fn is not None:
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
