import math
from functools import partial
from typing import Literal
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

# color codes for terminal output
MG = '\033[35m'     # magenta
X  = '\033[0m'      # reset


def _inverse_sqrt_factor(step: int, warmup_iters: int) -> float:
    t = step + 1
    if t <= warmup_iters:
        return t / warmup_iters
    return math.sqrt(warmup_iters / t)


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
        Number of optimizer steps in one cosine cycle (e.g., len(dataloader)).

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
        betas: tuple[float, float] = (0.9, 0.98),
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

        if self.schedule == 'inverse_sqrt':
            warmup_iters = max(1, warmup_steps)
            lambda_fn = partial(_inverse_sqrt_factor, warmup_iters = warmup_iters)
            print(f'  [scheduler]: inverse-square-root')
            print(f'  [warmup steps]: {warmup_iters}')

        elif self.schedule == 'cosine':
            if total_steps is None:
                raise ValueError('schedule="cosine" requires total_steps (steps per epoch).')
            if warmup_steps > 0:
                print(f'  [warmup warning]: CosineAnnealingWarmRestarts ignores warmup_steps; set schedule=\"none\" to use warmup-only.')
            print(f'  [scheduler]: CosineAnnealingWarmRestarts')
            print(f'  [steps per restart]: {total_steps}')
            self.scheduler = CosineAnnealingWarmRestarts(
                self,
                T_0 = total_steps,
                T_mult = 1,
                eta_min = 1e-5
            )

        elif self.schedule == 'none':
            if warmup_steps > 0:
                warmup_iters = warmup_steps
                lambda_fn = partial(_warmup_only_factor, warmup_iters = warmup_iters)
                print(f'  [scheduler]: warmup only → constant')
                print(f'  [warmup steps]: {warmup_iters}')
            else:
                print(f'  [scheduler]: none (constant LR)')

        else:
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
