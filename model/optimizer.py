import math
from typing import Literal, Optional

import torch

# color codes for terminal output
MG = '\033[35m'     # magenta
X  = '\033[0m'      # reset

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
            - 'none'        : constant LR (except optional warmup)
        The legacy `use_cosine_decay` flag maps to 'inverse_sqrt' when True.

    Attributes:
    -----------
    base_lr : float
        The base learning rate (used for warmup reference).

    warmup_steps : int
        Number of warmup steps configured.

    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        The underlying PyTorch learning rate scheduler.
        Can be LinearLR (warmup only), CosineAnnealingLR (decay only), or SequentialLR (both).

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
        use_cosine_decay: Optional[bool] = None,
        schedule: Literal['inverse_sqrt', 'none'] = 'inverse_sqrt'
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

        # legacy compatibility: reuse old flag but map to new behavior
        if use_cosine_decay is not None:
            if use_cosine_decay and schedule != 'inverse_sqrt':
                print(f'  [schedule override]: legacy use_cosine_decay=True -> inverse_sqrt')
                self.schedule = 'inverse_sqrt'
            elif not use_cosine_decay:
                print(f'  [schedule override]: legacy use_cosine_decay=False -> none')
                self.schedule = 'none'

        valid_schedules = {'inverse_sqrt', 'none'}
        if self.schedule not in valid_schedules:
            raise ValueError(f"Unsupported schedule '{self.schedule}'. Choose from {valid_schedules}.")

        # ---- setup learning rate scheduler based on configuration ----

        if self.schedule == 'inverse_sqrt':
            from torch.optim.lr_scheduler import LambdaLR
            warmup_iters = max(1, warmup_steps)
            print(f'  [scheduler]: inverse-square-root')
            print(f'  [warmup steps]: {warmup_iters}')

            def lr_lambda(step: int) -> float:
                """
                Transformer-style schedule:
                    scale linearly during warmup
                    decay as 1/sqrt(step) afterwards, anchored at warmup boundary
                """
                t = step + 1  # schedulers are 0-indexed; avoid division by zero
                if t <= warmup_iters:
                    return t / warmup_iters
                return math.sqrt(warmup_iters / t)

            self.scheduler = LambdaLR(self, lr_lambda = lr_lambda)

        elif self.schedule == 'none':
            if warmup_steps > 0:
                from torch.optim.lr_scheduler import LambdaLR
                warmup_iters = warmup_steps
                print(f'  [scheduler]: warmup only → constant')
                print(f'  [warmup steps]: {warmup_iters}')

                def warmup_only(step: int) -> float:
                    t = step + 1
                    if t <= warmup_iters:
                        return t / warmup_iters
                    return 1.0

                self.scheduler = LambdaLR(self, lr_lambda = warmup_only)
            else:
                print(f'  [scheduler]: none (constant LR)')

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
