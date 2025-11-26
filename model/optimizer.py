import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# color codes for terminal output
MG = '\033[35m'     # magenta
X  = '\033[0m'      # reset


class AdamWarlock(torch.optim.AdamW):
    '''
    AdamW optimizer with integrated CosineAnnealingLR scheduling.

    Attributes:
    -----------
    base_lr : float
        The base learning rate.

    scheduler : CosineAnnealingLR
        The underlying PyTorch learning rate scheduler.
    '''
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        total_steps: int = None,
        t_max: int | None = None,
        eta_min: float = 1e-5
    ):
        '''
        Encapsulates optimizer configuration and scheduling logic to simplify the training loop.

        Parameters:
        -----------
        params : iterable
            Iterable of model parameters to optimize or dicts defining parameter groups.

        lr : float, optional
            Base learning rate (default: 1e-4).

        betas : tuple[float, float], optional
            Coefficients for computing running averages of gradient and its square (default: (0.9, 0.98)).

        eps : float, optional
            Term added to denominator for numerical stability (default: 1e-8).

        weight_decay : float, optional
            Weight decay coefficient for L2 regularization (default: 0.01).

        total_steps : int, optional
            Backward-compat alias for t_max; use t_max for clarity.

        t_max : int, optional
            Total number of scheduler steps before reaching eta_min (monotonic cosine).

        eta_min : float, optional
            Minimum learning rate (default: 1e-5).
        '''

        print(f'\n{MG}[Adam Warlock Init]{X}')

        # initialize parent AdamW optimizer
        super().__init__(params, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        print(f' [base learning rate]: {lr}')
        print(f' [weight decay]: {weight_decay}')
        print(f' [betas]: {betas}')

        # store base learning rate for reference
        self.base_lr = lr

        # setup cosine annealing scheduler
        # if total_steps is None:
        #   raise ValueError('total_steps is required (steps per epoch).')

        # resolve T_max for monotonic cosine decay
        if t_max is None:
            if total_steps is None:
                raise ValueError('t_max or total_steps is required (total scheduler steps).')
            t_max = total_steps

        print(f' [scheduler]: CosineAnnealingLR')
        print(f' [T_max]: {t_max} steps')
        print(f' [min learning rate]: {eta_min}')

        self.scheduler = CosineAnnealingLR(
            self,
            T_max = t_max,
            eta_min = eta_min
        )

        # previous scheduler (kept for reference)
        # self.scheduler = CosineAnnealingWarmRestarts(
        #     self,
        #     T_0 = total_steps,
        #     T_mult = 1,
        #     eta_min = eta_min
        # )


    @property
    def lr(self) -> float:
        '''
        Get current learning rate.

        Returns:
        --------
        float
            Current learning rate from the first parameter group.
        '''
        return self.param_groups[0]['lr']
