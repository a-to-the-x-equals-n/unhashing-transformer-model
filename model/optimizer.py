import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# color codes for terminal output
MG = '\033[35m'     # magenta
X  = '\033[0m'      # reset


class AdamWarlock(torch.optim.AdamW):
    '''
    AdamW optimizer with integrated CosineAnnealingWarmRestarts scheduling.

    Attributes:
    -----------
    base_lr : float
        The base learning rate.

    scheduler : CosineAnnealingWarmRestarts
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

        total_steps : int
            Number of optimizer steps in one cosine cycle (e.g., len(dataloader)).

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
        if total_steps is None:
            raise ValueError('total_steps is required (steps per epoch).')

        print(f' [scheduler]: CosineAnnealingWarmRestarts')
        print(f' [steps per restart]: {total_steps}')
        print(f' [min learning rate]: {eta_min}')

        self.scheduler = CosineAnnealingWarmRestarts(
            self,
            T_0 = total_steps,
            T_mult = 1,
            eta_min = eta_min
        )


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
