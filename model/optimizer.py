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
        Computed as: epochs * batches_per_epoch
        Required if use_cosine_decay is True.

    use_cosine_decay : bool, optional
        Whether to apply cosine annealing learning rate decay after warmup (default: False).
        If True, learning rate decays from base_lr to 0 following a cosine curve.
        Helps model converge to flatter minima.

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
        use_cosine_decay: bool = False
    ):

        print(f'\n{MG}[ADAM WARLOCK INIT]{X}')

        # initialize parent AdamW optimizer with standard parameters
        super().__init__(params, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        print(f'  [base learning rate]: {lr:.2e}')
        print(f'  [weight decay]: {weight_decay}')
        print(f'  [betas]: {betas}')

        # store base learning rate for reference (used in warmup)
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.scheduler = None

        # ---- setup learning rate scheduler based on configuration ----

        # case 1: cosine decay enabled
        # (with or without warmup)
        if use_cosine_decay:
            if total_steps is None:
                raise ValueError("total_steps required when use_cosine_decay=True")

            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

            print(f'  [scheduler]: cosine annealing decay')
            print(f'  [total steps]: {total_steps:,}')

            # case 1a: warmup + cosine decay
            # (two-phase schedule)
            if warmup_steps > 0:
                print(f'  [warmup steps]: {warmup_steps:,}')
                print(f'  [decay steps]: {total_steps - warmup_steps:,}')
                print(f'  [schedule]: warmup → cosine decay')

                # phase 1: linear warmup from low LR to base_lr
                # start_factor = 0.01 means we start at 1% of base_lr
                warmup_scheduler = LinearLR(
                    self,
                    start_factor = 0.01,      # LR starts at base_lr * 0.01 (1% of base_lr)
                    end_factor = 1.0,         # LR reaches base_lr * 1.0 = base_lr
                    total_iters = warmup_steps
                )

                # phase 2: cosine annealing from base_lr to 0
                # T_max is the number of steps for the cosine curve
                # (excluding warmup)
                cosine_scheduler = CosineAnnealingLR(
                    self,
                    T_max = total_steps - warmup_steps,  # decay over remaining steps
                    eta_min = 0                          # minimum LR at end of schedule
                )

                # combine both phases:
                #   warmup for first N steps, then cosine decay
                #   milestones = [warmup_steps] tells SequentialLR when to switch schedulers
                self.scheduler = SequentialLR(
                    self,
                    schedulers = [warmup_scheduler, cosine_scheduler],
                    milestones = [warmup_steps]
                )

            # case 1b: cosine decay only
            # (no warmup)
            else:
                print(f'  [schedule]: cosine decay only (no warmup)')

                # just apply cosine annealing over all training steps
                self.scheduler = CosineAnnealingLR(
                    self,
                    T_max = total_steps,
                    eta_min = 0
                )

        # case 2: warmup only
        # (no decay after warmup)
        elif warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR

            print(f'  [scheduler]: warmup only')
            print(f'  [warmup steps]: {warmup_steps:,}')
            print(f'  [schedule]: warmup → constant LR')

            # linear warmup from low LR to base_lr
            # after warmup completes, LR stays constant at base_lr
            self.scheduler = LinearLR(
                self,
                start_factor = 0.01,      # LR starts at base_lr * 0.01 (1% of base_lr)
                end_factor = 1.0,         # LR reaches base_lr * 1.0 = base_lr
                total_iters = warmup_steps
            )

        # case 3: no scheduling (constant learning rate)
        # self.scheduler remains None, step_scheduler() becomes a no-op
        else:
            print(f'  [scheduler]: none (constant LR)')


    def step_scheduler(self):
        '''
        Update the learning rate schedule.

            call this method after each optimizer.step() to advance the learning rate schedule.
            handles warmup, cosine decay, or combined schedules automatically based on initialization parameters.

            this should be called once per training step (batch), not per epoch.

        Notes:
        ------
            does nothing if no scheduler was configured (constant LR)
            automatically handles transitions between warmup and decay phases
            safe to call even if scheduler is None
        '''
        if self.scheduler is not None:
            self.scheduler.step()

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