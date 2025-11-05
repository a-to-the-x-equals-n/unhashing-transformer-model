
import torch
from torch.optim import Adam
from model import OptimusPrime
from data import Bumblebee, collate_batch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

GR = '\033[32m'     # green
BU = '\033[34m'     # blue
RD = '\033[31m'     # red
CY = '\033[36m'     # cyan
YW = '\033[33m'     # yellow
MG = '\033[35m'     # magenta
X  = '\033[0m'      # reset

class Trainer:
    '''
    Trainer class for OptimusPrime model.

    Handles:
    - Model initialization and device placement
    - Dataset loading and batching
    - Training loop with progress tracking
    - Checkpoint saving/loading
    - TensorBoard logging
    '''

    def __init__(
        self,
        path: str | Path,
        epochs: int = 1,
        batch_size: int = 1024,
        lr: float = 1e-4,
        checkpoint_dir: Path = Path('checkpoints'),
        checkpoint_interval: int = 1,
        logs: str = 'runs/optimus'
    ):
        '''
        Initialize trainer with configuration.

        Parameters:
        -----------
        path : str | Path
            Path to training data TSV file
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        lr : float
            Learning rate for Adam optimizer
        checkpoint_dir : Path
            Directory to save checkpoints
        checkpoint_interval : int
            Save checkpoint every N epochs
        logs : str
            TensorBoard log directory
        '''

        # configuration
        self.path = path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.logs = logs

        # components 
        # (initialized in setup)
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.writer = None

        # training state
        self.start_epoch = 0
        self.best_loss = float('inf')

        # model config
        self.model_config = {
            'vocab_size': 257,
            'pw_vocab_size': 75,
            'pad_id': 74,
            'hash_pad_id': 256,
            'd_model': 256,
            'n_heads': 8,
            'num_layers': 4,
            'ff_dim': 512,
            'dropout': 0.1
        }


    def setup(self):
        '''
        Setup dataset, model, optimizer, and TensorBoard writer
        Initializes all components required for training

            loads dataset from TSV file
            creates DataLoader with specified batch size
            instantiates OptimusPrime model and moves to device
            creates Adam optimizer
            initializes TensorBoard SummaryWriter
            creates checkpoint directory if it doesn't exist
        '''
        print(f'\n{BU} [SETUP]{X}')
        print(f'  [total epochs]: {self.epochs}')
        print(f'  [batch size]: {self.batch_size}')
        print(f'  [learning rate]: {self.lr}')
        print(f'  [loading file]: {self.path.name}')

        # device detection
        print(f'  [cuda??]')
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f'{GR}   [cuda found]{X}')
            print(f'{GR}   [using GPU]{X}')
        else:
            self.device = 'cpu'
            print(f'{RD}   [cuda NOT found]{X}')
            print(f'{RD}   [using CPU]{X}')

        # build dataset and dataloader
        print(f'  [building dataset]')
        self.dataset = Bumblebee(self.path)

        print(f'  [building dataloader]')
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_batch
        )

        # build model
        print(f'  [forging Optimus Prime]')
        self.model = OptimusPrime(**self.model_config).to(self.device)

        # build optimizer
        print(f'  [building Adam]')
        self.optimizer = Adam(self.model.parameters(), lr = self.lr)

        # initialize TensorBoard
        print(f'  [initializing TensorBoard]')
        self.writer = SummaryWriter(log_dir =self.logs)

        # create checkpoint directory
        print(f'  [creating checkpoint directory]')
        self.checkpoint_dir.mkdir(exist_ok = True)


    def load_checkpoint(self):
        '''
        Load the most recent checkpoint if available to resume training

            searches the checkpoint directory for saved checkpoints and loads the latest one based on epoch number
            restores model weights, optimizer state, starting epoch, and best loss value
            if no checkpoint is found, training starts from scratch at epoch 0
        '''
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f'\n{YW} [CHECKPOINT FOUND]{X}')
            print(f'  [loading]: {latest_checkpoint.name}')

            checkpoint = torch.load(latest_checkpoint, map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))

            print(f'  [resuming from epoch]: {self.start_epoch}')
            print(f'  [previous best loss]: {self.best_loss:.4f}')
        else:
            print(f'\n{YW} [NO CHECKPOINT FOUND]{X}')
            print(f'  [starting fresh training]')


    def save_checkpoint(self, epoch: int, loss: float, global_step: int):
        '''
        Save a training checkpoint to disk

            saves model state, optimizer state, training metadata, and model configuration to a .pt file
            checkpoint can be used to resume training from this exact point

        Parameters:
        -----------
        epoch : int
            Current epoch number (0-indexed).

        loss : float
            Final average loss for this epoch.

        global_step : int
            Global training step across all epochs (for TensorBoard continuity).
        '''
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'global_step': global_step,
            'config': self.model_config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'    [checkpoint saved]: {checkpoint_path.name}')


    def save_best_model(self, epoch: int, loss: float):
        '''
        Save the best model encountered so far based on lowest loss.

            updates the best loss value and saves model state, optimizer state, 
                and configuration to a dedicated "best_model.pt" file
            this file always contains the model with the lowest validation loss

        Parameters:
        -----------
        epoch : int
            Epoch number where this best loss was achieved.

        loss : float
            The new best (lowest) loss value.
        '''
        self.best_loss = loss
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'config': self.model_config
        }, best_model_path)
        print(f'{GR}    [new best saved]: loss = {self.best_loss:.4f}{X}')


    def train_epoch(self, epoch: int) -> float:
        '''
        Train for one epoch.

        Parameters:
        -----------
        epoch : int
            Current epoch number

        Returns:
        --------
        float
            Average loss for this epoch
        '''
        self.model.train()
        total_loss = 0.0
        epoch_start_time = time.time()

        # progress bar
        progress = tqdm(self.dataloader, desc = f'Epoch {epoch + 1}/{self.start_epoch + self.epochs}', leave = True, unit = ' batch')

        for i, batch in enumerate(progress):
            batch_start_time = time.time()

            # move data to GPU/CPU
            hashes = batch['hash'].to(self.device)
            pw = batch['password'].to(self.device)

            # forward pass
            logits = self.model(hashes, pw)
            loss = self.model.compute_loss(logits, pw)

            # backward pass
            self.optimizer.zero_grad()  # reset old gradients
            loss.backward()             # compute new gradients via backprop
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0) # clip grad norms to stabilize training
            self.optimizer.step()       # update weights

            # accumulate loss
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1) # average over batches processed so far

            # log to TensorBoard
            global_step = epoch * len(self.dataloader) + i
            self.writer.add_scalar('Loss/batch', loss.item(), global_step)
            self.writer.add_scalar('Loss/epoch_avg', avg_loss, global_step)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
            self.writer.add_scalar('Gradient/norm', grad_norm.item(), global_step)
            batch_time = time.time() - batch_start_time
            self.writer.add_scalar('Time/batch_seconds', batch_time, global_step)

            # update progress bar
            progress.set_postfix_str(f'loss={avg_loss:.4f}, grad={grad_norm.item():.3f}')

        # close progress bar for this epoch
        progress.close()

        # log epoch-level metrics
        epoch_time = time.time() - epoch_start_time
        final_epoch_loss = total_loss / len(self.dataloader)
        self.writer.add_scalar('Loss/epoch_final', final_epoch_loss, epoch)
        self.writer.add_scalar('Time/epoch_minutes', epoch_time / 60, epoch)

        print(f'\n [epoch {epoch + 1} / {self.epochs} complete]')
        print(f'  [loss]: {final_epoch_loss:.4f}')
        print(f'  [time]: {epoch_time / 60:.2f} minutes')

        return final_epoch_loss, global_step


    def train(self):
        '''
        Execute the main training loop across all epochs

            for each epoch:
                calls train_epoch() to perform forward/backward passes
                saves checkpoint every N epochs (based on self.checkpoint_interval)
                saves best model if current epoch loss is lowest so far

            all training metrics are logged to TensorBoard
            tqdm shows progress bars with loss and grad norms
        '''
        print(f'\n{BU} [START]{X}\n')

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            # train one epoch
            epoch_loss, global_step = self.train_epoch(epoch)

            # save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch, epoch_loss, global_step)

            # save best
            if epoch_loss < self.best_loss:
                self.save_best_model(epoch, epoch_loss)

            print()


    def cleanup(self):
        '''
        Close writer and print summary
        
            displays the terminal commands for tensorboard
            local and the shared upload
        '''
        self.writer.close()
        print(f'\n{BU} [COMPLETE]{X}')
        print(f'  [best loss]: {self.best_loss:.4f}')
        print(f'  [checkpoints saved]: {self.checkpoint_dir}\n')
        print(f' [TensorBoard]: {YW}tensorboard --logdir=runs{X}')
        print(f'  [TensorBoard dashboard]: http://localhost:6006')
        print(f'  [upload TensorBoard logs]:\n\t{YW}tensorboard dev upload --logdir runs/optimus_prime{X} \\\
            \n\t\t{YW}--name {X}{CY}"Optimus Prime - MD5 Hash Inversion"{X} \\\
            \n\t\t{YW}--description {X}{CY}"Training run with 1M password dataset"{X}')




OPTIMUS = f'''
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠀⠀⠀⠀⠀⣤⣤⣤⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⣠⡶⢿⡇⢿⣿⡏⢳⣦⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⡛⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⣼⣿⣴⣋⡽⠮⠿⢭⣟⣏⣷⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣧⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡼⣇⣿⡿⠶⣶⣿⣟⡛⣷⣿⢠⠙⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡈⣏⠇⢹⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⢹⠁⣿⠋⠉⢹⠉⠙⣿⡇⣾⣀⣾⠀⢀⣤⡀⢀⡀⠀⠀⢀⣠⣴⣾⠛⢻⡛⢻⡄⢀⣳⡀⢀⣠⠄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣷⣾⢀⣿⡇⠀⠸⠀⠀⣿⣧⡽⠿⣟⣺⣭⠴⢿⡏⣩⣷⡾⢛⣭⣴⣿⣇⠘⣿⣷⣿⡛⠉⢻⣟⣷⠄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠿⢿⣟⣿⣿⡦⣶⣪⡭⠿⣚⣫⣭⣽⣶⡄⠀⢸⡇⣿⡙⣿⣿⣿⣿⣿⣿⣆⠹⣿⣿⣷⡀⠀⢿⡉⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣤⣶⣿⠿⠛⣉⣭⣶⣾⣿⠿⠟⠛⠉⠉⢻⠀⢸⣷⣿⣇⢻⡿⣿⣿⣿⣿⠟⠀⠹⣿⣿⠃⠀⠘⣷⡀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣦⣼⣿⠿⠛⣋⡁⣼⢠⣿⡿⠛⠉⠁⠀⠀⢀⡀⢀⣴⣾⠀⢸⣿⡇⢻⡄⠙⠿⠻⠛⠁⠀⢀⣠⣽⣿⣇⡀⠀⠸⣧⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⠿⣛⣭⣴⡾⠟⠛⣧⣿⢸⡿⠀⠀⠀⠀⣰⣿⣿⣷⣾⣿⣿⠀⢸⡏⣇⢸⣷⡀⠀⢀⣠⣴⣾⠿⠛⣿⢻⣿⣹⡀⠀⢻⣆⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⡟⣦⠀⠀⠀⢀⡿⣵⡿⠛⠉⣡⣶⣤⣄⣿⣯⢸⣇⠀⠀⢠⣾⣿⡿⣿⣿⣿⣿⡿⠀⢸⡇⢻⡼⣿⣷⣶⠿⠛⠉⠀⠀⠀⠸⡇⣿⣿⣧⠀⠘⣿⡀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡇⢹⠀⢀⣠⣼⣿⣿⠀⢀⣼⣿⣿⣿⣿⡇⣿⢸⣿⣀⣀⣿⡿⠿⠶⠚⠛⠉⠉⠀⠀⢸⡇⠀⢻⣾⣝⣿⡆⠀⢀⣠⡴⠖⠛⢻⡾⣿⣿⣆⠀⢹⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣇⣼⡾⠟⠋⣿⢻⣇⣤⣌⠻⢿⣿⣿⣿⠃⢿⠀⠉⠉⠁⠀⠀⠀⣀⣤⡤⠶⠶⠒⠚⣻⣷⣄⠈⣿⣿⣿⣿⡞⠉⠀⠀⠀⠀⠀⣿⢿⣿⣾⣋⣽⠇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣹⠏⠀⠀⠀⣿⢿⣿⣿⣯⡴⠾⠛⢋⣡⠶⠛⠛⠋⣉⣉⣉⣙⢻⣿⠀⠀⠀⠀⠀⢠⡟⠀⠈⠻⢦⣈⣿⣿⣧⠀⠀⢀⣠⣴⡾⢿⣿⣿⣿⣿⣿⡀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡟⣿⡟⠀⠀⠀⣿⠈⠋⠉⢀⣠⠴⣛⣩⣤⣶⣞⣭⣿⢿⣿⣿⣻⣼⣿⣆⣀⣤⣤⣴⣿⣄⣠⣶⣦⣀⣙⣿⣿⣿⡶⣿⠟⠋⣁⣶⠟⢻⣽⣿⣿⣿⠇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢠⣿⣇⠀⠀⠀⢹⣠⡴⠖⢻⣷⢫⣿⣿⣿⣯⣿⣟⣿⣿⣭⣽⣿⡿⣿⣿⣿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⠋⠉⣿⠀⢸⣿⣿⣿⣿⣷⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣼⣿⣿⣤⣴⣾⢿⡅⠀⣀⣾⢿⣿⣿⣿⣿⣿⣿⡿⣿⣷⣿⣿⣿⡇⣿⣿⡇⠀⠀⢸⣿⣿⡟⢿⣿⣿⣿⣿⣿⣣⣿⠁⣿⣀⣤⡿⠀⢀⣿⣿⣿⣿⣿⡇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠻⣿⠛⠉⠀⠈⣿⠛⢽⣿⢻⣿⣿⢿⣿⣿⣿⡇⣿⠿⣶⣶⣚⣧⣿⣿⡇⠀⠀⣸⣿⣿⣿⣄⣈⢿⣿⢿⣷⣿⣿⠀⠉⠉⠀⠀⠀⠘⡇⣿⣿⣿⣿⡇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⡀⣷⡆⠀⠀⠀⠸⣧⣻⣿⢸⣿⣿⡿⢿⣾⣻⡇⣿⣿⣿⣿⣿⣿⣿⠿⠷⠾⠛⠛⠿⢿⣿⣿⣿⣄⣿⠿⠋⢸⣿⠀⠀⠀⠀⠀⠀⠀⡇⣿⣿⣿⣿⣿⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⡇⣿⡇⠀⠀⠀⠀⣿⣿⣿⡾⢿⣿⣿⣿⣿⡶⠷⠾⠛⠛⠉⠁⢀⣠⠤⠴⠒⡆⢠⠀⢰⡉⠻⣿⣽⡏⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⣿⡿⣿⣿⣿⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣧⣿⠿⢀⣀⣤⣴⣿⣿⣿⡷⠾⠛⠋⠉⢀⣀⣠⠤⠴⠒⠻⡆⢸⠀⠀⢀⡠⠇⠸⡄⠈⣇⠀⠈⡻⢦⡀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⣿⣧⡘⠿⢻⡆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣆⣿⣿⣿⣿⣿⡿⠛⣉⣀⡀⣠⠴⠒⠋⠉⠁⠀⠀⠀⠀⠀⡇⢸⣠⠴⣫⡄⠀⠀⡇⠀⢹⠀⠀⣿⠦⢿⡀⢸⡇⠀⠀⣀⣤⣤⣿⠀⡇⣿⣿⣿⣆⢸⡇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⢿⡟⣽⣿⠀⣏⠁⠀⡇⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⠀⡖⣻⠋⠀⠀⠈⢻⠀⢈⡇⠀⠸⡄⠘⣧⢸⡇⠀⢸⣷⣾⣿⠏⠀⡇⣿⣿⣿⣿⢸⡇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⠏⠛⠋⢡⣿⠀⠸⣿⣟⡃⣇⠀⠀⠀⠀⠀⣀⣠⡤⠶⠒⠋⠀⠛⠁⠀⣀⣤⣶⣿⣿⣿⣿⣷⣤⡈⠁⢻⡞⣿⠀⠈⠻⣴⠏⠀⠀⠿⢹⣿⣎⢻⣿⡇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⡟⠀⠀⢀⡿⣿⠀⠀⠈⠳⡇⠻⠤⠶⠚⠋⠉⠁⠀⠀⠀⠀⠀⣀⣤⣶⣿⣿⣿⣿⣿⠿⠛⠻⣿⣿⣿⣷⣜⣷⣿⠀⠀⢀⣀⣤⣤⣶⣾⣶⣿⣿⠃⢸⡇
⠀⠀⠀⠀⠀⠀⣀⣤⡶⠶⠖⠚⢛⠛⠳⢶⣼⡟⠀⠀⢀⣼⣹⣿⢀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⢀⣀⣠⡤⢤⣾⣿⣿⣿⡿⠿⠛⠉⠹⡇⠀⠀⣿⣿⣟⢿⣿⣿⠹⣶⣿⡿⠛⠻⣏⠀⠉⠉⡛⣿⡿⣾⡇
⠀⠀⠀⢀⣴⠞⠋⢰⡇⢰⣿⢻⢻⢻⢶⣦⠙⣷⡀⠀⣸⢧⠟⢿⣿⣿⣿⣷⣶⣶⣤⣴⣲⡾⠿⠟⠒⠒⠛⡇⠙⣿⠉⠀⢧⠀⠀⠀⠀⣧⠀⠀⢸⣿⣿⡎⣿⠁⢀⣼⣏⢀⣠⣤⣸⣶⠀⠀⣿⣿⣿⠛⠁
⠀⠀⠀⣾⠃⠀⣠⡬⣤⣼⣛⠾⣼⣞⡾⡟⠀⠘⣧⣠⣏⡞⠀⠈⠻⣿⡏⢹⡟⠛⠻⣿⠁⠀⠀⠀⠀⠀⠀⣇⠀⣿⠀⠀⢸⡄⠀⠀⠀⢸⠀⠀⠘⣿⣿⣇⣿⣴⡞⢣⣽⣿⣿⣿⣿⣿⠀⠀⣿⣿⡟⠀⠀
⠀⠀⠀⣿⡶⣿⣿⣸⣿⣿⣿⠿⠷⠾⢽⣅⡲⠶⢻⣿⣼⢁⣠⣤⣶⣿⣿⠘⡇⠀⠀⢻⡆⠀⠀⠀⠀⠀⢀⣸⡀⢹⡇⠀⠈⡇⠀⠀⠀⠈⡇⠀⠀⢿⣿⣿⢹⣿⣤⣿⣿⣿⣿⡿⢿⣟⡀⠀⣿⣿⡇⠀⠀
⠀⠀⠀⠈⠛⠿⢯⣜⣿⠏⠀⠀⠀⢀⡿⣨⣿⣶⣤⣿⣷⣯⣿⣿⣿⣿⣿⠀⡇⠀⠀⠐⡿⣦⣰⣒⣶⣿⣿⣿⣷⣾⣇⠀⠀⢻⠀⠀⠀⠀⢷⠀⠀⢸⣿⣿⣾⣿⣸⣿⡏⢠⠟⣠⣿⣿⣿⣦⡈⢹⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢸⡟⣾⠄⠀⠀⣸⡇⣿⣿⣿⠟⠋⠛⢿⣿⣿⣿⣿⣿⡄⢻⠀⠀⠀⡇⠈⠙⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⢸⡆⠀⠀⠀⢸⡄⠀⠀⣿⣿⣇⣿⠛⠛⠻⣿⣺⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣼⢧⡇⠀⠀⠀⣿⢸⣿⣿⡿⢦⣴⣿⣿⣷⡿⣿⡿⣿⡇⢸⡄⠀⠀⢹⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⣇⠀⠀⠀⠀⣇⠀⠀⢸⣿⣟⢿⡀⠀⠀⠈⠉⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣿⣨⡧⠤⠤⢤⣇⡾⣿⣿⣠⣿⣿⣿⣿⣿⣿⣽⣿⣿⣷⠀⣇⠀⠀⢸⠀⠀⢸⢻⣿⣿⣿⣿⡇⣿⣿⠀⠀⢹⡄⠀⠀⢀⣸⠀⠀⠸⣿⣿⣼⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⡿⣧⣤⠶⠦⣼⣿⣿⣿⡏⠈⣿⣿⢿⣿⣿⣿⣏⠉⢹⣿⡀⢻⠀⠀⠘⡇⠀⠸⡄⠙⢿⣿⣿⠇⣿⣿⡄⠀⠈⠓⠒⠋⠉⠀⠀⠀⠀⢿⠹⣯⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣸⣿⢃⡏⠀⠀⢻⣿⣿⣽⣿⣦⠘⣿⣿⣿⣿⣿⢻⣿⣾⣿⡇⠘⡇⠀⠀⣇⠀⠀⣇⠀⠀⠙⢿⡇⣿⢸⣧⠀⠀⠀⠀⡴⠒⢶⠀⠀⠀⠘⣆⠀⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⡿⡅⣸⢁⣄⡄⣾⣿⢿⣿⠿⣿⣿⢻⣿⣿⣟⣿⣸⣻⡿⣿⣧⠀⠙⠒⠛⠛⠀⠀⢿⣿⣄⠀⠀⠀⣿⠈⣿⡄⠀⠀⠀⡇⠀⠘⡇⠀⠀⠀⢿⣦⢸⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⣧⡇⣿⣼⣿⠃⣿⣿⣾⣿⣷⣤⡿⠿⢿⣿⣿⣇⣿⡟⠋⠀⣿⡀⠀⣴⠲⡆⠀⠀⠸⣿⣿⣦⠀⠀⢸⡀⢹⣧⠀⠀⠀⣇⠀⠀⢹⠀⠀⠀⠸⣿⡟⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀  ⢽⡿⣷⠏⠛⠿⢠⣿⣿⣿⣿⢿⣯⡇⠀⠀⠈⠁⠀⠀⠀⠀⠀⢸⣇⠀⢻⠀⢳⠀⠀⠀⣿⣿⣿⣷⣾⢸⡇⠈⣿⡀⠀⠀⢸⠀⠀⠈⡇⠀⠀⢀⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
░██████████                                              ░████                                                                    
    ░██                                                 ░██                                                                       
    ░██    ░██░████  ░██████   ░████████   ░███████  ░████████  ░███████  ░██░████ ░█████████████   ░███████  ░██░████  ░███████  
    ░██    ░███           ░██  ░██    ░██ ░██           ░██    ░██    ░██ ░███     ░██   ░██   ░██ ░██    ░██ ░███     ░██        
    ░██    ░██       ░███████  ░██    ░██  ░███████     ░██    ░██    ░██ ░██      ░██   ░██   ░██ ░█████████ ░██       ░███████  
    ░██    ░██      ░██   ░██  ░██    ░██        ░██    ░██    ░██    ░██ ░██      ░██   ░██   ░██ ░██        ░██             ░██ 
    ░██    ░██       ░█████░██ ░██    ░██  ░███████     ░██     ░███████  ░██      ░██   ░██   ░██  ░███████  ░██       ░███████  
                                                                                                                                                                     
'''

def intro():
    '''Display Optimus Prime ASCII art intro.'''
    import shutil, getpass

    print('\033c', end = '')
    width = shutil.get_terminal_size().columns
    for line in OPTIMUS.splitlines():
        print(line.center(width))
    print()
    print('-- press ENTER to continue --'.center(width), end = '', flush = True)
    getpass.getpass('', stream = None)
    print('\033c', end = '')



if __name__ == '__main__':

    intro()

    # initialize trainer
    trainer = Trainer(
        path = Path.cwd().parent / 'data' / 'training' / '1M_train.tsv',
        epochs = 10,
        batch_size = 1024,
        lr = 1e-4,
        checkpoint_dir = Path('checkpoints'),
        checkpoint_interval = 1,
        logs = 'runs/optimus'
    )

    trainer.setup()             # setup
    trainer.load_checkpoint()   # load checkpoint if exists
    trainer.train()             # train
    trainer.cleanup()           # cleanup