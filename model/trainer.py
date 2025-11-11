
import torch
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
    - Training loop with progress tracking
    - Checkpoint saving/loading
    - TensorBoard logging
    '''
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        epochs: int = 1,
        checkpoint_dir: Path = Path('checkpoints'),
        checkpoint_interval: int = 1,
        logs: str = 'runs/optimus',
        save: bool = True,
        load_mode: str = 'latest',
        max_checkpoints: int = 5,
        eval_dataloader: torch.utils.data.DataLoader = None
    ):
        '''
        Initialize trainer with model, optimizer, and dataloader.

        Parameters:
        -----------
        model : torch.nn.Module
            The model to train

        optimizer : torch.optim.Optimizer
            Optimizer for training

        dataloader : torch.utils.data.DataLoader
            DataLoader with training data

        device : str
            Device to train on ('cuda' or 'cpu')

        epochs : int
            Number of training epochs

        checkpoint_dir : Path
            Directory to save checkpoints

        checkpoint_interval : int
            Save checkpoint every N epochs

        logs : str
            TensorBoard log directory

        save : bool
            Whether to save checkpoints (default: True)

        load_mode : str
            How to load checkpoints: 'latest' (most recent checkpoint), 'best' (lowest loss), or 'none' (start fresh)
            Default: 'latest'

        max_checkpoints : int
            Maximum number of checkpoint files to keep (oldest are deleted). Does not affect best_model.pt.
            Default: 5

        eval_dataloader : torch.utils.data.DataLoader, optional
            DataLoader with evaluation data. If provided, eval will run automatically every 10 epochs during training.
        '''

        # passed components
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device

        # training configuration
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.logs = logs
        self.save = save
        self.load_mode = load_mode
        self.max_checkpoints = max_checkpoints

        # training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.writer = None


    def setup(self):
        '''
        Setup training environment.

            moves model to device
            initializes TensorBoard SummaryWriter
            creates checkpoint directory if it doesn't exist
        '''
        print(f'\n{BU} [SETUP]{X}')
        print(f'  [total epochs]: {self.epochs}')
        print(f'  [device]: {self.device}')

        # move model to device
        print(f'  [moving model to device]')
        self.model = self.model.to(self.device)

        # initialize TensorBoard
        print(f'  [initializing TensorBoard]')
        self.writer = SummaryWriter(log_dir = self.logs)

        # create checkpoint directory
        print(f'  [creating checkpoint directory]')
        self.checkpoint_dir.mkdir(exist_ok = True)


    def load(self, load_mode: str | None = None):
        '''
        Load checkpoint based on self.load_mode setting.

            Modes:
            - 'latest': Load most recent checkpoint_epoch_*.pt
            - 'best': Load best_model.pt (lowest loss), with epoch from latest checkpoint
            - 'none': Start fresh training from epoch 0

            When loading 'best' mode:
            - Model weights come from best_model.pt
            - Epoch number comes from the most recent checkpoint_epoch_*.pt
            - This ensures we continue from the correct epoch while using best weights

            Restores model weights, optimizer state, starting epoch, and best loss value
        '''
        self.load_mode = self.load_mode if not load_mode else load_mode
        
        if self.load_mode == 'none':
            print(f'\n{YW} [CHECKPOINT LOADING DISABLED]{X}')
            print(f'  [load_mode]: none')
            print(f'  [starting fresh training]')
            return

        # Get checkpoints and sort by epoch number (not alphabetically)
        # This prevents "epoch_9" from coming after "epoch_19" in string sorting
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))

        # MODE: 'latest' - load most recent checkpoint
        if self.load_mode == 'latest':
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                print(f'\n{YW} [LOADING LATEST CHECKPOINT]{X}')
                print(f'  [load_mode]: latest')
                print(f'  [loading]: {latest_checkpoint.name}')

                checkpoint = torch.load(latest_checkpoint, map_location = self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                print(f'  [resuming from epoch]: {self.start_epoch}')
                print(f'  [best loss so far]: {self.best_loss:.4f}')
            else:
                print(f'\n{YW} [NO CHECKPOINT FOUND]{X}')
                print(f'  [load_mode]: latest')
                print(f'  [starting fresh training]')

        # MODE: 'best' - load best model weights, but get epoch from latest checkpoint
        elif self.load_mode == 'best':
            best_model_path = self.checkpoint_dir / 'best_model.pt'

            if not best_model_path.exists():
                print(f'\n{RD} [BEST MODEL NOT FOUND]{X}')
                print(f'  [load_mode]: best')
                print(f'  [path]: {best_model_path}')
                print(f'  [starting fresh training]')
                return

            # Load best model weights
            print(f'\n{GR} [LOADING BEST MODEL]{X}')
            print(f'  [load_mode]: best')
            print(f'  [loading weights]: {best_model_path.name}')

            best_checkpoint = torch.load(best_model_path, map_location = self.device)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
            self.best_loss = best_checkpoint.get('loss', float('inf'))

            # Get epoch from latest checkpoint (for proper continuation)
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                print(f'  [getting epoch from]: {latest_checkpoint.name}')

                epoch_checkpoint = torch.load(latest_checkpoint, map_location = self.device)
                self.start_epoch = epoch_checkpoint['epoch'] + 1

                print(f'  [resuming from epoch]: {self.start_epoch}')
                print(f'  [best model loss]: {self.best_loss:.4f}')
            else:
                print(f'  {YW}[WARNING]: No checkpoint found for epoch tracking{X}')
                print(f'  [starting from epoch]: 0')
                self.start_epoch = 0

        else:
            print(f'\n{RD} [INVALID LOAD MODE]{X}')
            print(f'  [load_mode]: {self.load_mode}')
            print(f'  [valid modes]: latest, best, none')
            print(f'  [starting fresh training]')


    def load_best_model(self):
        '''
        Load the best model (lowest loss) from disk.

            loads model weights and optimizer state from best_model.pt
            useful for resuming training from the best checkpoint or for inference
            restores best_loss and start_epoch metadata

        Returns:
        --------
        bool
            True if best model was loaded successfully, False otherwise
        '''
        best_model_path = self.checkpoint_dir / 'best_model.pt'

        if not best_model_path.exists():
            print(f'\n{RD} [BEST MODEL NOT FOUND]{X}')
            print(f'  [path]: {best_model_path}')
            return False

        print(f'\n{GR} [LOADING BEST MODEL]{X}')
        print(f'  [loading]: {best_model_path.name}')

        checkpoint = torch.load(best_model_path, map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('loss', float('inf'))
        self.start_epoch = checkpoint.get('epoch', 0) + 1

        print(f'  [best loss]: {self.best_loss:.4f}')
        print(f'  [from epoch]: {checkpoint.get("epoch", "unknown")}')

        return True


    def save_checkpoint(self, epoch: int, loss: float, global_step: int):
        '''
        Save a training checkpoint to disk and cleanup old checkpoints.

            saves model state, optimizer state, and training metadata to a .pt file
            checkpoint can be used to resume training from this exact point
            automatically removes oldest checkpoints if more than max_checkpoints exist
            skips saving if self.save is False

        Parameters:
        -----------
        epoch : int
            Current epoch number (0-indexed).

        loss : float
            Final average loss for this epoch.

        global_step : int
            Global training step across all epochs (for TensorBoard continuity).
        '''
        if not self.save:
            return

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'global_step': global_step
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'    [checkpoint saved]: {checkpoint_path.name}')

        # Cleanup old checkpoints (keep only max_checkpoints most recent)
        self._cleanup_old_checkpoints()


    def _cleanup_old_checkpoints(self):
        '''
        Remove old checkpoint files to keep only the most recent N checkpoints.

            keeps the max_checkpoints most recent checkpoint_epoch_*.pt files
            does NOT delete best_model.pt
            silently handles errors (e.g., file already deleted)

        Example:
            If max_checkpoints = 5 and there are 7 checkpoints, the oldest 2 are deleted
        '''
        # Get checkpoints and sort by epoch number (not alphabetically)
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))

        # Only cleanup if we exceed the limit
        if len(checkpoints) <= self.max_checkpoints:
            return

        # Delete oldest checkpoints
        num_to_delete = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:num_to_delete]:
            try:
                checkpoint.unlink()
                print(f'    [cleanup]: removed {checkpoint.name}')
            except Exception as e:
                print(f'    {YW}[WARNING]: Could not delete {checkpoint.name}: {e}{X}')


    def save_best_model(self, epoch: int, loss: float):
        '''
        Save the best model encountered so far based on lowest loss.

            updates the best loss value and saves model state and optimizer state
                to a dedicated "best_model.pt" file
            this file always contains the model with the lowest validation loss
            skips saving if self.save is False

        Parameters:
        -----------
        epoch : int
            Epoch number where this best loss was achieved.

        loss : float
            The new best (lowest) loss value.
        '''
        previous_best = self.best_loss
        self.best_loss = loss

        if not self.save:
            print(f'{GR}    [new best]: loss = {previous_best:.4f} -> {self.best_loss:.4f} (not saved){X}')
            return

        best_model_path = self.checkpoint_dir / 'best_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss
        }, best_model_path)

        print(f'{GR}    [new best saved]: {previous_best:.4f} -> {self.best_loss:.4f}{X}')


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

        print(f'\n [epoch {epoch + 1} / {self.start_epoch + self.epochs} complete]')
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
                runs evaluation every 10 epochs if eval_dataloader is provided

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

            # run evaluation every 10 epochs
            if self.eval_dataloader is not None and (epoch + 1) % 10 == 0:
                print(f'\n{CY} [STARTING EVALUATION at epoch {epoch + 1}]{X}')
                self.eval(self.eval_dataloader, step = epoch + 1)

            print()


    def eval(self, eval_dataloader: torch.utils.data.DataLoader = None, step: int = None) -> dict:
        '''
        Evaluate the model on a dataset.

            computes loss, exact match accuracy, and similarity metrics on evaluation data
            uses greedy decoding (argmax) to generate predictions
            optionally logs metrics to TensorBoard if step is provided
            saves predictions to TSV every 10 epochs when step is divisible by 10

        Parameters:
        -----------
        eval_dataloader : torch.utils.data.DataLoader, optional
            DataLoader with evaluation data. If None, uses self.dataloader

        step : int, optional
            Global step or epoch number for TensorBoard logging. If None, metrics are not logged.

        Returns:
        --------
        dict
            Dictionary containing:
                - 'loss': average loss across all batches
                - 'exact_match': exact match accuracy (fraction of passwords predicted correctly)
                - 'char_similarity': average character-level similarity
                - 'levenshtein': average normalized Levenshtein similarity
                - 'jaccard': average Jaccard similarity
                - 'total_samples': total number of samples evaluated
        '''
        self.model.eval()
        dataloader = eval_dataloader if eval_dataloader is not None else self.dataloader

        total_loss = 0.0
        correct_predictions = 0
        total_char_sim = 0.0
        total_levenshtein = 0.0
        total_jaccard = 0.0
        total_samples = 0

        # determine if we should save predictions (every 10 epochs)
        save_predictions = step is not None and step % 10 == 0
        predictions_list = [] if save_predictions else None

        print(f'\n{BU} [EVALUATION]{X}')
        if save_predictions:
            print(f'  [saving predictions for epoch {step}]')

        with torch.no_grad():
            progress = tqdm(dataloader, desc = 'Evaluating', leave = True, unit = ' batch')

            for batch in progress:
                # move data to device
                hashes = batch['hash'].to(self.device)
                pw = batch['password'].to(self.device)

                # forward pass for loss computation (teacher forcing)
                logits = self.model(hashes, pw)
                loss = self.model.compute_loss(logits, pw)

                # accumulate loss
                total_loss += loss.item()

                # AUTOREGRESSIVE GENERATION (true inference, no teacher forcing)
                # generate predictions token-by-token using model's own outputs
                generated = self.model.generate(hashes, max_length = 32, temperature = 1.0)  # [B, T]

                # remove <SOS> token from generated sequences for comparison
                predictions = generated[:, 1:]  # [B, T-1] (skip <SOS>)

                # targets: skip <SOS> token
                targets = pw[:, 1:]  # [B, T-1]

                # align lengths 
                # (predictions may be shorter/longer than targets)
                max_len = max(predictions.size(1), targets.size(1))
                if predictions.size(1) < max_len:
                    # pad predictions with PAD tokens
                    padding = torch.full((predictions.size(0), max_len - predictions.size(1)),
                                        self.model.pad_id, dtype=torch.long, device=self.device)
                    predictions = torch.cat([predictions, padding], dim=1)
                if targets.size(1) < max_len:
                    # pad targets with PAD tokens
                    padding = torch.full((targets.size(0), max_len - targets.size(1)),
                                        self.model.pad_id, dtype=torch.long, device=self.device)
                    targets = torch.cat([targets, padding], dim=1)

                # truncate to same length 
                # (take minimum)
                min_len = min(predictions.size(1), targets.size(1))
                predictions = predictions[:, :min_len]
                targets = targets[:, :min_len]

                # exact match accuracy: all tokens must match
                matches = (predictions == targets).all(dim = 1)
                correct_predictions += matches.sum().item()

                # compute similarity metrics per sample
                batch_size = hashes.size(0)
                for i in range(batch_size):
                    pred_str = dataloader.dataset.decode(predictions[i])
                    truth_str = dataloader.dataset.decode(targets[i])

                    total_char_sim += char_similarity(pred_str, truth_str)
                    total_levenshtein += levenshtein(pred_str, truth_str)
                    total_jaccard += jaccard(pred_str, truth_str)

                    # collect predictions for saving if needed
                    if save_predictions:
                        # convert hash bytes back to hex string
                        hash_bytes = hashes[i].cpu().numpy()
                        hash_hex = ''.join(f'{byte:02x}' for byte in hash_bytes)

                        predictions_list.append({
                            'epoch': step,
                            'hash': hash_hex,
                            'ground_truth': truth_str,
                            'prediction': pred_str
                        })

                total_samples += batch_size

                # update progress bar
                current_acc = correct_predictions / total_samples
                current_char_sim = total_char_sim / total_samples
                num_batches_processed = progress.n if progress.n > 0 else 1
                progress.set_postfix_str(
                    f'loss = {total_loss / num_batches_processed:.4f}, '
                    f'exact = {current_acc:.4f}, '
                    f'char = {current_char_sim:.4f}'
                )

            progress.close()

        avg_loss = total_loss / len(dataloader)
        exact_match = correct_predictions / total_samples
        avg_char_sim = total_char_sim / total_samples
        avg_levenshtein = total_levenshtein / total_samples
        avg_jaccard = total_jaccard / total_samples

        print(f'\n [evaluation complete]')
        print(f'  [loss]: {avg_loss:.4f}')
        print(f'  [exact match]: {exact_match:.4f} ({correct_predictions}/{total_samples})')
        print(f'  [char similarity]: {avg_char_sim:.4f}')
        print(f'  [levenshtein]: {avg_levenshtein:.4f}')
        print(f'  [jaccard]: {avg_jaccard:.4f}')

        # save predictions to TSV if this is a milestone epoch
        if save_predictions and predictions_list:
            import pandas as pd
            predictions_dir = Path(__file__).parent / 'predictions'
            predictions_dir.mkdir(exist_ok = True)

            predictions_file = predictions_dir / f'predictions_ep_{step}.tsv'
            df = pd.DataFrame(predictions_list)
            df.to_csv(predictions_file, sep = '\t', index = False)

            print(f'  [predictions saved]: {predictions_file.name}')

        # log to TensorBoard if step is provided
        if step is not None and self.writer is not None:
            self.writer.add_scalar('Eval/loss', avg_loss, step)
            self.writer.add_scalar('Eval/exact_match', exact_match, step)
            self.writer.add_scalar('Eval/char_similarity', avg_char_sim, step)
            self.writer.add_scalar('Eval/levenshtein', avg_levenshtein, step)
            self.writer.add_scalar('Eval/jaccard', avg_jaccard, step)

        return {
            'loss': avg_loss,
            'exact_match': exact_match,
            'char_similarity': avg_char_sim,
            'levenshtein': avg_levenshtein,
            'jaccard': avg_jaccard,
            'total_samples': total_samples
        }


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


# ============================================================================
# Similarity Metrics
# ============================================================================

def char_similarity(pred: str, truth: str) -> float:
    '''
    Calculate character-level positional similarity between prediction and ground truth.

        compares characters at each position and returns the fraction of matching characters
        shorter string is padded with spaces to match the length of the longer string

    Parameters:
    -----------
    pred : str
        Predicted password string

    truth : str
        Ground truth password string

    Returns:
    --------
    float
        Similarity score in range [0.0, 1.0] where:
            1.0 = all characters match at corresponding positions
            0.0 = no characters match (or prediction is empty)
    '''
    if pred == '':
        return 0.0
    length = max(len(pred), len(truth))
    pred = pred.ljust(length)
    truth = truth.ljust(length)
    return sum(p == t for p, t in zip(pred, truth)) / length


def _levenshtein_helper(a: str, b: str) -> int:
    '''
    Compute raw Levenshtein edit distance between two strings using dynamic programming.

        calculates the minimum number of single-character edits (insertions, deletions, substitutions)
        required to transform string a into string b

    Parameters:
    -----------
    a : str
        First string (automatically swapped to be longer if needed)

    b : str
        Second string

    Returns:
    --------
    int
        Minimum number of edits required to transform a into b
    '''
    if len(a) < len(b):
        return _levenshtein_helper(b, a)
    if len(b) == 0:
        return len(a)

    previous_row = range(len(b) + 1)

    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein(pred: str, truth: str) -> float:
    '''
    Calculate normalized Levenshtein similarity between prediction and ground truth.

        computes edit distance and normalizes by the maximum string length
        accounts for insertions, deletions, and substitutions

    Parameters:
    -----------
    pred : str
        Predicted password string

    truth : str
        Ground truth password string

    Returns:
    --------
    float
        Similarity score in range [0.0, 1.0] where:
            1.0 = strings are identical (zero edit distance)
            0.0 = maximum edit distance (completely different)

    Notes:
    ------
    Normalized as: 1.0 - (edit_distance / max_length)
    '''
    # if both are empty, they're identical
    if len(pred) == 0 and len(truth) == 0:
        return 1.0
    # if only prediction is empty, complete failure
    if len(pred) == 0:
        return 0.0
    dist = _levenshtein_helper(pred, truth)
    max_len = max(len(pred), len(truth))
    return 1.0 - (dist / max_len)


def jaccard(pred: str, truth: str) -> float:
    '''
    Calculate Jaccard similarity coefficient based on unique character sets.

        measures the overlap of unique characters between prediction and ground truth
        ignores character order and frequency
        only considers presence/absence

    Parameters:
    -----------
    pred : str
        Predicted password string

    truth : str
        Ground truth password string

    Returns:
    --------
    float
        Similarity score in range [0.0, 1.0] where:
            1.0 = identical character sets (all unique chars match)
            0.0 = disjoint character sets (no common characters)

    Notes:
    ------
    Computed as: |intersection| / |union| of character sets
    '''
    if pred == '':
        return 0.0
    set_pred = set(pred)
    set_truth = set(truth)
    intersection = set_pred & set_truth
    union = set_pred | set_truth
    return len(intersection) / len(union) if union else 1.0


if __name__ == '__main__':
    intro()
    print(f'\n{YW}[NOTE]{X}: Use run.ipynb to train the model')
    print(f'  trainer.py is now a class that needs model/optimizer/dataloader passed to it')