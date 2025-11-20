'''
Training Profiler for OptimusPrime

This script profiles your training loop to identify bottlenecks:
- Data loading time
- Forward pass time
- Backward pass time
- GPU utilization

Usage:
    python profile_training.py
'''

import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# local imports
from model import OptimusPrime
from optimizer import AdamWarlock
from data import Bumblebee, collate_batch, ALLOWED_PW_CHARS as PW_VOCAB, PAD_ID, SOS_ID, EOS_ID

# Colors
GR = '\033[32m'
BU = '\033[34m'
RD = '\033[31m'
CY = '\033[36m'
YW = '\033[33m'
MG = '\033[35m'
X  = '\033[0m'


def profile_training(num_batches: int = 100):
    '''
    Profile training loop for a small number of batches.

    Parameters:
    -----------
    num_batches : int
        Number of batches to profile (default: 100)
    '''
    print(f'\n[{BU}PROFILING TRAINING{X}]')
    print(f' [batches to profile]: {num_batches}')

    # Setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAIN_DATA_PATH = Path(__file__).parent.parent / 'data' / 'training' / '10MIL_train.tsv'

    # Load small subset for profiling
    print(f'\n[{CY}loading data{X}]')
    dataset = Bumblebee(TRAIN_DATA_PATH, sample_fraction=0.01)  # 1% for quick profiling
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_batch)

    # Create model
    print(f'[{CY}creating model{X}]')
    model = OptimusPrime(
        vocab_size=256,
        pw_vocab_size=len(PW_VOCAB),
        pad_id=PAD_ID,
        sos_id=SOS_ID,
        eos_id=EOS_ID,
        d_model=512,
        n_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        label_smoothing=0.06
    ).to(DEVICE)

    # Create optimizer
    optimizer = AdamWarlock(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=0,
        total_steps=len(dataloader),
        schedule='cosine'
    )

    # Mixed precision
    use_amp = DEVICE.startswith('cuda') and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    # Profiling metrics
    times = {
        'data_loading': [],
        'forward_pass': [],
        'backward_pass': [],
        'optimizer_step': [],
        'total_batch': []
    }

    print(f'\n[{GR}starting profiling{X}]')
    model.train()

    batch_iter = iter(dataloader)
    for i in range(min(num_batches, len(dataloader))):
        batch_start = time.time()

        # Data loading
        data_start = time.time()
        batch = next(batch_iter)
        hashes = batch['hash'].to(DEVICE, non_blocking=True)
        pw = batch['password'].to(DEVICE, non_blocking=True)
        data_time = time.time() - data_start

        # Forward pass
        forward_start = time.time()
        with autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
            logits = model(hashes, pw)
            loss = model.compute_loss(logits, pw)
        forward_time = time.time() - forward_start

        # Backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        backward_time = time.time() - backward_start

        # Optimizer step
        optim_start = time.time()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if optimizer.scheduler is not None:
            optimizer.scheduler.step()
        optim_time = time.time() - optim_start

        batch_time = time.time() - batch_start

        # Record times
        times['data_loading'].append(data_time)
        times['forward_pass'].append(forward_time)
        times['backward_pass'].append(backward_time)
        times['optimizer_step'].append(optim_time)
        times['total_batch'].append(batch_time)

        if (i + 1) % 20 == 0:
            print(f' [{i+1}/{num_batches}] batch_time: {batch_time*1000:.1f}ms')

    # Calculate averages
    print(f'\n[{GR}PROFILING RESULTS{X}]')
    print(f'{"="*60}')
    print(f'Average times per batch (ms):')
    print(f'{"="*60}')

    for key, values in times.items():
        avg_time = sum(values) / len(values) * 1000  # convert to ms
        percentage = (sum(values) / sum(times['total_batch'])) * 100
        print(f'  {key:20s}: {avg_time:6.2f} ms  ({percentage:5.1f}%)')

    print(f'{"="*60}')

    # GPU stats (if available)
    if torch.cuda.is_available():
        print(f'\n[{CY}GPU STATS{X}]')
        print(f'  Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
        print(f'  Current memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
        print(f'  Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')

    # Recommendations
    print(f'\n[{YW}RECOMMENDATIONS{X}]')
    data_pct = (sum(times['data_loading']) / sum(times['total_batch'])) * 100
    if data_pct > 20:
        print(f'  {RD}⚠ {X} Data loading takes {data_pct:.1f}% of time')
        print(f'     → Consider using num_workers > 0 in DataLoader')
        print(f'     → Consider pin_memory=True for faster GPU transfers')

    forward_pct = (sum(times['forward_pass']) / sum(times['total_batch'])) * 100
    backward_pct = (sum(times['backward_pass']) / sum(times['total_batch'])) * 100
    compute_pct = forward_pct + backward_pct

    if compute_pct < 60:
        print(f'  {YW}ℹ {X} Compute time is only {compute_pct:.1f}% of total')
        print(f'     → Training is not fully GPU-bound')
        print(f'     → Data loading or CPU overhead may be limiting speed')

    # Throughput
    avg_batch_time = sum(times['total_batch']) / len(times['total_batch'])
    samples_per_sec = 256 / avg_batch_time  # assuming batch_size=256
    print(f'\n[{GR}THROUGHPUT{X}]')
    print(f'  Samples/second: {samples_per_sec:.0f}')
    print(f'  Batches/second: {1/avg_batch_time:.2f}')

    # Estimate time for full epoch
    batches_per_epoch = len(dataloader)
    estimated_epoch_time = (batches_per_epoch * avg_batch_time) / 60
    print(f'\n[{MG}ESTIMATES{X}]')
    print(f'  Time per epoch (1% sample): {estimated_epoch_time:.2f} minutes')
    print(f'  Time per epoch (100% data): ~{estimated_epoch_time * 100:.1f} minutes')


if __name__ == '__main__':
    profile_training(num_batches=100)
