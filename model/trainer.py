
import torch
from torch.optim import Adam
from model import OptimusPrime
from data import Bumblebee, collate_batch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


EPOCHS = 1
shard_path = Path.cwd().parent / 'data' / 'training' / 'shards' / 'train_000.tsv'
full_path = Path.cwd().parent / 'data' / 'training' / '1M_train.tsv'
shard_path = full_path

print(f'  training file: {shard_path.name}')
dataset = Bumblebee(shard_path)

if torch.cuda.is_available():
    device = 'cuda'
    print(f'  cuda found / using GPU')
else:
    device = 'cpu'
    print(f'  cuda NOT found / using CPU')

# -- BUILD MODELS --
dloader = DataLoader(
    dataset,
    batch_size = 1024, 
    shuffle = True,
    collate_fn = collate_batch
)

model = OptimusPrime(
    vocab_size = 257,
    pw_vocab_size = 75,
    pad_id = 74,
    hash_pad_id = 256,
    d_model = 256,
    n_heads = 8,
    num_layers = 4, 
    ff_dim = 512, 
    dropout = 0.1
).to(device)

optimizer = Adam(model.parameters(), lr = 1e-4)

# training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    # tracks batches in current epoch
    progress = tqdm(dloader, desc = f'Epoch {epoch + 1}/{EPOCHS}', leave = False, unit = ' batch')

    for batch in progress:
        # move data to GPU/CPU
        hashes = batch['hash'].to(device)
        pw = batch['password'].to(device)

        # forward pass
        logits = model(hashes, pw)
        loss = model.compute_loss(logits, pw)   # compute loss 

        # backward pass
        optimizer.zero_grad()   # reset old gradients
        loss.backward()         # compute new gradients via backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)  # clip grad norms to stabilize training
        optimizer.step()        # update weights

        total_loss += loss.item()

        # current batch loss=
        avg_loss = total_loss / len(dloader)
        progress.set_postfix_str(f'loss = {avg_loss:.4f}')

# close progress bar
progress.close()
