
import torch
from torch.optim import Adam
from optimus_prime import OptimusPrime
from data import HashPwDataset, collate_batch
from torch.utils.data import DataLoader
from pathlib import Path


EPOCHS = 1
shard_path = Path.cwd().parent.parent / 'project' / 'data' / 'training' / 'shards' / 'toy_shard.tsv'
dataset = HashPwDataset(shard_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- BUILD MODELS --
dloader = DataLoader(
    dataset,
    batch_size = 8,
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

    for batch in dloader:
        # move data to GPU/CPU
        hashes = batch['hash'].to(device)
        pw = batch['password'].to(device)

        # forward pass
        logits = model(hashes, pw)

        # compute loss (ignoring padding)
        loss = model.compute_loss(logits, pw)

        # reset old gradients
        optimizer.zero_grad()

        # compute new gradients via backprop
        loss.backward()

        # optional: clip gradient norms to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        # apply weight updates
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1} | Avg loss: {total_loss / len(dloader):.4f}')
