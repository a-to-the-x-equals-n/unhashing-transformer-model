
import torch
from torch.optim import Adam
from optimus_prime import OptimusPrime
from data import dataset, collate_batch, dataset
from torch.utils.data import DataLoader

# TODO: setup imports / or reorganize for:
#   dataset, collate_batch, and OptimusPrime model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




model = OptimusPrime(
    vocab_size = 257,
    pw_vocab_size = len(dataset._ALLOWED_PW_CHARS), # 74
).to(device)




# build DataLoader with custom collate function
dloader = DataLoader(
    dataset,
    batch_size = 8,
    shuffle = True,
    collate_fn = collate_batch
)

optimizer = Adam(model.parameters(), lr = 1e-4)

EPOCHS = 1

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
