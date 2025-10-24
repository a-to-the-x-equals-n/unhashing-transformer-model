
import torch
from torch.optim import Adam

# TODO: setup imports / or reorganize for:
#   dataset, collate_batch, and OptimusPrime model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = OptimusPrime(
    vocab_size = 257,
    pw_vocab_size = len(dataset._ALLOWED_PW_CHARS),
    pad_id = dataset.pad_id
).to(device)

optimizer = Adam(model.parameters(), lr = 1e-4)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in loader:
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

    print(f'Epoch {epoch+1} | Avg loss: {total_loss / len(loader):.4f}')
