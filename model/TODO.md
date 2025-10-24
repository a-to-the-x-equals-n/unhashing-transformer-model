## Optuna

Implement it!

## Counting Trainable Parameters

```py
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params:,}")
```

per-layer counts

```py
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name:40s} {param.numel():,}")
```

## Saving and Loading Model Weights

Save

```py
torch.save(model.state_dict(), "optimusprime_weights.pt")
```
Load

```py
model = OptimusPrime(vocab_size = 257, pw_vocab_size = len(dataset._ALLOWED_PW_CHARS), pad_id = dataset.pad_id)
model.load_state_dict(torch.load("optimusprime_weights.pt"))
model.eval()  # switch to inference mode
```

### Saving Full Checkpoints

Save

```py
checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': total_loss
}

torch.save(checkpoint, "checkpoint_epoch_3.pt")
```

Load

```py
checkpoint = torch.load("checkpoint_epoch_3.pt")
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_epoch = checkpoint['epoch'] + 1
```