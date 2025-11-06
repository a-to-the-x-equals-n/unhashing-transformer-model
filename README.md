# <font color='#ffb733'>Password Hash Inversion with Seq2Seq Transformers</font> <!-- omit in toc -->

_Given that hashing functions are deterministic—each input consistently maps to a single output—this raises the question of whether a model could leverage subtle statistical patterns within these mappings to reconstruct or approximate the original plaintexts from their hashes._

_A custom Transformer encoder–decoder trained from scratch is the cleanest scientific experiment. It strips away irrelevant pretraining and directly asks: “Does the MD5 transformation leak enough statistical structure that a neural seq2seq model can learn to reverse it?”_

## <font color='#ffb733'>Contents</font> <!-- omit in toc -->

- [Requirements \& Setup](#requirements--setup)
    - [Cloning the Repository](#cloning-the-repository)
      - [(_Optional_) Virtual Environment](#optional-virtual-environment)
    - [Python Dependencies](#python-dependencies)
  - [Large File Tracking (Git LFS)](#large-file-tracking-git-lfs)
    - [Setup](#setup)
    - [Cloning and Pulling LFS Files](#cloning-and-pulling-lfs-files)
- [Running the Model](#running-the-model)
  - [Quick Start](#quick-start)
  - [TensorBoard Monitoring](#tensorboard-monitoring)
  - [Checkpoints](#checkpoints)
- [Project Introduction](#project-introduction)
  - [Goals](#goals)
  - [Dataset](#dataset)
  - [Model](#model)
    - [Architecture Components:](#architecture-components)
    - [Scalable Parameters:](#scalable-parameters)
  - [Training](#training)
    - [Training Features:](#training-features)
    - [Key Components:](#key-components)
    - [Example Training Configuration:](#example-training-configuration)
  - [Evaluation](#evaluation)
    - [Evaluation Metrics:](#evaluation-metrics)
    - [Example Evaluation:](#example-evaluation)
  - [Resources](#resources)
    - [Supplemental Information](#supplemental-information)
    - [Python Docs](#python-docs)
    - [Sources](#sources)
  - [Disclaimer](#disclaimer)

# <font color = '#ffb733'>Requirements & Setup</font>

This project depends on __Python__ and a few external libraries.

This project has been tested on:  
- __`WSL 2.6`__  
- __`Ubuntu 22.04`__  
- __`Python 3.10`__

### <font color = '#ffb733'>Cloning the Repository</font>

First, clone this repository and move into its directory:

```bash
git clone https://github.com/a-to-the-x-equals-n/unhashing-transformer-model.git
cd unhashing-transformer-model
```

#### <font color = '#ffb733'>(_Optional_) Virtual Environment</font>

_Sometimes recommended so you don't disturb your local setup._

Create virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

- On Linux/WSL/macOS:

  ```bash
  source .venv/bin/activate
  ```

- On Windows (PowerShell):

  ```powershell
  .venv\Scripts\Activate
  ```

When you’re done, deactivate with:

```bash
deactivate
```

### <font color = '#ffb733'>Python Dependencies</font>

CPU-only install (default):

```bash
pip install -r requirements.txt
```

GPU install (CUDA-enabled PyTorch):  

>__NOTE__: _If installed on a machine w/out an NVIDIA GPU, PyTorch will just fall back to CPU mode at runtime. But you're still left with the "dead weight" from the heavier GPU install._

```bash
pip install -r requirements-gpu.txt
```

## <font color = '#ffb733'>Large File Tracking (Git LFS)</font>

This project uses [Git Large File Storage (LFS)](https://git-lfs.github.com/) to handle very large datasets that should not be committed directly into the repository.

### Setup

1. Install Git LFS if you haven’t already:

```bash
git lfs install
```

2. Track the large dataset files and model checkpoints:

```bash
# Source data (YAML files)
git lfs track "data/source_data/dirty/*.yaml"
git lfs track "data/source_data/clean/*.yaml"

# Training data
git lfs track "data/training/1M_train.tsv"
git lfs track "data/training/1M_train.json"
git lfs track "data/training/shards/*.tsv"

# Model checkpoints
git lfs track "*.pt"
```

**Note**:
- The `*.yaml` patterns track all YAML source files (password lists and character frequencies)
- The `*.pt` pattern tracks all PyTorch model checkpoint files
- Evaluation data (`data/eval/1K_eval.tsv`) is small enough to be tracked normally

3. Commit the `.gitattributes` file that Git LFS generates:

```bash
git add .gitattributes
git commit -m "configure git-lfs tracking for large dataset files"
```
4. Once the files are tracked, you can `git add <filename>` as normal.

```bash
git add data/source_data/dirty/1mil_pw.yaml
git add data/training/1M_train.tsv
git add model/checkpoints/*.pt
```
### Cloning and Pulling LFS Files

When someone clones this repository, Git will only fetch lightweight pointers to the large files.  
To download the actual file contents, run:

```bash
git lfs pull
```

# <font color ='#ffb733'>Running the Model</font>

The model is trained and evaluated using the Jupyter notebook `model/run.ipynb`. This notebook provides a complete workflow from data loading through evaluation.

## <font color='#ffb733'>Quick Start</font>

1. **Navigate to the model directory:**

```bash
cd model
```

2. Open `run.ipynb` with preferred Jupyter Notebook environment.


3. **Run the notebook cells sequentially** (or use "Run All"):
   - **Configuration**: Set device (CPU/GPU), paths, hyperparameters, and model architecture
   - **Dataset Loading**: Load hash-password pairs from TSV files
   - **Model Creation**: Initialize the OptimusPrime transformer
   - **Training**: Run full training loop with TensorBoard logging and checkpointing
   - **Evaluation**: Test on held-out data with multiple similarity metrics
   - **Cleanup**: Close TensorBoard writer and display summary

## <font color='#ffb733'>TensorBoard Monitoring</font>

During training, metrics are logged to TensorBoard. To monitor training in real-time:

```bash
tensorboard --logdir=runs/optimus
```

Then navigate to `http://localhost:6006` in your browser.

Logged metrics include:
- **Loss/train**: Training loss per batch
- **Gradient/norm**: Gradient norms for monitoring training stability
- **Time/batch**: Batch processing time
- **Eval/loss**: Evaluation loss
- **Eval/exact_match**: Perfect password matches
- **Eval/char_similarity**: Positional character matching
- **Eval/levenshtein**: Normalized edit distance
- **Eval/jaccard**: Character set overlap

## <font color='#ffb733'>Checkpoints</font>

Model checkpoints are automatically saved to `model/checkpoints/` during training:
- `checkpoint_epoch_N.pt`: Saved every N epochs (configurable via `checkpoint_interval`)
- `best_model.pt`: Best model based on validation loss

To disable checkpoint saving/loading (useful for testing):

```python
trainer = Trainer(
    # ... other parameters ...
    save=False,  # Disable checkpoint saving
    load=False   # Disable checkpoint loading
)
```

# <font color ='#ffb733'>Project Introduction</font>

This project explores whether weak password hashing algorithms (e.g., MD5) preserve statistical patterns that can be learned by machine learning models. The task is framed as a __sequence-to-sequence problem__: given a hash (input sequence), predict the original password (output sequence).  

A __Transformer encoder–decoder model__ implemented in PyTorch is used, trained from scratch with randomly initialized weights. The architecture is __scalable__ (layers, hidden size, attention heads adjustable) to allow experiments across different model capacities depending on available hardware.

## <font color='#ffb733'>Goals</font>

- Build a dataset of `(password, hash)` pairs using weak hash functions (MD5 as primary, with optional comparisons to SHA-1, CRC32, or MurmurHash).  
- Train supervised seq2seq models to predict plaintext passwords from hashes.  
- Measure model performance using multiple metrics:  
  - Exact Match Accuracy  
  - Top-k Accuracy (e.g., top-5, top-10)  
  - Edit Distance (Levenshtein)  
  - Optional: Jaccard similarity (character set overlap)  
- Analyze whether larger models show improved generalization and whether any patterns in hash outputs can be exploited.

## <font color='#ffb733'>Dataset</font>

- Base source: cleaned/common password lists (e.g., RockYou, SecLists, or Kaggle-provided password datasets).  
- Preprocessing steps:
  - Deduplicate entries.  
  - Balance character composition (letters, digits, symbols).  
  - Generate hashes using Python’s `hashlib`.  

Example (MD5 hash generation in Python):

```python
    import hashlib

    pw = "password123"
    h = hashlib.md5(pw.encode("utf-8")).hexdigest()
    print(h)  # e.g., 482c811da5d5b4bc6d497ffa98491e38
```

The resulting dataset is stored as a CSV / TXT / JSON / YAML with two columns: `password, hash`.

## <font color='#ffb733'>Model</font>

The **OptimusPrime** model is a custom Transformer encoder-decoder architecture built from scratch in PyTorch.

### Architecture Components:

1. **Embedding Layers**:
   - Hash byte embedding (vocab size: 257 = 256 byte values + 1 padding)
   - Password character embedding with special tokens: `<PAD>`, `<SOS>`, `<EOS>`

2. **Transformer Encoder**:
   - Processes fixed-length hash sequences (16 bytes for MD5)
   - Multi-head self-attention to learn hash byte relationships
   - Stacked encoder layers with feedforward networks

3. **Nonlinear Encoder Projection**:
   - MLP applied after encoder to enrich representations
   - Learns higher-order statistical dependencies

4. **Transformer Decoder**:
   - Generates variable-length password sequences
   - Cross-attention to encoded hash representation
   - Causal masking for autoregressive generation

5. **Deep Output Projection Head**:
   - Multi-layer MLP transforms decoder outputs to vocabulary logits
   - Prevents reliance on shallow linear mappings

### Scalable Parameters:

- `vocab_size`: Hash byte vocabulary (default: 257)
- `pw_vocab_size`: Password character vocabulary (71 chars including special tokens)
- `d_model`: Hidden dimension size (default: 256)
- `n_heads`: Number of attention heads (default: 8)
- `num_layers`: Encoder and decoder depth (default: 4)
- `ff_dim`: Feedforward dimension (default: 512)
- `dropout`: Regularization dropout rate (default: 0.1)

Example model instantiation:

```python
from model import OptimusPrime
from data import ALLOWED_PW_CHARS, PAD_ID, SOS_ID, EOS_ID

model = OptimusPrime(
    vocab_size=257,
    pw_vocab_size=len(ALLOWED_PW_CHARS),
    pad_id=PAD_ID,
    sos_id=SOS_ID,
    eos_id=EOS_ID,
    d_model=256,
    n_heads=8,
    num_layers=4,
    ff_dim=512,
    dropout=0.1
)
```

## <font color='#ffb733'>Training</font>

The **Trainer** class orchestrates the complete training workflow with built-in checkpointing, TensorBoard logging, and gradient monitoring.

### Training Features:

- **Objective**: Supervised sequence-to-sequence learning on `(hash → password)` pairs
- **Loss Function**: Cross-entropy with padding mask (ignores `<PAD>` tokens)
- **Teacher Forcing**: Decoder receives ground-truth tokens during training
- **Causal Masking**: Prevents decoder from attending to future tokens
- **Optimization**: Adam optimizer (configurable learning rate)
- **Batch Processing**: Variable-length sequences padded dynamically via custom collate function

### Key Components:

1. **Automatic Checkpointing**:
   - Saves model state every N epochs
   - Tracks best model based on validation loss
   - Resume training from last checkpoint

2. **TensorBoard Integration**:
   - Real-time loss monitoring
   - Gradient norm tracking
   - Batch timing metrics

3. **Progress Tracking**:
   - tqdm progress bars with loss, samples/sec, and time estimates
   - Epoch summaries with average loss

### Example Training Configuration:

```python
from trainer import Trainer
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=1e-4)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=dataloader,
    device='cuda',
    epochs=10,
    checkpoint_dir=Path('checkpoints'),
    checkpoint_interval=1,
    logs='runs/optimus',
    save=True,  # Enable checkpoint saving
    load=True   # Resume from checkpoint if exists
)

trainer.setup()
trainer.load_checkpoint()
trainer.train()
```  

## <font color='#ffb733'>Evaluation</font>

The **Trainer.eval()** method provides comprehensive model evaluation with multiple similarity metrics and TensorBoard logging.

### Evaluation Metrics:

1. **Exact Match Accuracy**
   Percentage of predictions that perfectly match the ground truth password.
   ```
   accuracy = (number of correct predictions) / (total predictions)
   ```

2. **Character Similarity**
   Positional character matching - measures how many characters match at the same position.
   ```
   char_sim = Σ(pred[i] == truth[i]) / max(len(pred), len(truth))
   ```

3. **Levenshtein Distance (Edit Distance)**
   Normalized edit distance - minimum insertions, deletions, or substitutions needed.
   ```
   levenshtein = 1.0 - (edit_distance / max_length)
   ```
   Higher values indicate closer matches (1.0 = perfect match, 0.0 = completely different).

4. **Jaccard Similarity**
   Character set overlap - measures similarity of character sets regardless of position.
   ```
   jaccard = |A ∩ B| / |A ∪ B|
   ```
   where A and B are the sets of characters in predicted and true passwords.

### Example Evaluation:

```python
from torch.utils.data import DataLoader
from data import Bumblebee, collate_batch

# Load evaluation dataset
eval_dataset = Bumblebee('data/eval/eval.tsv')
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collate_batch
)

# Run evaluation with TensorBoard logging
results = trainer.eval(eval_dataloader, step=trainer.start_epoch + trainer.epochs)

print(f"Loss: {results['loss']:.4f}")
print(f"Exact Match: {results['exact_match']:.4f}")
print(f"Char Similarity: {results['char_similarity']:.4f}")
print(f"Levenshtein: {results['levenshtein']:.4f}")
print(f"Jaccard: {results['jaccard']:.4f}")
```

All evaluation metrics are automatically logged to TensorBoard under the `Eval/` namespace for visualization.

## <font color='#ffb733'>Resources</font>

### <font color = '#ffb733'>Supplemental Information</font>

[Transformer Model From Scratch - YT](https://www.youtube.com/watch?v=kCc8FmEb1nY)  

### <font color = '#ffb733'>Python Docs</font>

[Pytorch Transformer Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)    
[Python's Hashlib](https://docs.python.org/3/library/hashlib.html#)

### <font color = '#ffb733'>Sources</font>

[Top 1 Million Passwords From Data Dumps](https://github.com/danielmiessler/SecLists/blob/master/Passwords/Common-Credentials/Pwdb_top-1000000.txt)
[Attention is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## <font color='#ffb733'>Disclaimer</font>

This project is for __academic and research purposes only__. No live password dumps or sensitive user data are used. All datasets are sourced from public, cleaned lists (e.g., RockYou on Kaggle, SecLists) or synthetically generated.