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
- [Project Introduction](#project-introduction)
  - [Goals](#goals)
  - [Dataset](#dataset)
  - [Model](#model)
  - [Training](#training)
  - [Evaluation](#evaluation)
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

2. Track the large password and frequency files:

```bash
git lfs track "data/raw/plaintext/1mil_pw.txt"
git lfs track "data/raw/plaintext/char_frequencies.txt"
git lfs track "data/raw/yaml/1mil_pw.yaml"
git lfs track "data/raw/yaml/char_freq.yaml"
```

3. Commit the `.gitattributes` file that Git LFS generates:

```bash
git add .gitattributes
git commit -m "configure git-lfs tracking for large dataset files"
```
4. Once the files are tracked, you can `git add <filename>` as normal.

```bash
git add raw/plaintext/1mil_pw.txt
git add raw/yaml/1mil_pw.yaml
```
### Cloning and Pulling LFS Files

When someone clones this repository, Git will only fetch lightweight pointers to the large files.  
To download the actual file contents, run:

```bash
git lfs pull
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

- __Architecture:__ Transformer encoder–decoder (PyTorch `nn.Transformer` or Hugging Face configs).  
- __Input:__ Fixed-length hash (e.g., 32 hex characters).  
- __Output:__ Variable-length password sequence.  
- __Scalable parameters:__  
  - Hidden size (`d_model`)  
  - Number of encoder/decoder layers  
  - Attention heads (`nhead`)  
  - Feedforward layer size (`dim_feedforward`)  

Example PyTorch model instantiation:

```python
    import torch.nn as nn

    model = nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048
    )
```

## <font color='#ffb733'>Training</font>

- __Objective:__ supervised learning on `(hash → password)` pairs.  
- __Loss function:__ cross-entropy over output characters.  
- __Optimization:__ Adam or AdamW.  
- __Batching:__ input/output sequences padded to max length per batch.  
- __Hardware:__ train with scalable configurations depending on GPU memory.  

## <font color='#ffb733'>Evaluation</font>

Metrics used to assess model performance:

1. __Exact Match Accuracy__  
```
    accuracy = (number of correct predictions) / (total predictions)
```

2. __Top-k Accuracy__ (k=5,10)  
```
    top_k_accuracy = (# predictions where true password is in top-k guesses) / (total predictions)
```

3. __Edit Distance (Levenshtein)__  
    - Counts how many insertions, deletions, or substitutions needed to match prediction to truth.

4. __Jaccard Similarity__  
```
    J(A, B) = |A ∩ B| / |A ∪ B|
```

where A and B are the sets of characters in the true and predicted password.

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