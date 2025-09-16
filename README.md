# <font color='#ffb733'>Password Hash Inversion with Seq2Seq Transformers</font> <!-- omit in toc -->

## <font color='#ffb733'>Overview</font> <!-- omit in toc -->

This project explores whether weak password hashing algorithms (e.g., MD5) preserve statistical patterns that can be learned by machine learning models. The task is framed as a __sequence-to-sequence problem__: given a hash (input sequence), predict the original password (output sequence).  

A __Transformer encoder–decoder model__ implemented in PyTorch is used, trained from scratch with randomly initialized weights. The architecture is __scalable__ (layers, hidden size, attention heads adjustable) to allow experiments across different model capacities depending on available hardware.

## <font color='#ffb733'>Contents</font> <!-- omit in toc -->

- [Goals](#goals)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Why This Matters](#why-this-matters)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)
  - [Resources](#resources)

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

The resulting dataset is stored as a CSV with two columns: `password, hash`.

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

## <font color='#ffb733'>Why This Matters</font>

Hashing functions are deterministic; the same input always produces the same output. This raises an open question: could a model exploit subtle statistical regularities in weak hashes to reconstruct the original plaintexts?  

This project provides a systematic study of whether machine learning can mine patterns in weak hash outputs and what effect model capacity has on learnability.

## <font color='#ffb733'>Future Work</font>

- Compare across different hashing algorithms (MD5, SHA-1, CRC32, MurmurHash).  
- Test with synthetic password datasets designed to control character distribution and entropy.  

## <font color='#ffb733'>Disclaimer</font>

This project is for __academic and research purposes only__. No live password dumps or sensitive user data are used. All datasets are sourced from public, cleaned lists (e.g., RockYou on Kaggle, SecLists) or synthetically generated.


# <font color='#ffb733'>Resources</font>

[Official Pytorch Transformer Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)  
[Transformer Model From Scratch - YT](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
[Attention is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)