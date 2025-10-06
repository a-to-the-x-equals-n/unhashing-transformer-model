### 0. Reverse Engineering Hashes with Seq2Seq Transformers
This project investigates whether the MD5 password hashing algorithm preserves underlying statistical patterns from the original plaintext passwords in a way that can be learned by machine learning models. 

### 1. Team Members
__Logan Reine__ - reinel22@students.ecu.edu   
__Caleb Gilbert__ - gilbertr25@students.ecu.edu   
__Paolo Imperi__ - imperiop24@students.ecu.edu  

### 2. Data Source
[Top 1 Million Passwords From Data Dumps](https://github.com/danielmiessler/SecLists/blob/master/Passwords/Common-Credentials/Pwdb_top-1000000.txt)

### 3. Objectives
- Build a dataset of `(password, hash)` pairs using MD5.
- Train supervised seq2seq models to predict plaintext passwords from hashes.  
- Measure model performance.
- Analyze whether model showed improved generalization.
  
### 4. Motivation 
Hashing functions are deterministic; the same input always produces the same output. This raises an open question: _could a model exploit subtle statistical regularities in weak hashes to reconstruct the original plaintexts_?   
MD5 was chosen for its fixed 32-character hex output and fast performance when bulk hashing. Though broken for security, it remains a practical/ethical choice for research.

### 5. Approach
The task is framed as a __sequence-to-sequence problem__: given a hash (input sequence), predict the original password (output sequence).  

#### Preprocessing
- Deduplicate entries.
- Remove passwords with undesired characters.
- Balance character composition (letters, digits, symbols)
- Generate hashes using Python’s `hashlib`.

#### Model
- __Architecture:__ Transformer encoder–decoder.
- __Input:__ Fixed-length 32 hex character hash.
- __Output:__ Variable-length password sequence.  

#### Training
- __Objective:__ supervised learning on `(hash → password)` pairs.  
- __Loss function:__ cross-entropy over output characters.  
- __Optimization:__ Adam or AdamW.  

#### Evaluation
- __Top-k Accuracy__: Measures how often the true label appears within a model’s top-k predicted guesses.
- __Edit Distance (Levenshtein)__: How many insertions, deletions, or substitutions needed to match prediction to truth.
- __Jaccard Similarity__: Measures the overlap between two sets as the ratio of their intersection to their union.  