import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import string
from pathlib import Path



# def decode_password(tensor):
#     return ''.join(dataset.itos[i.item()] for i in tensor if i.item() != pad_id)

class HashPwDataset(Dataset):
    '''
    A PyTorch Dataset that pairs fixed-length hash digests with variable-length plaintext passwords.
    
        This dataset prepares the data for a neural network that learns to associate (or predict) passwords from their hash digests.
        It converts both the hashes and plaintexts into numeric tensors that can be efficiently processed by PyTorch models.
        Each hash in the dataset is assumed to be a fixed-length hexadecimal string (e.g., MD5 → 32 hex chars).
        Each plaintext password can vary in length and consists only of characters from a defined "allowed" set.
    
    Parameters:
    -----------
    shard : str | Path
        The file path to a TSV (tab-separated) dataset, where each line contains: <hash>\t<password>

    Attributes:
    -----------
    vocab_size : int
        The number of unique byte values (0–255) plus one padding token (for alignment in batching).

    pad_id : int
        The numeric ID representing a padding value (used when aligning different-length sequences).

    htoi : dict[str, int]
        A mapping from 2-character hex strings to integer byte values (0–255).

    itoh : dict[int, str]
        The reverse of `htoi`: maps integers (0–255) back to hex strings.

    stoi : dict[str, int]
        A mapping from allowed plaintext characters to their integer token IDs.

    itos : dict[int, str]
        The reverse of `stoi`: maps integer token IDs back to plaintext characters.

    hashes : np.ndarray
        A NumPy array of shape [N, 16] (for MD5 hashes) containing encoded byte values for each hash.

    passwords : list[np.ndarray]
        A list of NumPy arrays of variable lengths, each containing tokenized password characters.
    '''
        
    # allowed plaintext characters
    # only letters, digits, and specific punctuation
    _ALLOWED_PW_CHARS = sorted(set(string.ascii_letters + string.digits + '!@#$%^&*()-_')) + ['<EOS>']

    def __init__(self, shard: str | Path) -> None:
        '''
        Initialize dataset by loading and encoding hash-password pairs.

        Parameters:
        -----------
        shard : str | Path
            Path to a TSV file with two columns: <hash>\t<password>
        '''
                
        # each hash byte can take values 0–255 (256 total)
        # we add 1 more "padding" token for sequence alignment
        self.vocab_size = 74 + 1
        self.pad_id = 74

        # build a lookup table for hexadecimal conversion
        # "htoi" = hex → integer
        # "itoh" = integer → hex
        # 02x formats integers as 2-digit lowercase hex strings, e.g., 5 → "05"
        self.htoi = {f'{i:02x}': i for i in range(256)}
        self.itoh = {i: f'{i:02x}' for i in range(256)}

        # build a lookup table for plaintext characters
        # stoi = "string to index" → maps each allowed character to a number
        # itos = "index to string" → reverse mapping for decoding
        self.stoi = {ch: i for i, ch in enumerate(self._ALLOWED_PW_CHARS)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # read the input TSV file into a pandas DataFrame
        # columns: [hash, password]
        df = pd.read_csv(shard, sep = '\t', names = ['hash', 'password'])


        # convert each hash (hexadecimal string) into a NumPy array of bytes
        # MD5 hashes are 32 hex characters → 16 bytes total
        # we iterate over the hash string in chunks of 2 characters
        self.hashes = np.stack([
            np.array(
                [self.htoi[digest[i : i + 2]] for i in range(0, len(digest), 2)],
                  dtype = np.uint8 # each element fits into one byte → saves memory
                  ) 
            for digest in df['hash']
        ])

        # convert each password (plaintext string) into a sequence of token IDs
        # lengths vary → we store them as a list of arrays, not a single 2D matrix
        self.passwords = [
            np.array(
                [self.stoi[ch] for ch in pw],
                dtype = np.uint8
            )
            for pw in df['password']
        ]

        del df # free weezy


    def __len__(self) -> int:
        '''
        Return the total number of examples in the dataset.

        Returns:
        --------
        int
            The number of hash-password pairs available.
        '''
        return len(self.hashes)


    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        '''
        Retrieve a single (hash, password) pair from the dataset and return it as PyTorch tensors.

        Parameters:
        -----------
        i : int
            The index of the desired dataset sample.

        Returns:
        --------
        dict[str, torch.Tensor]
            - 'hash' : torch.LongTensor of shape [16]
                    The 16-byte encoded hash.
            - 'password' : torch.LongTensor of variable length
                    The encoded plaintext password as integer tokens.
        '''

        # convert the pre-encoded NumPy arrays into PyTorch tensors
        # dtype long is used because most PyTorch layers (e.g., Embedding, CrossEntropyLoss)
        # expect integer indices of type torch.long
        return {
            'hash': torch.from_numpy(self.hashes[i]).long(),
            'password': torch.from_numpy(self.passwords[i]).long()
        }
    

def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    '''
    Custom collate function to combine individual dataset samples into a padded batch.

        Hashes are fixed-length (easy to batch)
        Passwords vary in length (need padding to make them same size for tensors)

    Parameters:
    -----------
    batch : list[dict[str, torch.Tensor]]
        A list of items returned by `HashPwDataset.__getitem__`

    Returns:
    --------
    dict[str, torch.Tensor]
        - 'hash' : torch.LongTensor of shape [B, 16]
                Batched fixed-size hash tensors.
        - 'password' : torch.LongTensor of shape [B, max_pw_len]
                Padded password sequences (shorter ones padded with `pad_id`).
        - 'lengths' : torch.LongTensor of shape [B]
                The true lengths of each password before padding.
    '''

    # extract hash tensors and stack them directly 
    # (they all have shape [16])
    hashes = torch.stack([item['hash'] for item in batch], dim = 0)

    # extract password tensors (variable length)
    passwords = [item['password'] for item in batch]
    lengths = torch.tensor([len(pw) for pw in passwords], dtype = torch.long)

    # find longest password length in this batch
    max_len = max(lengths).item()

    # create a padded tensor for all passwords
    # fill with `pad_id` so model knows which positions are "not real"

    '''NOTE: change back to 256'''
    pad_id = 74
    padded = torch.full((len(passwords), max_len), fill_value = pad_id, dtype = torch.long)

    # copy each password tensor into the padded batch tensor
    for i, pw in enumerate(passwords):
        padded[i, :len(pw)] = pw  # left-align and pad the rest

    return {
        'hash': hashes,           # [B, 16]
        'password': padded,       # [B, max_len]
        'lengths': lengths        # [B]
    }


shard_path = Path.cwd().parent.parent / 'project' / 'data' / 'training' / 'shards' / 'toy_shard.tsv'
dataset = HashPwDataset(shard_path)


if __name__ == '__main__':
    # toy dataset shard
    shard_path = Path.cwd().parent.parent / 'project' / 'data' / 'training' / 'shards' / 'toy_shard.tsv' 

    # initialize dataset
    dataset = HashPwDataset(shard_path)

    # build DataLoader with custom collate function
    dloader = DataLoader(
        dataset,
        batch_size = 8,
        shuffle = True,
        collate_fn = collate_batch
    )

    # fetch one batch to inspect
    batch = next(iter(dloader))

    print('hash shape     :', batch['hash'].shape)       # [B, 16]
    print('password shape :', batch['password'].shape)   # [B, max_pw_len]
    print('lengths        :', batch['lengths'])