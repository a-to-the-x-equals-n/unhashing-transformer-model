import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import string



class HashPwDataset(Dataset):

    _ALLOWED_PW_CHARS = sorted(set(string.ascii_letters + string.digits + '!@#$%^&*()-_'))

    def __init__(self, shard: str) -> None:
        
        # hex <-> byte
        self._htoi = {f'{i:02x}': i for i in range(256)}
        self._itoh = {i: f'{i:02x}' for i in range(256)}

        # plaintext char <-> token
        self._stoi = {ch: i for i, ch in enumerate(self._ALLOWED_PW_CHARS)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

        # load the dataset
        df = pd.read_csv(shard, sep = '\t', names = ['hash', 'password'])

        # pre-encode digest to uint8 numpy arrays (shape: [N, 16])
        self.hashes = np.stack([
            np.array([self._htoi[digest[i : i + 2]] for i in range(0, len(digest), 2)], dtype = np.uint8) # we get discounted memory space for dtype, might as well use it
            for digest in df['hash']
        ])
        # pre-encode plaintext to uint8 (ragged, so store list of arrays)
        self.passwords = [
            np.array([self._stoi[ch] for ch in pw], dtype = np.uint8) # same here since all chars are represented by ~1 byte
            for pw in df['password']
        ]

        del df

    def __len__(self) -> int:
        return len(self.hashes)


    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:

        # return as tensors
        return {
            # entropy loss and nn.Embedding expect dtype long
            'hash': torch.from_numpy(self.hashes[i]).long(),
            'password': torch.from_numpy(self.passwords[i]).long()
        }