from pathlib import Path
import string


class Token:
    _ABSOLUTE = Path(__file__).resolve()
    _DIR = _ABSOLUTE.parent
    _ROOT = _DIR.parent
    _DATA_DIR = _ROOT / 'data' / 'structured'
    _VOCAB_DIR = _DIR / 'vocab'
    _ALLOWED_TEXT = sorted(set(string.ascii_letters + string.digits + '!@#$%^&*()-_'))

    def __init__(self):
        # hex / byte mappings
        self._htoi = {f'{i:02x}': i for i in range(256)}
        self._itoh = {i: f'{i:02x}' for i in range(256)}

        # character / byte mappings
        self._stoi = {ch: i for i,ch in enumerate(self._ALLOWED_TEXT)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

    @property
    def root(self):
        return str(self._ROOT)
    
    @property
    def where(self):
        return str(self._DIR)

    @property
    def htoi(self) -> dict[str, int]:
        '''hexadecimal string → byte value'''
        return self._htoi

    @property
    def itoh(self) -> dict[int, str]:
        '''byte value → hexadecimal string'''
        return self._itoh

    @property
    def stoi(self) -> dict[str, int]:
        '''character → integer index'''
        return self._stoi

    @property
    def itos(self) -> dict[int, str]:
        '''integer index → character'''
        return self._itos


    def encode_hex(self, h: str) -> list[int]:
        return [self._htoi[h[i : i + 2]] for i in range(0, len(h), 2)]


    def decode_hex(self, encoding: list[int]) -> str:
        return ''.join(self._itoh[i] for i in encoding)


    def encode_txt(self, s: str) -> list[int]:
        return [self._stoi[ch] for ch in s]


    def decode_txt(self, encoding: list[int]) -> str:
        return ''.join(self._itos[i] for i in encoding)


    def build_text_vocab(self):
        pass


    def save_vocab(self, fname: str, save_to: str | Path = '') -> None:
        import sys
        sys.path.append(str(self._ROOT))
        from util.fileio import FileIO

        vocab = {
            'htoi': self._htoi,
            'itoh': self._itoh,
            'stoi': self._stoi,
            'itos': self._itos,
        }

        f = self._VOCAB_DIR / fname if not save_to else Path(save_to) / fname
        FileIO.save_json(vocab, path = f)

    
    def load_vocab(self, path: str | Path) -> dict:
        import sys
        sys.path.append(str(self._ROOT))
        from util.fileio import FileIO

        vocab = FileIO.load_json(path)

        # re-cast integer-keyed maps
        self._itoh = vocab['itoh'] = {int(k): v for k, v in vocab['itoh'].items()}
        self._itos = vocab['itos'] = {int(k): v for k, v in vocab['itos'].items()}

        self._htoi = vocab['htoi']
        self._stoi = vocab['stoi']

        return vocab
