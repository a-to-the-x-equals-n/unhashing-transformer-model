from pathlib import Path
import string


class Token:
    _ABSOLUTE = Path(__file__).resolve()
    _DIR = _ABSOLUTE.parent
    _ROOT = _DIR.parent
    _DATA_DIR = _ROOT / 'data' / 'structured'
    _VOCAB_DIR = _DIR / 'vocab'
    _ALLOWED_TEXT = set(string.ascii_letters + string.digits + '!@#$%^&*()-_')

    def __init__(self):
        pass

    @property
    def root(self):
        return str(self._ROOT)
    
    @property
    def where(self):
        return str(self._DIR)
    
    def save_vocab(self, vocab: dict, fname: str, save_to: str | Path = ''):
        import sys
        sys.path.append(str(self._ROOT / 'util'))
        from fileio import FileIO

        f = self._VOCAB_DIR / fname if not save_to else Path(save_to / fname)
        FileIO.save_json(vocab, path = f)


    def load_corpus(self):
        pass


    def build_hex_vocab(self) -> dict[str, int]:
        hex_vocab = {f'{i:02x}': i for i in range(256)}
        return hex_vocab


    def build_text_vocab(self):
        pass


    def _decode_hex(self):
        pass


    def encode(self):
        pass


    def decode(self):
        pass



