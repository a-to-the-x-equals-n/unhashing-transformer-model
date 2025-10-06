import json
import yaml
import math
from pathlib import Path

class FileIO:
    # global root directory one level up from cwd
    ROOT = Path(__file__).resolve().parent.parent

    @staticmethod
    def resolve(path: str | Path) -> Path:
        '''
        prepend ROOT to the given relative path
        '''
        return FileIO.ROOT / Path(path)

    # yaml
    @staticmethod
    def load_yaml(path: str | Path) -> dict | list:
        p = FileIO.resolve(path)

        with p.open('r', encoding = 'utf-8', errors = 'ignore') as f:
            data = yaml.safe_load(f)

        return data

    @staticmethod
    def save_yaml(obj: dict | list, path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        with p.open('w', encoding = 'utf-8') as f:
            yaml.dump(obj, f, allow_unicode = True, sort_keys = False)

        print(f'saved {type(obj)} to {p}')


    # json
    @staticmethod
    def load_json(path: str | Path) -> dict | list:
        p = FileIO.resolve(path)

        with p.open('r', encoding = 'utf-8') as f:
            data =  json.load(f)

        return data

    @staticmethod
    def save_json(obj: dict | list, path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)
        
        with p.open('w', encoding = 'utf-8') as f:
            json.dump(obj, f, indent = 2, ensure_ascii = False)
        print(f'saved {type(obj)} to {p}')

    # text
    @staticmethod
    def load_txt(path: str | Path) -> list[str]:
        p = FileIO.resolve(path)

        with p.open('r', encoding = 'utf-8', errors = 'ignore') as f:
            lines =  [line.rstrip('\n') for line in f]

        return lines

    @staticmethod
    def save_txt(lines: list[str], path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        with p.open('w', encoding = 'utf-8') as f:
            f.write('\n'.join(lines))
        print(f'saved {type(lines)} to {p}')

    @staticmethod
    def shard_file(filepath: str | Path, shards: int) -> None:
        '''
        Split a dataset into a specified number of JSON shards.

        Parameters:
        -----------
        filepath : str | Path
            Path to the input file (YAML, JSON, or TXT). Relative to ROOT.

        shards : int
            Number of shards to split the dataset into.

        Returns:
        --------
        None
            Writes the shards as JSON files into ROOT/data/training/shards,
            named hash_pw_trainer_00x.json
        '''
        p = FileIO.resolve(filepath)
        suffix = p.suffix.lower()

        # load using internal helpers
        if suffix in ('.yaml', '.yml'):
            data = FileIO.load_yaml(p)
        elif suffix == '.json':
            data = FileIO.load_json(p)
        else:
            raise ValueError("shard_file expects JSON or YAML dict input")

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for sharding, got {type(data)}")

        # compute chunk size
        keys = list(data.keys())
        n = len(data)
        size = math.ceil(n / shards)

        out_dir = FileIO.ROOT / 'data' / 'training' / 'shards'
        out_dir.mkdir(parents = True, exist_ok = True)

        for i in range(shards):
            start = i * size
            end = start + size
            chunk_keys = keys[start:end]
            if not chunk_keys:
                break

            chunk = {k: data[k] for k in chunk_keys}
            out_path = out_dir / f"hash_pw_trainer_{i:03d}.json"
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(chunk, f, indent = 2, ensure_ascii = False)

            print(f"wrote shard {i} → {out_path} ({len(chunk)} entries)")
    
if __name__ == '__main__':
    pass