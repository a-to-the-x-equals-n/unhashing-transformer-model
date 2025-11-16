import csv
import json
import yaml
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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
    def iter_txt(path: str | Path, *, drop_blank: bool = True, strip_whitespace: bool = True) -> Iterator[str]:
        '''
        Stream plaintext lines from disk and optionally clean up whitespace.

        Parameters
        ----------
        path : str | Path
            Input text file; resolved relative to project root.
        drop_blank : bool
            Skip empty strings entirely when True.
        strip_whitespace : bool
            Remove leading/trailing whitespace so passwords are clean.

        Yields
        ------
        str
            One plaintext line at a time.

        Example
        -------
        >>> pw_iter = FileIO.iter_txt('data/sample.txt')
        >>> next(pw_iter)
        'hunter2'
        >>> next(pw_iter)
        'letmein'
        '''
        p = FileIO.resolve(path)

        with p.open('r', encoding = 'utf-8', errors = 'ignore') as handle:
            for raw in handle:
                line = raw.rstrip('\n')
                if strip_whitespace:
                    line = line.strip()
                if drop_blank and not line:
                    continue
                yield line

    @staticmethod
    def load_txt(path: str | Path, **kwargs) -> list[str]:
        '''
        Convenience wrapper that accumulates iter_txt into a list.
        '''
        return list(FileIO.iter_txt(path, **kwargs))

    @staticmethod
    def save_txt(lines: list[str], path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        with p.open('w', encoding = 'utf-8') as f:
            f.write('\n'.join(lines))
        print(f'saved {type(lines)} to {p}')

    @staticmethod
    @contextmanager
    def tsv_writer(path: str | Path, header: Sequence[str] | None = None):
        '''
        Context manager that yields a CSV writer configured for TSV output.

        Example
        -------
        >>> with FileIO.tsv_writer('data/out.tsv') as writer:
        ...     writer.writerow(('hash', 'password'))
        ...     writer.writerow(('5f4dcc3b5aa765d61d8327deb882cf99', 'password'))
        '''
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        with p.open('w', encoding = 'utf-8', newline = '') as handle:
            writer = csv.writer(handle, delimiter = '\t', lineterminator = '\n')
            if header:
                writer.writerow(header)
            yield writer

    @staticmethod
    def save_tsv(rows: Iterable[Sequence[str]], path: str | Path, header: Sequence[str] | None = None) -> int:
        '''
        Write an iterable of rows to disk as TSV.

        Returns the number of rows written (excluding the header).

        Example
        -------
        >>> rows = [('5f4dcc3b5aa765d61d8327deb882cf99', 'password')]
        >>> FileIO.save_tsv(rows, 'data/train.tsv', header = ('hash', 'password'))
        saved 1 rows to ...
        1
        '''
        count = 0
        with FileIO.tsv_writer(path, header) as writer:
            for row in rows:
                writer.writerow(row)
                count += 1

        print(f'saved {count} rows to {FileIO.resolve(path)}')
        return count

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
