import json
import yaml
import math
import asyncio
from pathlib import Path

class FileIO:
    # global root directory one level up from cwd
    ROOT = Path.cwd().parent

    # spinner
    @staticmethod
    async def waiting(label: str, event: asyncio.Event) -> None:
        '''
        display a simple non-blocking progress ticker until the given event is set
        '''
        print()
        i = 0
        while not event.is_set():
            dots = '.' * (i % 4)                
            print(f'\r {label}{dots:<3}', end = '', flush = True)
            await asyncio.sleep(.5)
            i += 1
        print('\n')

    @staticmethod
    def resolve(path: str | Path) -> Path:
        '''
        prepend ROOT to the given relative path
        '''
        return FileIO.ROOT / Path(path)

    # yaml
    @staticmethod
    async def load_yaml(path: str | Path) -> dict | list:
        p = FileIO.resolve(path)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[loading yaml]', event))

        def read():
            with p.open('r', encoding = 'utf-8', errors = 'ignore') as f:
                return yaml.safe_load(f)

        data = await asyncio.to_thread(read)
        event.set()
        await task
        return data

    @staticmethod
    async def save_yaml(obj: dict | list, path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[saving yaml]', event))

        def write():
            with p.open('w', encoding = 'utf-8') as f:
                yaml.dump(obj, f, allow_unicode = True, sort_keys = False)

        await asyncio.to_thread(write)
        event.set()
        await task

    # json
    @staticmethod
    async def load_json(path: str | Path) -> dict | list:
        p = FileIO.resolve(path)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[loading json]', event))

        def read():
            with p.open('r', encoding = 'utf-8') as f:
                return json.load(f)

        data = await asyncio.to_thread(read)
        event.set()
        await task
        return data

    @staticmethod
    async def save_json(obj: dict | list, path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[saving json]', event))

        def write():
            with p.open('w', encoding = 'utf-8') as f:
                json.dump(obj, f, indent = 2, ensure_ascii = False)

        await asyncio.to_thread(write)
        event.set()
        await task

    # text
    @staticmethod
    async def load_txt(path: str | Path) -> list[str]:
        p = FileIO.resolve(path)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[loading txt]', event))

        def read():
            with p.open('r', encoding = 'utf-8', errors = 'ignore') as f:
                return [line.rstrip('\n') for line in f]

        lines = await asyncio.to_thread(read)
        event.set()
        await task
        return lines

    @staticmethod
    async def save_txt(lines: list[str], path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[saving txt]', event))

        def write():
            with p.open('w', encoding = 'utf-8') as f:
                f.write('\n'.join(lines))

        await asyncio.to_thread(write)
        event.set()
        await task

    @staticmethod
    async def shard_file(filepath: str | Path, shards: int) -> None:
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
            data = await FileIO.load_yaml(p)
        elif suffix == '.json':
            data = await FileIO.load_json(p)
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
    asyncio.run(FileIO.shard_file('data/structured/1mil_pw_structured.json', shards = 10))