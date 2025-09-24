import json
import yaml
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
            print(f'\r {label}{dots:<3}', end='', flush=True)
            await asyncio.sleep(.5)
            i += 1
        print('\r', end='', flush=True)         # clear line tail when finishing

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
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                return yaml.safe_load(f)

        data = await asyncio.to_thread(read)
        event.set()
        await task
        return data

    @staticmethod
    async def save_yaml(obj: dict | list, path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[saving yaml]', event))

        def write():
            with p.open('w', encoding='utf-8') as f:
                yaml.dump(obj, f, allow_unicode=True, sort_keys=False)

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
            with p.open('r', encoding='utf-8') as f:
                return json.load(f)

        data = await asyncio.to_thread(read)
        event.set()
        await task
        return data

    @staticmethod
    async def save_json(obj: dict | list, path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[saving json]', event))

        def write():
            with p.open('w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

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
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                return [line.rstrip('\n') for line in f]

        lines = await asyncio.to_thread(read)
        event.set()
        await task
        return lines

    @staticmethod
    async def save_txt(lines: list[str], path: str | Path) -> None:
        p = FileIO.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        event = asyncio.Event()
        task = asyncio.create_task(FileIO.waiting('[saving txt]', event))

        def write():
            with p.open('w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

        await asyncio.to_thread(write)
        event.set()
        await task


async def main():
    # yaml
    passwords = await FileIO.load_yaml('data/raw/yaml/1mil_pw.yaml')
    await FileIO.save_yaml(passwords, 'data/cleaned/yaml/1mil_pw_cleaned.yaml')

    # json
    obj = {'foo': 'bar'}
    await FileIO.save_json(obj, 'data/test/out.json')
    data = await FileIO.load_json('data/test/out.json')

    # txt
    lines = ['a', 'b', 'c']
    await FileIO.save_txt(lines, 'data/test/out.txt')
    txt = await FileIO.load_txt('data/test/out.txt')

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

