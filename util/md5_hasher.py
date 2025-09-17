import hashlib
import yaml
import json
import asyncio
from pathlib import Path

async def waiting(label : str, event : asyncio.Event) -> None:
    '''
    Display a simple, non-blocking progress ticker until the given event is set.

    Parameters:
    -----------
    label : str
        Text label to display next to the animated dots.

    event : asyncio.Event
        Event that, when set, stops the animation.
    '''
    print()
    i = 0
    while not event.is_set():
        dots = '.' * (i % 4)                  # cycles through '', '.', '..', '...'
        print(f'\r {label}{dots:<3}', end = '', flush = True)
        await asyncio.sleep(.5)               # yields back to the event loop
        i += 1
    print('\r', end = '', flush = True)       # clear the line tail when finishing

async def load_yaml(_in: str | Path, out: str | Path) -> list[str]:
    '''
    Load a YAML file of passwords, filter to an allowed character set, and save the cleaned list.

    The input YAML is expected to contain a top-level key "passwords" whose value is a list of strings.
    The output YAML will contain the cleaned list only.

    Parameters:
    -----------
    _in : str | Path
        Path to the input YAML file containing passwords under the key "passwords".

    out : str | Path
        Path where the cleaned YAML sequence will be written.

    Returns:
    --------
    list[str]
        The cleaned list of passwords that were written to the output file.
    '''
    _in = Path(_in)
    passwords = []

    # - LOADING - 
    # offload file i/o + yaml parsing to a thread
    # show a spinner while waiting
    event = asyncio.Event()
    task = asyncio.create_task(waiting('  \t[loading]', event))
    passwords = await asyncio.to_thread(lambda: yaml.safe_load(_in.open('r', encoding = 'utf-8', errors = 'ignore')))
    # passwords = data['passwords']   # expects the canonical schema: {'passwords': [ ... ]}
    event.set()
    await task

def hash_pw(s: str) -> str:
    # TODO: hash pw using md5
    pass

def save_json():
    pass