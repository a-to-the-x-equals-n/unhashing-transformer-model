import string
from pathlib import Path
import yaml
import asyncio
import sys

# a–z, A–Z, 0-9, !@#$%^&*()-_
ALLOWED = set(string.ascii_letters + string.digits + '!@#$%^&*()-_')

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


async def purify(_in: str | Path, out: str | Path) -> list[str]:
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
    out = Path(out)
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

    # - CLEANING - 
    # strip whitespace
    # keep only passwords composed entirely of allowed chars
    cleaned = [pw.strip() for pw in passwords if all(ch in ALLOWED for ch in pw)]
    sys.stdout.write('\033[2K\r')   # clear entire line and return to start
    sys.stdout.flush()
    sys.stdout.write(' -- cleaned --')
    await asyncio.sleep(.5)         # small pause to let the ui breathe
    sys.stdout.write('\033[2K\r')   # clear entire line and return to start
    sys.stdout.flush()

    # - SAVING -
    # ensure parent directories exist
    # write yaml sequence in a background thread
    # show spinner
    out.parent.mkdir(parents = True, exist_ok = True)
    event = asyncio.Event()
    task = asyncio.create_task(waiting('  \t[saving]', event))
    await asyncio.to_thread(lambda: out.open('w', encoding = 'utf-8').write(yaml.dump(cleaned, allow_unicode = True, sort_keys = False)))
    event.set()
    await task

    print('  ("\'`\( 一_一)/`\'")\n\t-- success\n')

if __name__ == '__main__':
    infile = Path.cwd().parent / 'data' / 'raw' / 'yaml' / '1mil_pw.yaml'
    outfile = Path.cwd().parent / 'data' / 'cleaned' / 'yaml' / '1mil_pw_cleaned.yaml'
    asyncio.run(purify(infile, outfile))