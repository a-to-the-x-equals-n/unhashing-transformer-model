import hashlib
import asyncio
from pathlib import Path
from fileio import FileIO

def hash_md5(pw: str) -> str:
    '''
    Hash a plaintext password using MD5.

    Parameters:
    -----------
    pw : str
        The plaintext password.

    Returns:
    --------
    str
        The MD5 hash of the password, encoded as a 32-character hexadecimal string.
    '''
    return hashlib.md5(pw.encode('utf-8')).hexdigest()

async def main(f: str | Path, out: str | Path) -> None:
    '''
    Load passwords from YAML, hash them with MD5, and save as a JSON mapping.

    Parameters:
    -----------
    f : str | Path
        Path to the YAML file containing a top-level "passwords" list.

    out : str | Path
        Path where the JSON mapping {hash: password} will be saved.

    Returns:
    --------
    None
        Writes the hash → password mapping to the output JSON file.
    '''
    passwords = await FileIO.load_yaml(f)

    hashed = {hash_md5(pw): pw for pw in passwords}
    await FileIO.save_json(hashed, out)

    print(f'saved {len(hashed)} entries to {out}')

if __name__ == '__main__':
    in_file = 'data/cleaned/yaml/1mil_pw_cleaned.yaml'
    out_file = 'data/structured/1mil_pw_structured.json'
    asyncio.run(main(in_file, out_file))
