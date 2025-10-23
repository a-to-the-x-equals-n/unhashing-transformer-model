import hashlib
from pathlib import Path
from fileio import FileIO

def hash_md5(s: str) -> str:
    '''
    Hash string with MD5.

    Parameters:
    -----------
    s : str
        Plaintext string.

    Returns:
    --------
    str
        MD5 hash as a 32-character hexadecimal string.
    '''
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def main(p: str | Path, out: str | Path) -> None:
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
    passwords = FileIO.load_yaml(p)

    hashed = {hash_md5(pw): pw for pw in passwords}
    FileIO.save_json(hashed, out)

    print(f'saved {len(hashed)} entries to {out}')

if __name__ == '__main__':
    in_file = 'data/cleaned/yaml/1mil_pw_cleaned.yaml'
    out_file = 'data/structured/1mil_pw_structured.json'
    main(in_file, out_file)
