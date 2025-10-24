import string
from pathlib import Path
from fileio import FileIO

ALLOWED = set(string.ascii_letters + string.digits + '!@#$%^&*()-_')

def klean(f: str | Path, out: str | Path) -> list[str]:
    '''
    Load a YAML file of passwords, filter to an allowed character set, and save the cleaned list.

    Parameters:
    -----------
    f : str | Path
        Path to the input YAML file containing a top-level "passwords" list.

    out : str | Path
        Path where the cleaned YAML file will be saved.

    Returns:
    --------
    list[str]
        The cleaned list of passwords written to the output file.
    '''
    passwords = FileIO.load_yaml(f)
    cleaned = [pw.strip() for pw in passwords if all(ch in ALLOWED for ch in pw)]
    print(' -- cleaned --')
    FileIO.save_yaml(cleaned, out)
    return cleaned

if __name__ == '__main__':
    infile = 'data/raw/yaml/1mil_pw.yaml'
    outfile = 'data/cleaned/yaml/1mil_pw_cleaned.yaml'
    klean(infile, outfile)
