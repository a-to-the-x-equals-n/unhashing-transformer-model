from pathlib import Path
import yaml

def txt_to_yaml(txt_path: str | Path, yaml_path: str | Path) -> None:
    '''
    Convert a plaintext password list into a YAML file with a top-level key "passwords".

    Parameters:
    -----------
    txt_path : str | Path
        Path to the plaintext file containing one password per line.

    yaml_path : str | Path
        Path where the output YAML file will be saved.

    Returns:
    --------
    None
        Writes the password list to a YAML file at the specified location.
    '''
    txt_path = Path(txt_path)
    yaml_path = Path(yaml_path)

    # read each line from plaintext file, strip whitespace, ignore blank lines
    with txt_path.open('r', encoding = 'utf-8', errors = 'ignore') as f:
        print(' -- reading --')
        passwords = [line.strip() for line in f if line.strip()]

    # write dictionary to yaml file with unicode allowed and stable ordering
    with yaml_path.open('w', encoding = 'utf-8') as f:
        print(' -- writing --')
        yaml.dump(passwords, f, allow_unicode = True, sort_keys = False)


if __name__ == '__main__':

    # paths are built relative to current working directory for portability
    txt = Path.cwd().parent / 'data' / 'raw' / 'plaintext' / '1mil_pw.txt'
    yml = Path.cwd().parent / 'data' / 'raw' / 'yaml' / '1mil_pw.yaml'

    txt_to_yaml(txt, yml)
    print(' -- success --')
