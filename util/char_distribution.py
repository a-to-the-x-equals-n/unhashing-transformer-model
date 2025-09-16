from pathlib import Path
from collections import Counter
import yaml

def get_char_distribution(fpath: str | Path, /, *, show : bool = False, plot : bool = False, save : bool = False) -> Counter:
    '''
    Compute a character-frequency distribution from a password dataset stored as either:
      (a) YAML with a top-level key "passwords" mapping to a list of strings, or
      (b) plain text with one password per line.

    Parameters:
    -----------
    fpath : str | Path
        Path to the input file. Accepts `.yaml`/`.yml` or `.txt` (any non-YAML path is treated as plain text).

    show : bool, default False
        If True, print the full character-frequency list in descending order.

    plot : bool, default False
        If True, render a simple bar chart of the character distribution with matplotlib.

    save : bool, default False
        If True, write the resulting distribution to a sibling `char_freq.yaml`. Refuses to overwrite if the file already exists.

    Returns:
    --------
    Counter
        A Counter mapping each character to its observed frequency in the dataset.
    '''
    counts = Counter()

    if fpath.suffix == '.yaml':
        print('-reading YAML-')
        with fpath.open('r', encoding = 'utf-8', errors = 'ignore') as y:
            data = yaml.safe_load(y)

        for pw in data.get('passwords', []):    # safely grab the list
            counts.update(pw)                   # update counts with chars in password

    else:
        print('-reading TXT-')
        with fpath.open('r', encoding = 'utf-8', errors = 'ignore') as f:
            for line in f:
                pw = line.strip()               # remove leading/trailing whitespace
                counts.update(pw)               # update counts with chars in password

    if show:
        for ch, cts in counts.most_common():
            print(repr(ch), cts)

    if plot:
        import matplotlib.pyplot as plt
        ch, cts = zip(*sorted(counts.items()))  # sort by character before plotting
        plt.bar(ch, cts)
        plt.xlabel('character')
        plt.ylabel('frequency')
        plt.title('character distribution in password dataset')
        plt.show()

    if save:
        data = dict(counts)
        yml_path = fpath.parent / 'char_freq.yaml'

        # guard I made to save myself from overwriting my YAML file that I've commented
        # comment out as you see fit
        if yml_path.exists():
            print('[WARNING]: File already exists; canceling overwrite')
            print("  --- Please don't overwrite this YAML file; I don't want to lose the comments ---")
            return counts
        
        with open(yml_path, 'w', encoding = 'utf-8') as y:
            yaml.dump(data, y, allow_unicode = True, sort_keys = False)

    return counts


if __name__ == '__main__':
    f = Path.cwd().parent / 'data' / 'raw' / 'yaml' / '1mil_pw.yaml'
    c = get_char_distribution(f, save = True)
