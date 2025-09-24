from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from fileio import FileIO

def get_char_distribution(path: str | Path, /, *, show: bool = False, plot: bool = False, save: bool = False) -> Counter:
    '''
    Compute a character-frequency distribution from a password dataset.

    Parameters:
    -----------
    path : str | Path
        Path to the input file. Accepts `.yaml`/`.yml` with a top-level "passwords" list,
        or `.txt` with one password per line.

    show : bool, default False
        If True, print the character counts in descending order.

    plot : bool, default False
        If True, render a bar chart of the character distribution.

    save : bool, default False
        If True, save the resulting distribution to `char_freq.yaml` alongside the input.

    Returns:
    --------
    Counter
        A Counter mapping each character to its observed frequency in the dataset.
    '''
    counts = Counter()
    p = FileIO.resolve(path)

    if p.suffix in ('.yaml', '.yml'):
        print('- reading YAML -')
        data = FileIO.load_yaml(p)  # synchronous fallback
        for pw in data.get('passwords', []):
            counts.update(pw)
    else:
        print('- reading TXT -')
        lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
        for pw in lines:
            counts.update(pw.strip())

    if show:
        for ch, cts in counts.most_common():
            print(repr(ch), cts)

    if plot:
        ch, cts = zip(*sorted(counts.items()))
        plt.bar(ch, cts)
        plt.xlabel('character')
        plt.ylabel('frequency')
        plt.title('character distribution in password dataset')
        plt.show()

    if save:
        out_path = p.parent / 'char_freq.yaml'
        FileIO.save_yaml(dict(counts), out_path)

    return counts

if __name__ == '__main__':
    f = 'data/cleaned/yaml/1mil_pw_cleaned.yaml'
    c = get_char_distribution(f, save=True)
