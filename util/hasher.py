import argparse
import hashlib
import random
from pathlib import Path
from typing import Iterable, Iterator

from fileio import FileIO


def hash_md5(plaintext: str) -> str:
    '''
    Hash a plaintext password using MD5 and return the hex digest.
    '''
    return hashlib.md5(plaintext.encode('utf-8')).hexdigest()


def hash_passwords(passwords: Iterable[str]) -> Iterator[tuple[str, str]]:
    '''
    Convert an iterable of plaintext passwords into (hash, password) pairs.
    '''
    for password in passwords:
        if not password:
            continue
        yield hash_md5(password), password


def build_datasets(
    source_txt: str | Path,
    train_out: str | Path,
    eval_out: str | Path,
    eval_size: int = 10_000,
    seed: int = 1337,
) -> tuple[int, int]:
    '''
    Stream the plaintext list, hash in-place, and emit both train + eval TSVs.

        the eval reservoir is sampled first
        then removed from the emitted training TSV so the model never sees those pairs

    Returns
    -------
    tuple[int, int]
        (total_training_rows, total_eval_rows)

    Example
    -------
    >>> build_datasets(
    ...     source_txt = 'data/Pwdb_top-10_000_000.txt',
    ...     train_out = 'data/training/10M_train.tsv',
    ...     eval_out = 'data/eval/10K_eval.tsv',
    ... )
    (9990000, 10000)
    '''
    if eval_size < 0:
        raise ValueError('eval_size must be non-negative')

    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    total_rows = 0

    # pass 1: build a uniform eval reservoir
    for hashed, plaintext in hash_passwords(FileIO.iter_txt(source_txt)):
        total_rows += 1
        if eval_size == 0:
            continue

        if len(reservoir) < eval_size:
            reservoir.append((hashed, plaintext))
        else:
            idx = rng.randint(0, total_rows - 1)
            if idx < eval_size:
                reservoir[idx] = (hashed, plaintext)

    if eval_size > total_rows:
        print(f'warning: eval_size ({eval_size}) exceeds dataset size ({total_rows}); clipping.')
        eval_size = min(eval_size, total_rows)
        reservoir = reservoir[:eval_size]

    # count occurrences so duplicates are removed exactly once.
    removal_budget: dict[tuple[str, str], int] = {}
    for pair in reservoir:
        removal_budget[pair] = removal_budget.get(pair, 0) + 1

    # pass 2: write training TSV skipping eval pairs
    written_train = 0
    with FileIO.tsv_writer(train_out) as train_writer:
        for hashed, plaintext in hash_passwords(FileIO.iter_txt(source_txt)):
            key = (hashed, plaintext)
            budget = removal_budget.get(key, 0)
            if budget > 0:
                removal_budget[key] = budget - 1
                continue

            train_writer.writerow(key)
            written_train += 1

    eval_rows = FileIO.save_tsv(reservoir, eval_out) if reservoir else 0
    return written_train, eval_rows


if __name__ == '__main__':
    print(
        'Usage example:\n'
        '  python util/hasher.py \\\n'
        '      --source data/Pwdb_top-10_000_000.txt \\\n'
        '      --train-out data/training/10M_train.tsv \\\n'
        '      --eval-out data/eval/10K_eval.tsv'
    )

    parser = argparse.ArgumentParser(
        description = 'Hash plaintext passwords and emit TSV datasets.'
    )
    parser.add_argument(
        '--source',
        default = 'data/Pwdb_top-10_000_000.txt',
        help = 'Path to the plaintext password list (one per line).',
    )
    parser.add_argument(
        '--train-out',
        default = 'data/training/10M_train.tsv',
        help = 'Where to write the hash/password TSV for training.',
    )
    parser.add_argument(
        '--eval-out',
        default = 'data/eval/10K_eval.tsv',
        help = 'Where to save the random evaluation subset.',
    )
    parser.add_argument(
        '--eval-size',
        type = int,
        default = 10_000,
        help = 'How many random pairs to copy into the eval TSV.',
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1337,
        help = 'Deterministic RNG seed for the 10K sample.',
    )
    args = parser.parse_args()

    train_total, eval_total = build_datasets(
        source_txt = args.source,
        train_out = args.train_out,
        eval_out = args.eval_out,
        eval_size = args.eval_size,
        seed = args.seed,
    )

    train_path = FileIO.resolve(args.train_out)
    eval_path = FileIO.resolve(args.eval_out)

    print(f'wrote {train_total:,} training pairs to {train_path}')
    if args.eval_size > 0:
        print(f'wrote {eval_total:,} eval pairs to {eval_path}')
    else:
        print('no eval sample requested (eval_size = 0)')
