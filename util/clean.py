import argparse
import csv
import hashlib
import random
import string
from collections import Counter
from pathlib import Path

from fileio import FileIO

ALLOWED = set(string.ascii_letters + string.digits + '!@#$%^&*()-_')
HEADER = ('hash', 'password')


def clean_passwords(
    source: str | Path,
    cleaned_out: str | Path,
    char_freq_out: str | Path,
) -> tuple[int, int]:
    '''
    Stream the dirty password file, drop lines containing disallowed characters,
    and emit a cleaned plaintext list plus a YAML character distribution.
    '''
    counts: Counter[str] = Counter()
    kept = 0
    dropped = 0

    clean_path = FileIO.resolve(cleaned_out)
    clean_path.parent.mkdir(parents = True, exist_ok = True)

    with clean_path.open('w', encoding = 'utf-8') as clean_handle:
        for password in FileIO.iter_txt(
            source,
            drop_blank = True,
            strip_whitespace = True,
        ):
            if not password:
                continue

            if all(ch in ALLOWED for ch in password):
                clean_handle.write(password + '\n')
                counts.update(password)
                kept += 1
            else:
                dropped += 1

    sorted_counts = dict(sorted(counts.items()))
    FileIO.save_yaml(sorted_counts, char_freq_out)
    print(
        f'cleaned {kept:,} passwords (dropped {dropped:,}) → '
        f'{clean_path} and {FileIO.resolve(char_freq_out)}'
    )
    return kept, dropped


def hash_cleaned_passwords(
    cleaned_txt: str | Path,
    hashed_out: str | Path,
) -> int:
    '''
    Hash the cleaned plaintext passwords and persist them as TSV rows.
    '''
    hashed_path = FileIO.resolve(hashed_out)
    hashed_path.parent.mkdir(parents = True, exist_ok = True)

    total = 0
    with hashed_path.open('w', encoding = 'utf-8', newline = '') as handle:
        writer = csv.writer(handle, delimiter = '\t', lineterminator = '\n')
        writer.writerow(HEADER)

        for password in FileIO.iter_txt(cleaned_txt):
            digest = hashlib.md5(password.encode('utf-8')).hexdigest()
            writer.writerow((digest, password))
            total += 1

    print(f'hashed {total:,} passwords → {hashed_path}')
    return total


def sample_eval_split(
    hashed_tsv: str | Path,
    eval_out: str | Path,
    eval_size: int = 10_000,
    seed: int = 1337,
) -> tuple[int, int]:
    '''
    Uniformly sample eval_size rows from the hashed TSV, remove them from
    the training file, and store them separately as evaluation data.
    '''
    hashed_path = FileIO.resolve(hashed_tsv)
    eval_path = FileIO.resolve(eval_out)
    rng = random.Random(seed)

    reservoir: list[tuple[str, str]] = []
    total_rows = 0

    with hashed_path.open('r', encoding = 'utf-8', newline = '') as handle:
        reader = csv.reader(handle, delimiter = '\t')
        header = next(reader, None)
        if header is None:
            return 0, 0

        for row in reader:
            total_rows += 1
            if eval_size <= 0:
                continue

            pair = (row[0], row[1])
            if len(reservoir) < eval_size:
                reservoir.append(pair)
            else:
                idx = rng.randint(0, total_rows - 1)
                if idx < eval_size:
                    reservoir[idx] = pair

    if not reservoir:
        print('no eval split requested; skipping removal')
        return total_rows, 0

    removal_budget: Counter[tuple[str, str]] = Counter(reservoir)
    temp_path = hashed_path.with_suffix(hashed_path.suffix + '.tmp')

    with hashed_path.open('r', encoding = 'utf-8', newline = '') as read_handle, \
            temp_path.open('w', encoding = 'utf-8', newline = '') as write_handle:
        reader = csv.reader(read_handle, delimiter = '\t')
        writer = csv.writer(write_handle, delimiter = '\t', lineterminator = '\n')
        header = next(reader, None)
        if header:
            writer.writerow(header)

        for row in reader:
            key = (row[0], row[1])
            budget = removal_budget.get(key, 0)
            if budget > 0:
                if budget == 1:
                    del removal_budget[key]
                else:
                    removal_budget[key] = budget - 1
                continue
            writer.writerow(row)

    temp_path.replace(hashed_path)
    eval_rows = FileIO.save_tsv(reservoir, eval_path, header = HEADER)
    kept_rows = total_rows - eval_rows
    print(
        f'sampled {eval_rows:,} eval rows (seed {seed}) → {eval_path}\n'
        f'kept {kept_rows:,} training rows → {hashed_path}'
    )
    return kept_rows, eval_rows


def shard_training_dataset(
    hashed_tsv: str | Path,
    shard_dir: str | Path,
    shard_size: int = 1_000_000,
) -> list[Path]:
    '''
    Split the hashed training TSV into ~shard_size chunks for faster loading.
    '''
    hashed_path = FileIO.resolve(hashed_tsv)
    shard_root = FileIO.resolve(shard_dir)
    shard_root.mkdir(parents = True, exist_ok = True)

    created: list[Path] = []
    shard_handle = None
    shard_writer = None
    rows_in_shard = shard_size  # force the first shard to open immediately

    with hashed_path.open('r', encoding = 'utf-8', newline = '') as handle:
        reader = csv.reader(handle, delimiter = '\t')
        header = next(reader, None)
        if header is None:
            return created

        try:
            for row in reader:
                if shard_writer is None or rows_in_shard >= shard_size:
                    if shard_handle:
                        shard_handle.close()

                    shard_index = len(created)
                    shard_path = shard_root / f'hash_pw_shard_{shard_index:03d}.tsv'
                    shard_handle = shard_path.open('w', encoding = 'utf-8', newline = '')
                    shard_writer = csv.writer(shard_handle, delimiter = '\t', lineterminator = '\n')
                    shard_writer.writerow(header)
                    created.append(shard_path)
                    rows_in_shard = 0

                shard_writer.writerow(row)
                rows_in_shard += 1
        finally:
            if shard_handle:
                shard_handle.close()

    print(f'wrote {len(created)} shards → {shard_root}')
    return created


def run_pipeline(
    source: str | Path,
    cleaned_out: str | Path,
    char_freq_out: str | Path,
    hashed_out: str | Path,
    eval_out: str | Path,
    shard_dir: str | Path,
    eval_size: int,
    shard_size: int,
    seed: int,
) -> None:
    '''
    End-to-end orchestration: clean → hash → eval split → shards.
    '''
    kept, dropped = clean_passwords(source, cleaned_out, char_freq_out)
    if kept == 0:
        print('no passwords survived cleaning; aborting downstream steps')
        return

    hashed_total = hash_cleaned_passwords(cleaned_out, hashed_out)
    if hashed_total != kept:
        print('warning: mismatch between cleaned and hashed counts')

    train_rows, eval_rows = sample_eval_split(
        hashed_tsv = hashed_out,
        eval_out = eval_out,
        eval_size = eval_size,
        seed = seed,
    )

    if train_rows <= 0:
        print('no training rows left after eval sampling; skipping sharding')
        return

    shard_training_dataset(
        hashed_tsv = hashed_out,
        shard_dir = shard_dir,
        shard_size = shard_size,
    )
    print(
        f'pipeline complete:\n'
        f'  cleaned passwords: {kept:,}\n'
        f'  dropped passwords: {dropped:,}\n'
        f'  eval rows: {eval_rows:,}\n'
        f'  training rows (post-removal): {train_rows:,}'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = 'Clean password lists, build hash/pw TSVs, carve eval splits, and shard the result.',
    )
    parser.add_argument(
        '--source',
        default = 'data/source_data/dirty/10MIL_pw_dirty.txt',
        help = 'Dirty plaintext password file (one per line).',
    )
    parser.add_argument(
        '--clean-out',
        default = 'data/source_data/clean/10MIL_pw_cleaned.txt',
        help = 'Destination for the cleaned plaintext list.',
    )
    parser.add_argument(
        '--char-freq-out',
        default = 'data/source_data/clean/10MIL_pw_cleaned_char_freq.yaml',
        help = 'Path for the character distribution YAML.',
    )
    parser.add_argument(
        '--hashed-out',
        default = 'data/training/10MIL_pw_hashes.tsv',
        help = 'Location of the hash/password TSV.',
    )
    parser.add_argument(
        '--eval-out',
        default = 'data/eval/10K_eval.tsv',
        help = 'Path for the held-out eval TSV.',
    )
    parser.add_argument(
        '--shard-dir',
        default = 'data/training/shards',
        help = 'Directory that will receive ~1M-row shard TSVs.',
    )
    parser.add_argument(
        '--eval-size',
        type = int,
        default = 10_000,
        help = 'Number of hash/password pairs to move into the eval split.',
    )
    parser.add_argument(
        '--shard-size',
        type = int,
        default = 1_000_000,
        help = 'Approximate number of rows per shard.',
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1337,
        help = 'Deterministic seed for the eval sampling reservoir.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        source = args.source,
        cleaned_out = args.clean_out,
        char_freq_out = args.char_freq_out,
        hashed_out = args.hashed_out,
        eval_out = args.eval_out,
        shard_dir = args.shard_dir,
        eval_size = args.eval_size,
        shard_size = args.shard_size,
        seed = args.seed,
    )
