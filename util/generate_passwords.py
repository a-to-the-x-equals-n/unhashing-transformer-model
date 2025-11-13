#!/usr/bin/env python3
"""
Strategic password generator for maximizing MD5 hash space coverage.

Generates diverse passwords across multiple pattern families to help
the model learn statistical relationships between password structure
and hash characteristics.
"""

import hashlib
import random
import string
from pathlib import Path
from typing import Iterator
import sys

# Allowed character set (must match your model's vocabulary)
ALLOWED_CHARS = string.ascii_letters + string.digits + '!@#$%^&*()-_'


class PasswordGenerator:
    """
    Generates diverse passwords across multiple structural patterns.
    """

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)

        # Common password components
        self.common_words = [
            'password', 'admin', 'user', 'test', 'welcome', 'hello',
            'dragon', 'monkey', 'letmein', 'football', 'iloveyou',
            'master', 'sunshine', 'princess', 'shadow', 'qwerty',
            'abc', 'love', 'secret', 'summer', 'winter', 'spring'
        ]

        self.names = [
            'john', 'mary', 'michael', 'sarah', 'david', 'emma',
            'james', 'emily', 'robert', 'linda', 'william', 'lisa',
            'tom', 'anna', 'chris', 'kate', 'alex', 'jane'
        ]

        self.adjectives = [
            'red', 'blue', 'big', 'small', 'happy', 'sad',
            'fast', 'slow', 'hot', 'cold', 'new', 'old'
        ]

        self.nouns = [
            'cat', 'dog', 'sun', 'moon', 'star', 'tree',
            'car', 'house', 'book', 'phone', 'rock', 'bird'
        ]


    def generate_dates(self, count: int) -> Iterator[str]:
        """
        Generate date-based passwords (19XX-20XX).
        Format: YYYYMMDD, MMDDYYYY, DDMMYYYY
        """
        formats = [
            lambda y, m, d: f"{y:04d}{m:02d}{d:02d}",  # YYYYMMDD
            lambda y, m, d: f"{m:02d}{d:02d}{y:04d}",  # MMDDYYYY
            lambda y, m, d: f"{d:02d}{m:02d}{y:04d}",  # DDMMYYYY
            lambda y, m, d: f"{y:04d}{m:02d}",          # YYYYMM
            lambda y, m, d: f"{m:02d}{y:04d}",          # MMYYYY
        ]

        generated = 0
        while generated < count:
            year = random.randint(1950, 2024)
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Safe for all months

            fmt = random.choice(formats)
            yield fmt(year, month, day)
            generated += 1


    def generate_numeric_patterns(self, count: int) -> Iterator[str]:
        """
        Generate numeric passwords with patterns.
        - Repeated digits: 111111, 888888
        - Sequential: 123456, 987654
        - Repeated patterns: 121212, 123123
        """
        patterns = []

        # Repeated single digits
        for digit in range(10):
            for length in range(4, 13):
                patterns.append(str(digit) * length)

        # Sequential ascending
        for start in range(10):
            for length in range(4, 10):
                seq = ''.join(str((start + i) % 10) for i in range(length))
                patterns.append(seq)

        # Sequential descending
        for start in range(10):
            for length in range(4, 10):
                seq = ''.join(str((start - i) % 10) for i in range(length))
                patterns.append(seq)

        # Repeated patterns (123123, 456456)
        for base_len in range(2, 5):
            for _ in range(20):
                base = ''.join(str(random.randint(0, 9)) for _ in range(base_len))
                repeat_count = random.randint(2, 4)
                patterns.append(base * repeat_count)

        # Random but structured lengths
        for length in range(4, 13):
            for _ in range(count // 50):
                patterns.append(''.join(str(random.randint(0, 9)) for _ in range(length)))

        # Shuffle and yield
        random.shuffle(patterns)
        for i, pwd in enumerate(patterns):
            if i >= count:
                break
            yield pwd


    def generate_word_plus_number(self, count: int) -> Iterator[str]:
        """
        Generate word+number combinations.
        Examples: password123, dragon99, admin2024
        """
        for _ in range(count):
            word = random.choice(self.common_words + self.names)

            # Different number patterns
            pattern = random.choice([
                lambda: str(random.randint(1, 9999)),           # Random number
                lambda: str(random.randint(1950, 2024)),        # Year
                lambda: str(random.randint(1, 99)),             # Small number
                lambda: str(random.randint(0, 9)) * random.randint(2, 4),  # Repeated digit
            ])

            number = pattern()

            # Different orderings
            if random.random() < 0.7:
                yield f"{word}{number}"
            else:
                yield f"{number}{word}"


    def generate_leetspeak(self, count: int) -> Iterator[str]:
        """
        Generate leetspeak variations.
        Examples: p4ssw0rd, h3ll0, 4dm1n
        """
        leet_map = {
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$'],
            'l': ['1'],
            't': ['7'],
            'g': ['9'],
        }

        for _ in range(count):
            word = random.choice(self.common_words + self.names)
            result = []

            for char in word:
                if char.lower() in leet_map and random.random() < 0.6:
                    result.append(random.choice(leet_map[char.lower()]))
                else:
                    result.append(char)

            # Add optional number suffix
            if random.random() < 0.5:
                result.append(str(random.randint(1, 999)))

            yield ''.join(result)


    def generate_combinations(self, count: int) -> Iterator[str]:
        """
        Generate word combinations.
        Examples: redcat, happydog, bighouse123
        """
        for _ in range(count):
            components = []

            # Pick 2-3 components
            num_components = random.randint(2, 3)

            for i in range(num_components):
                component_type = random.choice(['adj', 'noun', 'name', 'digit'])

                if component_type == 'adj':
                    components.append(random.choice(self.adjectives))
                elif component_type == 'noun':
                    components.append(random.choice(self.nouns))
                elif component_type == 'name':
                    components.append(random.choice(self.names))
                else:  # digit
                    components.append(str(random.randint(1, 999)))

            # Random capitalization
            result = ''.join(components)
            if random.random() < 0.3:
                result = result.capitalize()
            elif random.random() < 0.1:
                result = result.upper()

            yield result


    def generate_special_chars(self, count: int) -> Iterator[str]:
        """
        Generate passwords with special characters.
        Examples: pass@123, admin!2024, test#word
        """
        special = '!@#$%^&*()-_'

        for _ in range(count):
            base = random.choice(self.common_words + self.names)
            num = str(random.randint(1, 9999))
            spec = random.choice(special)

            # Different patterns
            pattern = random.choice([
                f"{base}{spec}{num}",
                f"{base}{num}{spec}",
                f"{spec}{base}{num}",
                f"{num}{spec}{base}",
            ])

            yield pattern


    def generate_random_alphanum(self, count: int) -> Iterator[str]:
        """
        Generate random alphanumeric passwords (high entropy).
        Different lengths to maximize hash diversity.
        """
        for _ in range(count):
            length = random.randint(6, 16)
            chars = random.choices(string.ascii_letters + string.digits, k=length)
            yield ''.join(chars)


    def generate_keyboard_patterns(self, count: int) -> Iterator[str]:
        """
        Generate keyboard walk patterns.
        Examples: qwerty, asdfgh, zxcvbn, 1q2w3e
        """
        rows = [
            'qwertyuiop',
            'asdfghjkl',
            'zxcvbnm',
            '1234567890'
        ]

        for _ in range(count):
            row = random.choice(rows)
            start = random.randint(0, len(row) - 4)
            length = random.randint(4, min(10, len(row) - start))

            pattern = row[start:start + length]

            # Add variations
            if random.random() < 0.3:
                pattern = pattern[::-1]  # Reverse
            if random.random() < 0.3:
                pattern += str(random.randint(1, 999))

            yield pattern


    def generate_all(self, total_count: int, distribution: dict = None) -> Iterator[tuple[str, str]]:
        """
        Generate passwords across all categories with specified distribution.

        Parameters:
        -----------
        total_count : int
            Total number of passwords to generate

        distribution : dict, optional
            Distribution of password types (defaults to balanced)
            Keys: 'dates', 'numeric', 'word_num', 'leet', 'combo', 'special', 'random', 'keyboard'

        Yields:
        -------
        tuple[str, str]
            (hash, password) pairs
        """
        if distribution is None:
            distribution = {
                'dates': 0.15,
                'numeric': 0.15,
                'word_num': 0.20,
                'leet': 0.10,
                'combo': 0.15,
                'special': 0.10,
                'random': 0.10,
                'keyboard': 0.05
            }

        # Calculate counts per category
        generators = [
            ('dates', self.generate_dates, int(total_count * distribution['dates'])),
            ('numeric', self.generate_numeric_patterns, int(total_count * distribution['numeric'])),
            ('word_num', self.generate_word_plus_number, int(total_count * distribution['word_num'])),
            ('leet', self.generate_leetspeak, int(total_count * distribution['leet'])),
            ('combo', self.generate_combinations, int(total_count * distribution['combo'])),
            ('special', self.generate_special_chars, int(total_count * distribution['special'])),
            ('random', self.generate_random_alphanum, int(total_count * distribution['random'])),
            ('keyboard', self.generate_keyboard_patterns, int(total_count * distribution['keyboard'])),
        ]

        all_passwords = []
        seen = set()

        print(f"Generating {total_count:,} passwords...")
        for category, gen_func, count in generators:
            print(f"  [{category}]: {count:,} passwords")
            category_passwords = []

            for pwd in gen_func(count):
                # Filter out passwords with invalid characters
                if all(c in ALLOWED_CHARS for c in pwd):
                    if pwd not in seen:  # Ensure uniqueness
                        seen.add(pwd)
                        category_passwords.append(pwd)

                if len(category_passwords) >= count:
                    break

            all_passwords.extend(category_passwords)

        # Shuffle to mix categories
        random.shuffle(all_passwords)

        print(f"\nGenerated {len(all_passwords):,} unique passwords")
        print("Computing MD5 hashes...")

        # Generate hash-password pairs
        for i, password in enumerate(all_passwords):
            if i > 0 and i % 100000 == 0:
                print(f"  Hashed {i:,}/{len(all_passwords):,}")

            hash_digest = hashlib.md5(password.encode('utf-8')).hexdigest()
            yield (hash_digest, password)


def save_to_tsv(output_path: Path, generator: Iterator[tuple[str, str]], count: int):
    """
    Save generated hash-password pairs to TSV file.

    Parameters:
    -----------
    output_path : Path
        Output file path
    generator : Iterator[tuple[str, str]]
        Generator yielding (hash, password) tuples
    count : int
        Expected number of entries (for progress tracking)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (hash_digest, password) in enumerate(generator):
            f.write(f"{hash_digest}\t{password}\n")

            if (i + 1) % 100000 == 0:
                print(f"  Wrote {i + 1:,} lines")

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Total entries: {i + 1:,}")


if __name__ == '__main__':
    """
    Usage:
        python generate_passwords.py <count> [output_file]

    Examples:
        python generate_passwords.py 1000000
        python generate_passwords.py 5000000 ../data/training/5M_generated.tsv
    """

    if len(sys.argv) < 2:
        print("Usage: python generate_passwords.py <count> [output_file]")
        print("\nExamples:")
        print("  python generate_passwords.py 1000000")
        print("  python generate_passwords.py 5000000 ../data/training/5M_generated.tsv")
        sys.exit(1)

    count = int(sys.argv[1])

    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path(__file__).parent.parent / 'data' / 'training' / f'{count//1000}K_generated.tsv'

    print("="*60)
    print("PASSWORD GENERATOR FOR MD5 HASH COVERAGE")
    print("="*60)
    print(f"Target count: {count:,}")
    print(f"Output file: {output_file}")
    print("="*60)

    gen = PasswordGenerator(seed=42)
    password_generator = gen.generate_all(count)

    save_to_tsv(output_file, password_generator, count)

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
