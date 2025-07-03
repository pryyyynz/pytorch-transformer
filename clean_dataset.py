#!/usr/bin/env python3
# clean_dataset.py - Data cleaning and preprocessing script for English-Twi translation dataset

import os
from pathlib import Path
import re
import unicodedata
import argparse
from tqdm import tqdm
import random


def normalize_text(text):
    """Normalize text by removing extra whitespace, controlling for unicode, etc."""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_line(line, lang):
    """Clean a single line based on language-specific rules"""
    # Common cleaning
    line = normalize_text(line)

    # Remove lines consisting of just punctuation or very short content
    # Using standard punctuation instead of \p{P} which requires Python 3.10+
    punctuation = r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    if all(c.isspace() or c in punctuation for c in line) or len(line) < 2:
        return ""

    # English-specific cleaning
    if lang == 'en':
        # Replace common contractions if needed
        contractions = {
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            # Add more as needed
        }
        for contraction, expansion in contractions.items():
            line = line.replace(contraction, expansion)

    # Twi-specific cleaning
    elif lang == 'tw':
        # Add any Twi-specific rules here
        pass

    return line


def process_datasets(input_en_path, input_tw_path, output_en_path, output_tw_path,
                     max_length=None, min_length=1, validation_split=0.0, seed=42):
    """
    Process English and Twi datasets in parallel, ensuring alignment and cleaning.

    Args:
        input_en_path: Path to input English file
        input_tw_path: Path to input Twi file
        output_en_path: Path to output English file
        output_tw_path: Path to output Twi file
        max_length: Maximum number of tokens in a sentence (optional)
        min_length: Minimum number of tokens in a sentence
        validation_split: Proportion of data to save as validation set (0.0-1.0)
        seed: Random seed for validation split
    """
    print(f"Processing {input_en_path} and {input_tw_path}...")

    # Count lines in both files first
    en_line_count = 0
    tw_line_count = 0

    with open(input_en_path, 'r', encoding='utf-8', errors='replace') as f:
        for _ in f:
            en_line_count += 1

    with open(input_tw_path, 'r', encoding='utf-8', errors='replace') as f:
        for _ in f:
            tw_line_count += 1

    print(f"English file has {en_line_count} lines")
    print(f"Twi file has {tw_line_count} lines")

    if en_line_count != tw_line_count:
        print(
            f"WARNING: Line count mismatch! English: {en_line_count}, Twi: {tw_line_count}")

    # Process the files line by line
    valid_pairs = []
    skipped = 0

    with open(input_en_path, 'r', encoding='utf-8', errors='replace') as en_file, \
            open(input_tw_path, 'r', encoding='utf-8', errors='replace') as tw_file:

        total_lines = min(en_line_count, tw_line_count)
        for i, (en_line, tw_line) in enumerate(tqdm(zip(en_file, tw_file), total=total_lines)):
            # Clean the lines
            en_cleaned = clean_line(en_line, 'en')
            tw_cleaned = clean_line(tw_line, 'tw')

            # Skip if either line is empty after cleaning
            if not en_cleaned or not tw_cleaned:
                skipped += 1
                continue

            # Check length constraints
            en_tokens = en_cleaned.split()
            tw_tokens = tw_cleaned.split()

            if len(en_tokens) < min_length or len(tw_tokens) < min_length:
                skipped += 1
                continue

            if max_length and (len(en_tokens) > max_length or len(tw_tokens) > max_length):
                skipped += 1
                continue

            # Add to valid pairs
            valid_pairs.append((en_cleaned, tw_cleaned))

    print(f"Found {len(valid_pairs)} valid sentence pairs")
    print(f"Skipped {skipped} pairs due to filtering criteria")

    # Optional: Split into training and validation sets
    train_pairs = valid_pairs
    val_pairs = []

    if validation_split > 0:
        random.seed(seed)
        random.shuffle(valid_pairs)
        split_idx = int(len(valid_pairs) * (1 - validation_split))
        train_pairs = valid_pairs[:split_idx]
        val_pairs = valid_pairs[split_idx:]

        print(
            f"Split dataset: {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_en_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_tw_path), exist_ok=True)

    # Write training data
    with open(output_en_path, 'w', encoding='utf-8') as en_out, \
            open(output_tw_path, 'w', encoding='utf-8') as tw_out:
        for en_line, tw_line in train_pairs:
            en_out.write(f"{en_line}\n")
            tw_out.write(f"{tw_line}\n")

    # Write validation data if split was requested
    if validation_split > 0:
        val_en_path = output_en_path.replace('.txt', '_val.txt')
        val_tw_path = output_tw_path.replace('.txt', '_val.txt')

        with open(val_en_path, 'w', encoding='utf-8') as en_out, \
                open(val_tw_path, 'w', encoding='utf-8') as tw_out:
            for en_line, tw_line in val_pairs:
                en_out.write(f"{en_line}\n")
                tw_out.write(f"{tw_line}\n")

    print(f"Cleaned datasets saved to {output_en_path} and {output_tw_path}")
    if validation_split > 0:
        print(f"Validation sets saved to {val_en_path} and {val_tw_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Clean and preprocess English-Twi translation datasets')
    parser.add_argument('--en-input', default='datasets/english',
                        help='Path to input English file')
    parser.add_argument('--tw-input', default='datasets/twi',
                        help='Path to input Twi file')
    parser.add_argument('--en-output', default='datasets/english_clean.txt',
                        help='Path to output English file')
    parser.add_argument(
        '--tw-output', default='datasets/twi_clean.txt', help='Path to output Twi file')
    parser.add_argument('--max-length', type=int, default=350,
                        help='Maximum sentence length in tokens')
    parser.add_argument('--min-length', type=int, default=1,
                        help='Minimum sentence length in tokens')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation set split (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for validation split')

    args = parser.parse_args()

    process_datasets(
        args.en_input,
        args.tw_input,
        args.en_output,
        args.tw_output,
        args.max_length,
        args.min_length,
        args.val_split,
        args.seed
    )


if __name__ == "__main__":
    main()
