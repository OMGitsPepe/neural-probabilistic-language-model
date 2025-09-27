# nplm/preprocess.py
import argparse
import json
import logging
import random
from pathlib import Path
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_split_proportions(value: str) -> Tuple[float]:
    """
    Helper function to parse comma-separated split proportions.
    :param value: Input string from the command line.
    :return: Tuple of floats representing the proportions.
    """
    try:
        # Split input string by commas and convert each part to a float
        proportions = tuple(map(float, value.split(',')))
        # Check if number of proportions is either 2 or 3
        if not (2 <= len(proportions) <= 3):
            # If not, raise error
            raise ValueError("Must provide 2 or 3 split proportions (train,test or train,val,test).")
        # Check if proportions sum up to 1
        if abs(sum(proportions) - 1.0) > 1e-6:
            # If not, raise error
            raise ValueError("Split proportions must sum to 1.")
        # If all checks pass, return the tuple of proportions
        return proportions
    except ValueError as e:
        # If any conversion or validation fails, raise an error
        raise argparse.ArgumentTypeError(str(e))


def main():
    """
    Main function to preprocess raw text data into sharded JSONL files.

    This function reads a raw test file, shuffles its lines, splits the into train,
    validation and test set based on provided proportions and writes the output
    as sharded JSONL files.
    """
    # Creates an ArgumenParser object to handle command line arguments
    parser = argparse.ArgumentParser(description="Preprocess raw text into sharded JSONL format.")

    # Define the --input_dir argument
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing the raw text file.")

    # Define the --output_dir
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the processed JSONL shards.")

    # Define the --shard_size with default value
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of documents per JSONL shard.")

    # Define the --splits argument using custom parsing function and default value
    parser.add_argument("--splits", type=parse_split_proportions, default="0.8,0.1,0.1", help="Comma-separated train,val,test proportions (e.g., '0.8,0.1,0.1').")

    # Define the --lowercase argument as a boolean flag
    parser.add_argument("--lowercase", action="store_true", help="If set, converts all text to lowercase.")

    # Define the --seed argument for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling to ensure reproducibility.")

    # Parse the arguments provided from the command line
    args = parser.parse_args()

    # Read raw data
    # Build the full path to the input file
    input_file = args.input_dir / f"{args.input_dir.name}_raw.txt"
    # Check if the constructed file path exists
    if not input_file.exists():
        # Log error and exit
        logging.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Expected raw text file at {input_file}")

    # Log start of reading process
    logging.info(f"Reading documents from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read all lines
        documents = [line.strip() for line in f if line.strip()]
    # Log how many documents were found
    logging.info(f"Found {len(documents)} total documents.")

    # Shuffle and split data
    logging.info(f"Shuffling data with random seed {args.seed}...")
    random.seed(args.seed)
    random.shuffle(documents)

    # Determine names of the splits based on the number of proportions provided
    split_names = ("train", "val", "test") if len(args.splits) == 3 else ("train", "test")

    # Calculate split indices
    total_docs = len(documents)
    split_indices = [0]
    cumulative_prop = 0.0
    for prop in args.splits:
        cumulative_prop += prop
        split_indices.append(int(cumulative_prop * total_docs))

    # Create a dictionary to hold list of documents for each split
    data_splits = {}
    # Loop through the split names
    for i, name in enumerate(split_names):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        data_splits[name] = documents[start_idx:end_idx]
        logging.info(f"Split '{name}' has {len(data_splits[name])} documents.")

    # Write JSONL shards
    logging.info(f"Writing JSONL shards to {args.output_dir} with shard size {args.shard_size}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each split and its corresponding documents
    for split_name, docs in data_splits.items():
        # Create a subdirectory for the current split
        split_dir = args.output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        # Initialize counters and variable holder
        shard_count = 0
        doc_count_in_shard = 0
        shard_file = None

        # Iterate through all documents in current split
        for i, doc_text in enumerate(docs):
            # Open a new shard file when needed
            if doc_count_in_shard == 0:
                shard_path = split_dir / f"shard_{shard_count}.jsonl"
                shard_file = open(shard_path, 'w', encoding='utf-8')

            # Process text lowercase and create JSON object
            text = doc_text.lower() if args.lowercase else doc_text
            json_line = json.dumps({"text": text})

            # Write to file
            shard_file.write(json_line + "\n")
            doc_count_in_shard += 1

            # Close the shard file when it's full
            if doc_count_in_shard == args.shard_size or (i + 1) == len(docs):
                shard_file.close()
                logging.info(
                    f"Wrote {doc_count_in_shard} docs to {shard_path}")
                shard_count += 1
                doc_count_in_shard = 0

    # Log that the entire preprocessing is complete
    logging.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
