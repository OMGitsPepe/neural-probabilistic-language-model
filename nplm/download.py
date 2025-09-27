# nplm/download.py
import argparse
import logging
import nltk
from pathlib import Path

# Set up logging to see the script progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_brown_corpus(output_dir: Path):
    """
    Download the Brown Corpus using NLTK and save it as a single raw test file
    :param output_dir: Path to the object pointing to the directory where the raw data is saved
    """
    # Log script is checking corpus
    logging.info("Checking for Brown Corpus...")
    try:
        # Check if the corpus is already downloaded to avoid re-downloading
        nltk.data.find('corpora/brown')
        logging.info("Brown Corpus already downloaded.")
    except LookupError:
        # If not found, download corpus
        logging.info("Brown Corpus not found. Downloading now...")
        nltk.download('brown')
        logging.info("Download complete.")

    # Creates specified output directory and construct full path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "brown_raw.txt"

    # Log path where file will be saved
    logging.info(f"Writing corpus to {output_file}...")

    # Import the corpus
    from nltk.corpus import brown

    # Write each sentence to file, joining words with a space
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in brown.sents():
            f.write(" ".join(sent) + "\n")

    # Log file has been successfully written
    logging.info(f"Successfully saved Brown Corpus to {output_file}.")


def main():
    """
    Main function to parse arguments and run the downloader."
    """
    # Create Argument Parser object to handle command line arguments
    parser = argparse.ArgumentParser(description="Download a text corpus for NPLM training.")

    # Define the --dataset argument
    parser.add_argument("--dataset",type=str, required=True, choices=["brown"], help="The name of the dataset to download.")

    # Define the --out_dir argument
    parser.add_argument("--out_dir", type=Path, required=True, help="The directory where the raw data will be saved.")

    # Parse arguments provided by the command line
    args = parser.parse_args()

    # Check which dataset the user requested
    if args.dataset == "brown":
        download_brown_corpus(args.out_dir)
    else:
        logging.error(f"Dataset '{args.dataset}' is not supported yet.")
        raise NotImplementedError(f"Support for dataset '{args.dataset}' is not implemented.")


if __name__ == "__main__":
    main()
