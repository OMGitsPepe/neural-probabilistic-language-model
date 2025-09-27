# LLM Log
The LLM I used to help build the Neural Probabilistic Language Model project was Gemini. I have had good experiences 
using Gemini in the past to build code and to correct bugs or errors. For this project, Gemini's role was that of a pair 
programmer: it helped create coding templates to serve as a base, assisted in fixing errors and bugs, and generated 
comprehensive code comments and documentation.

## How Gemini Was Used
The collaboration followed a step by step, conversational process. I would provide the project requirements for a 
specific part, and Gemini would generate the necessary code and explanations. We then iterated on that code to debug 
and refine it.

Below are a few concrete examples of this workflow.

## Example 1: Iterative Development and Boilerplate Generation
My collaboration with Gemini was iterative. Initially, I would describe the desired functionality for a script. After Gemini generated the initial code, I would write it in my own environment, making structural changes and ensuring the logic met my requirements.

I found that while Gemini could generate a complete, working script from the assignment instructions in a single step, I chose a more hands-on approach to ensure I was learning the implementation details. My process was to write the core logic for a script myself and then provide it to Gemini to handle boilerplate tasks, such as creating the argparse command-line interface.

My Prompt:
```commandline
"Here is my functional download.py script. Can you add the argparse CLI to it so I can run it from the command line with --dataset and --out_dir arguments?"

<download.py code>
```
Gemini's Output (Partial):
```commandline
import argparse
import logging
import nltk
from pathlib import Path

# ... (rest of the script) ...

def main():
    """Main function to parse arguments and run the downloader."""
    # Create a parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Download a text corpus for NPLM training.")
    
    # Add an argument for the dataset name
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["brown"],
        help="The name of the dataset to download."
    )
    # Add an argument for the output directory
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="The directory where the raw data will be saved."
    )
    
    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the download function based on the dataset argument
    if args.dataset == "brown":
        download_brown_corpus(args.out_dir)
    # ...

if __name__ == "__main__":
    main()
```

This workflow allowed me to focus on the core logic while offloading repetitive tasks to the LLM.

## Example 2: Error Diagnosis and Bug Fixing
During the training phase, I encountered a runtime error. I provided the full traceback to Gemini, which was able to 
diagnose the issue and provide the exact fix.

My Prompt:
```commandline
I'm getting this error when trying to run the train.py:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  ...
  token_ids = tokenizer.encode(doc["text"],
AttributeError: 'WordTokenizer' object has no attribute 'encode'
```

Gemini's Output (Explanation & Fix):
```commandline
The error message AttributeError: 'WordTokenizer' object has no attribute 'encode' means that the train.py script is trying 
to call a method named .encode() on your WordTokenizer object, but that method doesn't exist.

If we look back at the word_tokenizer.py file, the method for encoding a raw string of text is actually called encode_text().

The Fix
We just need to change that one line in nplm/train.py to use the correct method name...

# ... inside the data creation logic ...
# INCORRECT:
# token_ids = tokenizer.encode(doc["text"], add_special_tokens=True
```

## Example 3: Code Documentation
Once a script was fully functional, I would provide the finished code to Gemini and ask it to generate documentation. 
Gemini would add docstrings and line by line comments to explain the logic. I would then review these comments to ensure 
they were accurate and clear.

My Prompt:
```commandline
"Can you add simple comments to the evaluation.py script?"
```

Gemini's Output (Partial):
```commandline
# nplm/evaluate.py
import argparse
import json
import logging
import time
from pathlib import Path
import torch
# ...

def main():
    """The main function to run the evaluation process."""
    # --- Argument Parsing ---
    # Create a parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained NPLM model.")
    # Add an argument for the path to the model checkpoint file
    parser.add_argument(
        "--checkpoint", type=Path, required=True, 
        help="Path to the trained model checkpoint (e.g., best.pt)."
    )
    # ... more arguments ...

    # --- 1. Load Checkpoint and Config ---
    # Check if the specified checkpoint file actually exists
    if not args.checkpoint.exists():
        # Log an error and exit if the file is not found
        logging.error(f"Model checkpoint not found: {args.checkpoint}")
        raise FileNotFoundError(f"Model checkpoint not found at {args.checkpoint}")

    # Load the checkpoint file from disk
    logging.info(f"Loading checkpoint from {args.checkpoint}...")
    # Use 'cpu' to ensure it loads on any machine
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
```

This process automated the tedious task of writing comments, ensuring the final codebase was well documented and easy to 
understand, while still allowing me to maintain full control and verification over the final product.
