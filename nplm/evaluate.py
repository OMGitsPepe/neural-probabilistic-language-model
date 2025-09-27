# nplm/evaluation.py
import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse model and dataset classes from training script
from .train import NPLMModel, NPLMDataset, evaluate
from .word_tokenizer import WordTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Create an argument parser to handle command-line inputs
    parser = argparse.ArgumentParser(description="Evaluate a trained NPLM model.")

    # Defined --checkpoint argument
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint (e.g., best.pt).")

    # Define --data_dir argument
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory with processed JSONL data.")

    # Define --out_json argument
    parser.add_argument("--out_json", type=Path, help="Optional: Path to save metrics as a JSON file.")

    # Define --batch_size argument
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for evaluation.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Load Checkpoint and config
    # Check if specified checkpoint file exists
    if not args.checkpoint.exists():
        logging.error(f"Model checkpoint not found: {args.checkpoint}")
        raise FileNotFoundError(f"Model checkpoint not found at {args.checkpoint}")

    # Load checkpoint file
    logging.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Extract training configuration from checkpoint
    config = checkpoint['config']

    # Get best vocabulary perplexity recorde during training
    best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))

    # Use GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = WordTokenizer.load(config['tokenizer_path'])

    # Get vocabulary size from tokenizer
    vocab_size = len(tokenizer.id_to_token)
    logging.info(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Create test dataset using test data split
    test_dataset = NPLMDataset(args.data_dir / 'test', tokenizer, config['context_size'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create new model with same architecture
    model = NPLMModel(vocab_size=vocab_size, embedding_dim=config['embedding_dim'], context_size=config['context_size'], hidden_dim=config['hidden_dim']).to(device)

    # Load saved weights from checkpoint into model
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Model state loaded successfully.")

    # Run evaluation
    assert tokenizer.pad_id is not None, "<pad> token not found in tokenizer."
    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_id)

    # Log the start of evaluation and record start time
    logging.info(f"Evaluating on 'test' split...")
    start_time = time.time()

    # Call evaluation function to get loss and perplexity
    test_loss, test_ppl = evaluate(model, test_loader, loss_fn, device)

    # Calculate time
    eval_time_seconds = time.time() - start_time

    # Output results
    print("\n--- Final Evaluation Results ---")
    print(f"  Test Perplexity:      {test_ppl:.4f}")
    print(f"  Best Val Perplexity:  {best_val_ppl:.4f}")
    print(f"  Evaluation time:      {eval_time_seconds:.2f}s")
    print("--------------------------------\n")

    # Save metrics into JSON
    if args.out_json:
        metrics = {
            "train_ppl": best_val_ppl,
            "test_ppl": test_ppl,
            "tokenizer": "word",
            "vocab_size_or_merges": vocab_size,
            "context_size": config['context_size'],
            "train_time_sec": None
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved final metrics to {args.out_json}")


if __name__ == "__main__":
    main()
