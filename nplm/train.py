# nplm/train.py
import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from .word_tokenizer import WordTokenizer

# Setup logging
# Configure logging to display info, level, and messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# NPLM Model
class NPLMModel(nn.Module):
    """
    Neural Probabilistic Language Model (Bengio et al., 2003).

    A feed-forward network that predicts the next word based on a fixed context window.
    It includes a shared embedding matrix and direct connections from embeddings to the output,
    in addition to the standard hidden layer.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int,
                 hidden_dim: int):
        """
        Initializes the NPLM model layers.
        :param vocab_size: The number of unique tokens in the vocabulary.
        :param embedding_dim: The dimensionality of the word embeddings.
        :param context_size: The number of previous words to use as context.
        :param hidden_dim: The number of neurons in the hidden layer.
        """
        # Call the constructor of the parent class
        super(NPLMModel, self).__init__()
        # Store context size and embedding dimension for use in the forward pass
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        # Embedding matrix C, which maps word indices to dense vectors
        self.C = nn.Embedding(vocab_size, embedding_dim)

        # Layer 1: linear transformation for the hidden layer
        self.H = nn.Linear(context_size * embedding_dim, hidden_dim,bias=False)
        # The bias 'd' for the hidden layer, defined as a learnable parameter
        self.d = nn.Parameter(torch.randn(hidden_dim))

        # Layer 2: linear transformation from the hidden layer to the output vocabulary
        self.U = nn.Linear(hidden_dim, vocab_size, bias=False)
        # The bias 'b' for the output layer, defined as a learnable parameter
        self.b = nn.Parameter(torch.randn(vocab_size))

        # A direct linear connection from the input embeddings to the output vocabulary
        self.W = nn.Linear(context_size * embedding_dim, vocab_size, bias=False)

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        :param x_idx: A tensor of shape (batch_size, context_size) containing word indices.
        :return: A tensor of log probabilities over the vocabulary, shape (batch_size, vocab_size).
        """
        # Retrieve the embeddings for the context words from the embedding matrix C
        x_emb = self.C(x_idx)
        # Reshape the embeddings from by concatenating them
        x = x_emb.view(-1, self.context_size * self.embedding_dim)

        # Calculate hidden layer activation using tanh
        hidden_out = torch.tanh(self.d + self.H(x))

        # Calculate the final output logits by combining the three components:
        # 1. The output layer bias 'b'
        # 2. The direct connection from the input 'x' via matrix 'W'
        # 3. The connection from the hidden layer via matrix 'U'
        out = self.b + self.W(x) + self.U(hidden_out)

        # Apply log_softmax to the output logits to get log probabilities
        log_probs = F.log_softmax(out, dim=1)

        # Return final log probabilities
        return log_probs


# PyTorch Dataset for Sliding Window
class NPLMDataset(Dataset):
    """
    A PyTorch Dataset that creates sliding window examples from a corpus
    of tokenized documents.
    """

    def __init__(self, data_dir: Path, tokenizer: WordTokenizer,
                 context_size: int):
        """
        Initializes the dataset.
        :param data_dir: The directory containing the JSONL data shards.
        :param tokenizer: The WordTokenizer object to use for encoding.
        :param context_size: The number of context words for each example.
        """
        # Store tokenizer and context size
        self.tokenizer = tokenizer
        self.context_size = context_size
        # Get ID for the padding token, defaulting to 0 if not present
        self.pad_id = tokenizer.pad_id or 0
        # Get ID for beginning of sentence token
        self.bos_id = tokenizer.bos_id

        # Create all context, target pairs from the data
        self.examples = self._create_examples(data_dir)

    def _create_examples(self, data_dir: Path) -> list[tuple[list[int], int]]:
        """
        Reads a directory of JSONL files and creates sliding window examples.
        This version correctly pads the beginning of each sentence to create
        an example for every token.
        :param data_dir: Path to the directory with .jsonl files.
        :return: A list of context, target tuples.
        """
        # Log start of process
        logging.info(f"Creating examples from data in {data_dir}...")

        # Initialize list to store examples
        examples = []

        # Find all .jsonl files
        shard_paths = sorted(data_dir.glob("*.jsonl"))

        # Iterate over each shard file
        for path in shard_paths:
            with open(path, 'r', encoding='utf-8') as f:
                # Process each line
                for line in f:
                    doc = json.loads(line)
                    # Encode text into a list of token IDs, including BOS and EOS tokens
                    token_ids = self.tokenizer.encode_text(doc['text'], with_bos_eos=True)

                    # Skip documents too short to form a single example
                    if len(token_ids) < 2:
                        continue

                    # Pad left side of the token list
                    padded_ids = [self.pad_id] * self.context_size + token_ids

                    # Slide window across original token_ids to create examples
                    # Iterate through each token in original document
                    for i in range(len(token_ids)):
                        # Context is the window of size `context_size` from padded list
                        context = padded_ids[i: i + self.context_size]
                        # Target is actual token at current position
                        target = token_ids[i]
                        # Append (context, target) to list of examples
                        examples.append((context, target))

        # Log total number of examples created
        logging.info(f"Created {len(examples)} examples.")

        # Return complete list of examples
        return examples

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset.
        :return: The number of examples.
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single example from the dataset by its index.
        :param idx: The index of the example to retrieve.
        :return: A tuple containing the context and target as PyTorch tensors.
        """
        # Get pre-created context and target
        context, target = self.examples[idx]
        # Convert context list and target integer into PyTorch tensors
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
             device: str) -> tuple[float, float]:
    """
    Calculates the loss and perplexity on an evaluation set.
    The `@torch.no_grad()` decorator disables gradient calculations,
    which saves memory and computation since we are not training.
    :param model: The NPLM model to evaluate.
    :param dataloader: The DataLoader for the validation or test set.
    :param loss_fn: The loss function.
    :param device: The device to run evaluation on ('cpu' or 'cuda').
    :return: A tuple containing the average loss and the perplexity.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize total loss and total number of tokens to zero
    total_loss = 0
    total_tokens = 0

    # Iterate over all batches in dataloader
    for context, target in dataloader:
        # Move data to the specified device
        context, target = context.to(device), target.to(device)

        # Get model's predictions
        log_probs = model(context)

        # Calculate loss for current batch
        loss = loss_fn(log_probs, target)

        # Accumulate total loss
        total_loss += loss.item() * target.numel()

        # Accumulate total number of tokens processed
        total_tokens += target.numel()

    # Calculate average loss per token.
    avg_loss = total_loss / total_tokens

    # Calculate perplexity
    perplexity = math.exp(avg_loss)

    # Return metrics
    return avg_loss, perplexity


def load_config(path: Path) -> dict:
    """
    Loads a YAML configuration file.
    :param path: The path to the YAML file.
    :return: A dictionary containing the configuration.
    """
    # Open file for reading
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Return parsed config
    return config


def main():
    """
    The main function that orchestrates the entire training process.
    """
    # Create an argument parser to handle command-line inputs
    parser = argparse.ArgumentParser(description="Train a Neural Probabilistic Language Model.")

    # Define the --config argument
    parser.add_argument("--config", type=Path, required=True, help="Path to the config YAML file.")

    # Define the required --data_dir argument
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory with processed JSONL data.")

    # Define the --save_dir argument
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to save checkpoints and results.")

    # Define the --resume_from argument
    parser.add_argument("--resume_from", type=Path, default=None, help="Path to checkpoint to resume training from.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Load config and setup
    config = load_config(args.config)

    # Create directory for saving models
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Set the random seed for PyTorch
    torch.manual_seed(config['seed'])

    # Set device to use GPU if available or CPU
    if config['device'] == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['device']
    logging.info(f"Using device: {device}")

    # Load tokenizer and data
    tokenizer = WordTokenizer.load(config['tokenizer_path'])

    # Get total vocabulary size from tokenizer
    vocab_size = len(tokenizer.id_to_token)
    logging.info(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Create training dataset
    train_dataset = NPLMDataset(args.data_dir / 'train', tokenizer, config['context_size'])

    # Create validation dataset
    val_dataset = NPLMDataset(args.data_dir / 'val', tokenizer, config['context_size'])

    # Create DataLoader for the training set
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Create DataLoader for validation set
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'] * 2)

    # Create NPLMModel
    model = NPLMModel(vocab_size=vocab_size, embedding_dim=config['embedding_dim'], context_size=config['context_size'], hidden_dim=config['hidden_dim']).to(device)

    # Log total number of parameters in the model
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Select optimizer based on config file
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # Define the loss function
    assert tokenizer.pad_id is not None, "<pad> token not found in tokenizer."
    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_id)

    # Training loop
    # Initialize tracking variables
    start_step = 0
    best_val_ppl = float('inf')

    if args.resume_from and args.resume_from.exists():
        logging.info(f"Resuming from checkpoint: {args.resume_from}")
        # Load checkpoint
        checkpoint = torch.load(args.resume_from, map_location=device)

        # Restore model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore step number and best perplexity
        start_step = checkpoint['step']
        best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))

    logging.info("Starting training...")
    # Create an iterator for training data loader
    train_iter = iter(train_loader)
    # Record start time of the training loop
    start_time = time.time()

    # Loop from starting step to the maximum number of steps
    for step in range(start_step, config['max_steps']):
        # Set model to training mode
        model.train()

        # Get next batch of data
        try:
            context, target = next(train_iter)
        except StopIteration:
            # If the iterator is exhausted (end of an epoch), create a new one.
            train_iter = iter(train_loader)
            context, target = next(train_iter)

        # Move batch to selected device
        context, target = context.to(device), target.to(device)

        # Forward pass
        log_probs = model(context)
        # Calculate loss
        loss = loss_fn(log_probs, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        if config.get('clip_grad_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        # Update model's weights
        optimizer.step()

        # Evaluation
        # Check if it's time to evaluate model.
        if (step + 1) % config['eval_every'] == 0:
            # Perform evaluation on validation set
            val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device)

            # Calculate time since start of training
            elapsed_time = time.time() - start_time

            # Log progress
            logging.info(
                f"Step {step + 1}/{config['max_steps']} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"Time: {elapsed_time:.1f}s"
            )

            # Save checkpoint
            checkpoint = {
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_ppl': best_val_ppl,
                'config': config
            }

            # Save this latest checkpoint
            torch.save(checkpoint, args.save_dir / 'last.pt')

            # If the current validation perplexity is the best save it as "best.pt".
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                checkpoint['best_val_ppl'] = best_val_ppl
                torch.save(checkpoint, args.save_dir / 'best.pt')
                logging.info(
                    f"  -> New best model saved with PPL: {best_val_ppl:.2f}")

    # Log training finished
    logging.info("Training finished.")


# This ensures the main function is called only when the script is executed directly.
if __name__ == "__main__":
    main()

