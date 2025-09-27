# NPLM Project: Experiment Log
This document summarizes the choices, configurations, and results for the Neural Probabilistic Language Model project.

## Dataset Selection
Dataset: Brown Corpus (via NLTK)

Reasoning: The Brown Corpus was selected for two primary reasons. First, it is the same dataset used in the original 
Bengio et al. (2003) paper, allowing for a more direct and faithful implementation of their work. Second, its accessibility 
through the nltk library makes it easy to download and reproduce the experimental setup without requiring manual data sourcing.

Preprocessing
The raw corpus was preprocessed into a standardized format suitable for training.

- Document Definition: Each sentence in the Brown Corpus was treated as a single "document." This is a natural choice for 
a sentence-level corpus and provides a large number of training examples.

- Data Splits: The entire dataset was shuffled and split into training (80%), validation (10%), and test (10%) sets. 
- This standard split ensures that the model is trained on one set of data, tuned on another, and finally evaluated on a 
- completely unseen third set.

- Output Format: The splits were written to sharded .jsonl files, with each line containing a JSON object {"text": "..."}. 
- This format is efficient to read and parse during training. All text was lowercased during this step to reduce 
- vocabulary size.

## Tokenizer Choice
Tokenizer: Word-Level (Bengio-style)

Reasoning: A classic word-level tokenizer was chosen to align with the methodology of the original paper. This approach 
creates a vocabulary based on unique words in the training corpus.

- Vocabulary Size: The vocabulary was capped at the 20,000 most frequent words.

- Frequency Cutoff: A minimum frequency of 2 was used to filter out rare words and typos.

- Special Tokens: The vocabulary includes <pad>, <unk>, <bos>, and <eos> tokens to handle padding, unknown words, 
and sentence boundaries.

## Model Configurations
Two main configurations were used for training, defined in tiny.yaml and medium.yaml.

- tiny.yaml: This configuration was designed for rapid testing and reproducibility (< 5 minutes on a modern CPU). It uses a small 
context size (3), smaller embedding and hidden dimensions (64 and 128), and a limited number of training steps. Its purpose was to 
verify the end-to-end functionality of the training and evaluation pipeline.

- medium.yaml: This configuration uses a larger model (embedding_dim: 128, hidden_dim: 256) and more training steps to achieve better 
performance, at the cost of longer training time.

## Observations and Results
Both configurations were successfully trained and evaluated. Below is a summary of their performance.

### Tiny Model Results
The smaller model trained quickly and achieved solid results, verifying the correctness of the pipeline.

- Best Validation Perplexity: ~174.78
- Final Test Perplexity: ~173.84
- Generalization: The closeness of the validation and test perplexity scores indicates that the model generalized well to unseen data.

### Medium Model Results
The larger model showed clear signs of overfitting during its longer training run.

- Best Validation Perplexity: 195.74 (achieved at step 4,000)

- Final Test Perplexity: 195.04

- Overfitting: After reaching its best performance at step 4,000, the model's validation perplexity began to steadily 
increase for the remaining 16,000 steps, finishing at ~405. This is a classic sign of overfitting, where the model 
starts to memorize the training data rather than learning general patterns. The early stopping mechanism (saving the best.pt model) 
was crucial for capturing the model at its peak performance.

### Comparison and Conclusion

| Model Configuration | Best Validation Perplexity | Final Test Perplexity |
|---------------------|----------------------------|-----------------------|
| Tiny                | ~173.84                    | ~174.78               |
| Medium              | 195.74                     | 195.04                |

Interestingly, the smaller tiny model outperformed the larger medium model. The medium model, with its greater capacity, 
began to overfit the training data relatively early in the training process. This suggests that simply increasing model 
size is not a guaranteed path to better performance. The larger model would likely benefit from regularization techniques
(like dropout) or further hyperparameter tuning (e.g., a lower learning rate) to prevent overfitting and achieve 
its full potential.







