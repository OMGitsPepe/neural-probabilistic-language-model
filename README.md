# Neural Probabilistic Language Model

This project implements a feed-forward Neural Probabilistic Language (NPLM) inspired by the groundbreaking 2003 paper by
Bengio, "A Neural Probabilistic Language Model". The model learns distribution representations for words embeddings to
combat the curse of dimensionality that affects traditional n-gram models. By mapping discrete word indices to a continuous 
vector space, the model can generalize to unseen n-grams and predict the next word in a sequence based on a fixed size
context windows.

This implementation includes scripts for data downloading, preprocessing, vocabulary building, training and evaluation.

## Quickstart Guide

Follow these steps to set up the environment process the data and replicate the results.

### 1. Installation

First, clone the repository and install the required Python packages using the provided `requirement.txt` file.
```commandline
git clone https://github.com/OMGitsPepe/neural-probabilistic-language-model.git

cd nplm_project

pip install -r requirements.txt
```

### 2. Download Data

Download the Brown Corpus using the provided script. This will download the data via NLTK and save it as a raw text file.

```commandline
python -m nplm.download --dataset brown --out_dir data/raw/brown
```

### 3. Preprocess Data

Process the raw text into sharded JSON files suitable for training. This scripts splits the data into training (80%),
validation (10%) and test (10%) set.

```commandline
python -m nplm.preprocess --input_dir data/raw/brown --output_dir data/brown_jsonl --shard_size 5000 --lowercase
```

### 4. Build Vocabulary

Create the word level vocabulary from the training data. This artifact is important for both training and evaluation.

```commandline
python -m nplm.build_word_vocab --jsonl_dir data/brown_jsonl/train --output_path artifacts/brown_word_vocab.json --min_freq 2 --max_vocab 20000 --lowercase
```

### 5. Train Model

Train the model using the `tiny.yaml` configuration. This configuration is designed to run quickly on a CPU.

```commandline
python -m nplm.train --config configs/tiny.yaml --data_dir data/brown_jsonl --save_dir runs/brown_tiny
```

### 6. Evaluate Model

Finally, evaluate the best performing checkpoint on the test set to get the final perplexity score.

```commandline
python -m nplm.evaluate --checkpoint runs/brown_tiny/best.pt --data_dir data/brown_jsonl --out_json results/metrics.json
```

After running this command, the final metrics will be saved in results/metrics.json. The expected should be approximately **173.88**.

## Hardware and Usage Notes

**Hardware**: Macbook Pro Apple M1 Max chip and 32 GB RAM

**Environment**: PyCharm Pro and Anaconda environment

**CPU/GPU**: The code will automatically detect and use a CUDA-enable GPU if on is available. The `tiny.yaml` configuration 
is specifically designed for CPU execution and should complete training under 10 minutes on a standard modern laptop.
To reproduce the results, a standard CPU is sufficient. No special hardware is required.
