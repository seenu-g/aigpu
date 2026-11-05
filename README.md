# AI GPU Shakespeare Language Model

## 1. Project description

This folder contains a small, experimental autoregressive language model built
with PyTorch. It learns to predict the next token from a corpus of Shakespeare
texts and can generate a text continuation from a prompt.

The project includes a byte-level BPE tokenizer, learned token and position
embeddings, causal multi-head self-attention, a feed-forward network, model
training with train/validation loss reporting, periodic resumable checkpoints,
and a one-shot terminal generation tool. It generates text continuations; it is
not a chatbot.

The default corpus is the `.txt` files in `literature/` (currently 42 files).
The default training setup uses a vocabulary target of 512, an embedding
dimension of 128, 4 attention heads, a context length of 64, and one
Transformer block. These are starting settings, not tuned or guaranteed to be
optimal.

## 2. Project files

| File or folder | Description |
| --- | --- |
| `attention.py` | Causal attention heads, multi-head attention, feed-forward network, and Transformer block. |
| `checkpoint.py` | Saves and loads model/optimizer checkpoints, including iteration and random-number-generator state for resuming. |
| `data_loader.py` | Reads and concatenates `.txt` files from a data folder. |
| `embeddings.py` | Learned token and position embeddings. |
| `evaluation.py` | Estimates train/validation loss and evaluates fixed-prompt samples for repetition and source overlap. |
| `generate_text.py` | Generates a text continuation and saves per-token probability diagnostics. |
| `inference.py` | Token-by-token generation and sampling logic used by the generation CLI. |
| `inference_stats.py` | Saves inference token-probability reports and graphs. |
| `inspect_tokenizer.py` | Lists learned BPE pieces from a tokenizer cache, highlighting pieces containing whitespace. |
| `main.py` | Demonstration of tokenization, embeddings, and attention components; it does not train the language model. |
| `model.py` | `LanguageModel` architecture and forward pass. |
| `training_stats.py` | Stores and plots training loss history and dataset statistics. |
| `requirements.txt` | Python dependency list. |
| `tokenizers.py` | Character tokenizer and byte-level BPE build, cache, restore, encode, and decode functions. |
| `train_config.py` | Loads and validates training configuration and applies command-line overrides. |
| `training_data.py` | Loads and tokenizes the corpus, creates train/validation splits, and samples batches. |
| `training_config.json` | Default model, training, and dataset settings for `train_model.py`. |
| `train_model.py` | Training entry point and model training/evaluation loop; delegates config, data preparation, and stats to their own modules. |
| `literature/` | Default training corpus. Place `.txt` training documents here, or provide another folder with `--data-dir`. |
| `cache/` | Generated BPE tokenizer cache files, separated by target vocabulary size and validated against the corpus text. |
| `checkpoints/` | Default location for saved model checkpoints. Created when training saves its first checkpoint. |
| `output/` | Generated inference-probability reports and benchmark graphs. |

## 3. Requirements

- Python 3.10 or later is recommended.
- PyTorch is required. Install a build appropriate for your Python version and
  hardware; use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
  for CUDA support. CPU-only runs are also supported.

Install the PyTorch dependency listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Choose the PyTorch package appropriate for your Python version and hardware;
the official PyTorch installation selector provides CUDA-specific install
commands.

Before checking in or copying the project to another machine, compile-check
the Python files from the project root:

```bash
python -m compileall -q .
```

Generated checkpoints, tokenizer caches, loss history, and output graphs are
created at runtime; they are not required to start a fresh training run.

## 4. Training

Run commands from this folder with the checked-in JSON configuration:

```bash
python train_model.py
```

`training_config.json` holds the model, training, and data settings. Pass
`--config` to select a different JSON file. Command-line options override the
config file, which overrides built-in defaults.

For example, override the learning rate, model size, and evaluation frequency
without editing the config:

```bash
python train_model.py --learning-rate 0.0001 --embedding-dim 256 --eval-interval 50
```

To use a separate config on a server, copy `training_config.json`, edit its
data and output paths, and pass that file:

```bash
python train_model.py --config configs/server.json
```

Paths in config files are resolved relative to the config file. Paths passed
on the command line are resolved relative to the current working directory.
The configured data directory must contain `.txt` files; otherwise the current
loader prints a warning and uses a tiny fallback text, which is not suitable
for a real training run.

The default config:

1. Loads and joins all `.txt` files in `literature/`.
2. Loads or builds the BPE tokenizer cache in `cache/`.
3. Tokenizes the corpus and uses the first 90% for training and the last 10%
   for validation by default; change `data.train_fraction` in the config or
   override it with `--train-fraction`.
4. Trains on CUDA when available, otherwise on CPU.
5. Reports completed iterations and train/validation loss at the evaluation
   interval.
6. Records loss history in `checkpoints/training_loss.csv`.
7. Refreshes `output/training_validation_loss.png` after each evaluation so
   you can follow the loss trend while training.
8. Saves a resumable checkpoint every 100 iterations and at the end of the run.

The default checkpoint path is `checkpoints/language_model.pt`. The first run
may take longer because it has to build the tokenizer cache. The corpus is
still read and tokenized on later runs.

`training_data.py` currently loads the complete text corpus and tokenizes it
into an in-memory tensor before training. Each iteration samples a batch of
random token windows from that tensor; it does not stream the next data shard
from disk. This works for the current small corpus, but larger datasets may
need a streaming or sharded data pipeline to avoid holding the full corpus in
memory.

### Common training options

```text
--max-iters             Target total number of training iterations (default: 1000)
--eval-interval         Iterations between train/validation loss reports (default: 100)
--checkpoint-interval   Iterations between checkpoint saves (default: 100)
--sample-interval       Iterations between fixed-prompt sample evaluations (default: 500)
--sample-tokens         New tokens generated per evaluation prompt (default: 100)
--batch-size            Sequences per training batch (default: 32)
--block-size            Maximum context length (default: 64)
--learning-rate         AdamW learning rate (default: 0.0003)
--embedding-dim         Embedding width (default: 128)
--num-heads             Attention heads (default: 4; must divide embedding dimension)
--target-vocab-size     BPE vocabulary target (default: 512; minimum: 256)
--data-dir              Folder containing training .txt files
--train-fraction        Fraction of tokenized data for training (default: 0.9)
--checkpoint-path       Checkpoint file path
--loss-history-path     CSV file for train/validation loss history
--loss-plot-path        Loss graph refreshed after each evaluation
--resume                Resume from the checkpoint at --checkpoint-path
--no-resume             Do not resume, overriding a true config setting
```

For example, train toward 5,000 total iterations with the default config:

```bash
python train_model.py --max-iters 5000 --eval-interval 100
```

If training is interrupted, resume toward the same total target:

```bash
python train_model.py --max-iters 5000 --eval-interval 100 --resume
```

For a server run, the same command can point at its config and override only
the values you want to experiment with:

```bash
python train_model.py --config configs/server.json --max-iters 20000 --resume
```

`--max-iters` is the target total, so resuming at iteration 1,000 continues
with iteration 1,001. Resuming requires the same corpus, model settings, batch
size, learning rate, and train/validation split as the checkpoint. The script
checks these values and rejects incompatible checkpoints. Use a different
`--checkpoint-path` for a separate model run.

At each sample interval (and at the final iteration), the trainer saves a
checkpoint and evaluates three fixed prompts. The same per-prompt sampling
seed is used at each checkpoint to make comparisons more repeatable. Each
generated continuation is printed with a repeated 4-gram rate and the count of
40-character windows found verbatim in the training split. These are
diagnostic signals for comparison, not definitive measures of quality or
memorization. Exact overlap can be expected for common phrases in a small
corpus.

## 5. Generate a text continuation

After training has created a checkpoint, run:

```bash
python generate_text.py "To be, or not to be"
```

The command prints the prompt followed by a sampled continuation and exits.
Useful options include:

```bash
python generate_text.py "Good morrow" --max-new-tokens 100 --temperature 0.8 --top-k 40
```

Inference also writes timestamped CSV and PNG reports to `output/` by default.
The report captures
each sampled token's probability under both the raw model distribution and
the temperature/top-k sampling distribution, the model's highest next-token
probability, and distribution entropy. Use `--show-token-probabilities` to
print these diagnostics in the terminal, `--seed` to make sampling repeatable,
and `--output-dir` to choose another report folder. Timestamped reports are
preserved to support comparisons between runs.

Generation loads the checkpoint and uses the tokenizer associated with the
checkpoint's BPE vocabulary size. If using a different training corpus, pass
the same data directory used for training:

```bash
python generate_text.py "A prompt" --data-dir ../other-literature
```

## 6. Inspect the tokenizer

After a tokenizer cache has been created, inspect learned pieces containing whitespace:

The default cache is `cache/bpe_512.json`. To inspect a different vocabulary size, pass its cache path, and use `--all-tokens` to list every learned piece:

```bash
python inspect_tokenizer.py --cache-path cache/bpe_256.json --all-tokens
```

Whitespace-containing pieces show that BPE merged across a space or other
whitespace boundary. The tokenizer learns frequent byte sequences, not language labels, so this inspection cannot determine whether a piece is specifically Latin or English. Use representative words or phrases from each language and compare their tokenization to investigate that.

## 7. Plot dataset sequence lengths

Generate a histogram of token lengths, with one sequence per `.txt` file:

```bash
python training_stats.py
```

By default, the script reads `literature/`, uses the training tokenizer's
default BPE vocabulary size of 512, and saves
`output/sequence_length_histogram.png`. Options let you select another dataset
folder, vocabulary size, bin count, or output path:

```bash
python training_stats.py --data-dir ../other-literature --target-vocab-size 512 --bins 25 --output results/sequence_lengths.png
```

After training, the same command also plots the training and validation losses
from `checkpoints/training_loss.csv` to
`output/training_validation_loss.png`. Use `--loss-history` and `--loss-output`
to select different paths. When resuming training, the loss history is
appended; start a new run to replace it.

During training, the console reports validation loss at every evaluation
interval and refreshes the loss graph at `output/training_validation_loss.png`.
Reopen or refresh the image to see the latest point; use `--eval-interval` to
control how often the validation loss and graph are updated.

## 8. Data and generated artifacts

The data loader reads sorted `*.txt` files and joins them with newline
separators. If no text files are found, it warns and uses a short fallback
sample; use a real corpus for meaningful training.

The tokenizer cache is keyed by both the target vocabulary size and the
training text hash. A different corpus causes a new tokenizer to be built. A
checkpoint also records the training text hash, preventing accidental resume
with a different corpus.

The model is a learning project and the generated output may be repetitive,
incoherent, or unlike the source style. Loss values help track learning, but
inspect generated samples as well; this model has not been comprehensively
tuned or evaluated.
