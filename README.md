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
| `evaluation.py` | Generates fixed-prompt checkpoint samples and reports repetition and exact source-overlap signals. |
| `generate_text.py` | Command-line interface for generating one text continuation from a prompt. |
| `inference.py` | Token-by-token generation and sampling logic used by the generation CLI. |
| `inspect_tokenizer.py` | Lists learned BPE pieces from a tokenizer cache, highlighting pieces containing whitespace. |
| `main.py` | Demonstration of tokenization, embeddings, and attention components; it does not train the language model. |
| `model.py` | `LanguageModel` architecture and forward pass. |
| `requirements.txt` | Python dependency list. |
| `tokenizers.py` | Character tokenizer and byte-level BPE build, cache, restore, encode, and decode functions. |
| `train_model.py` | Training entry point, argument parsing, training loop, loss reporting, and model summary. |
| `literature/` | Default training corpus. Place `.txt` training documents here, or provide another folder with `--data-dir`. |
| `cache/` | Generated BPE tokenizer cache files, separated by target vocabulary size and validated against the corpus text. |
| `checkpoints/` | Default location for saved model checkpoints. Created when training saves its first checkpoint. |

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

## 4. Training

Run commands from this folder:

```bash
python train_model.py --max-iters 1000 --eval-interval 100
```

By default, the script:

1. Loads and joins all `.txt` files in `literature/`.
2. Loads or builds the BPE tokenizer cache in `cache/`.
3. Tokenizes the corpus and uses the first 90% for training and the last 10%
   for validation.
4. Trains on CUDA when available, otherwise on CPU.
5. Reports completed iterations and train/validation loss at the evaluation
   interval.
6. Saves a resumable checkpoint every 100 iterations and at the end of the run.

The default checkpoint path is `checkpoints/language_model.pt`. The first run
may take longer because it has to build the tokenizer cache. The corpus is
still read and tokenized on later runs.

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
--checkpoint-path       Checkpoint file path
--resume                Resume from the checkpoint at --checkpoint-path
```

For example, train toward 5,000 total iterations:

```bash
python train_model.py --max-iters 5000 --eval-interval 100
```

If training is interrupted, resume toward the same total target:

```bash
python train_model.py --max-iters 5000 --eval-interval 100 --resume
```

`--max-iters` is the target total, so resuming at iteration 1,000 continues
with iteration 1,001. Resuming requires the same corpus, model settings, batch
size, and learning rate as the checkpoint. The script checks these values and
rejects incompatible checkpoints. Use a different `--checkpoint-path` for a
separate model run.

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

## 7. Data and generated artifacts

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
