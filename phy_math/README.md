# Math and Physics Language Model

This folder contains an experimental autoregressive language model implemented
with PyTorch. It streams Cosmopedia parquet data, trains a byte-pair encoding
(BPE) tokenizer, and trains separate models for math and physics text. The model
generates text continuations; it is not a calculator or a reliable source of
scientific answers.

## Requirements

- Python 3.10 or later
- PyTorch (install a build suitable for your Python version and hardware; see
  the [official installation selector](https://pytorch.org/get-started/locally/))
- `datasets` and `tokenizers`

Install the Python dependencies:

```bash
python -m pip install torch datasets tokenizers
```

## Data

The scripts expect parquet files under a Cosmopedia data directory, organized
as follows:

```text
cosmopedia_raw/data/
├── auto_math_text/*.parquet
├── stanford/*.parquet
├── openstax/*.parquet
└── khanacademy/*.parquet
```

Each parquet record should have a non-empty `text` field. Math training uses
`auto_math_text`; physics training uses `stanford`, `openstax`, and
`khanacademy`, filtered by physics-related metadata keywords. A deterministic
10% of records is reserved for validation.

`tokenizer.py` accepts a custom data directory. Training currently uses the
default `/home/guru/data/cosmopedia_raw/data` path in `train.py`; if your data is
elsewhere, update `get_subject_data_config` there to point to the same directory
before training.

## Tokenizer

From this directory, train both subject tokenizers (default vocabulary size:
32,000):

```bash
python tokenizer.py
```

To select a data directory, output directory, or vocabulary size:

```bash
python tokenizer.py --data_dir "D:\data\cosmopedia_raw\data" --vocab_size 32000
```

This creates `checkpoints/math_tokenizer.json` and
`checkpoints/physics_tokenizer.json`. Use `--out_dir` to choose another output
directory. Keep the tokenizer used for training: checkpoints verify its
contents when loading a model.

## Training

The default is the math model with a limit of 10,000 steps:

```bash
python train.py
```

Select the subject and maximum number of steps with `--subject` and
`--max-steps`:

```bash
python train.py --subject physics --max-steps 10000
```

Resume from the latest checkpoint for that subject, or provide a specific
checkpoint:

```bash
python train.py --subject math --max-steps 20000 --resume
python train.py --subject math --max-steps 20000 --resume-from "checkpoints\checkpoint_math_step_10000.pt"
```

Training automatically uses CUDA when available and otherwise uses the CPU.
Progress is printed as `completed_steps/max_steps`; validation runs every 500
steps. Resumable checkpoints are saved every 1,000 steps. The final model is
saved to `checkpoints/final_llm_<subject>.pt`, with a model summary at
`checkpoints/model_summary.txt` and metrics in
`checkpoints/training_metrics_<subject>.csv`. That CSV records training loss
for each step and validation loss at evaluation steps. Tokenizers are also
stored in `checkpoints/`.

## Generate text

After training, generate a continuation using the matching subject tokenizer
and final checkpoint:

```bash
python generate_text.py --subject math --prompt "Theorem: If a function is differentiable, then"
```

Useful options include `--max-new-tokens` (default 150) and `--temperature`
(default 0.5). Use `--subject physics` for the physics model.

## Evaluate generation

Generate evaluation samples for both subjects, or select one:

```bash
python evaluate_generation.py
python evaluate_generation.py --subject math --max-new-tokens 150 --temperature 0
```

The evaluation writes text reports under `checkpoints/generation_evals/`. It
requires the corresponding trained model checkpoint and tokenizer.

## Project files

| File | Purpose |
| --- | --- |
| `train.py` | Training loop, validation, checkpointing, and command-line options. |
| `tokenizer.py` | Builds math and physics BPE tokenizer JSON files. |
| `load_data.py` | Streams parquet records and creates fixed-length training sequences. |
| `model.py` | Model architecture and configuration. |
| `checkpoint.py` | Checkpoint compatibility, save/load, and training metrics helpers. |
| `generate_text.py` | Generates a text continuation from a trained model. |
| `evaluate_generation.py` | Generates repeatable evaluation samples and reports. |
| `test_pipeline.py` | Checks that a tokenizer can be loaded and used by the model. |
