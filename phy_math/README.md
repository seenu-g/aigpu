# Math and Physics Language Model

This folder contains an experimental autoregressive language model implemented
with PyTorch. It streams Cosmopedia parquet data, trains a byte-pair encoding
(BPE) tokenizer, and trains separate models for math and physics text. The model
generates text continuations; it is not a calculator or a reliable source of
scientific answers.

## Requirements

- Python 3.10 or later
- PyTorch (for AMD GPUs, use a ROCm-enabled build compatible with the server)
- `datasets`, `tokenizers`, and `matplotlib`

For ROCm PyTorch installation details, use the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).

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

## Training behavior and outputs

Training automatically uses a supported GPU when available and otherwise uses
the CPU. At startup, it prints the selected device and GPU name; AMD GPUs
supported by a ROCm-enabled PyTorch build are identified as ROCm.
Progress is printed as `completed_steps/max_steps`; validation runs every 500
steps. Resumable checkpoints are saved every 1,000 steps. The final model is
saved to `checkpoints/final_llm_<subject>.pt`, with a model summary at
`checkpoints/model_summary.txt` and metrics in
`checkpoints/training_metrics_<subject>.csv`. The CSV records training loss
for each step and validation loss at evaluation steps. Tokenizers are stored
in `checkpoints/`.

The tokenizer command creates both subject tokenizers, so the first run needs
all four dataset folders. Generation requires the matching subject tokenizer
and final checkpoint. Evaluation reports are written under
`checkpoints/generation_evals/`.

## Project files

| File | Purpose |
| --- | --- |
| `train.py` | Training loop, validation, checkpointing, and command-line options. |
| `tokenizer.py` | Builds math and physics BPE tokenizer JSON files. |
| `load_data.py` | Streams parquet records and creates fixed-length training sequences. |
| `model.py` | Model architecture and configuration. |
| `checkpoint.py` | Checkpoint compatibility, save/load, and training metrics helpers. |
| `training_stats.py` | Dataset token-length histograms, loss plots, and loss-trend diagnostics. |
| `generate_text.py` | Generates a text continuation from a trained model. |
| `evaluate_generation.py` | Generates repeatable evaluation samples and reports. |
| `test_pipeline.py` | Checks that a tokenizer can be loaded and used by the model. |
| `run_training.sh` | Creates/reuses the virtual environment and installs dependencies. |

## Command-line commands

Run these commands from this folder on the GPU server.

### Set up the environment


Optional ROCm-specific PyTorch index:
```bash
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/<matching-rocm-version>" 
```

Activate the environment:

```bash
source .venv/bin/activate
```

### Train the tokenizer

Train both subject tokenizers with the default data directory and vocabulary
size of 32,000:
```bash
python tokenizer.py
```

Choose another data directory, output directory, or vocabulary size:
```bash
python tokenizer.py --data_dir "/path/to/cosmopedia_raw/data" --out_dir checkpoints --vocab_size 32000
```
### Train the model

Train math for the default 10,000 steps:
```bash
python train.py
```

Specify the subject and step limit:
```bash
python train.py --subject math --max-steps 10000
python train.py --subject physics --max-steps 10000
```

Resume from the latest checkpoint for the subject, or from a specific checkpoint:
```bash
python train.py --subject math --max-steps 20000 --resume
python train.py --subject math --max-steps 20000 --resume-from "checkpoints/checkpoint_math_step_10000.pt"
```

Plot dataset token lengths and the training/validation losses already recorded
in the subject's metrics CSV:

```bash
python training_stats.py --subject math
python training_stats.py --subject physics --data-dir "/path/to/cosmopedia_raw/data"
```

Print loss-trend diagnostics without reading the dataset or creating plots:

```bash
python training_stats.py --subject math --analyze-loss-only
```

### Generate and evaluate text

Generate a continuation:

```bash
python generate_text.py --subject math --prompt "Theorem: If a function is differentiable, then"
```

`--max-new-tokens` defaults to 150 and `--temperature` defaults to 0.5. Evaluate
both subjects, or just math:

```bash
python evaluate_generation.py
python evaluate_generation.py --subject math --max-new-tokens 150 --temperature 0
```
