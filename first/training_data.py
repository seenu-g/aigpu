import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch

from data_loader import load_dataset
from tokenizers import load_or_build_bpe_tokenizer


def _start_timing(label):
    started_at = datetime.now().astimezone()
    started_counter = perf_counter()
    print(f"{label} started at {started_at.isoformat(timespec='seconds')}")
    return started_counter


def _finish_timing(label, started_counter):
    finished_at = datetime.now().astimezone()
    elapsed = perf_counter() - started_counter
    print(f"{label} finished at {finished_at.isoformat(timespec='seconds')} ({elapsed:.2f}s)")


@dataclass
class TrainingData:
    get_batch: Callable[[str], tuple[torch.Tensor, torch.Tensor]]
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    vocab_size: int
    dataset_hash: str
    training_text: str
    token_count: int


def prepare_training_data(
    data_dir,
    target_vocab_size,
    batch_size,
    block_size,
    device,
    train_fraction=0.9,
):
    """Load, tokenize, split the corpus, and provide random batch sampling."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    data_load_started = _start_timing("Corpus loading")
    raw_text = load_dataset(data_dir)
    _finish_timing("Corpus loading", data_load_started)
    dataset_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    encode, decode, vocab_size = load_or_build_bpe_tokenizer(
        raw_text, target_vocab_size=target_vocab_size
    )

    tokenization_started = _start_timing("Dataset tokenization")
    data = torch.tensor(encode(raw_text), dtype=torch.long)
    _finish_timing("Dataset tokenization", tokenization_started)
    split_index = int(train_fraction * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]
    if min(len(train_data), len(val_data)) <= block_size:
        raise ValueError(
            "The training and validation splits must each contain more tokens "
            "than the configured block size"
        )
    print(
        f"Dataset split: {train_fraction:.1%} training / "
        f"{1 - train_fraction:.1%} validation "
        f"({len(train_data):,} / {len(val_data):,} tokens)"
    )

    training_text_started = _start_timing(
        "Training split decode for source-overlap checks"
    )
    training_text = decode(train_data.tolist())
    _finish_timing(
        "Training split decode for source-overlap checks", training_text_started
    )

    def get_batch(split):
        if split == "train":
            split_data = train_data
        elif split == "val":
            split_data = val_data
        else:
            raise ValueError(f"Unknown data split: {split}")

        starts = torch.randint(
            len(split_data) - block_size,
            (batch_size,),
        )
        inputs = torch.stack(
            [split_data[start : start + block_size] for start in starts]
        )
        targets = torch.stack(
            [
                split_data[start + 1 : start + block_size + 1]
                for start in starts
            ]
        )
        return inputs.to(device), targets.to(device)

    return TrainingData(
        get_batch=get_batch,
        encode=encode,
        decode=decode,
        vocab_size=vocab_size,
        dataset_hash=dataset_hash,
        training_text=training_text,
        token_count=len(data),
    )
