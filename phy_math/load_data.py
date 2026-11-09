import glob
import hashlib
import os

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset


def get_parquet_files(data_dir, split_folders):
    """Return all parquet files from the requested dataset folders."""
    if isinstance(split_folders, str):
        split_folders = [split_folders]

    files = sorted(
        parquet_path
        for split_folder in split_folders
        for parquet_path in glob.glob(
            os.path.join(data_dir, split_folder, "*.parquet")
        )
    )
    if not files:
        folders = ", ".join(split_folders)
        raise FileNotFoundError(
            f"No parquet files found in {data_dir} for source folder(s): {folders}"
        )
    return files


def split_test_valid_data(row, validation_fraction=0.1):
    """Assign a record deterministically to validation or training."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not isinstance(row, dict):
        raise ValueError("Dataset row must be a mapping")

    content = "\0".join(
        str(row.get(field, "") or "")
        for field in ("text", "prompt", "audience", "format")
    )
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") < validation_fraction * (1 << 64)


def read_parquet_file_data(
    files, keywords=None, split="all", validation_fraction=0.1
):
    """Stream valid text rows filtered by metadata and deterministic split."""
    if split not in {"all", "train", "validation"}:
        raise ValueError("split must be 'all', 'train', or 'validation'")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not files:
        raise ValueError("At least one parquet file is required")

    try:
        dataset = load_dataset(
            "parquet",
            data_files=files,
            split="train",
            streaming=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not stream parquet files {files}: {exc}") from exc

    for row in dataset:
        if not isinstance(row, dict):
            raise ValueError("Parquet dataset yielded a row that is not a mapping")

        text = row.get("text", "")
        if not isinstance(text, str) or not text:
            continue

        if keywords:
            metadata = " ".join(
                str(row.get(field, "") or "")
                for field in ("prompt", "audience", "format")
            ).lower()
            if not any(keyword in metadata for keyword in keywords):
                continue

        is_validation = split_test_valid_data(row, validation_fraction)
        if split == "train" and is_validation:
            continue
        if split == "validation" and not is_validation:
            continue

        yield text


class TrainingDataset(IterableDataset):
    """Stream parquet text as fixed-length next-token training sequences."""

    def __init__(
        self,
        data_dir,
        split_folders,
        tokenizer_path,
        max_seq_len=2048,
        keywords=None,
        split="train",
        validation_fraction=0.1,
    ):
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")

        self.max_seq_len = max_seq_len
        self.split = split
        self.validation_fraction = validation_fraction
        try:
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Could not load tokenizer from {tokenizer_path}: {exc}"
            ) from exc

        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        if self.bos_id is None or self.eos_id is None:
            raise ValueError("Tokenizer must define both <s> and </s> special tokens")

        self.keywords = keywords
        self.files = get_parquet_files(data_dir, split_folders)

    def __iter__(self):
        token_buffer = []
        yielded_any = False

        for text in read_parquet_file_data(
            self.files,
            keywords=self.keywords,
            split=self.split,
            validation_fraction=self.validation_fraction,
        ):
            try:
                ids = [self.bos_id] + self.tokenizer.encode(text).ids + [self.eos_id]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Could not tokenize a parquet row's text field"
                ) from exc
            token_buffer.extend(ids)

            while len(token_buffer) >= self.max_seq_len + 1:
                chunk = token_buffer[: self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len :]
                inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yielded_any = True
                yield inputs, targets

        if not yielded_any:
            raise ValueError(
                f"No training sequences were produced from {self.files}; "
                "check the text fields, filters, and max_seq_len."
            )
