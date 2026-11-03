import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter


def _log_timing(label, started_counter):
    finished_at = datetime.now().astimezone()
    elapsed = perf_counter() - started_counter
    print(f"{label} finished at {finished_at.isoformat(timespec='seconds')} ({elapsed:.2f}s)")


def build_character_tokenizer(raw_text):
    """Create character-level encode and decode functions."""
    chars = sorted(set(raw_text))
    char_to_int = {char: index for index, char in enumerate(chars)}
    int_to_char = {index: char for index, char in enumerate(chars)}

    def encode(text):
        return [char_to_int[char] for char in text]

    def decode(ids):
        return "".join(int_to_char[index] for index in ids)

    return encode, decode, len(chars)


def build_bpe_tokenizer(raw_text, target_vocab_size=512):
    """Train a byte-pair tokenizer and return its encode/decode functions."""
    merges, vocab = _build_bpe_learned_data(raw_text, target_vocab_size)
    return _make_bpe_tokenizer(merges, vocab)


def _build_bpe_learned_data(raw_text, target_vocab_size):
    started_at = datetime.now().astimezone()
    started_counter = perf_counter()
    print(f"BPE learned-data build started at {started_at.isoformat(timespec='seconds')}")

    token_ids = list(raw_text.encode("utf-8"))
    base_vocab_size = 256
    num_merges = target_vocab_size - base_vocab_size

    merges = {}
    vocab = {index: bytes([index]) for index in range(base_vocab_size)}

    print(f"Training BPE... Initial sequence length: {len(token_ids)}")

    for merge_index in range(num_merges):
        stats = Counter(zip(token_ids, token_ids[1:]))
        if not stats:
            break

        top_pair = stats.most_common(1)[0][0]
        new_id = base_vocab_size + merge_index
        vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
        merges[top_pair] = new_id

        new_ids = []
        index = 0
        while index < len(token_ids):
            if index < len(token_ids) - 1 and (token_ids[index], token_ids[index + 1]) == top_pair:
                new_ids.append(new_id)
                index += 2
            else:
                new_ids.append(token_ids[index])
                index += 1
        token_ids = new_ids

        if (merge_index + 1) % 50 == 0 or merge_index + 1 == num_merges:
            print(f"  Merge {merge_index + 1}/{num_merges} complete. Current length: {len(token_ids)}")
    print(f"BPE training finished. Compressed sequence down to: {len(token_ids)}")
    _log_timing("BPE learned-data build", started_counter)
    return merges, vocab


def _make_bpe_tokenizer(merges, vocab):
    def encode(text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = Counter(zip(tokens, tokens[1:]))
            pair = min(stats, key=lambda item: merges.get(item, float("inf")))
            if pair not in merges:
                break

            new_id = merges[pair]
            new_ids = []
            index = 0
            while index < len(tokens):
                if index < len(tokens) - 1 and (tokens[index], tokens[index + 1]) == pair:
                    new_ids.append(new_id)
                    index += 2
                else:
                    new_ids.append(tokens[index])
                    index += 1
            tokens = new_ids
        return tokens

    def decode(ids):
        text_bytes = b"".join(vocab[index] for index in ids)
        return text_bytes.decode("utf-8", errors="replace")

    return encode, decode, len(vocab)


def load_or_build_bpe_tokenizer(raw_text, target_vocab_size=512, cache_path=None):
    """Load a compatible saved BPE tokenizer, or train and save one."""
    started_at = datetime.now().astimezone()
    started_counter = perf_counter()
    print(f"Tokenizer initialization started at {started_at.isoformat(timespec='seconds')}")

    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "cache" / f"bpe_{target_vocab_size}.json"
    else:
        cache_path = Path(cache_path)

    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cache.get("format_version") == 1
            and cache.get("target_vocab_size") == target_vocab_size
            and cache.get("training_text_sha256") == text_hash
        ):
            merges, vocab = restore_bpe_learned_data(cache["merges"])
            print(f"Loaded BPE tokenizer from {cache_path}")
            tokenizer = _make_bpe_tokenizer(merges, vocab)
            _log_timing("Tokenizer initialization", started_counter)
            return tokenizer

    print("No matching BPE tokenizer cache found; training a new tokenizer.")
    merges, vocab = _build_bpe_learned_data(raw_text, target_vocab_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "format_version": 1,
        "target_vocab_size": target_vocab_size,
        "training_text_sha256": text_hash,
        "merges": [
            [left, right, token_id]
            for (left, right), token_id in sorted(merges.items(), key=lambda item: item[1])
        ],
    }
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    print(f"Saved BPE tokenizer to {cache_path}")
    tokenizer = _make_bpe_tokenizer(merges, vocab)
    _log_timing("Tokenizer initialization", started_counter)
    return tokenizer


def restore_bpe_learned_data(serialized_merges):
    base_vocab_size = 256
    merges = {}
    vocab = {index: bytes([index]) for index in range(base_vocab_size)}

    for merge_index, serialized_merge in enumerate(serialized_merges):
        if (
            not isinstance(serialized_merge, list)
            or len(serialized_merge) != 3
            or any(not isinstance(value, int) for value in serialized_merge)
        ):
            raise ValueError("Invalid BPE tokenizer cache: malformed merge entry")

        left, right, token_id = serialized_merge
        expected_id = base_vocab_size + merge_index
        if left not in vocab or right not in vocab or token_id != expected_id:
            raise ValueError("Invalid BPE tokenizer cache: inconsistent merge sequence")

        pair = (left, right)
        if pair in merges:
            raise ValueError("Invalid BPE tokenizer cache: duplicate merge")
        merges[pair] = token_id
        vocab[token_id] = vocab[left] + vocab[right]

    return merges, vocab
