import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "training_config.json"

DEFAULT_SETTINGS = {
    "model": {
        "embedding_dim": 128,
        "num_heads": 4,
        "target_vocab_size": 512,
        "block_size": 64,
    },
    "training": {
        "batch_size": 32,
        "max_iters": 1000,
        "learning_rate": 3e-4,
        "eval_interval": 100,
        "checkpoint_interval": 100,
        "sample_interval": 500,
        "sample_tokens": 100,
        "checkpoint_path": PROJECT_DIR / "checkpoints" / "language_model.pt",
        "loss_history_path": PROJECT_DIR / "checkpoints" / "training_loss.csv",
        "loss_plot_path": PROJECT_DIR / "output" / "training_validation_loss.png",
        "resume": False,
    },
    "data": {
        "data_dir": PROJECT_DIR / "literature",
        "train_fraction": 0.9,
    },
}

SETTING_TYPES = {
    "model": {
        "embedding_dim": int,
        "num_heads": int,
        "target_vocab_size": int,
        "block_size": int,
    },
    "training": {
        "batch_size": int,
        "max_iters": int,
        "learning_rate": (int, float),
        "eval_interval": int,
        "checkpoint_interval": int,
        "sample_interval": int,
        "sample_tokens": int,
        "checkpoint_path": str,
        "loss_history_path": str,
        "loss_plot_path": str,
        "resume": bool,
    },
    "data": {
        "data_dir": str,
        "train_fraction": (int, float),
    },
}

PATH_SETTINGS = {
    ("training", "checkpoint_path"),
    ("training", "loss_history_path"),
    ("training", "loss_plot_path"),
    ("data", "data_dir"),
}

CLI_SETTINGS = {
    "model": {
        "embedding_dim": "embedding_dim",
        "num_heads": "num_heads",
        "target_vocab_size": "target_vocab_size",
        "block_size": "block_size",
    },
    "training": {
        "batch_size": "batch_size",
        "max_iters": "max_iters",
        "learning_rate": "learning_rate",
        "eval_interval": "eval_interval",
        "checkpoint_interval": "checkpoint_interval",
        "sample_interval": "sample_interval",
        "sample_tokens": "sample_tokens",
        "checkpoint_path": "checkpoint_path",
        "loss_history_path": "loss_history_path",
        "loss_plot_path": "loss_plot_path",
        "resume": "resume",
    },
    "data": {
        "data_dir": "data_dir",
        "train_fraction": "train_fraction",
    },
}

CONFIG_GROUPS = {
    "model": ("embedding_dim", "num_heads", "target_vocab_size", "block_size"),
    "training": (
        "batch_size",
        "max_iters",
        "learning_rate",
        "eval_interval",
        "checkpoint_interval",
        "sample_interval",
        "sample_tokens",
        "checkpoint_path",
        "loss_history_path",
        "loss_plot_path",
        "resume",
    ),
    "data": ("data_dir", "train_fraction"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a small BPE language model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config file (default: {DEFAULT_CONFIG_PATH.name}).",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--sample-interval", type=int)
    parser.add_argument("--sample-tokens", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--target-vocab-size", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--train-fraction",
        type=float,
        help="Fraction of tokenized data used for training (between 0 and 1).",
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--loss-history-path",
        type=Path,
        help="CSV file for training and validation loss history.",
    )
    parser.add_argument(
        "--loss-plot-path",
        type=Path,
        help="Loss curve image, refreshed after each evaluation.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume model and optimizer state from --checkpoint-path",
    )
    parsed = parser.parse_args(argv)

    try:
        settings = _load_settings(parsed.config)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))

    resolved = {}
    for section, keys in CLI_SETTINGS.items():
        resolved.update(settings[section])
        for key, argument_name in keys.items():
            value = getattr(parsed, argument_name)
            if value is not None:
                resolved[key] = value
    if not 0 < resolved["train_fraction"] < 1:
        parser.error("--train-fraction and data.train_fraction must be between 0 and 1")
    return argparse.Namespace(**resolved)


def save_settings_snapshot(settings):
    checkpoint_path = Path(settings.checkpoint_path)
    snapshot_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.settings.json"
    )
    snapshot = {
        section: {
            key: _json_value(getattr(settings, key))
            for key in keys
        }
        for section, keys in CONFIG_GROUPS.items()
    }

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(snapshot, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(snapshot_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Saved effective settings to: {snapshot_path.resolve()}")
    return snapshot_path


def _json_value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def _load_settings(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(
            "Config root must be a JSON object with model, training, and data sections"
        )

    unknown_sections = set(config) - set(DEFAULT_SETTINGS)
    if unknown_sections:
        raise ValueError(
            f"Unknown config section(s): {', '.join(sorted(unknown_sections))}"
        )

    settings = {
        section: dict(values) for section, values in DEFAULT_SETTINGS.items()
    }
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a JSON object")
        unknown_keys = set(values) - set(SETTING_TYPES[section])
        if unknown_keys:
            raise ValueError(
                f"Unknown setting(s) in '{section}': {', '.join(sorted(unknown_keys))}"
            )
        for key, value in values.items():
            expected_type = SETTING_TYPES[section][key]
            if (
                isinstance(value, bool) and expected_type is not bool
            ) or not isinstance(value, expected_type):
                raise ValueError(
                    f"Config setting '{section}.{key}' must be "
                    f"{_type_name(expected_type)}"
                )
            if (section, key) in PATH_SETTINGS:
                path = Path(value)
                settings[section][key] = (
                    path if path.is_absolute() else config_path.parent / path
                )
            else:
                settings[section][key] = value
    train_fraction = settings["data"]["train_fraction"]
    if not 0 < train_fraction < 1:
        raise ValueError("Config setting 'data.train_fraction' must be between 0 and 1")
    return settings


def _type_name(expected_type):
    if isinstance(expected_type, tuple):
        return " or ".join(item.__name__ for item in expected_type)
    return expected_type.__name__
