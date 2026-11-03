from pathlib import Path


def load_dataset(folder_path):
    """Load and concatenate all text files in a folder."""
    folder = Path(folder_path)
    file_list = sorted(folder.glob("*.txt"))

    if not file_list:
        print(f"Warning: No files found in '{folder}'. Using fallback text.")
        return "To be, or not to be, that is the question. Whether 'tis nobler in the mind to suffer."

    print(f"Loading and joining {len(file_list)} files...")
    return "\n".join(path.read_text(encoding="utf-8") for path in file_list)
