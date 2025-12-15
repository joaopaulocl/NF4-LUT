#!/usr/bin/env python3
"""
Download one or multiple Hugging Face models to local storage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one or multiple Hugging Face models."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model IDs (e.g. meta-llama/Llama-2-7b-hf) to download.",
    )
    parser.add_argument(
        "--cache-dir",
        default="models",
        help="Directory to store downloaded snapshots (default: %(default)s).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit) to pin.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional glob pattern(s) of files to keep. Repeatable.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional glob pattern(s) of files to skip. Repeatable.",
    )
    return parser.parse_args()


def download_model(model_id: str, cache_dir: Path, args: argparse.Namespace) -> Path:
    """
    Downloads a single model snapshot and returns the local path.
    """
    return Path(
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            revision=args.revision,
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
        )
    )


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        print(f"Downloading {model_id} to {cache_dir} ...")
        local_path = download_model(model_id, cache_dir, args)
        print(f"✓ Saved snapshot for {model_id} in {local_path}")


if __name__ == "__main__":
    main()
