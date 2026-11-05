#!/usr/bin/env python3
"""Create a source/config freeze from a clean committed implementation."""

import argparse
from pathlib import Path

from broader_common import write_json
from broader_provenance import create_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/inverse/broader_comparison.json")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/BROADER_COMPARISON_FREEZE.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = create_freeze(root, args.config, args.out)
    write_json(args.out, record)
    print(f"Created {args.out}. Commit this record before starting evaluation.")


if __name__ == "__main__":
    main()
