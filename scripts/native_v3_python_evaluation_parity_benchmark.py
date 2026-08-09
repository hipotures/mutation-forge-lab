#!/usr/bin/env python3
"""Run the provider-free ordinary-Python evaluation parity benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mutation_forge.native_v3_python.evaluation_benchmark import (
    benchmark_evaluation_profiles,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heg-repo", type=Path, default=Path("../heg"))
    parser.add_argument("--sample-cases", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_evaluation_profiles(args.heg_repo, sample_cases=args.sample_cases),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
