#!/usr/bin/env python
"""Compatibility entry point for `splm align` (CLAlign, global, gap=0)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm.alignment_cli import fasta_mapping, main, read_candidates

if __name__ == "__main__":
    raise SystemExit(main())
