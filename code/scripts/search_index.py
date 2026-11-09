#!/usr/bin/env python
"""Direct script entry point for persistent library construction and search."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "build":
        args = ["index", "build", *args[1:]]
    elif args and args[0] == "query":
        args = ["search", *args[1:]]
    raise SystemExit(main(args))
