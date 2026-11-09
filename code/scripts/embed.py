#!/usr/bin/env python
"""Compatibility entry point for `splmtm embed`, with the same full-length default."""
from pathlib import Path
import sys

# 让 `import splm_tm` 在不安装的情况下也能工作
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm.embed_cli import build_parser, load_inputs, main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
