#!/usr/bin/env python3
"""Check a directory against the reviewer's documented input contract.

    python scripts/validate_inputs.py /path/to/session

A thin wrapper around ``adroit_qc.validate``; see that module for what is checked.
Exit status is 0 when no errors were found, 1 otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adroit_qc.validate import ArrayUnreadable, Report, main, validate  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
