"""Deprecated alias for :mod:`adroit_qc.summary`; the package was renamed adroit_qc."""

import sys

from adroit_qc import summary as _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
sys.modules[__name__] = _module
