"""Deprecated alias for :mod:`adroit_qc.roi_metrics`; the package was renamed adroit_qc."""

import sys

from adroit_qc import roi_metrics as _module

sys.modules[__name__] = _module
