# Two copies of the generator

The generator exists in two places that are kept identical apart from one import line:

| here | laboratory copy in [najafi-laboratory/2p_imaging](https://github.com/najafi-laboratory/2p_imaging) |
|---|---|
| `adroit_qc/summary.py` | `utils_2p/processing_summary.py` |
| `adroit_qc/roi_metrics.py` | `utils_2p/roi_labels.py` |

This repository does not depend on the laboratory one: nothing here imports `utils_2p`,
and it runs for anyone. The laboratory pipeline calls its copy as its `summary` stage.

The import rewrite is `from adroit_qc.roi_metrics import` here and
`from utils_2p.roi_labels import` there. The first lines of each file name its
counterpart. (The package here was called `roi_reviewer` until 2026-11-21;
`roi_reviewer` remains as an import alias.)

## Keeping them in step

Since adroit-qc 0.2.0 this repository is the source: make interface changes here, run
`pytest tests`, then copy to the laboratory repository:

```bash
python - <<'EOF'
from pathlib import Path
src = Path("adroit_qc/summary.py").read_text().split("\n")
lab = [src[0],
       "# Synced from najafi-laboratory/adROIt-qc adroit_qc/summary.py (adroit-qc <version>).",
       "# Make interface changes there and copy them here; see docs/adroit-qc-lab-guide.md."] + src[5:]
text = "\n".join(lab).replace("from adroit_qc.roi_metrics import (", "from utils_2p.roi_labels import (", 1)
Path("/path/to/2p_imaging/utils_2p/processing_summary.py").write_text(text)
EOF
```

`roi_metrics.py` and `roi_labels.py` differ only in their header comments; copy the body
the same way when it changes.

## Deliberate divergences

- The saved-filter schema identifier is still `utils_2p_roi_metric_filter_v1`, so filter
  files are interchangeable between this tool and the laboratory pipeline.
- Both copies generate identical pages; every help link in the page points at this
  repository's documentation site.
