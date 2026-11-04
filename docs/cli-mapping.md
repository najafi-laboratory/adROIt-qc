# Command line, configuration and interface

A setting reaches the reviewer by one of three routes:

1. **A configuration field.** `scripts/build_reviewer.py` translates each field
   into exactly one generator flag.
2. **A generator flag.** Pass it directly to `python -m roi_reviewer.summary`,
   or let the configuration supply it.
3. **A control in the open reviewer.** These act on the page in the browser, and
   several have a build-time counterpart.

The first two routes are equivalent. A configuration file and the matching
command produce byte-identical reviewer HTML. The tables below show, for every
setting, its name on each route and where it appears in the interface.

## Build settings

| Configuration field | Generator flag | Where it shows in the reviewer |
|:--|:--|:--|
| `session_path` | positional argument | Everything: images, masks, traces and metrics are read from here |
| `output_dir` | `--output-dir` | Where the HTML, PDF and sidecar arrays are written |
| `input_layout` | `--input-layout` | Which files supply masks and traces; `auto` omits the flag |
| `roi_index_label` | `--roi-index-label` | The selected-ROI header ("Original Index 48, sort position 49 of 225"), the labeler's "Selected ROI (Original Index)" field, and the first sort key |
| `target_structure` | `--target-structure` | The **Target structure** menu at the top of **ROI Metric Filters**: the preset it opens on |
| `initialize_labels_from_model_scores` | `--initialize-labels-from-roi-model-scores` | The labeler's starting counts (otherwise every ROI opens as not labeled) |
| `custom_roi_metrics` | `--custom-roi-metrics` (once per file) | **Uploaded / Custom ROI Metrics** in the filter menu, the sort menu, and the metric spreadsheet |
| `html_name` | `--html-name` | Filename of the reviewer HTML |
| `pdf_name` | `--pdf-name` | Filename of the static summary PDF |
| `archive_output` | `--archive-output` | Outputs go into a per-session subfolder, which is also zipped |
| `archive_name` | `--archive-name` | Name of that zip |
| `name`, `notes` | none | Recorded in `run.json` only |

`target_structure` accepts `all_rois`, `soma`, `dendrite` or `dendrite_relaxed`.
When it is omitted, the menu opens on whatever the session's own pipeline
parameters recorded. `all_rois` applies no thresholds.

The build script's own options, `--dry-run` and `--skip-validation`, have no
generator equivalent. `--dry-run` prints the generator command a configuration
resolves to without running it, which makes it the quickest way to translate a
configuration into a command.

!!! warning "Unrecognized configuration fields are an error"
    A misspelled field (`custom_metrics:` for `custom_roi_metrics:`) stops the
    build and lists the known fields, rather than being dropped and producing a
    reviewer without the setting.

## Session files read automatically

Some inputs take no flag at all: the generator looks for them in the session
directory and, if present, enables the matching part of the interface.

| File in the session | Where it shows in the reviewer |
|:--|:--|
| `dff.h5` (dataset `dff`) | The selected-ROI trace and the stacked traces. Large sessions are written out as a `<session>_dff.npy` sidecar |
| `spikes.h5` (dataset `spikes`) | The inferred-spike overlay, the spike diagnostics panel and every inferred-spike metric. Large sessions are written out as `<session>_oasis_spikes.npy` |
| `roi_cell_type_labels.npy` | **Categorical ROI Variables** in the filter menu, and the categorical sort key |
| `roi_model_scores.h5` or `roi_qc_predictions.h5` | **Model Derived Metrics** in the filter and sort menus; also the labeler's starting state when `initialize_labels_from_model_scores` is set |
| `masks.h5` (`mean_anat`, `masks_anat`) | The **Anatomical mean** panel, and the **Cellpose ROIs** option in its **Overlay** menu, which is offered only when the anatomical image is present |

## Controls in the open reviewer

| Reviewer control | What it loads | Build-time equivalent |
|:--|:--|:--|
| **Upload Custom ROI Metrics/Cell Identity Labels** → *QC metric* | CSV, TSV, TXT or JSON, one value per ROI per column | `custom_roi_metrics` / `--custom-roi-metrics`, which also accepts NPY and NPZ |
| **Upload Custom ROI Metrics/Cell Identity Labels** → *Cell identity label* | A table with `cell_identity_code` and/or `cell_identity_label` | `roi_cell_type_labels.npy` in the session; no flag |
| **Load dF/F file** | A `.npy` dF/F array | The `<session>_dff.npy` sidecar the generator writes |
| **Load inferred spikes file** | A `.npy` inferred-spike array | The `<session>_oasis_spikes.npy` sidecar the generator writes |
| **Filter set name** menu | A built-in or saved filter set | `filter_set` / `--filter-set` sets which built-in filter set it starts from |
| **Import filter set JSON** | A filter set and sort order exported from a reviewer, or written by hand | `filter_settings` / `--filter-settings` |

### Upload file formats

`roi_index` is the ROI's 0-based original index, the number the reviewer shows for the selected ROI. It is optional: without it, row *k* is ROI *k* and the file needs exactly one row per ROI. No other name is accepted for this column; a file with a `suite2p_index`, `roi` or `index` column is rejected with a message asking for `roi_index`.

**QC metric** (CSV, TSV or TXT): one row per ROI, one column per metric, numeric values.

```csv
roi_index,local_contrast,tdtomato_intensity
0,0.82,1530
1,0.41,212
2,0.67,880
```

The same as JSON, one array per metric in original-index order (a list of row objects
with a `roi_index` key also works):

```json
{
  "local_contrast": [0.82, 0.41, 0.67],
  "tdtomato_intensity": [1530, 212, 880]
}
```

**Cell identity label:**

```csv
roi_index,cell_identity_code,cell_identity_label
0,1,inhibitory
1,-1,excitatory
2,1,inhibitory
```

**Filter set JSON** (Import filter set JSON, or `--filter-settings`). Only `filter` is
required; leave out thresholds you do not use, and write `null` for an unused minimum
or maximum. Uploaded metrics are keyed by their column name in lower case, and are
sorted with `custom:` plus that name.

```json
{
  "name": "strict soma",
  "filter": {
    "skewMin": 0.5,
    "aspectMax": 1.5,
    "roiAreaMin": 40,
    "cellTypeFilter": ["1"],
    "customMetrics": {"local_contrast": {"min": 0.3, "max": null}}
  },
  "sort": {"metrics": ["custom:local_contrast"], "direction": "desc"}
}
```

**Load dF/F file** and **Load inferred spikes file:** NumPy arrays, C-ordered,
little-endian float32 (float64 is converted on load), named after the session so
they load automatically when the page is served:

```python
import numpy as np

# one row per ROI in original-index order, one column per frame; both arrays the same shape
np.save("SESSION_dff.npy", np.ascontiguousarray(dff, dtype=np.float32))
np.save("SESSION_oasis_spikes.npy", np.ascontiguousarray(spikes, dtype=np.float32))
```

**Embedding a metric at build time and uploading it in the browser are not the
same in what persists.** A build-time metric is part of the file for everyone who
opens it. A browser upload exists in that open page until you use **Save
current state into HTML**, which writes it, together with labels and thresholds,
into the downloaded copy. The two trace arrays are the exception: arrays loaded
with **Load dF/F file** or **Load inferred spikes file** are not saved into the
HTML, so a saved copy still needs its sidecars beside it.

## The same build, both ways

A configuration file:

```yaml
name: example_session
session_path: data/example_session
input_layout: external_rois
output_dir: outputs/example_session
roi_index_label: "Original Index"
target_structure: all_rois
custom_roi_metrics:
  - data/example_session/burst_rate.csv
html_name: reviewer.html
archive_output: true
```

```bash
python scripts/build_reviewer.py config/example_session.yaml
```

and the equivalent command, run from the repository root:

```bash
python -m roi_reviewer.summary data/example_session \
    --output-dir outputs/example_session \
    --input-layout external_rois \
    --roi-index-label "Original Index" \
    --target-structure all_rois \
    --custom-roi-metrics data/example_session/burst_rate.csv \
    --html-name reviewer.html \
    --archive-output
```

The configuration route also validates `external_rois` inputs before building,
resolves relative paths against the repository root, and writes `run.json`
recording the exact command it ran. The direct command does none of these.
To see the command for any configuration:

```bash
python scripts/build_reviewer.py config/example_session.yaml --dry-run
```
