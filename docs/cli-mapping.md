# Build settings explained

A setting reaches the reviewer by one of three routes:

1. **A configuration field.** `scripts/build_reviewer.py` translates each field into
   exactly one generator flag.
2. **A generator flag.** Pass it directly to `python -m adroit_qc.summary`, or let
   the configuration supply it.
3. **A control in the open reviewer.** These act on the page in the browser; several
   have a build-time counterpart, listed [below](#controls-with-a-build-time-counterpart).

This page explains the choices behind the settings; the settings themselves are listed
in one table on [Inputs, settings and helpers](reference.md#build-settings).

The first two routes are equivalent: a configuration file and the matching command
produce byte-identical reviewer HTML. One session, one configuration file: copy
`config/template.yaml`, which documents every field in comments, edit the paths, and
build.

```yaml
name: my_session
session_path: /path/to/processed/session
input_layout: auto
output_dir: outputs/my_session
roi_index_label: "Original Index"
filter_set: null
initialize_labels_from_model_scores: false
custom_roi_metrics: []
force_embed: null
embed_limit_mb: 80
notes: ""
```

!!! warning "Unrecognized configuration fields are an error"
    A misspelled field (`custom_metrics:` for `custom_roi_metrics:`) stops the build and
    lists the known fields, rather than being dropped and producing a reviewer without the
    setting.

## Every setting

The full table of configuration fields, flags and Python keywords is on
[Inputs, settings and helpers](reference.md#build-settings).

`--force-embed` also accepts `1`/`0` and `yes`/`no`, in any capitalization. The older
field name `target_structure` is still accepted for `filter_set`.

The build script's own options have no generator equivalent. `--dry-run` resolves every
path, reports the layout that would be used and prints the generator command, writing
nothing. `--skip-validation` skips the input check that runs before an `external_rois`
build.

## Choosing an input layout

`auto` infers the layout and prints what it chose, which is right for most sessions.
Name it explicitly (`suite2p` or `external_rois`) when a directory contains more than one
candidate.

!!! warning "Deprecated pre-filtered layouts"
    `qc_results` and `manual_qc_results` describe pre-filtered ROI sets. Select them
    explicitly rather than relying on `auto`: a session containing both a full and a
    pre-filtered set resolves to the pre-filtered one, and the reviewer then shows fewer
    ROIs than the segmentation produced with nothing to indicate it.

## Filter sets and masks from other software

The built-in filter sets' morphology windows were calibrated on Suite2p's own values. For
masks from other software the generator computes the same shape statistics the Suite2p
way ([definitions](interface-reference.md#morphology-metrics)), but binary masks carry no
pixel weights and no mask format carries Suite2p's detection center, so compactness in
particular can differ. `filter_set: all_rois` applies no thresholds and is the safest
start; set thresholds from the [distribution plots](interface-reference.md#metric-distributions)
instead.

## Embedded or separate trace arrays

By default dF/F and inferred spikes are embedded in the HTML when ROIs × frames is at
most 15,335,424 (about 80 MB of encoded data per array), and otherwise written beside it
as `SESSION_dff.npy` and `SESSION_inferred_spikes.npy`. Separate files load automatically
only when the page is opened through a web server; see
[Open the result](tutorial.md#open-it). `--embed-limit-mb 200` raises the
cutoff to 38,928,384; `--force-embed true` embeds regardless, with a warning above 512 MB
per array, which browsers may not open.

## Rebuilding only the page

`--html-only` reads the data payload out of an existing reviewer in `--output-dir` and
writes it into the current page template, without recomputing metrics. Use it after
updating the generator. Combined with `--force-embed` it converts a reviewer between
embedded and separate arrays from its own data; embedding leaves the `.npy` files in
place.

## Controls with a build-time counterpart

| Reviewer control | Build-time equivalent |
|:--|:--|
| **Upload Custom ROI Metrics/Cell Identity Labels** | `custom_roi_metrics` / `--custom-roi-metrics`, same [file format](inputs.md#uploaded-metrics-and-categorical-labels) |
| **Load dF/F**, **Load inferred spikes** | The `SESSION_dff.npy` and `SESSION_inferred_spikes.npy` files the generator writes |
| **Filter set name** menu | `filter_set` / `--filter-set` |
| **Import filter set JSON** | `filter_settings` / `--filter-settings` |

Inputs that need no setting at all, such as `spikes.h5`, mean images, motion offsets,
cell type labels and model scores, are found in the session directory automatically;
[Preparing your inputs](inputs.md) lists them.

**A build-time metric and a browser upload differ in what persists.** A build-time metric
is part of the file for everyone who opens it. A browser upload exists in the open page
until you use **Save current state into HTML**. Arrays loaded with **Load dF/F** or
**Load inferred spikes** are never saved into the HTML, so a saved copy still needs them
beside it.

## The same build, both ways

```yaml
name: example_session
session_path: data/example_session
input_layout: external_rois
output_dir: outputs/example_session
roi_index_label: "Original Index"
filter_set: all_rois
custom_roi_metrics:
  - data/example_session/qc_metrics.csv
force_embed: true
```

```bash
python scripts/build_reviewer.py config/example_session.yaml
```

and the equivalent command, run from the repository root:

```bash
python -m adroit_qc.summary data/example_session \
    --output-dir outputs/example_session \
    --input-layout external_rois \
    --roi-index-label "Original Index" \
    --filter-set all_rois \
    --custom-roi-metrics data/example_session/qc_metrics.csv \
    --force-embed true
```

The configuration route also validates `external_rois` inputs before building, resolves
relative paths against the repository root, and writes `run.json` recording the exact
command. `--dry-run` prints that command for any configuration.
