# Inputs, settings and helpers at a glance

Every programmatic input, build setting and helper function on one page. Each row links
to the full description. Shapes use `n_rois` (ROIs), `n_frames` (frames) and `(Ly, Lx)`
(image height and width in pixels).

## Input files

A session directory is read in one of two layouts. In both, row *i* of every per-ROI
file is ROI *i*, the ROI's 0-based **original index**.

### Suite2p layout (`input_layout: suite2p`)

| File | Required | Format |
|:---|:---|:---|
| `suite2p/plane0/stat.npy` | yes | Suite2p ROI statistics (pixels, weights, shape statistics) |
| `suite2p/plane0/F.npy`, `Fneu.npy` | yes; `Fneu` optional | `(n_rois, n_frames)` raw and neuropil fluorescence |
| `suite2p/plane0/ops.npy` | yes | frame rate `fs`, `neucoeff`, mean images `meanImg` / `meanImg_chan2`, `max_proj`, `xoff` / `yoff`, `diameter` |
| `suite2p/plane0/iscell.npy` | no | `(n_rois, 2)` Suite2p accept/reject and probability; shown, never filters |
| Every optional file in the next table except the masks, traces and metadata | no | placed at the session root |

### Documented layout (`input_layout: external_rois`), for masks from any software

| File | Required | Format | Helper |
|:---|:---|:---|:---|
| `roi_mask.npy` | one of these two | `(Ly, Lx)` integer label image, 0 = background; sorted positive labels are rows 0, 1, … ([details](inputs.md#dense-label-masks)) | `write_label_mask` |
| `spatial_components.npz` | | SciPy sparse `(Ly·Lx, n_rois)` pixel weights, column *k* = ROI *k* ([details](inputs.md#overlapping-rois)) | `write_spatial_components` |
| `F.npy` | yes | `(n_rois, n_frames)` float32 raw fluorescence | `write_traces` |
| `Fneu.npy` | no | same shape, neuropil; without it nothing is subtracted | `write_traces` |
| `dff.h5`, dataset `dff` | no | same shape; displayed and measured as given instead of dF/F computed from `F.npy` | `write_dff` |
| `spikes.h5`, dataset `spikes` | no | same shape; inferred spikes from any method in their own units; attribute `event_threshold` sets the opening threshold | `write_spikes` |
| `mean_func.npy`, `mean_anat.npy`, `max_func.npy`, `max_anat.npy` | no | `(Ly, Lx)` images ([details](inputs.md#mean-images)) | `write_images` |
| `move_offset.h5`, datasets `xoff`, `yoff` | no | `(n_frames,)` rigid shift per frame in pixels ([details](inputs.md#motion-offsets)) | `write_motion_offsets` |
| `roi_cell_type_labels.npy` | no | `(n_rois,)` float: 1 inhibitory, −1 excitatory, 0 unsure, NaN unknown | `write_cell_types` |
| `iscell.npy` | no | `(n_rois, 2)` upstream accept (1) / reject (0) and probability | `write_iscell` |
| `roi_model_scores.h5` | no | classifier output, below | `write_classifier_scores` |
| `external_roi_metadata.json` | needed for sparse masks without an image | below | `write_metadata` |

**`external_roi_metadata.json` fields.** `image_shape` `[Ly, Lx]`; `flatten_order` `"F"`
(column-major, CaImAn) or `"C"`; `frame_rate` in Hz (default 30); `diameter`, expected ROI
diameter in pixels, one number or `[dy, dx]`; `neuropil_coefficient` (default 0.7).

**`roi_model_scores.h5`, from any classifier** ([details](inputs.md#roi-model-scores)).
Dataset `classifier_score` (or `probability`), 0–1 per ROI, higher = more likely good;
`state` per ROI, `good`, `bad` or `gray`; optional `roi` (or `suite2p_roi`) original
index per row. Attributes `good_threshold`, `bad_threshold`, `model_source`,
`target_structure`. The reviewer also reads `roi_qc_predictions.h5` and
`roi_predictions.h5` under the same names.

### Per-ROI tables (build time or in the browser)

One table format serves `--custom-roi-metrics` and the
**Upload Custom ROI Metrics/Cell Identity Labels** dialog
([details](inputs.md#uploaded-metrics-and-categorical-labels)): a `.csv`, `.tsv` or `.txt`
with a header row; optional `roi_index` (0-based original index; without it row *k* is
ROI *k*); `NAME_code` integer columns, with optional `NAME_label`, are a categorical
variable `NAME`, and `cell_identity` is the cell type variable; every other column is a
numeric metric. Helper: `write_metric_table`.

### Files the open reviewer loads

| Control | File |
|:---|:---|
| **Load dF/F**, **Load inferred spikes** | `(n_rois, n_frames)` float32 `.npy` ([format](interface-reference.md#loading-dff-and-inferred-spike-files)) |
| **Upload Custom ROI Metrics/Cell Identity Labels** | the per-ROI table above |
| **Import filter set JSON** | a file from **Export current filter/sorting settings JSON** |

## Build settings

The same setting has a configuration field (for `scripts/build_reviewer.py`), a command
line flag (for `python -m adroit_qc.summary` or `adroit-qc`) and a Python keyword (for
`adroit_qc.inputs.build_reviewer` or `adroit_qc.summary.create_processing_summary`).
[Build settings](cli-mapping.md) explains each in full.

| Configuration field | Flag | Python keyword | Values |
|:---|:---|:---|:---|
| `session_path` | positional | first argument | session root, `suite2p/plane0`, or a documented-layout directory |
| `output_dir` | `--output-dir` | `output_dir` | directory; default the session root |
| `input_layout` | `--input-layout` | `input_layout` | `auto` (default), `suite2p`, `external_rois`; deprecated `qc_results`, `manual_qc_results` |
| `filter_set` | `--filter-set` | `target_structure` | `all_rois`, `soma`, `dendrite`, `dendrite_relaxed` |
| `filter_settings` | `--filter-settings` | `filter_settings` | exported filter/sorting JSON (or YAML) |
| `initialize_labels_from_model_scores` | `--initialize-labels-from-roi-model-scores` | `initialize_labels_from_roi_model_scores` | open with labels from the classifier states |
| `custom_roi_metrics` | `--custom-roi-metrics` (repeat) | `custom_roi_metrics` | list of per-ROI tables |
| `custom_roi_metric_sections` | `--custom-roi-metric-section NAME=SECTION` (repeat) | `custom_roi_metric_sections` | `uploaded`, `model_derived`, `morphology`, `fluorescence`, `inferred_spike` (config: a mapping; Python: `"NAME=SECTION"` strings) |
| `roi_index_label` | `--roi-index-label` | not available | wording of the ROI index in the page |
| `force_embed` | `--force-embed true`/`false` | `force_embed` | `null`/`None` decides by size; true embeds; false writes `.npy` sidecars |
| `embed_limit_mb` | `--embed-limit-mb` | `embed_limit_mb` | MB per array, default 80, must exceed 2 |
| `title` | `--title` | `title` | page title; default the session name |
| `html_only` | `--html-only` | `html_only` | rewrite only the page from an existing reviewer's data |
| `html_name`, `pdf_name` | `--html-name`, `--pdf-name` | `html_name`, `pdf_name` | output filenames |
| `archive_output`, `archive_name` | `--archive-output`, `--archive-name` | not available | zip the outputs |

`scripts/build_reviewer.py` also takes `--dry-run` (print the command, write nothing) and
`--skip-validation`.

## Helper functions

`adroit_qc.inputs` writes each input file with the name, shape and container the
generator reads, and checks shapes as it goes. Every function takes the session
directory first and creates it if needed.

| Function | Writes |
|:---|:---|
| `write_session(out_dir, F=..., masks=... or spatial_components=..., ...)` | a whole session; every other argument is optional and maps to one helper below; `row_order` renumbers a label image to match the trace rows |
| `write_label_mask(out_dir, masks)` | `roi_mask.npy` |
| `write_spatial_components(out_dir, components, image_shape, flatten_order="F")` | `spatial_components.npz` and the metadata fields |
| `write_traces(out_dir, F, Fneu=None)` | `F.npy`, `Fneu.npy` |
| `write_dff(out_dir, dff)` | `dff.h5` |
| `write_spikes(out_dir, spikes, event_threshold=None)` | `spikes.h5` |
| `write_images(out_dir, mean_func=, mean_anat=, max_func=, max_anat=)` | the image `.npy` files |
| `write_motion_offsets(out_dir, xoff, yoff)` | `move_offset.h5` |
| `write_cell_types(out_dir, codes)` | `roi_cell_type_labels.npy` |
| `write_iscell(out_dir, accepted, probability=None)` | `iscell.npy` |
| `write_classifier_scores(out_dir, scores, good_threshold, bad_threshold, states=None, roi=None, model_source="", target_structure="")` | `roi_model_scores.h5` |
| `write_metadata(out_dir, image_shape=, flatten_order=, frame_rate=, diameter=, neuropil_coefficient=)` | merges fields into `external_roi_metadata.json` |
| `write_metric_table(path, metrics=, categorical=, category_names=, roi_index=)` | a per-ROI table |
| `validate(session_dir)` | nothing; prints and returns the validator's report (`report.errors` empty when buildable) |
| `build_reviewer(session_dir, output_dir=None, **settings)` | the reviewer; returns `(pdf_path, html_path)` |

Other entry points:

| Tool | Use |
|:---|:---|
| `python -m adroit_qc.validate DIR` or `scripts/validate_inputs.py DIR` | check a directory before building; exit status 1 on errors |
| `scripts/make_standard_inputs.py --masks ... --traces ... --out ...` | convert array files to the documented layout from the command line ([details](inputs.md#converting-from-another-pipeline)) |
| `examples/caiman/caiman_to_reviewer.py` | convert a CaImAn CNMF result ([details](inputs.md#converting-from-caiman)) |
| `adroit_qc.roi_metrics.load_reviewed_dff` | load reviewed ROIs and their dF/F downstream ([details](export.md#loading-reviewed-rois)) |

## Outputs

| File | Contents |
|:---|:---|
| `SESSION_interactive_fov_roi_dff.html` | the reviewer |
| `SESSION_dff.npy`, `SESSION_inferred_spikes.npy` | trace arrays, when not embedded |
| `SESSION_processing_summary.pdf`, `preprocessing_fov_summary.png`, `motion_correction_summary.png` | static summaries |
| `run.json` | resolved command and paths, from `scripts/build_reviewer.py` |

The reviewer's own exports (labels, spreadsheet, filter settings) are described in
[Export and downstream use](export.md).
