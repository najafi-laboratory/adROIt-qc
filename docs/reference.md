# Programmatic reference: build settings and helpers

Every build setting, Python helper and command-line tool on one page. Input files, their
shapes and containers are in [Input formats](inputs.md); the [Tutorial](tutorial.md)
shows them in use.

## Build settings

The same setting has a configuration field (for `scripts/build_reviewer.py`), a command
line flag (for `python -m adroit_qc.summary` or `adroit-qc`) and a Python keyword (for
`adroit_qc.inputs.build_reviewer` or `adroit_qc.summary.create_processing_summary`).
[Build settings explained](cli-mapping.md) covers the choices behind them: input layouts, filter sets for masks from other software, embedding and page-only rebuilds. The last column links to the part of the [Interface reference](interface-reference.md) each setting controls.

| Configuration field | Flag | Python keyword | Values | In the reviewer |
|:---|:---|:---|:---|:---|
| `session_path` | positional | first argument | session root, `suite2p/plane0`, or a documented-layout directory |  |
| `output_dir` | `--output-dir` | `output_dir` | directory; default the session root |  |
| `input_layout` | `--input-layout` | `input_layout` | `auto` (default), `suite2p`, `external_rois`; deprecated `qc_results`, `manual_qc_results` |  |
| `filter_set` | `--filter-set` | `target_structure` | `all_rois`, `soma`, `dendrite`, `dendrite_relaxed` | [Filter sets](interface-reference.md#filter-sets) |
| `filter_settings` | `--filter-settings` | `filter_settings` | exported filter/sorting JSON (or YAML) | [Saving and importing filter sets](interface-reference.md#saving-and-importing-filter-sets) |
| `initialize_labels_from_model_scores` | `--initialize-labels-from-roi-model-scores` | `initialize_labels_from_roi_model_scores` | open with labels from the classifier states | [Classifier score](interface-reference.md#classifier-score) |
| `custom_roi_metrics` | `--custom-roi-metrics` (repeat) | `custom_roi_metrics` | list of per-ROI tables | [Uploaded ROI metrics](interface-reference.md#uploaded-roi-metrics) |
| `custom_roi_metric_sections` | `--custom-roi-metric-section NAME=SECTION` (repeat) | `custom_roi_metric_sections` | `uploaded`, `model_derived`, `morphology`, `fluorescence`, `inferred_spike` (config: a mapping; Python: `"NAME=SECTION"` strings) | [Uploaded ROI metrics](interface-reference.md#uploaded-roi-metrics) (**Section** menu) |
| `roi_index_label` | `--roi-index-label` | not available | wording of the ROI index in the page | [Original index](interface-reference.md#original-index-and-sort-position) |
| `force_embed` | `--force-embed true`/`false` | `force_embed` | `null`/`None` decides by size; true embeds; false writes `.npy` sidecars | [Loading dF/F and inferred spike files](interface-reference.md#loading-dff-and-inferred-spike-files) |
| `embed_limit_mb` | `--embed-limit-mb` | `embed_limit_mb` | MB per array, default 80, must exceed 2 | [Loading dF/F and inferred spike files](interface-reference.md#loading-dff-and-inferred-spike-files) |
| `title` | `--title` | `title` | page title; default the session name | page heading |
| `html_only` | `--html-only` | `html_only` | rewrite only the page from an existing reviewer's data |  |
| `html_name`, `pdf_name` | `--html-name`, `--pdf-name` | `html_name`, `pdf_name` | output filenames |  |
| `archive_output`, `archive_name` | `--archive-output`, `--archive-name` | not available | zip the outputs |  |

`scripts/build_reviewer.py` also takes `--dry-run` (print the command, write nothing) and
`--skip-validation`.

## Helper functions

`adroit_qc.inputs` writes each input file with the name, shape and container the
generator reads, and checks shapes as it goes. Every function takes the session
directory first and creates it if needed.

| Function | Writes |
|:---|:---|
| `write_session(out_dir, F=..., masks=... or spatial_components=..., ...)` | a whole session; every other argument is optional and maps to one helper below; `row_order` renumbers a label image to match the trace rows |
| `write_label_mask(out_dir, masks)` | `roi_mask.npy` ([format](inputs.md#dense-label-masks)) |
| `write_spatial_components(out_dir, components, image_shape, flatten_order="F")` | `spatial_components.npz` and the metadata fields ([format](inputs.md#overlapping-rois)) |
| `write_traces(out_dir, F, Fneu=None)` | `F.npy`, `Fneu.npy` ([format](inputs.md#the-documented-layout)) |
| `write_dff(out_dir, dff)` | `dff.h5` ([format](inputs.md#dff-and-inferred-spikes)) |
| `write_spikes(out_dir, spikes, event_threshold=None)` | `spikes.h5` ([format](inputs.md#dff-and-inferred-spikes)) |
| `write_images(out_dir, mean_func=, mean_anat=, max_func=, max_anat=)` | the image `.npy` files ([format](inputs.md#mean-images)) |
| `write_motion_offsets(out_dir, xoff, yoff)` | `move_offset.h5` ([format](inputs.md#motion-offsets)) |
| `write_cell_types(out_dir, codes)` | `roi_cell_type_labels.npy` ([format](inputs.md#uploaded-metrics-and-categorical-labels)) |
| `write_iscell(out_dir, accepted, probability=None)` | `iscell.npy` ([format](inputs.md#the-documented-layout)) |
| `write_classifier_scores(out_dir, scores, good_threshold, bad_threshold, states=None, roi=None, model_source="", target_structure="")` | `roi_model_scores.h5` ([format](inputs.md#roi-model-scores)) |
| `write_metadata(out_dir, image_shape=, flatten_order=, frame_rate=, diameter=, neuropil_coefficient=)` | merges fields into `external_roi_metadata.json` ([format](inputs.md#the-documented-layout)) |
| `write_metric_table(path, metrics=, categorical=, category_names=, roi_index=)` | a per-ROI table ([format](inputs.md#uploaded-metrics-and-categorical-labels)) |
| `validate(session_dir)` | nothing; prints and returns the validator's report (`report.errors` empty when buildable) |
| `build_reviewer(session_dir, output_dir=None, **settings)` | the reviewer; returns `(pdf_path, html_path)` |

Other entry points:

| Tool | Use |
|:---|:---|
| `python -m adroit_qc.validate DIR` or `scripts/validate_inputs.py DIR` | check a directory before building; exit status 1 on errors |
| `scripts/make_standard_inputs.py --masks ... --traces ... --out ...` | convert array files to the documented layout from the command line ([details](tutorial.md#converting-from-another-pipeline)) |
| `examples/caiman/caiman_to_reviewer.py` | convert a CaImAn CNMF result ([details](tutorial.md#converting-from-caiman)) |
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
