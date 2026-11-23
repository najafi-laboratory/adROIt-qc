# Tutorial: from installation to a reviewed session

This tutorial goes from an empty terminal to reviewers built from every kind of input
adROIt-qc accepts. Part 1 installs the package. Part 2 builds one reviewer from the
command line and loads extra inputs into it by hand, in the browser. Part 3 prepares
inputs from your own software, and Part 4 runs one script that builds a reviewer for each
input route.

[Input formats](inputs.md) is the reference for every input file, and the
[Programmatic reference](reference.md) lists every build setting and helper function.

## The example data

The examples use two laboratory recordings, each in two input formats, plus a CaImAn
result for the first one:

| Folder | Contents |
|:---|:---|
| `sessions/ppc_suite2p/VTYH01_PPC_20250212_1451ShortLong/` | Posterior parietal cortex, 225 ROIs, Suite2p output with two channels, excitatory/inhibitory cell type codes and inferred spikes |
| `sessions/ppc_masks/VTYH01_PPC_20250212_1451ShortLong/` | The same session in the [documented layout](inputs.md#the-documented-layout): sparse masks, traces, images, offsets |
| `sessions/crbl_suite2p/E5LG_CRBL_crux1_20250520_EBC-334/` | Cerebellar Purkinje cell dendrites, 2,695 ROIs, Suite2p output with classifier scores and inferred spikes |
| `sessions/crbl_masks/E5LG_CRBL_crux1_20250520_EBC-334/` | The same session in the documented layout |
| `sessions/caiman_ppc/VTYH01_PPC_20250212_1451ShortLong_caiman/` | The PPC recording segmented by CaImAn and converted with `examples/caiman/caiman_to_reviewer.py`, with CaImAn's own metrics in `caiman_metrics.csv` |
| `extras/VTYH01_PPC_20250212_1451ShortLong_cell_identity_labels.csv` | A cell identity table in the upload format |
| `caiman_run/` | The CaImAn result the converted session was made from (`cnmf_results.hdf5`, shifts, parameters) |

!!! note "Where the data are"
    **[PLACEHOLDER: Zenodo record DOI and download command for the example sessions.]**
    Until that record is public, the folder is on the Georgia Tech PACE cluster at
    `/storage/project/r-fnajafi3-0/shared/2P_Imaging/adroit_qc_examples`, readable by
    members of the Najafi laboratory. Everything below works the same with your own
    sessions in place of these.

The CaImAn session is an example of adapting another tool's output and metrics to these
inputs, not an optimized CaImAn analysis.

### Try the reviewer without installing anything

Two reviewers built from these sessions are on the Zenodo sandbox, with their trace
arrays, so you can explore the interface first:

| Session | What it shows | Record |
|:---|:---|:---|
| Posterior parietal cortex, 225 ROIs | Two channels and excitatory/inhibitory cell type labels; the session in the screenshots on this site | [sandbox.zenodo.org/records/620871](https://sandbox.zenodo.org/records/620871) |
| Cerebellar dendrites | Classifier scores; set **Classifier score** Min to 0.8 and press **Apply Filters** | [sandbox.zenodo.org/records/620873](https://sandbox.zenodo.org/records/620873) |

Download the three files of a record into one folder and open the HTML as in
[Open it](#open-it). These are test records built from an earlier version of the
interface; they will be replaced by records on zenodo.org with permanent DOIs.

## Part 1: Install

You need Python 3.9 or newer.

```bash
git clone https://github.com/najafi-laboratory/adROIt-qc
cd adROIt-qc
pip install -e .
```

or, with conda:

```bash
conda env create -f environment.yml
conda activate adroit-qc
pip install -e .
```

Check that the generator runs, and point the examples at the data:

```bash
python -m adroit_qc.summary --help
export ADROIT_QC_EXAMPLES=/storage/project/r-fnajafi3-0/shared/2P_Imaging/adroit_qc_examples
ls $ADROIT_QC_EXAMPLES/sessions
```

The only dependencies are numpy, scipy, h5py, matplotlib, Pillow and PyYAML. The
package installs two equivalent commands, `adroit-qc` and `python -m adroit_qc.summary`;
the older name `roi_reviewer` still works as an alias.

To check an installation, run the test suite. It builds small synthetic sessions in
every input layout and checks that each input reaches the reviewer unchanged:

```bash
pip install pytest
pytest -q tests        # browser tests are skipped unless Playwright is installed
```

## Part 2: One reviewer, built and loaded by hand

### Build it

```bash
python -m adroit_qc.summary $ADROIT_QC_EXAMPLES/sessions/ppc_suite2p/VTYH01_PPC_20250212_1451ShortLong \
    --output-dir tutorial_out/manual \
    --input-layout suite2p \
    --filter-set soma
```

The generator finds `suite2p/plane0/`, the cell type codes in
`roi_cell_type_labels.npy` and the inferred spikes in `spikes.h5`, and prints the files
it wrote: `VTYH01_PPC_20250212_1451ShortLong_interactive_fov_roi_dff.html`, a PDF
summary and, because this session is larger than the embedding limit, two arrays beside
the HTML, `..._dff.npy` and `..._inferred_spikes.npy`.

**The same build from a configuration file.** One session, one file: copy
`config/template.yaml` (every field is documented in its comments), set `session_path`
and `output_dir`, and build. `--dry-run` resolves the paths and prints the generator
command without writing anything; a real build also writes `run.json`, recording the
resolved paths and options, beside the outputs.

```bash
cp config/template.yaml config/my_session.yaml
python scripts/build_reviewer.py config/my_session.yaml --dry-run
python scripts/build_reviewer.py config/my_session.yaml
```

Every field and flag is listed in the [Programmatic reference](reference.md#build-settings).

### Open it

Copy `tutorial_out/manual/` to your computer if you built it on a cluster. Move the
whole folder, not the HTML alone: the two `.npy` arrays beside it hold the traces.

#### Whether the arrays load automatically {#whether-the-sidecars-load-automatically}

That depends on how the page is opened:

| How you open the HTML | Address starts with | Arrays beside the HTML |
|:---|:---|:---|
| Through a web server, for example `python -m http.server 8000` run in the folder, then `http://localhost:8000/<name>_interactive_fov_roi_dff.html` | `http://` or `https://` | Load automatically |
| By double-clicking it, or from any downloaded, copied or mounted folder | `file://` | Not loaded, because browsers do not let a page opened from disk read files beside it; select them with **Load dF/F** and **Load inferred spikes** (next step) |

The dF/F panel's **(i)** popover says which case applies and whether each array loaded.
Sessions small enough to embed their arrays have no separate files and open the same
way from anywhere; `--force-embed true` makes any session self-contained
([details](cli-mapping.md#embedded-or-separate-trace-arrays)).

### Load inputs by hand

Each of these acts on the open page only. **Save Labels → Save current state into HTML**
writes a new copy with them included.

1. **dF/F and inferred spikes.** In the dF/F panel, **Load dF/F** and **Load inferred
   spikes** take any `(n_rois, n_frames)` float32 `.npy` array, so you can load the
   sidecars or an array from your own spike-inference method
   ([format](interface-reference.md#loading-dff-and-inferred-spike-files)).
2. **Cell identity labels.** In **Upload Custom ROI Metrics/Cell Identity Labels**,
   choose `$ADROIT_QC_EXAMPLES/extras/VTYH01_PPC_20250212_1451ShortLong_cell_identity_labels.csv`.
   Its `cell_identity_code` and `cell_identity_label` columns become the cell type
   variable; the dialog reports what it loaded.
3. **Your own metrics.** The same dialog takes any table with a `roi_index` column and
   numeric columns; each becomes a metric under **Uploaded ROI Metrics** in the Filter
   and Sort menus. Categorical columns are `NAME_code` with an optional `NAME_label`
   ([format](inputs.md#uploaded-metrics-and-categorical-labels)).
4. **A filter set.** **Import filter set JSON** loads thresholds and a sort order exported
   from another reviewer with **Export current filter/sorting settings JSON**.

Then label ROIs ([Interface reference](interface-reference.md)) and save with **Save Labels**
([what each export contains](export.md)).

## Part 3: Preparing your own inputs

The reviewer needs two things, where each ROI is and what its fluorescence did over
time, and can use more: inferred spikes from any method, mean images, motion offsets,
classifier scores and your own per-ROI metrics and labels. A Suite2p session needs no
preparation. For any other software, write the session in the
[documented layout](inputs.md#the-documented-layout): one directory of plain arrays.
[Input formats](inputs.md) gives every file's name, shape and container.

### Masks

The simplest mask format is a dense label image, one integer per pixel, which Cellpose,
scikit-image's `label()` and rasterized ImageJ ROI sets all produce. ROI rows follow the
sorted label values:

<div class="screenshot" markdown>
![A dense label mask with labels 3, 7 and 12. Each pixel holds its ROI's label and 0 is background. The positive labels are sorted ascending, so 3, 7 and 12 become rows 0, 1 and 2, and row i of the trace files belongs to ROI i. The traces are illustrative.](img/label-mask-diagram.png)
</div>

Overlapping or weighted footprints (CaImAn, NMF-style output) go in a sparse
pixels × ROIs matrix instead ([format](inputs.md#overlapping-rois)).

### Writing a session from Python arrays

Step 2 of the [example script](#part-4-every-input-route-from-one-script) does this from a Suite2p session: it reads the Suite2p session into
plain arrays only to have something to convert; replace `suite2p_arrays()` with whatever
loads your own masks and traces:

```python
from adroit_qc import inputs

session = inputs.write_session(
    "data/my_session",
    masks=label_image,      # (Ly, Lx) integers, 0 = background; sorted labels are ROI rows
    F=raw_traces,           # (n_rois, n_frames)
    Fneu=neuropil,          # optional; without it nothing is subtracted
    spikes=deconvolved,     # optional, any method, same shape as F
    mean_func=mean_image,   # optional (Ly, Lx)
    xoff=x_shift, yoff=y_shift,  # optional, pixels per frame
    cell_types=codes,       # optional: 1 inhibitory, -1 excitatory, 0 unsure, NaN unknown
    frame_rate=30.0,
)
inputs.validate(session)
inputs.build_reviewer(session, output_dir="outputs/my_session",
                      input_layout="external_rois", target_structure="all_rois")
```

For overlapping or weighted footprints use `spatial_components=` (a sparse pixels × ROIs
matrix) with `image_shape=` instead of `masks=`.

### Converting from another pipeline

```bash
python scripts/make_standard_inputs.py \
    --masks  segmentation/cell_masks.npy \
    --traces segmentation/traces.npy \
    --out    data/my_session
```

| Option | Use |
|---|---|
| `--masks` | dense 2-D integer label image (`.npy`, `.npz`, `.csv`) |
| `--traces` | raw fluorescence, ROIs by frames or frames by ROIs |
| `--neuropil` | optional neuropil traces, same shape as `--traces` |
| `--spikes` | optional inferred spike amplitudes, same shape as `--traces` |
| `--dff` | optional precomputed dF/F; omit to let the generator compute it |
| `--fov` | optional functional mean image |
| `--cell-types` | optional per-ROI cell type codes |
| `--row-order` | the mask label belonging to each trace row, in row order |
| `--out` | output directory |

The converter transposes a frames-by-ROIs trace matrix, compacts mask labels that are
not a gap-free run from 1, and writes dF/F and spikes as HDF5.

### Converting from CaImAn

`examples/caiman/` holds a worked CaImAn path, run on the posterior parietal cortex
demo session.

!!! note "An example of adapting another workflow, not a recommended CaImAn analysis"
    This example shows how the output, file formats and quality metrics of a different
    tool can be mapped onto the reviewer's inputs. It is not a comprehensive or optimized
    CaImAn session: it processes only the first four files (16,356 frames) with
    near-default parameters, no parameter tuning and no neuropil estimate, and its
    component counts say nothing about CaImAn's performance. Tune CaImAn for your own
    data as its documentation describes, then use the same conversion.

The two scripts:

* `run_caiman.py` reads raw Bruker OME-TIFFs for one channel, runs CaImAn rigid motion
  correction and CNMF, evaluates the components, and saves the result with
  `cnm.save(...)`, the motion-corrected memmap and the rigid shifts;
* `caiman_to_reviewer.py` converts that result into the [documented layout](inputs.md#the-documented-layout)
  without importing CaImAn.

```bash
python examples/caiman/run_caiman.py --tiff-dir raw/SESSION --channel Ch2 \
    --n-files 4 --frame-rate 30 --out caiman_out
python examples/caiman/caiman_to_reviewer.py --cnmf caiman_out/cnmf_results.hdf5 \
    --movie caiman_out/memmap_*order_C_frames_*.mmap --shifts caiman_out/rigid_shifts.npy \
    --frame-rate 30 --out data/SESSION_caiman
python scripts/validate_inputs.py data/SESSION_caiman
python -m adroit_qc.summary data/SESSION_caiman --input-layout external_rois \
    --custom-roi-metrics data/SESSION_caiman/caiman_metrics.csv
```

| CaImAn output | Written as | Notes |
|:---|:---|:---|
| `estimates.A` | `spatial_components.npz` | unchanged; pixel order F, recorded in `external_roi_metadata.json` |
| motion-corrected movie | `F.npy` | footprint-weighted mean of the movie per component, so dF/F is computed as for Suite2p's `F.npy` |
| `estimates.S` | `spikes.h5` | deconvolved activity; set the [spike threshold](interface-reference.md#inferred-spike-threshold) in its units |
| mean of the movie, correlation image | `mean_func.npy`, `max_func.npy` | |
| `shifts_rig` | `move_offset.h5` | `xoff`, `yoff` in pixels |
| `idx_components`, `SNR_comp`, `r_values` | `iscell.npy`, `caiman_metrics.csv` | rejected components are kept and marked, so they can be reviewed rather than dropped |

Component *k* becomes reviewer ROI *k*, so `roi_index` *k* in every export refers back to
CaImAn component *k*. `caiman_metrics.csv` adds `caiman_snr`, `caiman_r_value` and the
categorical `caiman_status` (accepted or rejected) as uploaded metrics.

The same pattern applies to any other tool: write its masks and traces in the documented layout, and put whatever per-ROI quality measures or decisions it produces into an
uploaded metric file, numeric columns for scores and `NAME_code` columns for categorical
decisions, so they can be filtered, sorted and exported alongside the reviewer's own
metrics.

### Validate before you build

```bash
python scripts/validate_inputs.py data/my_session
```

It reports shape disagreements, a mask label count that does not match the trace row
count, arrays that are not C-ordered, a stray `.npy` where HDF5 is expected, and files
it cannot read at all. It exits non-zero when it finds an error, so it can gate a
pipeline step. `build_reviewer.py` runs the same checks unless you pass
`--skip-validation`.

### The one thing no tool can check

Whether mask label *k* really belongs to trace row *k−1*.

A mismatch produces a reviewer in which every mask is paired with the wrong trace.
Every array has the right shape, every check passes, and nothing in the interface
flags it. Pass `--row-order` if your segmentation numbers masks and traces
differently, and spot-check one known-active ROI before trusting a session: select it
in the field of view and confirm the trace panel shows the activity you expect in the
place you expect it.

### Where the defaults differ from Suite2p

For masks from other software, the morphology metrics are computed the Suite2p way from
the masks you supply, so they can differ from Suite2p's own values: binary masks carry
no pixel weights, and no mask format carries Suite2p's detection centre, which changes
compactness most. Start from the `all_rois` filter set and set thresholds from the
distribution plots ([details](cli-mapping.md#filter-sets-and-masks-from-other-software)).
dF/F is computed from `F.npy` and `Fneu.npy` with a neuropil coefficient of 0.7 unless you
set `neuropil_coefficient` in `external_roi_metadata.json` or supply `dff.h5`.

## Part 4: Every input route, from one script

`examples/tutorial/build_examples.py` builds six reviewers, one per route:

```bash
python examples/tutorial/build_examples.py --out tutorial_out          # all six
python examples/tutorial/build_examples.py --out tutorial_out --only 2  # one step
```

| Step | Route | Inputs | Build settings it shows |
|:---|:---|:---|:---|
| 1 | Suite2p session as it is | `suite2p/plane0/`, `spikes.h5`, `roi_cell_type_labels.npy` | `input_layout="suite2p"`, a filter set, `title` |
| 2 | Arrays from any software | a dense label image, F, Fneu, spikes, mean and max images, motion offsets, cell type codes, `iscell`, frame rate, all written with `inputs.write_session`, then validated | `input_layout="external_rois"`, `filter_set` `all_rois` |
| 3 | A session already in the documented layout | sparse masks, plus the cell identity table | `custom_roi_metrics` |
| 4 | Classifier scores | `roi_model_scores.h5` beside a Suite2p session | `initialize_labels_from_roi_model_scores=True`, the dendrite filter set |
| 5 | Another pipeline (CaImAn) | converted CaImAn output and CaImAn's own metrics | `custom_roi_metric_sections` |
| 6 | Your own derived inputs | your own dF/F (`dff.h5`), classifier scores from any model, a numeric metric and a categorical variable, written with `inputs.write_dff`, `write_classifier_scores` and `write_metric_table` | `force_embed=False` |
