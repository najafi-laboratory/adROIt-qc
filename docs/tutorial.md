# Tutorial: from installation to a reviewed session

This tutorial goes from an empty terminal to reviewers built from every kind of input
adROIt-qc accepts. Part 1 installs the package. Part 2 builds one reviewer from the
command line and loads extra inputs into it by hand, in the browser. Part 3 runs one
script that builds a reviewer for each input route, writing the input files from Python
arrays where a route needs them.

[Inputs, settings and helpers](reference.md) lists every file, setting and helper
function on one page; [Preparing your inputs](inputs.md) gives each format in full.

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

The package installs two equivalent commands, `adroit-qc` and `python -m
adroit_qc.summary`. The older name `roi_reviewer` still works as an alias.

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

The same build from a configuration file is
`python scripts/build_reviewer.py config/my_session.yaml`; see
[Build settings](cli-mapping.md).

### Open it

Copy `tutorial_out/manual/` to your computer if you built it on a cluster, then either

* run `python -m http.server 8000` in that folder and open
  `http://localhost:8000/VTYH01_PPC_20250212_1451ShortLong_interactive_fov_roi_dff.html`,
  so the two arrays load automatically; or
* double-click the HTML file and load the arrays by hand, as in the next step.

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

Then label ROIs ([interface tour](interface.md)) and save with **Save Labels**
([what each export contains](export.md)).

## Part 3: Every input route, from one script

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

Step 2 is the template for any segmentation software. It reads the Suite2p session into
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

!!! warning "Check the ROI pairing yourself"
    Row *i* of every per-ROI array must describe the ROI that sorts to row *i* in the
    masks. No tool can verify that from the files; open a reviewer and check that an ROI
    you know is active has an active trace ([why](inputs.md#the-one-thing-no-tool-can-check)).

### Where the defaults differ from Suite2p

For masks from other software, the morphology metrics are computed the Suite2p way from
the masks you supply, so they can differ from Suite2p's own values: binary masks carry
no pixel weights, and no mask format carries Suite2p's detection centre, which changes
compactness most. Start from the `all_rois` filter set and set thresholds from the
distribution plots ([details](cli-mapping.md#filter-sets-and-masks-from-other-software)).
dF/F is computed from `F.npy` and `Fneu.npy` with a neuropil coefficient of 0.7 unless you
set `neuropil_coefficient` in `external_roi_metadata.json` or supply `dff.h5`.
