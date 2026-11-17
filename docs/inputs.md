# Preparing your inputs

The reviewer needs two things, where each ROI is and what its fluorescence did over
time, and can use more: inferred spikes from any method, mean images, motion offsets,
classifier scores and your own per-ROI metrics and labels. This page is the single
reference for every input file; the other pages link here.

## The documented layout

`scripts/make_standard_inputs.py` writes a directory in this form, and
`scripts/validate_inputs.py` checks one against it.

| File | Meaning |
|---|---|
| `roi_mask.npy` | dense 2-D integer label image, one integer per pixel naming the ROI it belongs to; see [Dense label masks](#dense-label-masks) |
| `spatial_components.npz` | alternative to the above: sparse `(pixels, n_rois)` component matrix |
| `F.npy`, `Fneu.npy` | raw fluorescence and neuropil, `(n_rois, n_frames)` float32, C-ordered |
| `dff.h5` (dataset `dff`) | optional precomputed dF/F, `(n_rois, n_frames)` float32 |
| `spikes.h5` (dataset `spikes`) | optional inferred spikes from any method, same shape: amplitudes, rates, probabilities, counts or signed values |
| `mean_func.npy`, `mean_anat.npy` | optional functional and anatomical mean images, `(Ly, Lx)`; see [Mean images](#mean-images) |
| `roi_cell_type_labels.npy` | optional per-ROI cell type codes; see [Uploaded metrics and categorical labels](#uploaded-metrics-and-categorical-labels) |
| `move_offset.h5` | optional rigid shifts per frame; see [Motion offsets](#motion-offsets) |
| `iscell.npy` | optional `(n_rois, 2)` upstream accept/reject and probability |
| `external_roi_metadata.json` | `image_shape` and `flatten_order` (needed for sparse components without an image), and optionally `frame_rate` and `diameter` |

Sparse component matrices from CaImAn-style pipelines are accepted directly; supply
`flatten_order: "F"` in the metadata file unless your pixel indices are row-major.

!!! danger "HDF5 for dF/F and spikes, not .npy"
    The generator reads dF/F and inferred spikes **only** from HDF5, at the dataset
    names above. A `dff.npy` or `spikes.npy` placed beside them is ignored without
    warning, which is why the validator reports one as an error.

    ```python
    import h5py
    import numpy as np

    with h5py.File("dff.h5", "w") as handle:
        handle.create_dataset("dff", data=np.ascontiguousarray(dff, dtype=np.float32))

    with h5py.File("spikes.h5", "w") as handle:
        handle.create_dataset("spikes", data=np.ascontiguousarray(spikes, dtype=np.float32))
    ```

    These are HDF5 on the input side because they are multi-array stage outputs that
    carry companion arrays and file attributes. The `.npy` form appears only on the
    output side, in the sidecars the browser reads, where a header simple enough to
    parse in a few lines of JavaScript is what keeps the reviewer self-contained.

If `dff.h5` is absent the generator computes dF/F from `F.npy` and `Fneu.npy`.
Inferred spikes have no such fallback: without `spikes.h5`, the spike overlay and
every inferred-spike metric are unavailable, and the reviewer says so rather than
showing an empty panel.

## Dense label masks

A dense label mask is a single 2-D integer array with the same height and width as
the field of view. Each pixel holds the number of the ROI it belongs to, and `0` means
the pixel belongs to no ROI. It is the format most segmentation tools can export
directly: Cellpose's `masks` output, a scikit-image `label()` result, or an ImageJ ROI
set rasterized to one label image are all already in this form.

<div class="screenshot" markdown>
![A dense label mask with labels 3, 7 and 12. Each pixel holds its ROI's label and 0 is background. The positive labels are sorted ascending, so 3, 7 and 12 become rows 0, 1 and 2, and row i of the trace files belongs to ROI i. The traces are illustrative.](img/label-mask-diagram.png)
</div>

The example uses labels 3, 7 and 12 to show the general case. Label 7 sits
highest in the image, but label 3 is smallest, so label 3 becomes row 0: rows follow
label value, not position, and the gaps between labels are skipped.

**File and shape.** Saved as `roi_mask.npy` (also found as `roi_masks.npy`,
`label_mask.npy` or `masks.npy`) in the input directory. The array is indexed
`mask[y, x]`, rows first, so its shape is `(Ly, Lx)`: the same shape as
`mean_func.npy` and in the same pixel coordinates as the movie the traces were
extracted from. Use an integer dtype (`int32` is safe for any realistic ROI count).

**How labels become ROI rows.** The generator collects every distinct positive label,
sorts them in ascending order, and assigns them to rows `0, 1, 2, …` in that order. Row
*i* of the mask set is paired with row *i* of `F.npy`, `Fneu.npy`, `dff.h5` and
`spikes.h5`. When the labels run `1, 2, …, N` with no gaps, this reduces to the short
rule "label *k* is row *k − 1*". When they do not, the sorted order still applies:
labels `{3, 7, 12}` become rows `0, 1, 2`, so the traces must be in that same order.

**What the reviewer shows as the ROI's index.** The "Original Index" in the labeler, the
exported spreadsheet and `roi_manual_labels.npy` is the row number, not the label
value. If your labels have gaps and you need to map reviewed ROIs back to them, keep the
sorted label list (`np.unique(mask[mask > 0])`): entry *i* is the label of row *i*. The
generator records the label as `source_label` in its internal ROI table but does not
display it.

**One ROI per pixel, no weights.** Each pixel can belong to only one ROI, so a dense mask
cannot express overlapping ROIs, and every pixel of an ROI counts equally. If your
segmentation produces overlapping or weighted footprints, as CaImAn's spatial
components do, supply them as `spatial_components.npz` instead; see
[Overlapping ROIs](#overlapping-rois).

**What is ignored.** Zero and negative values are treated as background. A label may
cover several separate patches of pixels; they form one ROI, and its connectivity
metric reports how many patches there are.

**What is checked, and what is not.**

| Check | Where |
|:---|:---|
| The mask is 2-D | generator and validator, as an error |
| The number of labels equals the number of trace rows | validator, as an error |
| Labels run `1..N` without gaps | validator, as a warning (gaps are legal, but are the usual sign of a row-order mismatch) |
| The mask has the same shape as the mean image | not checked: a mismatch draws outlines in the wrong place |
| Label *k* really belongs to the trace in its row | not checkable; see [below](#the-one-thing-no-tool-can-check) |

If the counts disagree and validation is skipped, the generator uses only as many ROIs
as both the mask and the traces have, without warning.

**Making one.** From per-ROI boolean masks, in the same order as your traces:

```python
import numpy as np

mask = np.zeros((Ly, Lx), dtype=np.int32)
for row, roi in enumerate(roi_masks):      # roi: boolean (Ly, Lx) array
    mask[roi & (mask == 0)] = row + 1      # first ROI wins where two overlap
np.save("roi_mask.npy", mask)
```

From a label image whose numbering does not match your trace order, pass
`--row-order` to `scripts/make_standard_inputs.py` with the label of each trace row; it
renumbers the mask to `1..N` in trace order.

### Worked example: the PPC demo session

Download [`roi_mask.npy`](examples/ppc/roi_mask.npy) (512 KB) and
[`mean_func.npy`](examples/ppc/mean_func.npy) (1 MB) for the posterior parietal cortex
session shown throughout these docs.

<div class="screenshot" markdown>
![The PPC session's dense label mask, labels 1 to 225, on its functional mean image, with an 18 by 18 pixel region enlarged to show the stored integer values.](img/ppc-roi-mask.png)
</div>

| Property | Value |
|:---|:---|
| Shape, dtype | `(512, 512)`, `int16` |
| Labels | `1..225`, gap-free, so label *k* is row *k − 1* |
| Matches | the 225 rows of the session's `F.npy`, and of the `_dff.npy` and `_inferred_spikes.npy` arrays in its [archived record](demo-sessions.md#try-them-without-installing-anything) |
| ROI pixels | 12,614 of 262,144; median 56 per ROI |

```python
import numpy as np

mask = np.load("roi_mask.npy")              # (512, 512) int16
labels = np.unique(mask[mask > 0])          # array([1, 2, ..., 225])
roi_17 = mask == 18                         # boolean mask of ROI row 17
print(roi_17.sum(), "pixels")
```

The file was made from the session's Suite2p `stat.npy`, which stores each ROI as
pixel lists with a weight per pixel (`ypix`, `xpix`, `lam`). Suite2p lets ROIs overlap
(this session ran with `allow_overlap: True`), but a dense mask cannot, so:

* **452 pixels belong to more than one ROI** in `stat.npy`, touching 73 of the 225 ROIs.
* Each shared pixel was given to the ROI with the **larger weight** at that pixel.
* As a result 68 ROIs are slightly smaller in the mask than in Suite2p, by 458 pixels in
  total. No ROI disappeared.

```python
mask = np.zeros((ops["Ly"], ops["Lx"]), dtype=np.int16)
best = np.zeros(mask.shape, dtype=np.float32)
for i, roi in enumerate(stat):
    y, x, w = roi["ypix"], roi["xpix"], roi["lam"]
    win = w > best[y, x]                    # this ROI outweighs the current owner
    mask[y[win], x[win]] = i + 1
    best[y[win], x[win]] = w[win]
```

!!! note "This is an illustration; a Suite2p session does not need it"
    For a Suite2p session the generator reads `stat.npy` directly and keeps every
    overlapping pixel and weight. Converting to a dense mask only changes the outlines and
    the morphology metrics recomputed from them (area, compactness, aspect ratio) for the
    ROIs that shared pixels. The traces do not change, because they come from `F.npy`,
    which Suite2p extracted from the full, overlapping ROIs.

### Overlapping ROIs

A dense mask stores one label per pixel, so overlaps must be resolved before saving it.
Choose by what your segmentation produced:

| Your ROIs | Supply | What happens to shared pixels |
|:---|:---|:---|
| Suite2p `stat.npy` | the Suite2p layout | kept in every ROI, with their weights |
| Overlapping or weighted footprints (CaImAn `A`, NMF-style methods) | `spatial_components.npz` | kept in every ROI, with their weights |
| Non-overlapping labels (Cellpose, watershed, hand-drawn) | `roi_mask.npy` | none exist |
| Overlapping binary masks you must flatten | `roi_mask.npy` | your choice: highest weight wins (above), first ROI wins, or set to 0 so they belong to no ROI |

Whichever you choose, the traces should come from the same ROIs you are reviewing. If
they were extracted from overlapping masks, the reviewer still pairs each trace with its
row correctly; only the drawn outline and the shape metrics reflect the flattened mask.

## Mean images

The field-of-view panels show one mean image per channel, with every ROI outlined on
top. Both are 2-D float arrays of shape `(Ly, Lx)`, in the same pixel coordinates as the
masks.

| Panel | Suite2p session | Masks from other software (file in the input directory) |
|:---|:---|:---|
| Functional channel | `ops.npy` `meanImg` | `mean_func.npy` (also found as `meanImg.npy`, `mean_green.npy` or `functional_mean.npy`) |
| Anatomical channel, optional | `ops.npy` `meanImg_chan2` when `nchannels` is 2 | `mean_anat.npy` (also `mean_red.npy` or `anatomical_mean.npy`) |
| Maximum projections, optional | `ops.npy` `max_proj` | `max_func.npy`, `max_anat.npy`; the mean image is used when absent |

A `masks.h5` at the session root with `mean_func` or `mean_anat` datasets takes
precedence over both; its `masks_anat` dataset adds a **Cellpose ROIs** option to the
anatomical panel's **Overlay** menu. Without any functional image the panel is drawn on a blank
background of the masks' size, which still lets you select ROIs.

## Motion offsets

The motion correction panels plot the rigid shift applied to each frame and the
distribution of shifts over the session. They read, in order:

1. `move_offset.h5` at the session root, with datasets `xoff` and `yoff`;
2. otherwise Suite2p's `ops.npy` `xoff` and `yoff`.

Each is a 1-D array with one value per frame, in pixels, in the same frame order as the
traces. Where the lengths differ the panels use the frames both have; with no offsets the
panels are left empty. Non-rigid (patch-wise) shifts are not drawn: store their per-frame
mean, or the rigid shifts, as `xoff` and `yoff`.

```python
import h5py
import numpy as np

with h5py.File("move_offset.h5", "w") as handle:
    handle.create_dataset("xoff", data=np.asarray(x_shift, dtype=np.float32))  # (n_frames,)
    handle.create_dataset("yoff", data=np.asarray(y_shift, dtype=np.float32))  # (n_frames,)
```

CaImAn's `shifts_rig` is an `(n_frames, 2)` array of `(y, x)` shifts; the
[CaImAn converter](#converting-from-caiman) writes it in this form.

## Uploaded metrics and categorical labels

Your own per-ROI values can be added to a reviewer in two ways, with the same file
format:

* at build time, with `custom_roi_metrics` in the configuration or `--custom-roi-metrics`
  on the command line (once per file); these are part of the HTML for everyone who opens
  it;
* in an open reviewer, with **Upload Custom ROI Metrics/Cell Identity Labels**; these stay
  in that page until you use **Save current state into HTML**.

**File format.** A comma- or tab-separated table (`.csv`, `.tsv`, or `.txt` in either
layout) with a header row and one row per ROI. JSON, `.npy` and `.npz` are not accepted.

| Column | Required | Meaning |
|:---|:---|:---|
| `roi_index` | optional | The ROI's 0-based original index (see [Original index](interface-reference.md#original-index-and-sort-position)). Without it, row *k* is ROI *k* and the file needs exactly one row per ROI. A `suite2p_index`, `roi` or `index` column is rejected with a message asking for `roi_index` |
| `NAME_code` | one per categorical variable | Integer codes of a categorical variable called NAME. One file can hold several |
| `NAME_label` | optional | The name of each code in `NAME_code`. Without it you are asked to name the codes after upload. A `NAME_label` column with no `NAME_code` column is an error |
| `cell_identity_code`, `cell_identity_label` | optional | The cell type/compartment variable, a categorical variable with its own filter and spreadsheet columns |
| any other column | optional | One numeric metric, named by its header. Blank or non-numeric cells are missing values |

```csv
roi_index,cell_identity_code,cell_identity_label,layer_code,roi_area_px,snr
0,1,PV,2,112,4.2
1,2,SST,3,86,1.8
2,1,PV,2,140,3.1
```

This file adds the cell type/compartment labels PV and SST, a categorical variable
`layer` whose codes 2 and 3 the reviewer asks you to name, and two numeric metrics.

**Where they appear.** Numeric metrics and categorical variables are listed under
**Uploaded ROI Metrics** in the Filter and Sort menus (a numeric metric's **Section** menu,
or `--custom-roi-metric-section NAME=SECTION`, moves it to another section). Each
categorical variable gets a checkbox filter, and each is exported as `custom_NAME` (the
code) and `custom_NAME_label` (the name) in the [metric spreadsheet](export.md). Code
names can be edited at any time in the upload dialog.

**Cell type labels stored with the session.** Without an upload, the generator looks for
`suite2p/plane0/roi_cell_type_labels.npy`, then `roi_cell_type_labels.npy` at the
session root, then a `labels` dataset in `masks.h5`: one float per ROI, `1`
inhibitory/red, `-1` excitatory/non-red, `0` unsure, `NaN` unavailable. A
`cell_identity_code` column, uploaded or given at build time, replaces them.

## ROI model scores

Optional per-ROI classifier output, read from the first of `roi_model_scores.h5`,
`roi_qc_predictions.h5` or `roi_predictions.h5` at the session root. Any classifier can
supply it.

| Dataset or attribute | Content |
|:---|:---|
| `probability` | float per ROI: the classifier's probability that the ROI is good |
| `state` | string per ROI: `good`, `bad`, or `gray` for undetermined |
| `suite2p_roi` or `roi` | optional original ROI index per row; otherwise rows are in ROI order |
| attributes `good_threshold`, `bad_threshold` | the thresholds that produced `state` |
| attribute `model_source` | provenance carried into the reviewer |

```python
import h5py
import numpy as np

with h5py.File("roi_model_scores.h5", "w") as handle:
    handle.create_dataset("probability", data=probability.astype(np.float64))
    handle.create_dataset("state", data=np.array(states, dtype="S"))
    handle.create_dataset("suite2p_roi", data=np.arange(len(probability)))
    handle.attrs["good_threshold"] = 0.8
    handle.attrs["bad_threshold"] = 0.2
```

The probability appears as **ROI model probability** under **Model Derived Metrics** in
the Filter and Sort menus and as `model_probability` in the spreadsheet. The states
(`model_QC_label`) label ROIs only when the build sets
`initialize_labels_from_model_scores: true`.

## Converting from another pipeline

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

## Converting from CaImAn

`examples/caiman/` holds a worked CaImAn path, run on the posterior parietal cortex
demo session:

* `run_caiman.py` reads raw Bruker OME-TIFFs for one channel, runs CaImAn rigid motion
  correction and CNMF, evaluates the components, and saves the result with
  `cnm.save(...)`, the motion-corrected memmap and the rigid shifts;
* `caiman_to_reviewer.py` converts that result into this layout without importing
  CaImAn.

```bash
python examples/caiman/run_caiman.py --tiff-dir raw/SESSION --channel Ch2 \
    --n-files 4 --frame-rate 30 --out caiman_out
python examples/caiman/caiman_to_reviewer.py --cnmf caiman_out/cnmf_results.hdf5 \
    --movie caiman_out/memmap_*order_C_frames_*.mmap --shifts caiman_out/rigid_shifts.npy \
    --frame-rate 30 --out data/SESSION_caiman
python scripts/validate_inputs.py data/SESSION_caiman
python -m roi_reviewer.summary data/SESSION_caiman --input-layout external_rois \
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

## Validate before you build

```bash
python scripts/validate_inputs.py data/my_session
```

It reports shape disagreements, a mask label count that does not match the trace row
count, arrays that are not C-ordered, a stray `.npy` where HDF5 is expected, and files
it cannot read at all. It exits non-zero when it finds an error, so it can gate a
pipeline step. `build_reviewer.py` runs the same checks unless you pass
`--skip-validation`.

## The one thing no tool can check

Whether mask label *k* really belongs to trace row *k−1*.

A mismatch produces a reviewer in which every mask is paired with the wrong trace.
Every array has the right shape, every check passes, and nothing in the interface
flags it. Pass `--row-order` if your segmentation numbers masks and traces
differently, and spot-check one known-active ROI before trusting a session: select it
in the field of view and confirm the trace panel shows the activity you expect in the
place you expect it.
