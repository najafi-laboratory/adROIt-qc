# Input formats

This page is the reference for every file the generator reads: its name, shape, container
and the helper that writes it. The [Tutorial](tutorial.md#part-3-preparing-your-own-inputs)
walks through preparing a session from your own software, and the
[Programmatic reference](reference.md) lists the build settings and helper functions.

A session directory is read in one of two layouts. In both, row *i* of every per-ROI
file is ROI *i*, the ROI's 0-based **original index**. Shapes use `n_rois` (ROIs),
`n_frames` (frames) and `(Ly, Lx)` (image height and width in pixels).

## The Suite2p layout

Read with `input_layout: suite2p`.

| File | Required | Format |
|:---|:---|:---|
| `suite2p/plane0/stat.npy` | yes | Suite2p ROI statistics (pixels, weights, shape statistics) |
| `suite2p/plane0/F.npy`, `Fneu.npy` | yes; `Fneu` optional | `(n_rois, n_frames)` raw and neuropil fluorescence |
| `suite2p/plane0/ops.npy` | yes | frame rate `fs`, `neucoeff`, mean images `meanImg` / `meanImg_chan2`, `max_proj`, `xoff` / `yoff`, `diameter` |
| `suite2p/plane0/iscell.npy` | no | `(n_rois, 2)` Suite2p accept/reject and probability; shown, never filters |
| Every optional file of the [documented layout](#the-documented-layout) except the masks, traces and metadata | no | placed at the session root |

## The documented layout

Read with `input_layout: external_rois`, for masks from any software.
`adroit_qc.inputs` writes each file from Python arrays ([helpers](reference.md#helper-functions)),
`scripts/make_standard_inputs.py` writes a directory in this form from array files, and
`python -m adroit_qc.validate` checks one against it.

| File | Required | Format | Helper |
|:---|:---|:---|:---|
| `roi_mask.npy` | one of these two | `(Ly, Lx)` integer label image, 0 = background; sorted positive labels are rows 0, 1, … ([details](#dense-label-masks)) | `write_label_mask` |
| `spatial_components.npz` | | SciPy sparse `(Ly·Lx, n_rois)` pixel weights, column *k* = ROI *k* ([details](#overlapping-rois)) | `write_spatial_components` |
| `F.npy` | yes | `(n_rois, n_frames)` float32 raw fluorescence | `write_traces` |
| `Fneu.npy` | no | same shape, neuropil; without it nothing is subtracted | `write_traces` |
| `dff.h5`, dataset `dff` | no | same shape; displayed and measured as given instead of dF/F computed from `F.npy` | `write_dff` |
| `spikes.h5`, dataset `spikes` | no | same shape; inferred spikes from any method in their own units; attribute `event_threshold` sets the opening threshold | `write_spikes` |
| `mean_func.npy`, `mean_anat.npy`, `max_func.npy`, `max_anat.npy` | no | `(Ly, Lx)` images ([details](#mean-images)) | `write_images` |
| `move_offset.h5`, datasets `xoff`, `yoff` | no | `(n_frames,)` rigid shift per frame in pixels ([details](#motion-offsets)) | `write_motion_offsets` |
| `roi_cell_type_labels.npy` | no | `(n_rois,)` float: 1 inhibitory, −1 excitatory, 0 unsure, NaN unknown | `write_cell_types` |
| `iscell.npy` | no | `(n_rois, 2)` upstream accept (1) / reject (0) and probability | `write_iscell` |
| `roi_model_scores.h5` | no | classifier output, below | `write_classifier_scores` |
| `external_roi_metadata.json` | needed for sparse masks without an image | below | `write_metadata` |

**`external_roi_metadata.json` fields.** `image_shape` `[Ly, Lx]`; `flatten_order` `"F"`
(column-major, CaImAn) or `"C"`; `frame_rate` in Hz (default 30); `diameter`, expected ROI
diameter in pixels, one number or `[dy, dx]`; `neuropil_coefficient` (default 0.7).

**`roi_model_scores.h5`, from any classifier** ([details](#roi-model-scores)).
Dataset `classifier_score` (or `probability`), 0–1 per ROI, higher = more likely good;
`state` per ROI, `good`, `bad` or `gray`; optional `roi` (or `suite2p_roi`) original
index per row. Attributes `good_threshold`, `bad_threshold`, `model_source`,
`target_structure`. The reviewer also reads `roi_qc_predictions.h5` and
`roi_predictions.h5` under the same names.

Sparse component matrices from CaImAn-style pipelines are accepted directly; supply
`flatten_order: "F"` in the metadata file unless your pixel indices are row-major.

## dF/F and inferred spikes

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

**dF/F.** If `dff.h5` is present it is used as given, so dF/F from any method (your own
baseline, CaImAn's `F_dff`, a different neuropil model) can be reviewed; it must have the
same shape as `F.npy`, which is still required. Otherwise the generator computes dF/F
as Suite2p does: `F − c·Fneu`, divided by its Gaussian-filtered baseline, with `c` the
`neuropil_coefficient` from the metadata file (default 0.7) and no subtraction when
`Fneu.npy` is absent. `dff.h5` is read only in this layout; Suite2p sessions always
compute dF/F from their own traces.
Inferred spikes have no such fallback: without `spikes.h5`, the spike overlay and
every inferred-spike metric are unavailable, and the reviewer says so rather than
showing an empty panel.

The opening [spike threshold](interface-reference.md#inferred-spike-threshold) can be set
with the `event_threshold` attribute of `spikes.h5` (helper: `write_spikes(..., event_threshold=)`).

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
| Label *k* really belongs to the trace in its row | not checkable; see [below](tutorial.md#the-one-thing-no-tool-can-check) |

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
| Matches | the 225 rows of the session's `F.npy`, and of the `_dff.npy` and `_inferred_spikes.npy` arrays in its [archived record](tutorial.md#try-the-reviewer-without-installing-anything) |
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
[CaImAn converter](tutorial.md#converting-from-caiman) writes it in this form.

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
| `classifier_score` | float per ROI from 0 to 1: the classifier's score, higher for ROIs it rates as more likely good (files that name this dataset `probability` are also read) |
| `state` | string per ROI: `good`, `bad`, or `gray` for undetermined |
| `suite2p_roi` or `roi` | optional original ROI index per row; otherwise rows are in ROI order |
| attributes `good_threshold`, `bad_threshold` | the thresholds that produced `state` |
| attribute `model_source` | provenance carried into the reviewer |

```python
import h5py
import numpy as np

with h5py.File("roi_model_scores.h5", "w") as handle:
    handle.create_dataset("classifier_score", data=scores.astype(np.float64))
    handle.create_dataset("state", data=np.array(states, dtype="S"))
    handle.create_dataset("suite2p_roi", data=np.arange(len(scores)))
    handle.attrs["good_threshold"] = 0.8
    handle.attrs["bad_threshold"] = 0.2
```

`adroit_qc.inputs.write_classifier_scores` writes this file from an array of scores and
two thresholds ([helpers](reference.md#helper-functions)). The score appears as
**Classifier score** under **Model Derived Metrics** in the Filter and Sort menus and as
`classifier_score` in the spreadsheet.

!!! note "A score, not a probability"
    Most classifiers end in a sigmoid, so their output lies between 0 and 1, but unless the
    classifier has been calibrated on held-out labels the value ranks ROIs rather than
    estimating how often ROIs with that score are good. A score of 0.8 does not mean 80%
    of such ROIs are good. Treat thresholds on it as ranking cutoffs.

The states
(`model_QC_label`) label ROIs only when the build sets
`initialize_labels_from_model_scores: true`.

## Files the open reviewer loads


| Control | File |
|:---|:---|
| **Load dF/F**, **Load inferred spikes** | `(n_rois, n_frames)` float32 `.npy` ([format](interface-reference.md#loading-dff-and-inferred-spike-files)) |
| **Upload Custom ROI Metrics/Cell Identity Labels** | the [per-ROI table](#uploaded-metrics-and-categorical-labels) |
| **Import filter set JSON** | a file from **Export current filter/sorting settings JSON** |
