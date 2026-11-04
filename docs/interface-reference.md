# Interface reference

Each **Read more** box in the reviewer gives a two-sentence summary and links to
the matching section below, which has the full description.

## Inferred spike fit panel

- **Average transient:** the selected ROI's dF/F windows around inferred spikes,
  baseline-subtracted and averaged.
- **Exponential model:** a decay curve drawn from the average transient's peak
  using the empirical decay tau. That tau is the time the average transient
  takes to fall to 1/e of its peak amplitude.
- **Amplitude threshold:** only inferred spike amplitudes above the selected
  threshold contribute to the average transient.

The panel can be hidden with its **Hide** button and shown again from the dF/F
panel.

## Inferred spike amplitude threshold

Inferred spikes are loaded as an ROI-by-frame amplitude array. The viewer only
requires that array's shape and meaning, whatever method produced it, and says
which method produced it when the file records that. The values are inferred
spike amplitudes, not spike probabilities: larger values mean the inference
method assigned a stronger spike-like transient at that frame.

The amplitude threshold is a cutoff applied in the viewer. Frames with an
inferred spike amplitude above it are drawn as red dots on the dF/F trace and
are used for the average transient and the decay and rise taus. For Suite2p and
OASIS output, see
[Suite2p's spike deconvolution documentation](https://suite2p.readthedocs.io/en/latest/deconvolution/)
and the [original OASIS paper](https://doi.org/10.1371/journal.pcbi.1005423).

## Loading dF/F and inferred spike files

For larger sessions the dF/F and inferred-spike arrays are stored beside the
HTML rather than inside it. Opened through a web address they load
automatically; opened from disk, select them with **Load dF/F file** and **Load
inferred spikes file** (see
[Whether the sidecars load automatically](quickstart.md#whether-the-sidecars-load-automatically)).

Each file is a NumPy `.npy` array with one row per ROI, in original-index order,
and one column per frame, and both must have the same shape. Rows must be
C-ordered (each row one contiguous time series) and little-endian float32, or
float64, which is converted on load. To have them load automatically, name them
`SESSION_dff.npy` and `SESSION_oasis_spikes.npy`, with the reviewer's session
name.

```python
import numpy as np
# one row per ROI in original-index order, one column per frame
np.save("SESSION_dff.npy", np.ascontiguousarray(dff, dtype=np.float32))            # (n_rois, n_frames)
np.save("SESSION_oasis_spikes.npy", np.ascontiguousarray(spikes, dtype=np.float32))  # same shape
```

## Original index and sort position

The **Original Index** is the ROI's fixed identifier. It is 0-indexed, like
every input and output file the generator reads and writes. ROI *k* is row *k*
of each of these files:

- the session files `stat.npy`, `F.npy`, `Fneu.npy`, `iscell.npy`, `dff.h5` and
  `spikes.h5`;
- the `_dff.npy` and `_oasis_spikes.npy` sidecars;
- the exported `roi_manual_labels.npy`.

The same number appears in the `roi_index` column of the exported metric CSV,
and it is what `roi_index` means in uploaded metric and cell identity files.
With N ROIs it runs from 0 to N − 1, so the last of 225 ROIs is 224.

For masks from other segmentation software, the index is the ROI's row after
the mask labels are sorted in ascending order, not the label value stored in
the mask (see [Dense label masks](inputs.md#dense-label-masks)).

The **sort position** counts from 1 within the ROIs currently shown, in the
current sort order. It changes when you sort, filter or hide labels; the
original index never does. Type an original index in the box to jump to that
ROI.

## Summary of good, bad and unsure ROIs

These plots group every ROI in the session by its current label and update as
you label. ROIs that a metric filter labeled Bad are counted as Bad. The panel
is hidden when a reviewer opens; show it with **Show summary of good/bad/unsure
ROIs** in the dF/F panel, and hide it again with its **Hide** button.

- **Fraction of ROIs by label:** the share of all ROIs in the session with each
  label (Good, Bad, Unsure and Not labeled), with the count above each bar.
- **Average calcium transient by label:** each ROI's average dF/F transient,
  computed as in the inferred spike fit panel:
    - spikes are those above that ROI's default amplitude threshold, counted at
      least 0.25 s apart;
    - each spike's window runs from 0.5 s before to 2 s after it;
    - each window has the median of its own pre-spike part subtracted.

    The line is the mean of those per-ROI averages for each label. The band is
    the standard error of the mean across ROIs, so each ROI counts once however
    many spikes it has. Event windows whose baseline-subtracted dF/F exceeds 20
    in magnitude are left out of the average, as in the multi-session QC
    summary. Values that large come from near-zero baselines rather than
    calcium signals, and a few of them would otherwise set the scale for the
    whole label. Those events still count toward the firing rate. ROIs with no
    spike above their threshold, or with every window left out, have no
    transient; the note above the plots gives the counts.
- **Firing rate by label:** a kernel density estimate of the per-ROI firing
  rates for each label. An ROI's firing rate is its number of inferred spikes
  above threshold (counted as above) divided by the analyzed recording time, in
  events per second.
    - The curves use a Gaussian kernel with Silverman's bandwidth, reflected at
      zero because rates cannot be negative.
    - Each curve has area 1, so labels with different numbers of ROIs compare
      by shape rather than height.
    - The dashed line marks each label's median.
    - The x-axis runs to about the 99.5th percentile of all labeled ROIs.
    - A label whose ROIs all share one rate is drawn as a solid line at that
      rate.

The two lower panels need dF/F and inferred spikes. They are computed once both
are loaded and then only regrouped as labels change. Changing the amplitude
threshold for the selected ROI does not change them.

## Filter sets

A **filter set** is a named group of threshold values for the metric filters.
Choosing one fills in its thresholds, and any field can then be changed. The
built-in filter sets are `all_rois` (no thresholds), `soma`, `dendrite` and
`dendrite_relaxed`. Filter sets you save or import are listed with "(saved)".

Filters are conjunctive: an ROI passes only if it meets every threshold that is
set, so all conditions are combined with AND. A blank field is not used.

Filter sets can also be loaded when the reviewer is generated. Export one with
**Export current filter/sorting settings JSON** under View and Export, then pass
the file with `--filter-settings` or the `filter_settings` field of a YAML
config (see [Saving and importing filter sets](#saving-and-importing-filter-sets)).

## Metric distributions

The plot beside each metric's threshold fields is a histogram of that metric
across all ROIs in the session, over its 1st to 99th percentile range. Plots are
shown by default; click **Distribution** above a plot to hide it.

Vertical lines mark the thresholds, green for the minimum and red for the
maximum, and they move as you type. Where they are drawn depends on what is set:

1. Threshold values you enter.
2. Otherwise, the thresholds in the loaded filter set.
3. Otherwise, a starting value:
    - the mean for skew, aspect ratio, compactness, both SNRs, inferred spike
      SNR and the ROI model probability minimum;
    - the 25th and 75th percentiles for ROI area, inferred spike rise and
      decay tau and uploaded ROI metrics;
    - the 75th percentile for connectivity.

The value in parentheses beside a field is the one its line uses.

## Cell identity labels

Cell identity labels are cell-type labels (for example excitatory/inhibitory)
or compartment labels (for example soma/axon). Check one or more identities to
keep only ROIs with those labels; if none are checked, the filter is not used.

Labels are read from the session's pipeline outputs when present. They can also
be uploaded with **Upload Custom ROI Metrics/Cell Identity Labels** as a CSV,
TSV or TXT table or a JSON list. Give an integer code column, a name column, or
both: `cell_identity_code` and `cell_identity_label`. Legacy cell-type files
use `-1` excitatory/non-red, `0` unsure and `1` inhibitory/red; other files may
use any integer codes, with names supplied by the file.

`roi_index` is the ROI's 0-based original index (see
[Original index and sort position](#original-index-and-sort-position)). It is
optional: without it, row *k* is ROI *k* and the file needs exactly one row per
ROI. No other name is accepted for this column. The file formats are given in
[Upload file formats](cli-mapping.md#upload-file-formats).

## ROI model probability

ROI model probability is the trained classifier's output for p(good ROI),
loaded from `roi_model_scores.h5` when available. The currently registered
model was trained for cerebellar dendrite ROIs only. Treat it as a suggestion
specific to that target, not a general ROI quality score for other structures.
Its filter has a minimum only.

By default the scores are suggestions for filtering and sorting, and they do
not label ROIs when the page opens. If the reviewer was generated with label
initialization enabled, ROIs open with these labels:

- **Good** for a probability at or above the good threshold;
- **Bad** for a probability at or below the bad threshold;
- **Unsure** for probabilities between the two.

To label ROIs from score ranges in the open reviewer, use
[Assign Labels to Metric Ranges](#assigning-labels-to-metric-ranges).

## Uploaded ROI metrics

Uploaded ROI metrics are your own per-ROI values. They can be files given with
`--custom-roi-metrics` when the reviewer is generated, or QC metric files
uploaded with **Upload Custom ROI Metrics/Cell Identity Labels**. Every column
other than `roi_index` becomes a metric named after its header. Values must be
numbers, with blanks or `NaN` where missing. `roi_index` is optional, as for
cell identity labels. The accepted CSV, JSON and NumPy forms are given in
[Upload file formats](cli-mapping.md#upload-file-formats).

An uploaded metric stays in the Uploaded ROI Metrics section unless you move it
with its **Section** menu. That menu can place it under any section of the
Filter menu and the matching group of the Sort menu. The section can also be
set when the reviewer is generated, with
`--custom-roi-metric-section NAME=SECTION` or the `custom_roi_metric_sections`
field of a YAML config.

## Morphology metrics

- **ROI area:** the number of pixels in the ROI's mask, each pixel counted once
  whatever its weight.
    - For Suite2p ROIs these are the pixels listed in `stat.npy` (`xpix`),
      including any that Suite2p's soma crop leaves out of its shape
      statistics.
    - For masks from other software they are the pixels carrying the ROI's
      label in a dense label mask, or its nonzero pixels in a sparse
      spatial-component matrix.
- **Aspect ratio:** Suite2p's `aspect_ratio`. A two-dimensional Gaussian is
  fitted to the ROI's weighted pixels. When Suite2p's `soma_crop` option is on,
  the default, only the soma-cropped pixels are used. The ratio is
  2 × r_major / (r_major + r_minor + 0.01), from the radii of the fitted
  ellipse. It is about 1 for a round ROI and approaches 2 for a very elongated
  one.
- **Compactness:** Suite2p's `compact`. It is the mean distance of the ROI's
  soma-cropped pixels from their median position, divided by the mean distance
  for an ideal disk with the same number of pixels, with a minimum of 1. A disk
  scores 1 and less compact shapes score higher.
- **Connectivity:** the number of separate groups of connected pixels in the
  ROI's mask. ROIs with several disconnected pixel groups may indicate poor
  segmentation.

For Suite2p ROIs, aspect ratio and compactness are read from `stat.npy`. For
masks from other software the generator computes them the same way, with the
same soma crop. A binary label mask gives every pixel the same weight, whereas
Suite2p ROIs and sparse spatial components carry per-pixel weights. Values for
binary masks are therefore close to, but not identical with, what Suite2p would
report. An external `stat.npy` keeps the values it has. See also
[Suite2p's stat.npy field definitions](https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields)
and the [Metric reference](metrics.md).

## Fluorescence trace metrics

- **Skew:** Suite2p's `skew`, the skewness of the ROI's trace. It is a trace
  activity field, not a shape field
  ([Suite2p stat.npy fields](https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields)).
- **SNR: 95/50 percentile:** a robust transient amplitude, the 95th minus the
  50th percentile of dF/F, divided by the residual noise
  ([calculation code](https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_labels.py)).
- **SNR: CaImAn (large-transient score):** an exceptional-event score; larger
  values indicate stronger large-transient structure
  ([source code](https://github.com/farznaj/imaging_decisionMaking_exc_inh/blob/master/imaging/evaluate_components.py)).

## Inferred spike metrics

The inferred spike metrics are computed from dF/F windows around inferred spike
amplitudes above the threshold set when the reviewer was generated. They are
computed by `_oasis_event_waveform_metrics` in the generator. The current
pipeline produces inferred spike amplitudes with Suite2p/OASIS, but these
metrics only need an ROI-by-frame inferred spike amplitude array in the
documented format. Changing the amplitude threshold in the viewer updates the
selected ROI's display and diagnostics, not these metrics.

## Assigning labels to metric ranges

**Assign Labels to Metric Ranges** labels every ROI whose value of one metric
falls in a range you set. For example, you could label ROIs good for a model
probability of 0.85 or more, bad for 0.15 or less and unsure in between.
**Use model score thresholds** fills in exactly these, from the thresholds
stored with the scores.

Each range works as follows:

- It needs a minimum, a maximum or both. A blank side is open, so a range with
  only a minimum covers every value at or above it.
- Both bounds are inclusive.
- Ranges are checked from the top, and an ROI takes the first one it falls in.
- ROIs outside every range, or with no value for the metric, are left
  unchanged.

There are two ways to apply the ranges:

- **Good/bad/unsure labels** sets the review labels directly, as the G/B/U keys
  do.
- **New variable** adds an uploaded ROI metric with the name you give. Its
  value is the number of the matching range (1 for the first, 2 for the second,
  and so on, blank outside every range), and each range's name is kept for the
  exported spreadsheet. The new metric can then be filtered, sorted and
  exported like any other.

Neither option depends on the threshold filters or needs **Apply Filters**.

## Saving and importing filter sets

- **Save new filter set:** saves the current thresholds under the name you
  enter, inside the open reviewer. It appears in the **Filter set name** list
  until the page is closed.
- **Save filter set into HTML:** saves the filter set the same way, then
  downloads a copy of the HTML with it, the current labels and the current sort
  order embedded. Reopening that copy keeps them.
- **Import filter set JSON:** loads a file exported with **Export current
  filter/sorting settings JSON**, including its sort order, or one written by
  hand. The rules for the file are:
    - Only `filter` is required.
    - Leave out any threshold you do not want to use, and use `null` for an
      unused minimum or maximum.
    - `cellTypeFilter` lists the cell identity codes to keep.
    - Custom metrics are keyed by their column name in lower case, with spaces
      and punctuation replaced by underscores.
    - Sort metrics for uploaded metrics are written `custom:` plus that name.

The same file can be given at generation with `--filter-settings`.

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

## Sort metrics

The Sort menu offers the same metrics as the Filter menu, with the same
definitions:

- [Morphology metrics](#morphology-metrics);
- [Cell identity labels](#cell-identity-labels);
- [Fluorescence trace metrics](#fluorescence-trace-metrics);
- [Inferred spike metrics](#inferred-spike-metrics);
- the [ROI model probability](#roi-model-probability), when the session has
  scores.

Each checked metric is converted to a 0–1 normalized value, and the values are
summed with equal weight. Cell identity code sorts by the integer code.

## Saving labels and exports

- **Save current state into HTML:** downloads a reviewed copy of the HTML. It
  keeps the current labels, uploaded metrics, saved filter sets and sort order
  inside the file.
- **Export metric spreadsheet CSV:** saves one row per ROI. Each row has
  code-friendly column names, the current labels, the ROI metrics, filter
  failures and exclusion reasons. Its first column, `roi_index`, is the ROI's
  original index.
- **Export labels NPY:** saves `roi_manual_labels.npy`, a one-dimensional NumPy
  label array with one entry per ROI, in original-index order. The values are
  `NaN` not labeled, `0` bad, `1` good and `2` unsure.
- **Export current filter/sorting settings JSON:** saves the current
  thresholds, with the filter set name and the current sort order, for reuse
  or documentation.

How to load each export downstream is described in
[Export and downstream use](export.md#what-each-export-contains).

## Uploading files

**Upload Custom ROI Metrics/Cell Identity Labels** accepts two kinds of file:

- **QC metric:** your own per-ROI values to filter and sort by. The file has
  one row per ROI, one column per metric, and a header row whose names become
  the metric names. Values must be numbers, with blanks or `NaN` where missing.
- **Cell identity label:** one row per ROI with an integer
  `cell_identity_code`, a `cell_identity_label` name, or both. Names can be
  edited in the dialog after loading.

Both kinds accept CSV, TSV, TXT and JSON. `roi_index` is optional. Without it,
row *k* is ROI *k*, and the viewer reports an error if the number of rows does
not match the number of ROIs. Examples of every form are in
[Upload file formats](cli-mapping.md#upload-file-formats).
