# Interface reference

Every panel, menu card and menu section in the reviewer has an **(i)** icon in its
top-right corner (beside the title on the full-width stacked-trace and motion panels).
It opens a short bulleted summary that links to the matching section below. The
sections follow the page from top to bottom, then the menu cards on the right.

## Field of view

- Each image is the session's mean frame for one channel, with every visible ROI
  outlined in its current label's color. The anatomical panel's **Overlay** menu picks
  which ROI set it draws.
- Click an ROI to select it; every linked panel follows.
- Ctrl/⌘ + scroll or a trackpad pinch zooms, drag pans, and double-click resets. A plain
  scroll always scrolls the page.
- The corner icon or the divider beside the summary column resizes the field of view;
  double-click either to return to the default size. A size you drag to is remembered.

The image inputs are listed in [Mean images](inputs.md#mean-images).

## Summary by QC label {#summary-of-good-bad-and-unsure-rois}

The column beside the field of view groups every ROI in the session by its current
label and updates as you label. It is visible when the reviewer opens; **Hide** closes
it and the **Show summary by QC label** tab brings it back. ROIs that a metric filter
labeled Bad count as Bad.

- **ROIs by label:** the count and share of all ROIs with each label (Good, Bad, Unsure,
  Not labeled).
- **Firing rate by label:** a kernel density estimate of the per-ROI firing rates for
  each label. An ROI's firing rate is its number of inferred spikes above threshold,
  counted at least 0.25 s apart, divided by the analyzed recording time, in events per
  second. The curves use a Gaussian kernel with Silverman's bandwidth, reflected at zero,
  and each has area 1 so labels compare by shape; the dashed line marks each label's
  median.
- **Average transient**, with a **By label** / **Selected ROI** switch:
    - *By label:* each ROI's spike-triggered average dF/F (windows from 0.5 s before to 2 s
      after each spike, each with the median of its own pre-spike part subtracted),
      then the mean ± s.e.m. of those per-ROI averages across ROIs, so each ROI counts
      once. Windows whose baseline-subtracted dF/F exceeds 20 in magnitude are left out,
      since values that large come from near-zero baselines rather than calcium.
    - <a id="inferred-spike-fit-panel"></a>*Selected ROI:* the selected ROI's transient
      estimated by regression (see [Inferred spike metrics](#inferred-spike-metrics)),
      with the fitted exponential decay in red. It is the same estimate the inferred
      spike metrics use, so the tau shown is the ROI's inferred spike decay tau.

The firing-rate and transient plots need dF/F and inferred spikes; until both are
loaded a short message says what is missing. The label groups use each ROI's default
threshold, or the threshold set for every ROI when **Default for all ROIs** is set to
the same setting for every ROI (see [Inferred spike threshold](#inferred-spike-threshold)).
The selected-ROI transient follows the threshold as you move it.

## Selected ROI trace

The dF/F trace of the selected ROI, with inferred spikes above threshold drawn as red
dots when **Show Inferred Spikes** is checked. Ctrl/⌘ + scroll or pinch zooms time, drag
pans, and the **Start s** / **End s** fields set the window, which the motion panels
follow. The panel's (i) popover holds this section's summary and the loading status of
the two arrays.

### Inferred spike threshold

The inferred-spike array can hold any per-frame estimate from any method: deconvolved
amplitudes (for example Suite2p's OASIS or CaImAn's `S`), spike rates or probabilities
(for example CASCADE), counts, or signed values. The threshold is a cutoff in the units
of that array. Frames above it are the red dots and feed the average transient, its taus
and the firing rate.

**Spike threshold** (the button shows the current value) opens a menu with:

- **The slider and value box**, with the value converted to the other unit beside them.
- **Units:** **Raw values** of the array, or **SD**, *k* standard deviations. An SD
  threshold is center + *k* × scale of the selected ROI's own values, computed separately
  for each ROI and never pooled across the session, so one *k* gives each ROI its own
  threshold.
- **SD measured as**, to suit the array's format:

    | Choice | Center and scale | Suits |
    |:---|:---|:---|
    | Non-zero values | mean and SD of the ROI's non-zero values | sparse non-negative output (OASIS, Suite2p, CaImAn `S`, counts) |
    | All frames | mean and SD of every frame | dense rates or probabilities (CASCADE) |
    | Robust | median and 1.4826 × median absolute deviation of every frame | signed or noisy output |
    | Auto | chosen from the session: robust if more than 1% of values are negative, non-zero if more than half are zero, otherwise all frames | |

  An ROI with too few distinct values for an SD falls back to its raw-value threshold.
- **Default for all ROIs:** either each ROI's own default (the generator's per-ROI value;
  the slider then changes only the selected ROI), or one setting for every ROI, in the
  current units, applied on selection and in the summary column (the slider then sets it
  for all ROIs).
- **Reset selected ROI to default.**

The [inferred spike metrics](#inferred-spike-metrics) in the Filter menu are computed by
the generator at each ROI's default threshold, so this menu does not change them. For
Suite2p and OASIS see
[Suite2p's deconvolution documentation](https://suite2p.readthedocs.io/en/latest/deconvolution/)
and the [OASIS paper](https://doi.org/10.1371/journal.pcbi.1005423).

### Loading dF/F and inferred spike files

Small sessions carry both arrays inside the HTML. Larger ones keep them beside it as
`SESSION_dff.npy` and `SESSION_inferred_spikes.npy` (see
[Embedded or separate trace arrays](cli-mapping.md#embedded-or-separate-trace-arrays)).
Opened through a web server they load automatically; opened from disk, choose them with
**Load dF/F** and **Load inferred spikes** ([why](tutorial.md#open-it)). Show
Inferred Spikes and the threshold menu appear once spikes are loaded.

Each file is a NumPy `.npy` array, one row per ROI in original-index order and one
column per frame, both the same shape, C-ordered little-endian float32 (float64 is
converted on load):

```python
import numpy as np
np.save("SESSION_dff.npy", np.ascontiguousarray(dff, dtype=np.float32))                 # (n_rois, n_frames)
np.save("SESSION_inferred_spikes.npy", np.ascontiguousarray(spikes, dtype=np.float32))  # same shape
```

## Stacked traces

- One dF/F trace per visible ROI, in the current sort order, sharing the selected ROI's
  time window. Two rows carrying the same events are usually two ROIs sharing a source.
- **First ROI** and **Last ROI** choose which sort positions are drawn; **Show all
  visible traces** draws every visible ROI.
- Ctrl/⌘ + scroll or pinch zooms time; double-click resets.

## Motion correction

The top plot shows the rigid x and y shift of each frame in the current time window, in
pixels; the bottom plots show the distribution of shifts over the session. A trace that
looks odd at one moment is often from a session that moved then. The input format is in
[Motion offsets](inputs.md#motion-offsets).

## Labeler

Good, Bad, Unsure and Not labeled, by button or with the G, B, U and N keys; the left and
right arrows step through the visible ROIs in the current sort order. **Show ROIs**
restricts the visible ROIs by label (for a second pass over the unsure ones), applied
after the metric filters.

### Original index and sort position

The **Original Index** is the ROI's fixed identifier. It is 0-based, like every file the
generator reads and writes: ROI *k* is row *k* of `stat.npy`, `F.npy`, `Fneu.npy`,
`iscell.npy`, `dff.h5`, `spikes.h5`, the `_dff.npy` and `_inferred_spikes.npy` files and
the exported `roi_manual_labels.npy`, and *k* in every `roi_index` column. With N ROIs it
runs from 0 to N − 1. For masks from other software it is the ROI's row after the mask
labels are sorted (see [Dense label masks](inputs.md#dense-label-masks)), and for CaImAn
it is the component number.

The **sort position** counts from 1 within the ROIs currently shown, in the current sort
order, and changes when you sort, filter or hide labels. Type an original index in the
box to jump to that ROI.

### Assigning labels to metric ranges

The **Bulk labeling menu** labels every visible ROI at once (**Label all visible ROIs
as ...**, including Not labeled) or labels ROIs from ranges of one metric:

- **Metric:** any numeric metric, including uploaded ones.
- **Assign good/bad/unsure labels** sets Good, Bad, Unsure or Not labeled, as the keys do;
  **Only change ROIs that are not labeled** protects labels you have already set.
- **New variable** instead stores each ROI's range number (1 for the first range, 2 for
  the second, blank outside every range) as a categorical variable with the name you
  give, keeping each range's name. It appears under **Uploaded ROI Metrics** in the
  **Filter ROIs** and **Sort filtered ROIs** menus and in the spreadsheet export.
- **The plot** is the metric's distribution. The highlighted range row's min and max are
  drawn on it and can be dragged; the other ranges are shaded in their label's color.
- **Ranges** need a minimum, a maximum or both (a blank side is open), have inclusive
  bounds, and are checked from the top; an ROI takes the first range it falls in. ROIs
  outside every range, or with no value, are left unchanged.
- Neither option depends on the threshold filters or needs **Apply Filters**.

## Filtering ROIs {#filtering-rois}

**Filter ROIs** opens the threshold menu. Editing a field immediately previews how many
ROIs pass; **Apply Filters** then labels every failing ROI Bad and passing ROIs Not
labeled. After manual labeling, refresh the view with a sort rather than pressing Apply
Filters again, which would reset those labels. An ROI passes only if it meets every
threshold that is set; a blank field is not used. The spreadsheet's `reason` column
records which thresholds each ROI failed.

### Filter sets

A **filter set** is a named group of thresholds. Choosing one in **Filter set name**
fills in its values, and any field can then be changed; **Restore selected filter set**
resets them. The built-in sets are `all_rois` (no thresholds), `soma`, `dendrite` and
`dendrite_relaxed`; sets you save or import are marked "(saved)". A build can open on any
of them with [`--filter-set` or `--filter-settings`](reference.md#build-settings).

### Metric distributions

Each metric's panel is titled with its name and shows a histogram of that metric across
all ROIs, over its 1st to 99th percentile range, above its **Min** and **Max** fields.

- For numeric metrics, the mean is a dark dotted line and the 25th and 75th percentiles
  are dashed gray lines.
- An unused min or max sits on its percentile line. Select or drag it and it becomes that
  threshold, drawn solid (green for a min, red for a max), with the field filled in;
  dragging a set line adjusts it.
- Double-click a set line to remove it, the same as leaving the field empty (Not Used).
- The text under the plot counts the ROIs below the min, between, and above the max as
  you adjust them.
- Click the plot away from the lines for a purple line reporting a value's percentile and
  how many ROIs fall below and at or above it. **Summary statistics** lists the count,
  minimum, quartiles, median and maximum.
- A value outside the plotted range is pinned to the nearer edge with an arrow.

### Morphology metrics

- **ROI area:** the number of pixels in the ROI, each counted once whatever its weight
  (`xpix` for Suite2p, labeled pixels of a dense mask, nonzero pixels of a sparse
  component).
- **Aspect ratio:** Suite2p's `aspect_ratio`: a 2-D Gaussian fitted to the ROI's weighted,
  soma-cropped pixels gives 2 × r_major / (r_major + r_minor + 0.01), about 1 for a round
  ROI and approaching 2 for an elongated one.
- **Compactness:** Suite2p's `compact`: the mean distance of the soma-cropped pixels from
  the ROI's center, divided by that of an ideal disk with the same number of pixels,
  minimum 1. A disk scores 1.
- **Connectivity:** the number of separate 4-connected pixel groups.

For Suite2p ROIs, aspect ratio and compactness are read from `stat.npy`. For masks from
other software the generator computes them the same way and matches Suite2p exactly
given the same pixel weights and ROI centers. Binary masks carry no weights, and no mask
format carries Suite2p's detection center (the generator centers the soma crop on the
median pixel), so values for such masks are close to, not identical with, Suite2p's.
Suite2p's `footprint` (a detection-scale index) is not defined for them. See
[Suite2p's stat.npy fields](https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields)
and the [Metric reference](metrics.md).

### Fluorescence trace metrics

- **Skew:** Suite2p's `skew`, the skewness of the ROI's trace; a trace activity field,
  not a shape field.
- **SNR: 95/50 percentile:** the 95th minus the 50th percentile of dF/F, divided by the
  residual noise.
- **SNR: CaImAn (large-transient score):** an exceptional-event score after CaImAn's
  component evaluation; larger values mean stronger large transients.

### Inferred spike metrics

All three come from one estimate of the ROI's average calcium transient. They need only
an ROI-by-frame inferred-spike array, from any method.

**The transient estimate.** Let *s*(*t*) be 1 at every frame whose inferred-spike value is
above the threshold and 0 otherwise. The ROI's dF/F is modelled as a constant plus one
copy of an unknown transient *k* placed at every such frame:

dF/F(*t*) = *c* + sum over lags ℓ of *k*(ℓ) · *s*(*t* − ℓ), with ℓ from −0.5 s to +2 s.

The 76 values of *k* (at 30 Hz) and the constant *c* are found by least squares. Where two
events overlap the model adds their transients, so each is assigned to the right lag
rather than piled into one average. A plain event-triggered average is the true transient blurred by
how events cluster in time, which distorts the decay when events come in bursts. In
simulated recordings with known decay times, the regression estimated decay tau to within
about 5% (median) for both independent and bursting spikes, against 6% and 18% for an
event-triggered average. At least 3 frames above the threshold are needed.

- **Inferred spike SNR:** (peak of *k* − mean of *k* before the event) ÷ SD of *k* before
  the event. The pre-event values of *k* should be near zero, so their spread measures
  how noisy the estimate is; the SNR grows with the number of events.
- **Rise tau:** time between the 10% and 90% crossings on the rising side of *k*, found
  backwards from the peak and interpolated between frames.
- **Decay tau:** *k* from its peak is fitted with *b* + *A* exp(−*t*/τ), where *b* is the
  mean of *k* before the event; τ is the decay tau (searched from half a frame to 10 s).
  If no fit is found, the interpolated time to fall to 1/e of the amplitude is used.

One inferred spike often spreads over several frames, and each such frame gets its own
copy of *k*. The shape and the time constants are unaffected, but the height of *k* is
an amplitude per frame above threshold, smaller than a whole transient. The spreadsheet's
`inferred_spike_frames_above_threshold` column gives the number of frames each ROI's
estimate used.

**Recomputing at another threshold.** The generator computes these at its default
threshold. The **Spike threshold** field at the top of this section and the dF/F panel's
spike threshold are one setting: changing either changes the other and applies it to every
ROI, in raw values or per-ROI SD units. **Recompute inferred spike metrics**, here or in the
dF/F panel's threshold menu, recalculates all three metrics for every ROI at that setting
with the same definitions, then updates the filters, distributions, sort order and
spreadsheet. A note under the button says which threshold the current values were computed
at, and warns when the threshold has changed since. The spreadsheet's
`inferred_spike_metric_threshold` column records the threshold used for each ROI, and
**Save current state into HTML** keeps the recomputed values and the threshold.

### Cell identity labels

Cell type or compartment labels (for example PV/SST or soma/axon). Check one or more to
keep only ROIs with those labels; with none checked the filter is not used. They come
from the session's files or from a `cell_identity_code` column; see
[Uploaded metrics and categorical labels](inputs.md#uploaded-metrics-and-categorical-labels).
Other categorical variables have the same kind of filter under Uploaded ROI Metrics.

### Classifier score

- A classifier's per-ROI output from 0 to 1, from `roi_model_scores.h5`
  ([format](inputs.md#roi-model-scores)); higher means the classifier rates the ROI as more
  likely good. Its filter has a minimum only.
- It ranks ROIs but is not a calibrated probability: a score of 0.8 does not mean that 80%
  of such ROIs are good.
- The scores do not label ROIs unless the reviewer was built with label initialization,
  in which case ROIs open Good at or above the good threshold, Bad at or below the bad
  threshold, and Unsure in between.
- To label ROIs from score ranges in the open reviewer, use
  [the Bulk labeling menu](#assigning-labels-to-metric-ranges).

### Uploaded ROI metrics

Your own numeric metrics and categorical variables, from a build-time file or an upload
([format](inputs.md#uploaded-metrics-and-categorical-labels)), plus variables made in the
Bulk labeling menu. A numeric metric's **Section** menu moves it to any section of the
Filter menu and the matching group of the Sort menu.

### Saving and importing filter sets

- **Save new filter set** keeps the current thresholds under a name in the open page.
- **Save filter set into HTML** also downloads a copy of the HTML with it, the current
  labels and the sort order embedded.
- **Import filter set JSON** loads a file exported with **Export current filter/sorting
  settings JSON** (under View and Export), or one written by hand; the same file can be
  given at build time with `--filter-settings`.

Only `filter` is required. Leave out thresholds you do not use, or write `null` for an
unused min or max. `cellTypeFilter` lists the cell identity codes to keep. Uploaded
metrics are keyed by their column name in lower case, with spaces and punctuation
replaced by underscores; a categorical one takes `{"codes": [...]}`. Sort metrics for
uploaded metrics are written `custom:` plus that name.

```json
{
  "name": "strict soma",
  "filter": {
    "skewMin": 0.5,
    "aspectMax": 1.5,
    "roiAreaMin": 40,
    "cellTypeFilter": ["1"],
    "customMetrics": {"snr": {"min": 2.0}, "layer": {"codes": ["2"]}}
  },
  "sort": {"metrics": ["custom:snr"], "direction": "desc"}
}
```

## Sorting ROIs {#sorting-rois}

**Sort filtered ROIs** orders the ROIs shown in the field of view, the stacked traces and
the arrow-key order. The current order is named beside **Sorting**.

### Sort metrics

The Sort menu offers the same metrics as the Filter menu, with the same definitions. Each
checked metric is converted to a 0 to 1 normalized value and the values are summed with
equal weight, so a reviewer can walk the session worst-first by any combination.
Categorical variables sort by their integer code.

## Uploading files

**Upload Custom ROI Metrics/Cell Identity Labels** loads one `.csv`, `.tsv` or `.txt`
file holding any mix of numeric metrics, categorical variables and cell identity labels;
the dialog lists the columns it requires, says what it loaded, and asks you to name any
codes that came without a `_label` column. The format is given once, in
[Uploaded metrics and categorical labels](inputs.md#uploaded-metrics-and-categorical-labels).

## Saving labels and exports

**View ROI metric spreadsheet** shows every ROI's labels and metrics in a new window, and
**Save Labels** saves the reviewer or exports the spreadsheet, labels or filter settings.
Each export, its columns and how to load it downstream are described in
[Export and downstream use](export.md).
