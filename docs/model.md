# Model inference and fine-tuning

This folder contains everything needed to run score-only inference with the two selected ROI-QC models, or fine-tune either existing model on new labeled sessions. 

## Included models

| Profile | Weights | Inputs | Default score thresholds |
|---|---|---|---|
| purkinje | weights/purkinje_all.pt | image + fluorescence/neuropil signals + 24 metrics | good >= 0.85; bad <= 0.15 |
| ppc_vis | weights/ppc_vis_signal_metrics.pt | fluorescence/neuropil signals + 24 metrics | good >= 0.85; bad <= 0.15 |

Only these two selected best pretrained checkpoints are included. Profiles and relative weight paths are in profiles.json.

Both models predict P(good ROI), not cell identity. 

## Installation and portability

Use Python 3.10 or later in a suitable environment:

    python -m pip install -r /path/to/model/requirements.txt

PyTorch CPU inference works; a compatible CUDA-enabled PyTorch installation can use a GPU. No model weights are downloaded at runtime.

You may copy this entire folder to another machine. Session locations are command-line arguments. 

## Inference commands

From any directory:

    python /path/to/model/run.py infer \
      --model ppc_vis \
      --session /path/to/session

For Purkinje dendrites:

    python /path/to/model/run.py infer \
      --model purkinje \
      --session /path/to/session


Options:

    --checkpoint /path/to/compatible/best_model.pt
    --output /path/to/output/roi_model_scores.h5
    --batch-size 32
    --device auto|cpu|cuda
    --good-threshold 0.85
    --bad-threshold 0.15
    --overwrite

The default output is <session>/roi_model_scores.h5. Existing score files are not overwritten unless --overwrite is supplied. A custom output directory is useful if the input session is read-only. The file must be placed at the reviewer session root before building the interface.

Inference never opens manual label files to get labels, never updates ROI_label.h5 or roi_manual_labels.npy, and never trains the model. It loads the selected existing model weights and checkpoint normalization, then evaluates all available QC ROIs.

## Required input layout

This initial package supports the same processed-session layout as the trained models:

    session/
      suite2p/
        plane0/
          ops.npy                  # must contain meanImg
          stat.npy                 # original Suite2p ROI rows
      qc_results/
        stat.npy                   # QC ROI rows
        fluo.npy                   # N x T
        neuropil.npy               # N x T
        masks.npy                  # 2-D ROI labels 1..N; zero is background

Each QC row must represent the same ROI across stat, fluo and neuropil. Mask label i+1 belongs to QC row i. Images/masks must share coordinates.

The original suite2p/plane0/stat.npy is used to map QC rows back to original ROI indices by exact ordered xpix/ypix arrays. Missing, ambiguous or repeated matches fail explicitly. Counts alone are never used to infer identity.

## Output consumed by interface

The HDF5 score file contains:

- suite2p_roi: original zero-based Suite2p ROI index for each row.
- qc_roi: corresponding zero-based QC row, for audit/debugging.
- probability: P(good ROI), float32.
- state: UTF-8 good, bad or gray.
- attributes: index_space=suite2p_original, model_name, model_source,
  model_path, checkpoint_sha256, target_structure, good_threshold,
  bad_threshold, preprocessing, source session, counts and creation time.

Only QC ROIs are scored. Original ROIs not present in qc_results have no score; they are not silently labeled bad.

IMPORTANT: use original Suite2p layout for these original-index score files. The current demo score loader does not translate original indices for a filtered qc_results reviewer layout. It can display the precomputed scores without importing this model package.


## Preserved preprocessing

The supplied checkpoints depend on the original preprocessing:
- 64 x 64 image patches: z-score mean-image channel, binary-mask channel,
  followed by division of BOTH channels by 255.
- 2048-point resampled fluorescence and neuropil traces.
- Independent robust signal normalization using median and 1.4826 * MAD,
  with standard-deviation and constant-trace fallbacks.
- The original 24 QC metrics in the original column order.
- Metric normalization using metric_mean/metric_std stored in the checkpoint.

PPC/VIS inference skips image-patch construction, but still uses the QC/original ROI identity information. Purkinje inference uses every branch.

## Fine-tuning commands

Fine-tuning loads the selected EXISTING checkpoint before the first optimization step. It does not start a new randomly initialized ROI classifier.

Supply at least two labeled sessions:

    python /path/to/model/run.py finetune \
      --model ppc_vis \
      --sessions /path/to/session_A /path/to/session_B /path/to/session_C

Or supply a JSON file containing a list of session-path strings:

    python /path/to/model/run.py finetune \
      --model purkinje \
      --sessions-file /path/to/sessions.json \
      --epochs 25 --batch-size 32 --lr 0.0001

Relative paths in the JSON file resolve relative to that JSON file.

Use --checkpoint to initialize from a compatible previously fine-tuned checkpoint.
The profile must still match the checkpoint architecture and feature count.

### Supported labels

Inference needs no labels. Fine-tuning reads, but never changes:

1. <session>/ROI_label.h5, when present, takes priority.
   good_roi and bad_roi are zero-based QC indices by default.
   An explicit index_space=suite2p_original attribute is also supported.
   Conflicting, duplicate, noninteger and out-of-range indices are errors.
2. Otherwise <session>/suite2p/plane0/roi_manual_labels.npy is read.
   It must have one row per ORIGINAL Suite2p ROI:
   0=bad, 1=good, 2=unsure, NaN=unlabeled.
   Labels are mapped exactly to QC rows.

Unsure/unlabeled ROIs are excluded from training. Use finalized independent human
labels rather than treating model-generated labels as ground truth.

### Training behavior and settings

- Default: 25 epochs, batch size 32, learning rate 1e-4, AdamW weight decay 1e-4.
  The fine-tuning learning rate is lower than the original from-scratch run.
- Session-aware 20% validation split, seed 42.
- At least two sessions must contribute labels, and both classes must exist in both splits.
- Checkpoint metric normalization is retained; no normalization is fitted on validation/test data.
- Training augmentation matches the original dataset implementation; validation has no augmentation.
- Image backbone: the original freezing convention trains the adapted first convolution and
  projection while freezing other ResNet parameters. Frozen-backbone running statistics remain fixed.
  --unfreeze-backbone makes the full image backbone trainable.
- For signal+metrics there is no image backbone.
- Best epoch: highest validation ROC-AUC; lower validation loss breaks ties.
- Early stopping patience 5, configurable.
- Architecture/profile and all preprocessing stay fixed.

Other options: --weight-decay, --patience, --val-ratio, --seed, --device, --output.
New output directories are required; existing training runs are never overwritten.

### Fine-tuning outputs

By default: model/runs/<profile>_finetune_<timestamp>/

- dataset/: generated training arrays; no input files are modified.
- run_config.json and split.json: source model, settings and exact split.
- history.json/history.csv: training and validation loss, accuracy, precision,
  recall, F1, ROC-AUC and confusion matrices saved after every epoch.
- summary.json: metrics at the selected best epoch.
- best_model.pt: complete reusable best checkpoint.
- last.pt: final epoch checkpoint plus optimizer/early-stopping state.
- best_validation_predictions.npz: selected validation predictions.

New checkpoints retain normalization, modality flags and feature schema, so inference
can use them directly with --checkpoint. Base weights under weights/ are never modified.
This package does not automatically resume an interrupted optimizer; --checkpoint starts
a new fine-tuning run from the supplied learned weights.