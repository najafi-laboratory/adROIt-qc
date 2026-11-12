#!/usr/bin/env python3
# Vendored from najafi-laboratory/2p_imaging: utils_2p/processing_summary.py
# Branch claude/reviewer-updates, commit d24d1ab776, copied 2026-10-15.
# This is a copy, not a dependency; nothing here imports utils_2p.
# See VENDOR.md before re-syncing.
"""Generate static and interactive processing summaries for one session."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from roi_reviewer.roi_metrics import (
    dff_qc_metrics,
    load_iscell,
    map_qc_to_suite2p_rois,
    morphology_exclusion_reasons,
    roi_morphology_metrics,
    suite2p_stat_fingerprint,
)


QC_PRESETS: dict[str, dict[str, Any]] = {}
SUMMARY_TARGET_STRUCTURES = ("all_rois", "soma", "dendrite", "dendrite_relaxed")
# Filter-menu sections an uploaded ROI metric can be shown under (--custom-roi-metric-section).
CUSTOM_METRIC_SECTIONS = ("uploaded", "model_derived", "morphology", "fluorescence", "inferred_spike")


def normalize_qc_target(target_structure: str) -> str:
    """Return the reviewer-facing target/preset name."""

    target = target_structure.strip()
    if target == "neuron":
        return "soma"
    if target == "cerebellum_lax":
        return "dendrite_relaxed"
    return target


PDF_NAME_TEMPLATE = "{session_name}_processing_summary.pdf"
HTML_NAME_TEMPLATE = "{session_name}_interactive_fov_roi_dff.html"
ROI_INDEX_LABEL = "Suite2p Original Index"  # override with --roi-index-label, e.g. "Original Index"
_DEFAULT_ROI_INDEX_LABEL = ROI_INDEX_LABEL
_LAST_BUILD_WAS_HTML_ONLY = False  # set by create_preprocessing_summary for main()'s message
EMBEDDED_DFF_BYTE_LIMIT = 80 * 1024 * 1024
EMBEDDED_DFF_BASE64_EXPANSION = 4 / 3
EMBEDDED_DFF_HTML_OVERHEAD = 2 * 1024 * 1024
# Largest ROI x frame count whose float32 array is embedded in the HTML rather than written as a .npy sidecar.
EMBEDDED_MAX_ROI_FRAMES = int((EMBEDDED_DFF_BYTE_LIMIT - EMBEDDED_DFF_HTML_OVERHEAD) / EMBEDDED_DFF_BASE64_EXPANSION / 4)


@dataclass(frozen=True)
class SummaryInputLayout:
    name: str
    stat_path: Path
    fluo_path: Path
    neuropil_path: Path | None
    iscell_path: Path
    spikes_paths: tuple[Path, ...]
    roi_mask_path: Path | None = None
    spatial_matrix_path: Path | None = None

    @property
    def input_dir(self) -> Path:
        for path in (self.stat_path, self.fluo_path, self.roi_mask_path, self.spatial_matrix_path):
            if path is not None:
                return path.parent
        return self.stat_path.parent

    @property
    def dff_label(self) -> str:
        if self.name == "suite2p":
            return "raw dF/F from Suite2p F.npy/Fneu.npy"
        if self.name == "external_rois":
            return "raw dF/F from external ROI fluorescence arrays"
        return f"raw dF/F from {self.name} fluo.npy/neuropil.npy"


def _path_status(path: Path) -> str:
    return str(path) if path.exists() else f"{path} (missing)"


def _path_status_optional(path: Path | None) -> str:
    if path is None:
        return "not supplied"
    return _path_status(path)


def _layout_complete(layout: SummaryInputLayout) -> bool:
    has_roi_geometry = layout.stat_path.exists() or (
        layout.name == "external_rois"
        and (
            (layout.roi_mask_path is not None and layout.roi_mask_path.exists())
            or (layout.spatial_matrix_path is not None and layout.spatial_matrix_path.exists())
        )
    )
    has_neuropil = layout.neuropil_path is not None and layout.neuropil_path.exists()
    return has_roi_geometry and layout.fluo_path.exists() and (layout.name == "external_rois" or has_neuropil)


def _direct_suite2p_layout(plane_dir: Path, session_dir: Path) -> SummaryInputLayout:
    return SummaryInputLayout(
        name="suite2p",
        stat_path=plane_dir / "stat.npy",
        fluo_path=plane_dir / "F.npy",
        neuropil_path=plane_dir / "Fneu.npy",
        iscell_path=plane_dir / "iscell.npy",
        spikes_paths=(session_dir / "spikes.h5", session_dir / "qc_results" / "spikes.h5"),
    )


def _direct_filtered_layout(qc_dir: Path, session_dir: Path) -> SummaryInputLayout:
    return SummaryInputLayout(
        name=qc_dir.name,
        stat_path=qc_dir / "stat.npy",
        fluo_path=qc_dir / "fluo.npy",
        neuropil_path=qc_dir / "neuropil.npy",
        iscell_path=qc_dir / "iscell.npy",
        spikes_paths=(qc_dir / "spikes.h5", session_dir / "spikes.h5"),
    )


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _direct_external_layout(input_dir: Path, session_dir: Path) -> SummaryInputLayout:
    return SummaryInputLayout(
        name="external_rois",
        stat_path=input_dir / "stat.npy",
        fluo_path=_first_existing([input_dir / "F.npy", input_dir / "fluo.npy"]) or input_dir / "F.npy",
        neuropil_path=_first_existing([input_dir / "Fneu.npy", input_dir / "neuropil.npy"]),
        iscell_path=input_dir / "iscell.npy",
        spikes_paths=(input_dir / "spikes.h5", session_dir / "spikes.h5"),
        roi_mask_path=_first_existing(
            [
                input_dir / "roi_mask.npy",
                input_dir / "roi_masks.npy",
                input_dir / "label_mask.npy",
                input_dir / "masks.npy",
            ]
        ),
        spatial_matrix_path=_first_existing(
            [
                input_dir / "spatial_components.npz",
                input_dir / "roi_spatial_components.npz",
                input_dir / "A.npz",
            ]
        ),
    )


def _suite2p_layout(session_dir: Path) -> SummaryInputLayout:
    plane_dir = session_dir / "suite2p" / "plane0"
    return _direct_suite2p_layout(plane_dir, session_dir)


def _qc_results_layout(session_dir: Path, dirname: str) -> SummaryInputLayout:
    qc_dir = session_dir / dirname
    return _direct_filtered_layout(qc_dir, session_dir)


def _resolve_summary_session_and_layout(
    input_path: Path, requested: str = "auto"
) -> tuple[Path, SummaryInputLayout]:
    """Resolve a session root and input layout from either a session or data directory."""

    if input_path.name == "plane0" and input_path.parent.name == "suite2p":
        session_dir = input_path.parents[1]
        return session_dir, _direct_suite2p_layout(input_path, session_dir)
    if input_path.name == "suite2p" and (input_path / "plane0").is_dir():
        session_dir = input_path.parent
        return session_dir, _direct_suite2p_layout(input_path / "plane0", session_dir)
    if input_path.name in {"qc_results", "manual_qc_results"}:
        session_dir = input_path.parent
        return session_dir, _direct_filtered_layout(input_path, session_dir)
    if (input_path / "stat.npy").exists() and (input_path / "F.npy").exists() and (input_path / "Fneu.npy").exists():
        session_dir = input_path.parents[1] if input_path.name == "plane0" else input_path.parent
        return session_dir, _direct_suite2p_layout(input_path, session_dir)
    if (
        (input_path / "stat.npy").exists()
        and (input_path / "fluo.npy").exists()
        and (input_path / "neuropil.npy").exists()
    ):
        return input_path.parent, _direct_filtered_layout(input_path, input_path.parent)
    external_layout = _direct_external_layout(input_path, input_path.parent)
    if _layout_complete(external_layout):
        return input_path, _direct_external_layout(input_path, input_path)
    session_dir = input_path
    return session_dir, _resolve_summary_input_layout(session_dir, requested)


def _resolve_summary_input_layout(session_dir: Path, requested: str = "auto") -> SummaryInputLayout:
    layouts = {
        "suite2p": _suite2p_layout(session_dir),
        "qc_results": _qc_results_layout(session_dir, "qc_results"),
        "manual_qc_results": _qc_results_layout(session_dir, "manual_qc_results"),
        "external_rois": _direct_external_layout(session_dir, session_dir),
    }
    if requested != "auto":
        layout = layouts[requested]
        if not _layout_complete(layout):
            raise FileNotFoundError(
                f"Requested input layout '{requested}' is incomplete. Required files:\n"
                f"  stat: {_path_status(layout.stat_path)}\n"
                f"  roi mask: {_path_status_optional(layout.roi_mask_path)}\n"
                f"  spatial matrix: {_path_status_optional(layout.spatial_matrix_path)}\n"
                f"  fluorescence: {_path_status(layout.fluo_path)}\n"
                f"  neuropil: {_path_status_optional(layout.neuropil_path)}"
            )
        return layout

    for name in ("suite2p", "qc_results", "manual_qc_results", "external_rois"):
        layout = layouts[name]
        if _layout_complete(layout):
            return layout
    searched = "\n".join(
        f"{name}: stat={layout.stat_path}, fluorescence={layout.fluo_path}, neuropil={layout.neuropil_path}"
        for name, layout in layouts.items()
    )
    raise FileNotFoundError(f"Could not detect a complete summary input layout. Checked:\n{searched}")


def _log_summary_input_layout(layout: SummaryInputLayout, requested: str) -> None:
    mode = "auto-detected" if requested == "auto" else "explicit"
    print(f"Using {mode} preprocessing summary input layout: {layout.name}")
    print(f"  stat: {layout.stat_path}")
    if layout.name == "external_rois":
        print(f"  roi mask: {_path_status_optional(layout.roi_mask_path)}")
        print(f"  spatial matrix: {_path_status_optional(layout.spatial_matrix_path)}")
    print(f"  fluorescence: {layout.fluo_path}")
    print(f"  neuropil: {_path_status_optional(layout.neuropil_path)}")
    print(f"  iscell: {_path_status(layout.iscell_path)}")
    existing_spikes = [path for path in layout.spikes_paths if path.exists()]
    if existing_spikes:
        print(f"  inferred spikes: {existing_spikes[0]}")
    else:
        print("  inferred spikes: not found")


def _estimate_embedded_dff_bytes(n_rois: int, n_frames: int) -> int:
    raw_bytes = max(0, int(n_rois)) * max(0, int(n_frames)) * 4
    return int(raw_bytes * EMBEDDED_DFF_BASE64_EXPANSION + EMBEDDED_DFF_HTML_OVERHEAD)


def _load_npy(paths: list[Path], allow_pickle: bool = False) -> Any:
    for path in paths:
        if path.exists():
            return np.load(path, allow_pickle=allow_pickle)
    searched = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find any of:\n{searched}")


def _load_ops(session_dir: Path) -> dict[str, Any]:
    try:
        ops = _load_npy(
            [session_dir / "ops.npy", session_dir / "suite2p" / "plane0" / "ops.npy"],
            allow_pickle=True,
        ).item()
    except FileNotFoundError:
        ops = {}
    ops["save_path0"] = str(session_dir)
    return ops


def _load_stat(session_dir: Path) -> np.ndarray:
    return _load_npy(
        [session_dir / "qc_results" / "stat.npy", session_dir / "suite2p" / "plane0" / "stat.npy"],
        allow_pickle=True,
    )


def _load_roi_mask(session_dir: Path) -> np.ndarray:
    masks_h5 = session_dir / "masks.h5"
    if masks_h5.exists():
        with h5py.File(masks_h5, "r") as h5:
            if "masks_func" in h5:
                return np.asarray(h5["masks_func"])
    return _load_npy([session_dir / "qc_results" / "masks.npy", session_dir / "masks.npy"])


def _load_masks_h5_image(session_dir: Path, key: str) -> np.ndarray | None:
    path = session_dir / "masks.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if key not in h5:
            return None
        return np.asarray(h5[key])


def _load_layout_image(session_dir: Path, layout: SummaryInputLayout, key: str) -> np.ndarray | None:
    image = _load_masks_h5_image(session_dir, key)
    if image is not None:
        return image
    aliases = {
        "mean_func": ("mean_func.npy", "meanImg.npy", "mean_green.npy", "functional_mean.npy"),
        "max_func": ("max_func.npy", "max_proj.npy", "maxImg.npy", "functional_max.npy"),
        "mean_anat": ("mean_anat.npy", "mean_red.npy", "anatomical_mean.npy"),
        "max_anat": ("max_anat.npy", "max_red.npy", "anatomical_max.npy"),
    }
    for name in aliases.get(key, ()):
        path = layout.input_dir / name
        if path.exists():
            return np.asarray(np.load(path, allow_pickle=False))
    return None


def _load_masks_h5_label_mask(
    session_dir: Path,
    key: str,
    target_shape: tuple[int, int],
    ops: dict[str, Any],
) -> np.ndarray | None:
    path = session_dir / "masks.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if key not in h5:
            return None
        mask = np.asarray(h5[key])
    if mask.shape[:2] == target_shape:
        return mask
    xrange = ops.get("xrange")
    yrange = ops.get("yrange")
    if xrange is None or yrange is None:
        return None
    x1, x2 = map(int, xrange)
    y1, y2 = map(int, yrange)
    if mask.shape[:2] != (y2 - y1, x2 - x1):
        return None
    expanded = np.zeros(target_shape, dtype=mask.dtype)
    expanded[y1:y2, x1:x2] = mask
    return expanded


def _coerce_cell_type_label_values(labels: np.ndarray, source: str) -> list[int | None]:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    finite = labels[np.isfinite(labels)]
    if not np.isin(finite, [-1, 0, 1]).all():
        raise ValueError(f"Skipped {source}: values must be -1, 0, 1, or NaN.")
    return [None if not np.isfinite(value) else int(value) for value in labels]


def _load_cell_type_labels(
    session_dir: Path,
    n_rois: int,
    suite2p_stat: np.ndarray,
) -> tuple[list[int | None], str]:
    """Load optional excitatory/inhibitory labels indexed by original Suite2p ROI."""
    candidates = [
        session_dir / "suite2p" / "plane0" / "roi_cell_type_labels.npy",
        session_dir / "roi_cell_type_labels.npy",
    ]
    for path in candidates:
        if not path.exists():
            continue
        labels = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if labels.shape != (n_rois,):
            return [None] * n_rois, f"Skipped {path.name}: expected {n_rois} labels, found {labels.size}."
        try:
            return _coerce_cell_type_label_values(labels, path.name), str(path)
        except ValueError as exc:
            return [None] * n_rois, str(exc)

    path = session_dir / "masks.h5"
    if not path.exists():
        return [None] * n_rois, "not loaded"
    with h5py.File(path, "r") as h5:
        if "labels" not in h5:
            return [None] * n_rois, "not loaded"
        labels = np.asarray(h5["labels"], dtype=np.float64).reshape(-1)
    if labels.shape != (n_rois,):
        qc_stat_path = session_dir / "qc_results" / "stat.npy"
        if not qc_stat_path.exists():
            return [None] * n_rois, f"Skipped masks.h5 labels: expected {n_rois} labels, found {labels.size}."
        qc_stat = np.load(qc_stat_path, allow_pickle=True)
        if labels.shape != (len(qc_stat),):
            return [None] * n_rois, (
                f"Skipped masks.h5 labels: expected {n_rois} original Suite2p labels "
                f"or {len(qc_stat)} QC labels, found {labels.size}."
            )
        try:
            mapped_indices = map_qc_to_suite2p_rois(qc_stat, suite2p_stat)
            mapped = np.full(n_rois, np.nan, dtype=np.float64)
            mapped[mapped_indices] = labels
            return _coerce_cell_type_label_values(mapped, "masks.h5 labels"), "masks.h5:labels mapped from qc_results/stat.npy"
        except ValueError as exc:
            return [None] * n_rois, f"Skipped masks.h5 labels: {exc}"
    try:
        return _coerce_cell_type_label_values(labels, "masks.h5 labels"), "masks.h5:labels"
    except ValueError as exc:
        return [None] * n_rois, str(exc)


def _map_legacy_roi_label_indices(
    session_dir: Path,
    good_roi: set[int],
    bad_roi: set[int],
    n_rois: int,
    stat: np.ndarray,
) -> tuple[set[int], set[int]]:
    """Map legacy ROI_label.h5 indices to original Suite2p rows when needed."""

    all_indices = good_roi | bad_roi
    if not all_indices:
        return set(), set()
    if max(all_indices) >= n_rois:
        return {idx for idx in good_roi if idx < n_rois}, {idx for idx in bad_roi if idx < n_rois}
    qc_stat_path = session_dir / "qc_results" / "stat.npy"
    suite2p_stat_path = session_dir / "suite2p" / "plane0" / "stat.npy"
    if not qc_stat_path.exists() or not suite2p_stat_path.exists():
        return good_roi, bad_roi
    try:
        qc_stat = np.load(qc_stat_path, allow_pickle=True)
        suite2p_stat = np.load(suite2p_stat_path, allow_pickle=True)
        if len(qc_stat) == n_rois:
            return good_roi, bad_roi
        if len(stat) == len(suite2p_stat):
            mapping = map_qc_to_suite2p_rois(qc_stat, suite2p_stat)
            return (
                {int(mapping[idx]) for idx in good_roi if 0 <= idx < len(mapping)},
                {int(mapping[idx]) for idx in bad_roi if 0 <= idx < len(mapping)},
            )
    except Exception:
        return good_roi, bad_roi
    return good_roi, bad_roi


def _load_roi_label_h5(session_dir: Path, n_rois: int, stat: np.ndarray) -> tuple[list[int | None], str]:
    """Load legacy ``ROI_label.h5`` good/bad labels for viewer initialization."""

    label_path = session_dir / "ROI_label.h5"
    labels: list[int | None] = [None] * n_rois
    if not label_path.exists():
        return labels, "No ROI_label.h5 loaded."
    good_roi: set[int] = set()
    bad_roi: set[int] = set()
    with h5py.File(label_path, "r") as h5:
        index_space = str(h5.attrs.get("index_space", ""))
        if "good_roi" in h5:
            good_roi.update(int(value) for value in np.asarray(h5["good_roi"]).reshape(-1))
        if "bad_roi" in h5:
            bad_roi.update(int(value) for value in np.asarray(h5["bad_roi"]).reshape(-1))
    if index_space == "suite2p_original":
        mapped_good, mapped_bad = good_roi, bad_roi
    else:
        mapped_good, mapped_bad = _map_legacy_roi_label_indices(session_dir, good_roi, bad_roi, n_rois, stat)
    for roi in mapped_good:
        if 0 <= roi < n_rois:
            labels[roi] = 1
    for roi in mapped_bad:
        if 0 <= roi < n_rois:
            labels[roi] = 0
    return labels, str(label_path)


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    decoded = []
    for value in values.reshape(-1):
        decoded.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return decoded


def _load_roi_model_scores(session_dir: Path, n_rois: int) -> tuple[list[dict[str, Any] | None], str]:
    """Load trained ROI model score predictions indexed by original Suite2p ROI."""

    path = _first_existing(
        [
            session_dir / "roi_model_scores.h5",
            session_dir / "roi_qc_predictions.h5",
            session_dir / "roi_predictions.h5",
        ]
    )
    predictions: list[dict[str, Any] | None] = [None] * n_rois
    if path is None:
        return predictions, "No ROI model score predictions loaded."
    with h5py.File(path, "r") as h5:
        probability = np.asarray(h5["probability"], dtype=np.float64).reshape(-1)
        states = _decode_h5_strings(np.asarray(h5["state"]))
        if "suite2p_roi" in h5:
            roi_indices = np.asarray(h5["suite2p_roi"], dtype=np.int64).reshape(-1)
        elif "roi" in h5:
            roi_indices = np.asarray(h5["roi"], dtype=np.int64).reshape(-1)
        else:
            roi_indices = np.arange(probability.size, dtype=np.int64)
        good_threshold = float(h5.attrs.get("good_threshold", np.nan))
        bad_threshold = float(h5.attrs.get("bad_threshold", np.nan))
        model_path = str(h5.attrs.get("model_path", ""))
        model_target = normalize_qc_target(str(h5.attrs.get("target_structure", "")))
        model_source = str(h5.attrs.get("model_source", ""))
    for roi, prob, state in zip(roi_indices, probability, states):
        roi = int(roi)
        if 0 <= roi < n_rois:
            predictions[roi] = {
                "probability": float(prob),
                "state": state,
                "good_threshold": good_threshold,
                "bad_threshold": bad_threshold,
                "model_path": model_path,
                "model_target_structure": model_target,
                "model_source": model_source,
            }
    return predictions, str(path)


def _initial_labels_from_roi_model_scores(
    roi_model_scores_labels: list[int | None],
    roi_model_scores: list[dict[str, Any] | None],
) -> list[int]:
    """Build the viewer's initial label vector from model/manual QC outputs.

    ``ROI_label.h5`` only stores good/bad sets. When the model prediction file
    is present, use its explicit state so gray predictions initialize as the
    viewer's Unsure label.
    """

    initial = [-1 if label is None else int(label) for label in roi_model_scores_labels]
    for roi, prediction in enumerate(roi_model_scores):
        if not prediction:
            continue
        state = str(prediction.get("state", "")).lower()
        if state == "good":
            initial[roi] = 1
        elif state == "bad":
            initial[roi] = 0
        elif state in {"gray", "grey", "unsure"}:
            initial[roi] = 2
    return initial


def _load_dff(session_dir: Path) -> np.ndarray:
    for path in [session_dir / "dff.h5", session_dir / "qc_results" / "dff.h5"]:
        if path.exists():
            with h5py.File(path, "r") as h5:
                if "dff" not in h5:
                    raise KeyError(f"{path} does not contain dataset 'dff'")
                return np.asarray(h5["dff"], dtype=np.float32)
    raise FileNotFoundError(f"Could not find dff.h5 in {session_dir} or {session_dir / 'qc_results'}")


def _load_oasis_spikes(
    session_dir: Path,
    n_rois: int,
    n_frames: int,
    input_layout: SummaryInputLayout | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    paths = input_layout.spikes_paths if input_layout is not None else (session_dir / "spikes.h5", session_dir / "qc_results" / "spikes.h5")
    for path in paths:
        if not path.exists():
            continue
        with h5py.File(path, "r") as h5:
            if "spikes" not in h5:
                continue
            spikes = np.asarray(h5["spikes"], dtype=np.float32)
            attrs = {key: h5.attrs[key].item() if hasattr(h5.attrs[key], "item") else h5.attrs[key] for key in h5.attrs}
        spikes = spikes[:n_rois, :n_frames]
        if spikes.shape != (n_rois, n_frames):
            padded = np.full((n_rois, n_frames), np.nan, dtype=np.float32)
            padded[: spikes.shape[0], : spikes.shape[1]] = spikes
            spikes = padded
        return spikes, attrs
    return None, {}


def _load_raw_fractional_dff(session_dir: Path, ops: dict[str, Any]) -> np.ndarray:
    fluo_path = session_dir / "qc_results" / "fluo.npy"
    neuropil_path = session_dir / "qc_results" / "neuropil.npy"
    if not fluo_path.exists() or not neuropil_path.exists():
        raise FileNotFoundError("Missing qc_results/fluo.npy or qc_results/neuropil.npy")

    fluo = np.load(fluo_path, allow_pickle=True).astype(np.float32, copy=False)
    neuropil = np.load(neuropil_path, allow_pickle=True).astype(np.float32, copy=False)
    signal = fluo - float(ops.get("neucoeff", 0.7)) * neuropil
    baseline = gaussian_filter(signal, [0.0, 600.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (signal - baseline) / baseline
    dff[~np.isfinite(dff)] = np.nan
    return dff.astype(np.float32, copy=False)


def _load_display_dff(session_dir: Path, ops: dict[str, Any]) -> tuple[np.ndarray, str]:
    try:
        return _load_raw_fractional_dff(session_dir, ops), "raw dF/F"
    except FileNotFoundError:
        return _load_dff(session_dir), "saved dF/F dataset"


def _load_offsets(session_dir: Path, ops: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = session_dir / "move_offset.h5"
    if path.exists():
        with h5py.File(path, "r") as h5:
            if "xoff" in h5 and "yoff" in h5:
                return np.asarray(h5["xoff"]), np.asarray(h5["yoff"])
    xoff = np.asarray(ops.get("xoff", []), dtype=float)
    yoff = np.asarray(ops.get("yoff", []), dtype=float)
    return xoff, yoff


def _load_qc_parameters(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / "qc_results" / "qc_parameters.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pipeline_parameters(session_dir: Path) -> dict[str, Any]:
    for filename in ("processing_pipeline_parameters.json", "preprocessing_pipeline_parameters.json"):
        path = session_dir / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _target_structure(pipeline_parameters: dict[str, Any], qc_parameters: dict[str, Any] | None) -> str:
    target = pipeline_parameters.get("target_structure")
    if target:
        return normalize_qc_target(str(target))
    source = str((qc_parameters or {}).get("source", "")).lower()
    if "cerebellum" in source:
        return "dendrite_relaxed"
    if "dendrite" in source:
        return "dendrite"
    if "neuron" in source:
        return "soma"
    return "all_rois"


def _all_rois_filter() -> dict[str, float | int | None]:
    return {
        "skewMin": None,
        "skewMax": None,
        "maxConnect": None,
        "aspectMin": None,
        "aspectMax": None,
        "compactMin": None,
        "compactMax": None,
        "eventSnrMin": None,
        "andreaPostdocSnrMin": None,
        "roiAreaMin": None,
        "roiAreaMax": None,
        "oasisEventSnrMin": None,
        "oasisRiseTauMin": None,
        "oasisRiseTauMax": None,
        "oasisDecayTauMin": None,
        "oasisDecayTauMax": None,
        "cellTypeFilter": None,
        "roiModelScoreProbabilityMin": None,
        "roiModelScoreProbabilityMax": None,
    }


def _morphology_preset_payload() -> dict[str, dict[str, float | int | None]]:
    presets: dict[str, dict[str, float | int | None]] = {"all_rois": _all_rois_filter()}
    for name, values in QC_PRESETS.items():
        presets[name] = {
            "skewMin": values["range_skew"][0],
            "skewMax": values["range_skew"][1],
            "maxConnect": values["max_connect"],
            "aspectMin": values["range_aspect"][0],
            "aspectMax": values["range_aspect"][1],
            "compactMin": values["range_compact"][0],
            "compactMax": values["range_compact"][1],
            "eventSnrMin": None,
            "andreaPostdocSnrMin": None,
            "roiAreaMin": None,
            "roiAreaMax": None,
            "oasisEventSnrMin": None,
            "oasisRiseTauMin": None,
            "oasisRiseTauMax": None,
            "oasisDecayTauMin": None,
            "oasisDecayTauMax": None,
            "cellTypeFilter": None,
            "roiModelScoreProbabilityMin": None,
            "roiModelScoreProbabilityMax": None,
        }
    return presets


def _load_suite2p_dff(session_dir: Path, ops: dict[str, Any], input_layout: SummaryInputLayout | None = None) -> np.ndarray:
    layout = input_layout or _suite2p_layout(session_dir)
    fluo = np.load(layout.fluo_path, allow_pickle=False).astype(np.float32, copy=False)
    if layout.neuropil_path is not None and layout.neuropil_path.exists():
        neuropil = np.load(layout.neuropil_path, allow_pickle=False).astype(np.float32, copy=False)
    else:
        neuropil = np.zeros_like(fluo, dtype=np.float32)
    signal = fluo - float(ops.get("neucoeff", 0.7)) * neuropil
    baseline = gaussian_filter(signal, [0.0, 600.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (signal - baseline) / baseline
    dff[~np.isfinite(dff)] = np.nan
    return dff.astype(np.float32, copy=False)


def _load_suite2p_fluorescence(
    session_dir: Path,
    input_layout: SummaryInputLayout | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    layout = input_layout or _suite2p_layout(session_dir)
    fluo = np.load(layout.fluo_path, allow_pickle=False, mmap_mode="r")
    neuropil = (
        np.load(layout.neuropil_path, allow_pickle=False, mmap_mode="r")
        if layout.neuropil_path is not None and layout.neuropil_path.exists()
        else None
    )
    return fluo, neuropil


def _dff_metrics_to_jsonable(dff_metrics: list[dict[str, float | int | np.floating | np.integer]]) -> list[dict[str, float | None]]:
    payload: list[dict[str, float | None]] = []
    for metrics in dff_metrics:
        payload.append(
            {
                key: (float(value) if np.isfinite(value) else None)
                for key, value in metrics.items()
            }
        )
    return payload


def _load_matching_sidecar(sidecar_path: Path, shape: tuple[int, int], *, label: str = "sidecar") -> np.ndarray | None:
    if not sidecar_path.exists():
        return None
    try:
        sidecar = np.load(sidecar_path, allow_pickle=False, mmap_mode="r")
    except Exception as error:
        print(f"Existing {label} sidecar could not be read and will be regenerated: {sidecar_path} ({error})")
        return None
    if tuple(sidecar.shape) != shape:
        print(
            f"Existing {label} sidecar shape does not match the selected ROI set and will be regenerated: "
            f"{sidecar_path} has {tuple(sidecar.shape)}, expected {shape}"
        )
        return None
    return sidecar


def _dff_metrics_from_array(
    dff_source: np.ndarray,
    *,
    n_rois: int,
    frame_rate: float,
    chunk_size: int,
) -> list[dict[str, float | None]]:
    metrics: list[dict[str, float | None]] = []
    for start in range(0, n_rois, chunk_size):
        stop = min(n_rois, start + chunk_size)
        metrics.extend(
            _dff_metrics_to_jsonable(
                list(dff_qc_metrics(np.asarray(dff_source[start:stop], dtype=np.float32), frame_rate=frame_rate))
            )
        )
    return metrics


def _build_dff_and_metrics(
    session_dir: Path,
    ops: dict[str, Any],
    input_layout: SummaryInputLayout,
    n_rois: int,
    n_frames: int,
    frame_rate: float,
    storage_mode: str,
    sidecar_path: Path | None,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray | None, list[dict[str, float | None]]]:
    if storage_mode == "embedded":
        dff = _load_suite2p_dff(session_dir, ops, input_layout)
        dff = dff[:n_rois, :n_frames]
        return dff, _dff_metrics_to_jsonable(list(dff_qc_metrics(dff, frame_rate=frame_rate)))

    if sidecar_path is None:
        raise ValueError("sidecar_path is required when storage_mode is file")

    existing_sidecar = _load_matching_sidecar(sidecar_path, (n_rois, n_frames), label="dF/F")
    if existing_sidecar is not None:
        return None, _dff_metrics_from_array(
            existing_sidecar,
            n_rois=n_rois,
            frame_rate=frame_rate,
            chunk_size=chunk_size,
        )

    fluo, neuropil = _load_suite2p_fluorescence(session_dir, input_layout)
    coeff = float(ops.get("neucoeff", 0.7))
    writer = np.lib.format.open_memmap(sidecar_path, mode="w+", dtype=np.float32, shape=(n_rois, n_frames))
    metrics: list[dict[str, float | None]] = []
    for start in range(0, n_rois, chunk_size):
        stop = min(n_rois, start + chunk_size)
        neuropil_chunk = (
            np.asarray(neuropil[start:stop], dtype=np.float32)
            if neuropil is not None
            else np.zeros_like(np.asarray(fluo[start:stop], dtype=np.float32))
        )
        signal = np.asarray(fluo[start:stop], dtype=np.float32) - coeff * neuropil_chunk
        baseline = gaussian_filter(signal, [0.0, 600.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            dff_chunk = (signal - baseline) / baseline
        dff_chunk[~np.isfinite(dff_chunk)] = np.nan
        writer[start:stop] = dff_chunk.astype(np.float32, copy=False)
        metrics.extend(_dff_metrics_to_jsonable(list(dff_qc_metrics(dff_chunk, frame_rate=frame_rate))))
    writer.flush()
    return None, metrics


def _add_roi_area_metrics(
    dff_metrics: list[dict[str, float | None]],
    stat: np.ndarray,
) -> list[dict[str, float | None]]:
    for metrics, entry in zip(dff_metrics, stat):
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        metrics["roi_area"] = float(xpix.size)
    return dff_metrics


def _custom_metric_key(name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip().lower()).strip("_")
    if not key:
        key = "metric"
    if key[0].isdigit():
        key = f"metric_{key}"
    return key


def _source_name(source: Any) -> Any:
    """File name only for a source path, so a shared reviewer does not embed the machine's paths."""
    if isinstance(source, (str, Path)) and ("/" in str(source) or "\\" in str(source)):
        return Path(str(source)).name
    return source


def _json_safe(value: Any) -> Any:
    """Replace NaN and infinities with None, recursively, so the payload is valid JSON.

    Browsers' JSON.parse rejects the bare NaN that json.dumps writes by default, and a
    single one leaves the whole reviewer page without data. Missing values (for example
    the footprint of masks from other software) are represented as null instead.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(_json_safe(payload), separators=(",", ":"), allow_nan=False)


def _jsonable_metric_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _empty_custom_metrics(n_rois: int) -> dict[str, Any]:
    return {"definitions": [], "values": [{} for _ in range(n_rois)], "sources": []}


def _add_custom_metric_column(
    payload: dict[str, Any],
    *,
    name: str,
    values: Any,
    source: str,
    n_rois: int,
    suite2p_indices: np.ndarray,
    roi_index_values: Any | None = None,
) -> None:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"Custom metric '{name}' must be one-dimensional after column selection.")
    metric_values: list[float | None] = [None] * n_rois
    if roi_index_values is not None:
        roi_by_index = {int(value): roi for roi, value in enumerate(np.asarray(suite2p_indices).astype(int))}
        for source_index, raw_value in zip(np.asarray(roi_index_values), array):
            try:
                roi = roi_by_index.get(int(source_index))
            except (TypeError, ValueError):
                roi = None
            if roi is not None:
                metric_values[roi] = _jsonable_metric_value(raw_value)
    else:
        if array.shape[0] != n_rois:
            raise ValueError(f"Custom metric '{name}' has {array.shape[0]} rows; expected {n_rois}.")
        metric_values = [_jsonable_metric_value(value) for value in array]
    if not any(value is not None for value in metric_values):
        return
    existing = {definition["key"] for definition in payload["definitions"]}
    base_key = _custom_metric_key(name)
    key = base_key
    suffix = 2
    while key in existing:
        key = f"{base_key}_{suffix}"
        suffix += 1
    payload["definitions"].append({"key": key, "label": str(name), "source": source, "origin": "command line"})
    for roi, value in enumerate(metric_values):
        payload["values"][roi][key] = value


RETIRED_ROI_INDEX_COLUMNS = ("suite2p_index", "roi", "index")


def _roi_index_header(headers: Sequence[str], source: Path | str) -> str | None:
    """The ``roi_index`` column of an uploaded table, or None when the table has none.

    ``roi_index`` is the only accepted name. A former name (suite2p_index, roi, index) is
    rejected rather than silently read as a metric column.
    """

    lower = {str(header).lower().strip(): header for header in headers}
    if "roi_index" in lower:
        return lower["roi_index"]
    retired = [lower[name] for name in RETIRED_ROI_INDEX_COLUMNS if name in lower]
    if retired:
        raise ValueError(
            f"{source}: the ROI index column must be named 'roi_index'; rename the column '{retired[0]}'."
        )
    return None


def _load_custom_metric_table(path: Path, n_rois: int, suite2p_indices: np.ndarray) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        return _empty_custom_metrics(n_rois)
    headers = list(rows[0].keys())
    index_header = _roi_index_header(headers, path)
    roi_index_values = [row.get(index_header) for row in rows] if index_header else None
    payload = _empty_custom_metrics(n_rois)
    payload["sources"].append(str(path))
    for header in headers:
        if header == index_header:
            continue
        column = [row.get(header, "") for row in rows]
        numeric = [_jsonable_metric_value(value) for value in column]
        if not any(value is not None for value in numeric):
            continue
        _add_custom_metric_column(
            payload,
            name=header,
            values=numeric,
            source=str(path),
            n_rois=n_rois,
            suite2p_indices=suite2p_indices,
            roi_index_values=roi_index_values,
        )
    return payload


def _load_custom_metric_json(path: Path, n_rois: int, suite2p_indices: np.ndarray) -> dict[str, Any]:
    with path.open() as handle:
        raw = json.load(handle)
    payload = _empty_custom_metrics(n_rois)
    payload["sources"].append(str(path))
    if isinstance(raw, dict) and "metrics" in raw:
        raw = raw["metrics"]
    if isinstance(raw, dict):
        for name, values in raw.items():
            if isinstance(values, dict) and "values" in values:
                _roi_index_header(list(values), path)  # rejects a former index key name
                _add_custom_metric_column(
                    payload,
                    name=values.get("label", name),
                    values=values["values"],
                    source=values.get("source", str(path)),
                    n_rois=n_rois,
                    suite2p_indices=suite2p_indices,
                    roi_index_values=values.get("roi_index"),
                )
            elif isinstance(values, list):
                _add_custom_metric_column(
                    payload,
                    name=name,
                    values=values,
                    source=str(path),
                    n_rois=n_rois,
                    suite2p_indices=suite2p_indices,
                )
        return payload
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        headers = sorted({key for row in raw for key in row})
        index_header = _roi_index_header(headers, path)
        roi_index_values = [row.get(index_header) for row in raw] if index_header else None
        for header in headers:
            if header == index_header:
                continue
            values = [row.get(header) for row in raw]
            if not any(_jsonable_metric_value(value) is not None for value in values):
                continue
            _add_custom_metric_column(
                payload,
                name=header,
                values=values,
                source=str(path),
                n_rois=n_rois,
                suite2p_indices=suite2p_indices,
                roi_index_values=roi_index_values,
            )
    return payload


def _load_custom_metric_numpy(path: Path, n_rois: int, suite2p_indices: np.ndarray) -> dict[str, Any]:
    payload = _empty_custom_metrics(n_rois)
    payload["sources"].append(str(path))
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            for name in arrays.files:
                array = np.asarray(arrays[name])
                if array.ndim == 1:
                    _add_custom_metric_column(
                        payload,
                        name=name,
                        values=array,
                        source=str(path),
                        n_rois=n_rois,
                        suite2p_indices=suite2p_indices,
                    )
                elif array.ndim == 2 and array.shape[0] == n_rois:
                    for col in range(array.shape[1]):
                        _add_custom_metric_column(
                            payload,
                            name=f"{name}_{col}",
                            values=array[:, col],
                            source=str(path),
                            n_rois=n_rois,
                            suite2p_indices=suite2p_indices,
                        )
        return payload
    array = np.load(path, allow_pickle=False)
    if array.ndim == 1:
        _add_custom_metric_column(
            payload,
            name=path.stem,
            values=array,
            source=str(path),
            n_rois=n_rois,
            suite2p_indices=suite2p_indices,
        )
    elif array.ndim == 2 and array.shape[0] == n_rois:
        for col in range(array.shape[1]):
            _add_custom_metric_column(
                payload,
                name=f"{path.stem}_{col}",
                values=array[:, col],
                source=str(path),
                n_rois=n_rois,
                suite2p_indices=suite2p_indices,
            )
    else:
        raise ValueError(f"Custom metric array {path} must have shape ({n_rois},) or ({n_rois}, n_metrics).")
    return payload


def _merge_custom_metric_payloads(payloads: list[dict[str, Any]], n_rois: int) -> dict[str, Any]:
    merged = _empty_custom_metrics(n_rois)
    for payload in payloads:
        for source in payload.get("sources", []):
            if source not in merged["sources"]:
                merged["sources"].append(source)
        for definition in payload.get("definitions", []):
            key = definition["key"]
            if any(existing["key"] == key for existing in merged["definitions"]):
                base_key = key
                suffix = 2
                while any(existing["key"] == f"{base_key}_{suffix}" for existing in merged["definitions"]):
                    suffix += 1
                key = f"{base_key}_{suffix}"
                definition = {**definition, "key": key, "original_key": definition["key"]}
            merged["definitions"].append(definition)
            original_key = definition.get("original_key", definition["key"])
            for roi in range(n_rois):
                value = payload["values"][roi].get(original_key, payload["values"][roi].get(definition["key"]))
                if value is not None:
                    merged["values"][roi][key] = value
    return merged


def _load_custom_roi_metrics(
    paths: list[str | os.PathLike[str]] | None,
    n_rois: int,
    suite2p_indices: np.ndarray,
) -> dict[str, Any]:
    if not paths:
        return _empty_custom_metrics(n_rois)
    payloads: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Custom ROI metric file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            payloads.append(_load_custom_metric_table(path, n_rois, suite2p_indices))
        elif suffix == ".json":
            payloads.append(_load_custom_metric_json(path, n_rois, suite2p_indices))
        elif suffix in {".npy", ".npz"}:
            payloads.append(_load_custom_metric_numpy(path, n_rois, suite2p_indices))
        else:
            raise ValueError(f"Unsupported custom ROI metric file extension: {path.suffix}")
    return _merge_custom_metric_payloads(payloads, n_rois)


def _oasis_event_waveform_metrics(
    trace: np.ndarray,
    spikes: np.ndarray,
    threshold: float,
    frame_rate: float,
) -> tuple[float, float, float]:
    event_frames = np.flatnonzero(np.isfinite(spikes) & (spikes > threshold))
    if event_frames.size == 0 or not np.isfinite(frame_rate) or frame_rate <= 0:
        return np.nan, np.nan, np.nan
    pre = max(2, int(round(frame_rate * 0.5)))
    post = max(4, int(round(frame_rate * 2.0)))
    windows = []
    for frame in event_frames:
        frame = int(frame)
        if frame - pre < 0 or frame + post >= trace.size:
            continue
        window = np.asarray(trace[frame - pre : frame + post + 1], dtype=np.float32)
        if np.all(np.isfinite(window)):
            windows.append(window)
    if len(windows) < 3:
        return np.nan, np.nan, np.nan
    event_mean = np.mean(np.stack(windows, axis=0), axis=0)
    baseline = float(np.mean(event_mean[:pre]))
    baseline_sd = float(np.std(event_mean[:pre], ddof=1))
    peak_index = int(np.nanargmax(event_mean[pre:])) + pre
    peak = float(event_mean[peak_index])
    amplitude = peak - baseline
    snr = amplitude / baseline_sd if baseline_sd > 0 and amplitude > 0 else np.nan

    rise_tau = np.nan
    if amplitude > 0 and peak_index > 0:
        low = baseline + 0.1 * amplitude
        high = baseline + 0.9 * amplitude
        rise_segment = event_mean[: peak_index + 1]
        low_hits = np.flatnonzero(rise_segment >= low)
        high_hits = np.flatnonzero(rise_segment >= high)
        if low_hits.size and high_hits.size:
            rise_tau = float(max(0, high_hits[0] - low_hits[0]) / frame_rate)

    decay_tau = np.nan
    if amplitude > 0:
        decay_target = baseline + amplitude / np.e
        decay_segment = event_mean[peak_index:]
        decay_hits = np.flatnonzero(decay_segment <= decay_target)
        if decay_hits.size:
            decay_tau = float(decay_hits[0] / frame_rate)
    return float(snr), rise_tau, decay_tau


def _add_oasis_event_metrics(
    dff_metrics: list[dict[str, float | None]],
    dff: np.ndarray | None,
    dff_sidecar_path: Path | None,
    oasis_spikes: np.ndarray | None,
    default_threshold: float,
    frame_rate: float,
) -> list[dict[str, float | None]]:
    if oasis_spikes is None:
        return dff_metrics
    source = dff
    if source is None and dff_sidecar_path is not None and dff_sidecar_path.exists():
        source = np.load(dff_sidecar_path, mmap_mode="r")
    if source is None:
        return dff_metrics
    n_rois = min(len(dff_metrics), int(source.shape[0]), int(oasis_spikes.shape[0]))
    for roi in range(n_rois):
        threshold = float(default_threshold)
        if not np.isfinite(threshold):
            threshold = 0.0
        dff_metrics[roi]["oasis_optimal_threshold"] = float(threshold) if np.isfinite(threshold) else None
        if np.isfinite(threshold):
            snr, rise_tau, decay_tau = _oasis_event_waveform_metrics(
                np.asarray(source[roi], dtype=np.float32),
                np.asarray(oasis_spikes[roi], dtype=np.float32),
                threshold,
                frame_rate,
            )
        else:
            snr, rise_tau, decay_tau = np.nan, np.nan, np.nan
        dff_metrics[roi]["oasis_event_snr"] = float(snr) if np.isfinite(snr) else None
        dff_metrics[roi]["oasis_rise_tau_seconds"] = float(rise_tau) if np.isfinite(rise_tau) else None
        dff_metrics[roi]["oasis_decay_tau_seconds"] = float(decay_tau) if np.isfinite(decay_tau) else None
    return dff_metrics


def _trace_skew(trace: np.ndarray | None) -> float:
    if trace is None:
        return np.nan
    values = np.asarray(trace, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    centered = values - float(np.mean(values))
    sd = float(np.std(centered))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float(np.mean((centered / sd) ** 3))


# Suite2p's ROI shape statistics, reimplemented from their definitions in
# suite2p.detection.stats (roi_stats, soma_crop, fitMVGaus, median_pix) so that masks
# from other software are scored on the same scale as Suite2p ROIs. Suite2p sessions
# never pass through this code: their stat.npy values are used as they are.
SUITE2P_SHAPE_FIELDS = (
    "med", "npix", "soma_crop", "npix_soma", "mrs", "mrs0", "compact",
    "radius", "aspect_ratio", "footprint", "npix_norm", "npix_norm_no_crop", "overlap",
)
DEFAULT_SHAPE_DIAMETER = (12.0, 12.0)  # Suite2p's roi_stats default, in pixels [dy, dx]
_DISK_REFERENCE_CACHE: dict[tuple[float, float], np.ndarray] = {}


def _shape_diameter(value: Any = None) -> tuple[float, float]:
    """Expected cell diameter [dy, dx] in pixels; a single number means the same in y and x."""

    if value is None:
        return DEFAULT_SHAPE_DIAMETER
    values = np.atleast_1d(np.asarray(value, dtype=np.float64)).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 2)
    if values.size != 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"ROI shape diameter must be one or two positive numbers, got {value!r}.")
    return float(values[0]), float(values[1])


def _center_pixel(ypix: np.ndarray, xpix: np.ndarray) -> list[float]:
    """The ROI pixel nearest the coordinate-wise median (Suite2p's ``med``)."""

    nearest = int(np.argmin((ypix - np.median(ypix)) ** 2 + (xpix - np.median(xpix)) ** 2))
    return [float(ypix[nearest]), float(xpix[nearest])]


def _soma_crop_mask(ypix: np.ndarray, xpix: np.ndarray, lam: np.ndarray, center: Sequence[float]) -> np.ndarray:
    """Pixels inside the soma boundary, as Suite2p's soma crop defines it.

    The weight enclosed within radius r of the center is evaluated at r = 0, 1, 2, ...
    The boundary is the first radius, after the enclosed weight has started growing by
    more than a third of its largest one-pixel increase, at which the growth falls
    back below that level. ROIs of 10 pixels or fewer are not cropped.
    """

    n = ypix.size
    keep_all = np.ones(n, dtype=bool)
    if n <= 10:
        return keep_all
    distance = np.hypot(ypix - center[0], xpix - center[1])
    radii = np.arange(0.0, float(distance.max()), 1.0)
    if radii.size < 2:
        return keep_all
    order = np.argsort(distance, kind="stable")
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(lam, dtype=np.float64)[order])))
    enclosed = cumulative[np.searchsorted(distance[order], radii, side="left")]  # weight strictly inside r
    growth = np.diff(enclosed)
    level = growth.max() / 3.0
    boundary = radii[-1]
    rising = np.flatnonzero(growth > level)
    if rising.size:
        falling = np.flatnonzero(growth[rising[0]:] < level)
        if falling.size:
            boundary = radii[rising[0] + falling[0]]
    crop = distance < boundary
    return crop if crop.any() else keep_all


def _disk_reference_distances(diameter: tuple[float, float]) -> np.ndarray:
    """Sorted diameter-normalized distances of a pixel grid spanning +/-3 diameters (the ideal disk)."""

    if diameter not in _DISK_REFERENCE_CACHE:
        dy, dx = diameter
        gy, gx = np.meshgrid(np.arange(-3 * dy, 3 * dy + 1) / dy, np.arange(-3 * dx, 3 * dx + 1) / dx, indexing="ij")
        _DISK_REFERENCE_CACHE[diameter] = np.sort(np.hypot(gy, gx).reshape(-1))
    return _DISK_REFERENCE_CACHE[diameter]


def _ellipse_radii(ypix: np.ndarray, xpix: np.ndarray, lam: np.ndarray, diameter: tuple[float, float], n_sd: float = 2.0) -> np.ndarray:
    """Major and minor radii (diameter units) of the weighted Gaussian fit, at ``n_sd`` standard deviations."""

    weights = np.asarray(lam, dtype=np.float64)
    positive = weights > 0
    if not positive.any():
        return np.array([np.nan, np.nan])
    coords = np.stack((ypix[positive] / diameter[0], xpix[positive] / diameter[1])).astype(np.float64)
    w = weights[positive] / weights[positive].sum()
    centered = coords - (coords * w).sum(axis=1, keepdims=True)
    covariance = (centered * w) @ centered.T
    variances = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    return np.sort(n_sd * np.sqrt(variances))[::-1]


def suite2p_shape_stats(
    ypix: np.ndarray,
    xpix: np.ndarray,
    lam: np.ndarray | None = None,
    *,
    diameter: Any = None,
    center: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Suite2p's per-ROI shape fields for one ROI.

    ``lam`` defaults to uniform weights, which is what a binary mask provides; Suite2p
    ROIs carry detection weights, so binary masks give approximate values (see docs).
    ``footprint`` (Suite2p's detection spatial scale) cannot be derived from a mask
    and is returned as NaN. ``npix_norm`` and ``overlap`` need all ROIs and are added
    by :func:`complete_suite2p_shape_fields`.
    """

    ypix = np.asarray(ypix, dtype=np.int64).reshape(-1)
    xpix = np.asarray(xpix, dtype=np.int64).reshape(-1)
    weights = np.ones(ypix.size, dtype=np.float32) if lam is None else np.asarray(lam, dtype=np.float32).reshape(-1)
    d = _shape_diameter(diameter)
    if ypix.size == 0:
        return {"med": [np.nan, np.nan], "npix": 0, "soma_crop": np.zeros(0, bool), "npix_soma": 0, "mrs": np.nan,
                "mrs0": np.nan, "compact": np.nan, "radius": np.nan, "aspect_ratio": np.nan, "footprint": np.nan}
    med = list(center) if center is not None else _center_pixel(ypix, xpix)
    crop = _soma_crop_mask(ypix, xpix, weights, med)
    yc, xc, wc = ypix[crop], xpix[crop], weights[crop]
    distance = np.hypot((yc - np.median(yc)) / d[0], (xc - np.median(xc)) / d[1])
    mrs = float(distance.mean())
    mrs0 = float(_disk_reference_distances(d)[: yc.size].mean())
    radii = _ellipse_radii(yc, xc, wc, d)
    return {
        "med": med,
        "npix": int(ypix.size),
        "soma_crop": crop,
        "npix_soma": int(crop.sum()),
        "mrs": mrs,
        "mrs0": mrs0,
        "compact": float(max(1.0, mrs / (1e-10 + mrs0))),
        "radius": float(radii[0] * np.mean(d)),
        "aspect_ratio": float(2.0 * radii[0] / (0.01 + radii[0] + radii[1])),
        "footprint": np.nan,
    }


def complete_suite2p_shape_fields(
    stat: Sequence[dict[str, Any]],
    *,
    image_shape: tuple[int, int] | None = None,
    diameter: Any = None,
) -> list[str]:
    """Fill in any Suite2p shape field an external stat entry lacks; existing values are kept.

    Returns the names of the fields that were computed for at least one ROI.
    ``npix_norm`` divides by the median over all ROIs (Suite2p's convention for
    Cellpose detections, since external ROIs have no detection order), and
    ``overlap`` marks pixels shared with another ROI.
    """

    filled: set[str] = set()
    for entry in stat:
        missing = [field for field in SUITE2P_SHAPE_FIELDS[:10] if field not in entry]
        if not missing:
            continue
        lam = entry.get("lam")
        computed = suite2p_shape_stats(entry["ypix"], entry["xpix"], lam, diameter=diameter,
                                       center=entry.get("med") if "med" in entry else None)
        for field in missing:
            entry[field] = computed[field]
            filled.add(field)
    if stat and any("npix_norm" not in entry or "npix_norm_no_crop" not in entry for entry in stat):
        npix_soma = np.array([float(entry.get("npix_soma", np.size(entry["ypix"]))) for entry in stat])
        npix = np.array([float(np.size(entry["ypix"])) for entry in stat])
        for entry, soma, whole in zip(stat, npix_soma / (np.median(npix_soma) + 1e-10), npix / (np.median(npix) + 1e-10)):
            if "npix_norm" not in entry:
                entry["npix_norm"] = float(soma); filled.add("npix_norm")
            if "npix_norm_no_crop" not in entry:
                entry["npix_norm_no_crop"] = float(whole); filled.add("npix_norm_no_crop")
    if stat and image_shape is not None and any("overlap" not in entry for entry in stat):
        counts = np.zeros(tuple(int(v) for v in image_shape), dtype=np.int32)
        for entry in stat:
            np.add.at(counts, (np.asarray(entry["ypix"], dtype=np.int64), np.asarray(entry["xpix"], dtype=np.int64)), 1)
        for entry in stat:
            if "overlap" not in entry:
                entry["overlap"] = counts[np.asarray(entry["ypix"], dtype=np.int64), np.asarray(entry["xpix"], dtype=np.int64)] > 1
                filled.add("overlap")
    return sorted(filled)


def _suite2p_like_stat_entry(
    ypix: np.ndarray,
    xpix: np.ndarray,
    lam: np.ndarray | None = None,
    trace: np.ndarray | None = None,
    diameter: Any = None,
) -> dict[str, Any]:
    ypix = np.asarray(ypix, dtype=np.int64).reshape(-1)
    xpix = np.asarray(xpix, dtype=np.int64).reshape(-1)
    if ypix.shape != xpix.shape:
        raise ValueError("External ROI ypix and xpix arrays must have the same length.")
    if lam is None:
        lam = np.ones(ypix.shape, dtype=np.float32)
    else:
        lam = np.asarray(lam, dtype=np.float32).reshape(-1)
        if lam.shape != ypix.shape:
            raise ValueError("External ROI lam/weight array must match ypix/xpix length.")
    order = np.lexsort((xpix, ypix))
    ypix = ypix[order]
    xpix = xpix[order]
    lam = lam[order]
    entry = {"ypix": ypix, "xpix": xpix, "lam": lam}
    entry.update(suite2p_shape_stats(ypix, xpix, lam, diameter=diameter))
    entry["skew"] = _trace_skew(trace)
    entry["external_roi"] = True
    return entry


def label_mask_to_suite2p_stat(
    label_mask: np.ndarray, traces: np.ndarray | None = None, diameter: Any = None
) -> np.ndarray:
    """Convert a dense integer ROI-label image into Suite2p-like stat entries.

    Background must be zero. Positive integer labels are converted in ascending
    label order, so row 0 of the returned stat array corresponds to the
    smallest positive label in the mask.
    """

    mask = np.asarray(label_mask)
    if mask.ndim != 2:
        raise ValueError(f"External label mask must be 2-D, got shape {mask.shape}.")
    stat: list[dict[str, Any]] = []
    labels = [int(value) for value in np.unique(mask) if value > 0]
    for row, label in enumerate(labels):
        ypix, xpix = np.where(mask == label)
        trace = None if traces is None or row >= traces.shape[0] else np.asarray(traces[row])
        entry = _suite2p_like_stat_entry(ypix, xpix, trace=trace, diameter=diameter)
        entry["source_label"] = label
        stat.append(entry)
    complete_suite2p_shape_fields(stat, image_shape=mask.shape, diameter=diameter)
    return np.asarray(stat, dtype=object)


def _load_external_metadata(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "external_roi_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_external_image_shape(layout: SummaryInputLayout, mean_green: np.ndarray | None) -> tuple[int, int] | None:
    if mean_green is not None:
        return tuple(map(int, np.asarray(mean_green).shape[:2]))
    metadata = _load_external_metadata(layout.input_dir)
    if "image_shape" in metadata:
        shape = tuple(map(int, metadata["image_shape"]))
        if len(shape) == 2:
            return shape
    for name in ("image_shape.npy", "fov_shape.npy"):
        path = layout.input_dir / name
        if path.exists():
            values = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
            if values.size >= 2:
                return int(values[0]), int(values[1])
    return None


def spatial_matrix_to_suite2p_stat(
    spatial_components: sparse.spmatrix | np.ndarray,
    image_shape: tuple[int, int],
    traces: np.ndarray | None = None,
    *,
    order: str = "F",
    diameter: Any = None,
) -> np.ndarray:
    """Convert an ROI spatial-component matrix into Suite2p-like stat entries.

    The matrix is expected to be shaped ``(pixels, rois)``. CaImAn-style
    matrices commonly use Fortran flattening order, so ``order='F'`` is the
    default for mapping flattened pixel indices back to ``(y, x)`` coordinates.
    """

    n_pixels = int(image_shape[0]) * int(image_shape[1])
    components = spatial_components
    if components.shape[0] != n_pixels and components.shape[1] == n_pixels:
        components = components.T
    if components.shape[0] != n_pixels:
        raise ValueError(
            f"External spatial matrix first dimension must equal Ly*Lx={n_pixels}; "
            f"got shape {components.shape}."
        )
    stat: list[dict[str, Any]] = []
    n_rois = int(components.shape[1])
    for row in range(n_rois):
        column = components[:, row]
        if sparse.issparse(column):
            column = column.tocoo()
            flat = np.asarray(column.row, dtype=np.int64)
            lam = np.asarray(column.data, dtype=np.float32)
        else:
            values = np.asarray(column).reshape(-1)
            flat = np.flatnonzero(values)
            lam = values[flat].astype(np.float32, copy=False)
        ypix, xpix = np.unravel_index(flat, image_shape, order=order)
        trace = None if traces is None or row >= traces.shape[0] else np.asarray(traces[row])
        stat.append(_suite2p_like_stat_entry(ypix, xpix, lam=lam, trace=trace, diameter=diameter))
    complete_suite2p_shape_fields(stat, image_shape=tuple(image_shape), diameter=diameter)
    return np.asarray(stat, dtype=object)


def _load_layout_stat(
    layout: SummaryInputLayout,
    traces: np.ndarray | None = None,
    mean_green: np.ndarray | None = None,
) -> np.ndarray:
    metadata = _load_external_metadata(layout.input_dir) if layout.name == "external_rois" else {}
    diameter = metadata.get("diameter")
    if layout.stat_path.exists():
        stat = np.load(layout.stat_path, allow_pickle=True)
        if layout.name == "external_rois":
            # Suite2p's own sessions are used as they are; an external stat.npy keeps every
            # field it has and gains only the Suite2p shape fields it lacks.
            shape = tuple(mean_green.shape) if mean_green is not None else None
            filled = complete_suite2p_shape_fields(stat, image_shape=shape, diameter=diameter)
            if filled:
                print(f"Computed Suite2p shape fields missing from {layout.stat_path.name}: {', '.join(filled)}")
        return stat
    if layout.roi_mask_path is not None and layout.roi_mask_path.exists():
        return label_mask_to_suite2p_stat(
            np.load(layout.roi_mask_path, allow_pickle=False), traces=traces, diameter=diameter
        )
    if layout.spatial_matrix_path is not None and layout.spatial_matrix_path.exists():
        image_shape = _load_external_image_shape(layout, mean_green)
        if image_shape is None:
            raise FileNotFoundError(
                "External spatial-component matrices require a FOV shape. Supply mean_func.npy, "
                "external_roi_metadata.json with image_shape, or image_shape.npy."
            )
        order = str(metadata.get("flatten_order", metadata.get("order", "F")))
        components = sparse.load_npz(layout.spatial_matrix_path)
        return spatial_matrix_to_suite2p_stat(components, image_shape, traces=traces, order=order, diameter=diameter)
    raise FileNotFoundError(
        f"Could not find ROI geometry for external layout in {layout.input_dir}. "
        "Expected stat.npy, roi_mask.npy, label_mask.npy, masks.npy, or spatial_components.npz."
    )


def _stat_image_shape(stat: np.ndarray) -> tuple[int, int]:
    max_y = 0
    max_x = 0
    for entry in stat:
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        if ypix.size:
            max_y = max(max_y, int(np.max(ypix)))
        if xpix.size:
            max_x = max(max_x, int(np.max(xpix)))
    return max_y + 1, max_x + 1


def _stat_to_mask(stat: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.int32)
    for index, entry in enumerate(stat):
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        mask[ypix, xpix] = index + 1
    return mask


def _robust_limits(image: np.ndarray, lower: float = 1, upper: float = 99.7) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = _robust_limits(image)
    return (np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _array_png_data_uri(array: np.ndarray) -> str:
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _channel_png_data_uri(image: np.ndarray, color: str) -> str:
    return _array_png_data_uri(_channel_rgb(image, color))


def _channel_rgb(image: np.ndarray, color: str) -> np.ndarray:
    uint8 = _normalize_uint8(image)
    rgb = np.zeros((*uint8.shape, 3), dtype=np.uint8)
    if color == "red":
        rgb[..., 0] = uint8
    elif color == "green":
        rgb[..., 1] = uint8
    else:
        rgb[...] = uint8[..., None]
    return rgb


def _mask_data_uri(mask: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    masked = np.ma.masked_where(mask <= 0, mask)
    ax.imshow(masked, cmap="turbo", interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, transparent=True)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _boundary(mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask)
    positive = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary & positive


def _roi_outline_path(xpix: np.ndarray, ypix: np.ndarray) -> str:
    segments: list[str] = []
    if not xpix.size:
        return ""
    pixels = {(int(x), int(y)) for x, y in zip(xpix, ypix)}
    for x, y in sorted(pixels, key=lambda item: (item[1], item[0])):
        if (x, y - 1) not in pixels:
            segments.append(f"M{x} {y}h1")
        if (x, y + 1) not in pixels:
            segments.append(f"M{x} {y + 1}h1")
        if (x - 1, y) not in pixels:
            segments.append(f"M{x} {y}v1")
        if (x + 1, y) not in pixels:
            segments.append(f"M{x + 1} {y}v1")
    return "".join(segments)


def _roi_hit_path(xpix: np.ndarray, ypix: np.ndarray) -> str:
    if not xpix.size:
        return ""
    return "".join(f"M{int(x)} {int(y)}h1v1h-1Z" for x, y in zip(xpix, ypix))


def _roi_table(stat: np.ndarray, mask: np.ndarray, n_rois: int) -> list[dict[str, float | int | str]]:
    rois: list[dict[str, float | int | str]] = []
    for idx in range(n_rois):
        entry = stat[idx]
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        rois.append(
            {
                "roi": idx,
                "path": _roi_outline_path(xpix, ypix),
                "hitPath": _roi_hit_path(xpix, ypix),
                "npix": int(xpix.size),
            }
        )
    return rois


def _label_mask_roi_table(mask: np.ndarray | None) -> list[dict[str, float | int | str]]:
    if mask is None:
        return []
    rois: list[dict[str, float | int | str]] = []
    for idx, label_id in enumerate(np.unique(mask)):
        if label_id <= 0:
            continue
        ypix, xpix = np.where(mask == label_id)
        rois.append(
            {
                "roi": idx,
                "label": int(label_id),
                "path": _roi_outline_path(xpix, ypix),
                "npix": int(xpix.size),
            }
        )
    return rois


def _float32_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes()).decode("ascii")


def _float64_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype("<f8", copy=False)).tobytes()).decode("ascii")


def _plot_channel_image(ax: plt.Axes, image: np.ndarray | None, title: str, color: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    if image is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
        return
    ax.imshow(_channel_rgb(image, color), interpolation="nearest")


def _plot_channel_overlay(ax: plt.Axes, image: np.ndarray | None, mask: np.ndarray, title: str, color: str) -> None:
    _plot_channel_image(ax, image, title, color)
    if image is None:
        return
    outline = np.ma.masked_where(~_boundary(mask), _boundary(mask))
    ax.imshow(outline, cmap="gray", vmin=0, vmax=1, alpha=1.0, interpolation="nearest")


def _build_fov_figure(
    *,
    session_name: str,
    n_rois: int,
    mean_green: np.ndarray,
    max_green: np.ndarray | None,
    mean_red: np.ndarray | None,
    max_red: np.ndarray | None,
    mask: np.ndarray,
) -> plt.Figure:
    if mean_red is None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.7), constrained_layout=True)
        fig.suptitle(
            f"{session_name} FOV summary ({n_rois} ROIs)\n"
            "* Anatomical channel not present for this recording",
            fontsize=14,
        )
        _plot_channel_image(axes[0], max_green, "Functional max projection", "green")
        _plot_channel_image(axes[1], mean_green, "Functional mean projection", "green")
        _plot_channel_overlay(axes[2], mean_green, mask, "Functional mean + white outlines", "green")
        axes[3].set_title("Raw white outlines", fontsize=10)
        axes[3].set_axis_off()
        axes[3].imshow(np.zeros_like(mask), cmap="gray", vmin=0, vmax=1)
        axes[3].imshow(
            np.ma.masked_where(~_boundary(mask), _boundary(mask)),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        return fig

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    fig.suptitle(f"{session_name} FOV summary ({n_rois} ROIs)", fontsize=14)

    _plot_channel_image(fig.add_subplot(gs[0, 0]), max_green, "Functional max projection", "green")
    _plot_channel_image(fig.add_subplot(gs[0, 1]), mean_green, "Functional mean projection", "green")
    _plot_channel_image(fig.add_subplot(gs[0, 2]), max_red, "Anatomical max projection", "red")
    _plot_channel_image(fig.add_subplot(gs[0, 3]), mean_red, "Anatomical mean projection", "red")

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Raw white outlines", fontsize=10)
    ax.set_axis_off()
    ax.imshow(np.zeros_like(mask), cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~_boundary(mask), _boundary(mask)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    _plot_channel_overlay(fig.add_subplot(gs[1, 1]), mean_green, mask, "Functional mean + white outlines", "green")

    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()

    _plot_channel_overlay(fig.add_subplot(gs[1, 3]), mean_red, mask, "Anatomical mean + white outlines", "red")

    return fig


def _build_motion_figure(
    *,
    session_name: str,
    n_rois: int,
    xoff: np.ndarray,
    yoff: np.ndarray,
    frame_rate: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.9])
    fig.suptitle(f"{session_name} motion correction summary ({n_rois} ROIs)", fontsize=14)

    ax = fig.add_subplot(gs[0, :])
    ax.set_title("X shift per frame", fontsize=10)
    if xoff.size:
        frames = np.arange(xoff.size)
        ax.plot(frames, xoff, lw=0.25, color="black")
        ax.axhline(0, color="#9ca3af", lw=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("X shift (pixels)")
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Y shift per frame", fontsize=10)
    if yoff.size:
        frames = np.arange(yoff.size)
        ax.plot(frames, yoff, lw=0.25, color="black")
        ax.axhline(0, color="#9ca3af", lw=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Y shift (pixels)")
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("Offset distributions", fontsize=10)
    if xoff.size and yoff.size:
        values = np.concatenate([np.asarray(xoff, dtype=float), np.asarray(yoff, dtype=float)])
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            bins = np.linspace(lo, hi, 80)
            ax.hist(xoff, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color="#1f77b4", label="x")
            ax.hist(yoff, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color="#d62728", label="y")
        ax.set_xlabel("Shift (pixels)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Shift cumulative distributions", fontsize=10)
    if xoff.size and yoff.size:
        for values, label, color in [(xoff, "x", "#1f77b4"), (yoff, "y", "#d62728")]:
            values = np.asarray(values, dtype=float)
            if values.size:
                sorted_values = np.sort(values)
                ax.plot(sorted_values, np.linspace(0, 1, sorted_values.size), label=label, color=color)
        ax.set_xlabel("Shift (pixels)")
        ax.set_ylabel("Cumulative fraction")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 2])
    ax.set_title("Absolute shift cumulative distributions", fontsize=10)
    if xoff.size and yoff.size:
        for values, label, color in [(np.abs(xoff), "|x|", "#1f77b4"), (np.abs(yoff), "|y|", "#d62728")]:
            values = np.asarray(values, dtype=float)
            if values.size:
                sorted_values = np.sort(values)
                ax.plot(sorted_values, np.linspace(0, 1, sorted_values.size), label=label, color=color)
        ax.axvline(16, color="#6b7280", lw=0.9, ls="--", label="16 px")
        ax.axvline(20, color="#111827", lw=0.9, ls=":", label="20 px")
        ax.set_xlabel("Absolute shift (pixels)")
        ax.set_ylabel("Cumulative fraction")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    return fig


def _write_summary_pdf(
    output_path: Path,
    *,
    session_name: str,
    n_rois: int,
    mean_green: np.ndarray,
    max_green: np.ndarray | None,
    mean_red: np.ndarray | None,
    max_red: np.ndarray | None,
    mask: np.ndarray,
    xoff: np.ndarray,
    yoff: np.ndarray,
    frame_rate: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fov_fig = _build_fov_figure(
        session_name=session_name,
        n_rois=n_rois,
        mean_green=mean_green,
        max_green=max_green,
        mean_red=mean_red,
        max_red=max_red,
        mask=mask,
    )
    motion_fig = _build_motion_figure(
        session_name=session_name,
        n_rois=n_rois,
        xoff=xoff,
        yoff=yoff,
        frame_rate=frame_rate,
    )
    with PdfPages(output_path) as pdf:
        pdf.savefig(fov_fig, dpi=200)
        pdf.savefig(motion_fig, dpi=200)
    fov_fig.savefig(output_path.with_name("preprocessing_fov_summary.png"), dpi=200)
    motion_fig.savefig(output_path.with_name("motion_correction_summary.png"), dpi=200)
    plt.close(fov_fig)
    plt.close(motion_fig)


def _extract_reviewer_payload(html_path: Path) -> dict[str, Any]:
    """Read the embedded data payload back out of a generated (or saved) reviewer HTML."""

    text = html_path.read_text(encoding="utf-8")
    match = re.search(r'<script[^>]*id="payload"[^>]*>(.*?)</script>', text, flags=re.S)
    if match is None:
        raise ValueError(f"No reviewer payload found in {html_path}; is it a reviewer HTML file?")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict) or "nRois" not in payload or "session" not in payload:
        raise ValueError(f"The payload in {html_path} is not a reviewer payload.")
    return payload


def _assign_custom_metric_sections(custom_metrics: dict[str, Any], assignments: list[str] | None) -> None:
    """Apply NAME=SECTION assignments (NAME is a metric's label or key) to custom metric definitions."""

    if not assignments:
        return
    definitions = custom_metrics.get("definitions") or []
    for raw in assignments:
        name, sep, section = str(raw).partition("=")
        name, section = name.strip(), section.strip()
        if not sep or not name or not section:
            raise ValueError(f"--custom-roi-metric-section expects NAME=SECTION, got {raw!r}.")
        if section not in CUSTOM_METRIC_SECTIONS:
            raise ValueError(
                f"Unknown section {section!r} for custom metric {name!r}; choose one of {', '.join(CUSTOM_METRIC_SECTIONS)}."
            )
        matches = [d for d in definitions if d.get("label") == name or d.get("key") == name]
        if not matches:
            available = ", ".join(str(d.get("label")) for d in definitions) or "none loaded"
            raise ValueError(f"No custom ROI metric named {name!r} to assign a section to (available: {available}).")
        for definition in matches:
            definition["section"] = section


def _load_filter_settings(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load filter/sorting settings exported from the reviewer (JSON) or written by hand (JSON or YAML)."""

    settings_path = Path(path).expanduser().resolve()
    if not settings_path.exists():
        raise FileNotFoundError(f"Filter settings file not found: {settings_path}")
    text = settings_path.read_text(encoding="utf-8")
    if settings_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("Reading YAML filter settings needs PyYAML (pip install pyyaml); or use JSON.") from error
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{settings_path}: filter settings must be a mapping.")
    filter_values = raw.get("filter", raw.get("morphology_filter"))
    if filter_values is None:
        meta = {"format", "name", "session", "saved_at", "sort", "target_structure", "preset_name", "filter_set"}
        filter_values = {key: value for key, value in raw.items() if key not in meta}
    if not isinstance(filter_values, dict):
        raise ValueError(f"{settings_path}: 'filter' must be a mapping of threshold names to values.")
    name = str(raw.get("name") or raw.get("preset_name") or raw.get("filter_set") or settings_path.stem).strip()
    sort = raw.get("sort")
    if sort is not None:
        if not isinstance(sort, dict):
            raise ValueError(f"{settings_path}: 'sort' must be a mapping with 'metrics' and 'direction'.")
        metrics = sort.get("metrics", ["original"])
        if isinstance(metrics, str):
            metrics = [metrics]
        direction = str(sort.get("direction", "asc")).lower()
        if direction not in {"asc", "desc"}:
            raise ValueError(f"{settings_path}: sort direction must be 'asc' or 'desc', got {direction!r}.")
        sort = {"metrics": [str(metric) for metric in metrics], "direction": direction}
    return {"name": name, "filter": filter_values, "sort": sort, "source": str(settings_path)}


def _apply_filter_settings(payload: dict[str, Any], settings: dict[str, Any]) -> None:
    """Embed loaded filter/sorting settings as the reviewer's opening filter set and sort order."""

    presets = dict(payload.get("customMorphologyPresets") or {})
    presets[settings["name"]] = settings["filter"]
    payload["customMorphologyPresets"] = presets
    payload["initialFilterSet"] = settings["name"]
    payload["initialMorphologyFilter"] = settings["filter"]
    if settings.get("sort") is not None:
        payload["initialSort"] = settings["sort"]
    payload["filterSettingsSource"] = settings.get("source")


def _rebuild_reviewer_html(
    html_path: Path,
    *,
    input_layout_name: str,
    target_structure: str | None,
    custom_roi_metrics: list[str | os.PathLike[str]] | None,
    custom_roi_metric_sections: list[str] | None,
    filter_settings: dict[str, Any] | None,
) -> None:
    """--html-only: rebuild an existing reviewer from its own embedded data and the current template."""

    payload = _extract_reviewer_payload(html_path)
    if ROI_INDEX_LABEL != _DEFAULT_ROI_INDEX_LABEL or not payload.get("roiIndexLabel"):
        payload["roiIndexLabel"] = ROI_INDEX_LABEL  # only when --roi-index-label was given
    payload["morphologyPresets"] = _morphology_preset_payload()
    layout_name = str(payload.get("inputLayout") or input_layout_name)
    payload["inputLayout"] = layout_name
    if target_structure:
        payload["targetStructure"] = target_structure
    if custom_roi_metrics:
        payload["customRoiMetrics"] = _load_custom_roi_metrics(
            custom_roi_metrics, int(payload["nRois"]), np.asarray(payload["suite2pIndices"], dtype=np.int64)
        )
        for definition in payload["customRoiMetrics"].get("definitions", []):
            definition.setdefault("origin", "command line")
    _assign_custom_metric_sections(payload.get("customRoiMetrics") or {}, custom_roi_metric_sections)
    if filter_settings is not None:
        _apply_filter_settings(payload, filter_settings)
    for mode_key, name_key in (("dffStorageMode", "dffSidecarName"), ("oasisStorageMode", "oasisSidecarName")):
        sidecar = payload.get(name_key)
        if payload.get(mode_key) == "file" and sidecar and not (html_path.parent / sidecar).exists():
            print(f"warning: {sidecar} is not beside {html_path.name}; the rebuilt reviewer will ask for it.")
    _render_reviewer_html(html_path, payload, input_layout_name=layout_name)


def _write_html(
    output_path: Path,
    *,
    input_layout_name: str = "suite2p",
    session_name: str,
    mean_green: np.ndarray,
    mean_red: np.ndarray | None,
    cellpose_mask: np.ndarray | None,
    mask: np.ndarray,
    stat: np.ndarray,
    suite2p_indices: np.ndarray,
    iscell: np.ndarray,
    suite2p_fingerprint: str,
    morphology_metrics: list[dict[str, float | int]],
    cell_type_labels: list[int | None],
    cell_type_label_source: str,
    roi_model_scores_labels: list[int | None],
    roi_model_score_label_source: str,
    roi_model_scores: list[dict[str, Any] | None],
    roi_model_score_source: str,
    custom_roi_metrics: dict[str, Any],
    preset_exclusion_reasons: list[list[str]],
    qc_parameters: dict[str, Any] | None,
    target_structure: str,
    n_rois: int,
    n_frames: int,
    dff_label: str,
    frame_rate: float,
    dff_metrics: list[dict[str, float | None]],
    dff: np.ndarray | None,
    dff_storage_mode: str,
    dff_sidecar_name: str | None,
    estimated_embedded_dff_bytes: int,
    xoff: np.ndarray,
    yoff: np.ndarray,
    oasis_spikes: np.ndarray | None,
    oasis_attrs: dict[str, Any],
    oasis_storage_mode: str,
    oasis_sidecar_name: str | None,
    initialize_labels_from_roi_model_scores: bool = False,
    filter_settings: dict[str, Any] | None = None,
) -> None:
    rois = _roi_table(stat, mask, n_rois)
    red_available = mean_red is not None
    cellpose_rois = _label_mask_roi_table(cellpose_mask)
    cellpose_available = red_available and bool(cellpose_rois)
    image_height, image_width = np.asarray(mean_green).shape[:2]
    xoff = np.asarray(xoff, dtype=np.float32)
    yoff = np.asarray(yoff, dtype=np.float32)
    motion_frame_count = min(xoff.size, yoff.size)
    motion_available = motion_frame_count > 0
    payload = {
        "session": session_name,
        "frameRate": float(frame_rate),
        "nRois": int(n_rois),
        "nFrames": int(n_frames),
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "green": _channel_png_data_uri(mean_green, "green"),
        "red": _channel_png_data_uri(mean_red, "red") if red_available else None,
        "redAvailable": red_available,
        "cellposeMaskAvailable": cellpose_available,
        "cellposeRois": cellpose_rois,
        "mask": _mask_data_uri(mask),
        "rois": rois,
        "suite2pIndices": suite2p_indices.tolist(),
        "suite2pRoiCount": int(iscell.shape[0]),
        "suite2pStatFingerprint": suite2p_fingerprint,
        "iscell": _float64_b64(iscell),
        "morphology": morphology_metrics,
        "cellTypeLabels": cell_type_labels,
        "cellTypeLabelSource": _source_name(cell_type_label_source),
        "roiModelLabels": roi_model_scores_labels,
        "roiModelLabelSource": roi_model_score_label_source,
        "roiModelScores": roi_model_scores,
        "roiModelScoreSource": _source_name(roi_model_score_source),
        "customRoiMetrics": custom_roi_metrics,
        "presetExclusionReasons": preset_exclusion_reasons,
        "qcParameters": qc_parameters,
        "targetStructure": target_structure,
        "initialLabels": (
            _initial_labels_from_roi_model_scores(roi_model_scores_labels, roi_model_scores)
            if initialize_labels_from_roi_model_scores
            else [-1] * int(n_rois)
        ),
        "initialLabelsSource": (
            "ROI model score/ROI_label.h5"
            if initialize_labels_from_roi_model_scores
            else "not labeled by default"
        ),
        "initialMorphologyFilter": None,
        "customMorphologyPresets": None,
        "roiIndexLabel": ROI_INDEX_LABEL,
        "dffMetrics": dff_metrics,
        "dffStorageMode": dff_storage_mode,
        "dffSidecarName": dff_sidecar_name,
        "estimatedEmbeddedDffBytes": int(estimated_embedded_dff_bytes),
        "motionAvailable": motion_available,
        "motionFrameCount": int(motion_frame_count),
        "xoff": _float32_b64(xoff[:motion_frame_count]) if motion_available else None,
        "yoff": _float32_b64(yoff[:motion_frame_count]) if motion_available else None,
        "morphologyPresets": _morphology_preset_payload(),
        "dff": _float32_b64(dff) if dff_storage_mode == "embedded" and dff is not None else None,
        "dffLabel": dff_label,
        "oasisAvailable": oasis_spikes is not None,
        "oasisSpikes": _float32_b64(oasis_spikes)
        if oasis_storage_mode == "embedded" and oasis_spikes is not None
        else None,
        "oasisStorageMode": oasis_storage_mode,
        "oasisSidecarName": oasis_sidecar_name,
        "oasisEventThreshold": float(oasis_attrs.get("event_threshold", 0.05)),
        "oasisAttrs": oasis_attrs,
        "inputLayout": input_layout_name,
    }
    if filter_settings is not None:
        _apply_filter_settings(payload, filter_settings)
    _render_reviewer_html(output_path, payload, input_layout_name=input_layout_name)


def _render_reviewer_html(output_path: Path, payload: dict[str, Any], *, input_layout_name: str = "suite2p") -> None:
    """Write the reviewer page for an already-built payload.

    The full build calls this from ``_write_html``; ``--html-only`` calls it with the
    payload read back out of an existing reviewer, so the page is rebuilt from the
    current template without recomputing any data.
    """

    session_name = str(payload["session"])
    n_rois = int(payload["nRois"])
    red_available = bool(payload.get("redAvailable"))
    fov_grid_class = "with-red" if red_available else "single-channel"
    fov_review_class = "with-red" if red_available else "single-channel"
    oasis_diagnostics_panel = (
        '<div class="panel oasis-diagnostics minimized">'
        '<div class="oasis-diagnostics-header">'
        '<div class="title">Selected ROI inferred spike fit metrics</div>'
        '<div class="oasis-diagnostics-actions">'
        '<button id="oasisDiagnosticsHide" type="button" title="Hide this panel; show it again from the dF/F panel">Hide</button>'
        '<button id="oasisDiagnosticsInfo" class="info-button" type="button" data-info-target="oasisDiagnosticsHelp" aria-expanded="false">Read more</button>'
        '</div>'
        '</div>'
        '<div id="oasisDiagnosticsHelp" class="info-box" hidden>'
        '<p>The blue line averages the selected ROI&apos;s baseline-subtracted dF/F around every frame whose inferred-spike value is above the threshold. The red curve is an exponential decay from its peak with the empirical decay tau, the time the average takes to fall to 1/e of its peak.</p>'
        '<p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#inferred-spike-fit-panel" target="_blank" rel="noopener noreferrer">The fit panel in full</a></p>'
        '</div>'
        '<div class="oasis-diagnostics-body">'
        '<div class="note" id="oasisDiagnosticsSummary">Inferred spikes not loaded for this session.</div>'
        '<canvas id="oasisTransientCanvas"></canvas>'
        '</div>'
        '</div>'
    )
    # FOV titles name the imaging channel; Suite2p sessions also say which role it plays.
    is_suite2p = input_layout_name == "suite2p"
    channel1_title = f"Channel 1 mean{' (functional)' if is_suite2p else ''} ({n_rois} ROIs)"
    channel2_title = f"Channel 2 mean{' (anatomical)' if is_suite2p else ''}"
    red_panel = (
        '<div class="panel"><div class="panel-title-row"><div class="title">' + channel2_title + '</div>'
        '<label class="overlay-mode-control">Overlay'
        '<select id="redOverlayMode">'
        '<option value="functional" selected>Suite2p ROIs</option>'
        '<option value="cellpose">Cellpose ROIs</option>'
        '</select></label></div>'
        '<div class="imagewrap"><img id="red"><svg class="overlay red-overlay" preserveAspectRatio="xMidYMid meet">'
        '</svg></div></div>'
        if red_available
        else ""
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{session_name} preprocessing QC</title>
<style>
html, body {{ overflow-y: auto; }}
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #202124; background: #f6f7f8; }}
.page {{ width: min(1600px, calc(100vw - 72px)); margin: 16px auto 26px; }}
.head {{ display: flex; justify-content: space-between; gap: 14px; align-items: end; margin-bottom: 12px; }}
h1 {{ margin: 0; font-size: 21px; letter-spacing: 0; }}
.grid {{ display: grid; gap: 6px; }}
.review-main {{ margin-top: 8px; display: grid; grid-template-columns: minmax(0, 1fr) clamp(270px, 22vw, 340px); gap: 8px; align-items: start; }}
.review-main.menu-collapsed {{ grid-template-columns: minmax(0, 1fr) 38px; }}
.viewer-column {{ display: flex; flex-direction: column; gap: 6px; }}
.fov-row {{ display: grid; grid-template-columns: 1fr; gap: 6px; align-items: start; }}
.fov-row, .fov-review {{ display: contents; }}
.trace-panel {{ position: relative; }}
.trace-panel .trace-header-row {{ padding-right: 30px; }}
.load-info-icon {{ position: absolute; top: 7px; right: 8px; width: 20px; height: 20px; padding: 0; border-radius: 50%; border: 1.5px solid #98a2b3; background: #fff; color: #475467; font: italic 700 12px Georgia, "Times New Roman", serif; line-height: 1; display: flex; align-items: center; justify-content: center; cursor: pointer; z-index: 3; }}
.load-info-icon:hover, .load-info-icon[aria-expanded="true"] {{ border-color: #175cd3; color: #175cd3; }}
.load-info-icon.warn {{ border-color: #dc6803; color: #b54708; background: #fffaeb; }}
.load-info-rule {{ margin: 6px 0 0; color: #475467; }}
.load-info-rule code {{ font-size: 11px; }}
.load-info-pop {{ position: absolute; top: 32px; right: 8px; max-width: min(460px, calc(100% - 16px)); padding: 8px 10px; background: #fff; border: 1px solid #d0d5dd; border-radius: 6px; box-shadow: 0 4px 12px rgba(16, 24, 40, 0.12); font-size: 12px; line-height: 1.4; color: #344054; z-index: 4; }}
.viewer-column {{ display: grid; grid-template-columns: auto 12px minmax(0, 1fr); grid-template-areas: "fov split side" "trace trace trace"; column-gap: 0; row-gap: 8px; align-items: start; }}
.viewer-column.summary-hidden {{ grid-template-columns: auto 0 34px; }}
#fovGrid {{ grid-area: fov; max-width: 100%; position: relative; }}
.fov-resize-icon {{ position: absolute; top: -1px; right: -15px; width: 18px; height: 18px; padding: 1px; display: flex; align-items: center; justify-content: center; border: 1px solid #d0d5dd; border-radius: 4px; background: #fff; color: #475467; cursor: nesw-resize; touch-action: none; z-index: 2; }}
.fov-resize-icon:hover, .fov-resize-icon.dragging {{ color: #175cd3; border-color: #175cd3; }}
.viewer-column > .panel {{ grid-area: trace; min-width: 0; }}
#summaryColumn {{ grid-area: side; }}
.summary-show-tab {{ grid-area: side; }}
.fov-splitter {{ grid-area: split; align-self: stretch; cursor: col-resize; position: relative; touch-action: none; }}
.fov-splitter::before {{ content: ""; position: absolute; top: 45%; height: 10%; left: 5px; width: 2px; border-radius: 1px; background: #d0d5dd; }}
.fov-splitter {{ margin-top: 24px; margin-bottom: 0; }}
.fov-splitter:hover::before, .fov-splitter.dragging::before {{ background: #475467; }}
.viewer-column.summary-hidden .fov-side, .viewer-column.summary-hidden .fov-splitter {{ display: none; }}
.summary-show-tab {{ display: none; writing-mode: vertical-rl; margin-left: 6px; padding: 10px 4px; font-size: 13px; align-self: start; }}
.viewer-column.summary-hidden .summary-show-tab {{ display: block; }}
.grid.with-red {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.grid.single-channel {{ grid-template-columns: minmax(0, 1fr); }}
.panel {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 7px; padding: 7px; box-sizing: border-box; }}
.title {{ font-size: 13px; font-weight: 700; margin-bottom: 4px; }}
.panel-title-row {{ height: 28px; display: flex; gap: 6px; align-items: center; justify-content: space-between; margin-bottom: 4px; overflow: hidden; }}
.panel-title-row .title {{ margin-bottom: 0; }}
.imagewrap {{ position: relative; width: 100%; aspect-ratio: 1/1; background: #111; overflow: hidden; }}
.imagewrap img, .imagewrap svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.imagewrap img {{ object-fit: contain; image-rendering: pixelated; }}
.imagewrap img, .imagewrap svg {{ transform-origin: 0 0; will-change: transform; }}
.imagewrap {{ cursor: grab; }}
.imagewrap.panning {{ cursor: grabbing; }}
.roi-hit {{ fill: rgba(255,255,255,0); stroke: none; cursor: pointer; pointer-events: all; }}
.roi {{ fill: none; stroke: rgba(255,255,255,.86); stroke-width: .7; vector-effect: non-scaling-stroke; pointer-events: none; }}
.roi-hit:hover + .roi {{ fill: none; stroke: #06b6d4; stroke-width: 1.6; }}
.roi.selected {{ fill: none; stroke: #00e5ff; stroke-width: 3.2; }}
.cellpose-roi {{ fill: none; stroke: #fbbf24; stroke-width: 1.4; vector-effect: non-scaling-stroke; pointer-events: none; }}
.overlay-mode-control {{ display: flex; flex: 0 0 auto; gap: 4px; align-items: center; margin: 0; font-size: 11px; color: #475467; white-space: nowrap; }}
.overlay-mode-control select {{ width: auto; max-width: 128px; min-width: 104px; padding: 2px 4px; font-size: 11px; }}
.controls {{ display: grid; grid-template-columns: 1fr repeat(5, auto); gap: 5px; align-items: center; margin-top: 6px; }}
.label-controls {{ display: flex; flex-direction: column; gap: 4px; align-items: stretch; }}
.label-controls .button-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
.label-controls .nav-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
.show-roi-menu {{ position: relative; display: inline-block; }}
.show-roi-menu > summary {{ list-style: none; cursor: pointer; border: 1px solid #d0d5dd; border-radius: 6px; padding: 4px 7px; background: #fff; }}
.show-roi-menu > summary::-webkit-details-marker {{ display: none; }}
.show-roi-options {{ position: absolute; z-index: 20; top: calc(100% + 4px); left: 0; display: grid; gap: 4px; min-width: 165px; padding: 8px; border: 1px solid #d0d5dd; border-radius: 7px; background: #fff; box-shadow: 0 12px 30px rgba(16,24,40,.18); }}
.show-roi-options label {{ display: flex; gap: 6px; align-items: center; white-space: nowrap; }}
.show-roi-options input {{ width: auto; margin: 0; }}
.roi-details summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
.roi-details #readout {{ margin-top: 5px; color: #475467; font-size: 14px; line-height: 1.35; }}
.save-option {{ display: grid; gap: 3px; justify-items: start; }}
.save-option button, .save-options .docs-link {{ width: fit-content; }}
.info-button {{ padding: 3px 7px; width: fit-content; font-size: 14px; color: #175cd3; font-weight: 600; }}
.nav-button {{ font-size: 15px; font-weight: 700; }}
.side-menu {{ position: fixed; top: 10px; right: max(16px, calc((100vw - 1600px) / 2)); width: clamp(270px, 22vw, 340px); z-index: 15; display: flex; flex-direction: column; gap: 6px; max-height: calc(100vh - 20px); overflow-y: auto; align-self: start; }}
.side-menu-toggle {{ align-self: stretch; font-weight: 700; }}
.control-column {{ display: flex; flex-direction: column; gap: 5px; align-items: stretch; min-height: 0; overflow: visible; font-size: 14px; }}
.review-main.menu-collapsed .control-column {{ display: none; }}
.review-main.menu-collapsed .side-menu {{ width: 38px; overflow: visible; }}
.review-main.menu-collapsed .side-menu-toggle {{ writing-mode: vertical-rl; min-height: 150px; padding: 8px 4px; }}
.menu-card > summary {{ cursor: pointer; font-weight: 700; color: #202124; }}
.menu-card > summary::marker {{ color: #667085; }}
.menu-card-content {{ display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }}
.morphology-card {{ display: flex; flex-direction: column; gap: 4px; }}
.qc-header {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.qc-current {{ color: #667085; font-size: 14px; }}
.sort-card {{ display: flex; flex-direction: column; gap: 4px; padding-top: 4px; border-top: 1px solid #eaecf0; }}
.sort-header {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.sort-current {{ color: #667085; font-size: 14px; }}
.trace-header-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }}
.trace-header-controls {{ display: none; flex: 1 1 100%; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: flex-start; font-size: 14px; }}
.trace-summary-toggle {{ flex: 0 0 auto; white-space: nowrap; margin-left: auto; font-size: 14px; }}
.info-box .info-source {{ color: #475569; font-size: 13px; }}
.trace-header-controls.visible {{ display: flex; }}
.trace-header-controls button {{ flex: 0 0 auto; white-space: nowrap; }}
.filter-controls {{ display: grid; grid-template-columns: repeat(3, minmax(130px, 1fr)); gap: 8px; margin-top: 10px; align-items: end; }}
.filter-controls label {{ font-size: 14px; color: #475467; }}
.filter-controls input {{ display: block; margin-top: 3px; width: 100%; box-sizing: border-box; }}
.metric-controls {{ display: grid; grid-template-columns: repeat(2, minmax(250px, 1fr)); gap: 10px; margin-top: 10px; }}
.metric-control {{ display: grid; grid-template-columns: minmax(150px, 1fr) minmax(130px, .8fr); gap: 8px; align-items: center; padding: 8px; border: 1px solid #eaecf0; border-radius: 7px; background: #fff; }}
.metric-control .threshold-inputs {{ display: grid; gap: 5px; }}
.threshold-default {{ color: #667085; font-size: 14px; }}
.metric-control canvas {{ height: 70px; }}
.metric-histogram-panel {{ min-width: 180px; }}
.metric-histogram-panel summary {{ cursor: pointer; color: #175cd3; font-size: 14px; font-weight: 600; }}
.metric-histogram {{ height: 92px; min-width: 180px; margin-top: 4px; }}
.filter-controls.metric-blocks {{ grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); align-items: start; gap: 12px; }}
.metric-block {{ display: flex; flex-direction: column; gap: 4px; min-width: 0; }}
.metric-block .custom-metric-header {{ grid-column: auto; }}
.metric-block .metric-histogram-panel {{ min-width: 0; }}
.metric-bounds {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; align-items: end; }}
.metric-bounds label {{ font-size: 13px; }}
.histogram-threshold-readout {{ font-size: 12px; color: #344054; margin-top: 2px; line-height: 1.3; }}
.histogram-threshold-readout .excluded {{ color: #b42318; font-weight: 600; }}
.histogram-threshold-readout .kept {{ color: #067647; font-weight: 600; }}
.histogram-probe-readout {{ font-size: 12px; color: #5b21b6; margin-top: 2px; line-height: 1.3; }}
.histogram-stats {{ margin-top: 2px; font-size: 12px; color: #344054; }}
.histogram-stats summary {{ cursor: pointer; color: #175cd3; font-weight: 600; }}
.histogram-stats-table {{ border-collapse: collapse; margin-top: 2px; }}
.histogram-stats-table th {{ text-align: left; font-weight: 400; color: #667085; padding: 1px 12px 1px 0; }}
.histogram-stats-table td {{ text-align: right; font-variant-numeric: tabular-nums; padding: 1px 0; }}
.filter-subsection-title {{ margin-top: 10px; font-size: 15px; font-weight: 700; color: #344054; }}
.custom-metric-header {{ grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 14px; color: #344054; }}
.custom-metric-header label {{ display: flex; gap: 6px; align-items: center; }}
.custom-metric-header select {{ width: auto; }}
.custom-metric-slot:empty {{ display: none; }}
.info-example-label {{ margin: 8px 0 2px; font-size: 13px; color: #475467; }}
.info-example {{ margin: 4px 0 6px; padding: 6px 8px; background: #fff; border: 1px solid #e4e7ec; border-radius: 4px; font-size: 13px; white-space: pre; overflow-x: auto; }}
.range-label-controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-top: 8px; font-size: 14px; color: #475467; }}
.range-label-controls input[type=text], .range-label-controls select {{ display: block; margin-top: 3px; }}
.range-label-controls .inline-check {{ display: flex; gap: 6px; align-items: center; }}
.range-label-controls select {{ min-width: 240px; }}
[hidden] {{ display: none !important; }}
.range-label-controls .inline-check input {{ width: auto; margin: 0; flex: none; }}
.range-label-controls .inline-check {{ align-self: end; padding-bottom: 8px; }}
.range-label-rows {{ display: grid; gap: 6px; margin-top: 8px; }}
.range-label-row {{ display: grid; grid-template-columns: repeat(3, minmax(110px, 1fr)) minmax(90px, auto) auto; gap: 8px; align-items: end; font-size: 14px; color: #475467; }}
.range-label-row input, .range-label-row select {{ display: block; width: 100%; margin-top: 3px; box-sizing: border-box; }}
.range-label-row .range-count {{ color: #667085; font-size: 13px; padding-bottom: 6px; }}
.range-label-actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
.labeler-keys {{ font-size: 14px; }}
.source-heading {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.filter-summary {{ color: #475467; font-size: 15px; }}
.dialog-actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; margin-top: 12px; }}
.dialog-header {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }}
.dialog-title {{ font-size: 18px; font-weight: 700; }}
.dialog-close {{ border: 0; background: transparent; color: #667085; font-size: 20px; line-height: 1; padding: 0 4px; }}
.dialog-close:hover {{ color: #111827; }}
.dialog-section {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid #eaecf0; }}
.dialog-section:first-child {{ margin-top: 0; padding-top: 0; border-top: 0; }}
.dialog-section-title {{ font-size: 16px; font-weight: 700; margin-bottom: 6px; }}
.info-box {{ margin-top: 8px; padding: 8px 10px; background: #f8fafc; border: 1px solid #d0d5dd; border-radius: 6px; color: #475467; font-size: 14px; line-height: 1.35; }}
.info-box p {{ margin: 0 0 6px; }}
.info-box p:last-child {{ margin-bottom: 0; }}
.save-options {{ display: grid; gap: 10px; margin-top: 10px; }}
.bulk-label-controls {{ display: grid; gap: 8px; margin-top: 10px; justify-items: start; }}
.bulk-label-controls select {{ width: auto; min-width: 160px; }}
.bulk-label-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
.bulk-label-row select {{ margin-left: 6px; }}
dialog {{ width: min(980px, calc(100vw - 40px)); border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; box-shadow: 0 24px 60px rgba(16,24,40,.24); }}
dialog::backdrop {{ background: rgba(15,23,42,.38); }}
button, input, select {{ font: inherit; }}
button {{ border: 1px solid #d0d5dd; background: #fff; border-radius: 6px; padding: 4px 7px; cursor: pointer; }}
button.good {{ border-color: #16a34a; color: #166534; }}
button.bad {{ border-color: #dc2626; color: #991b1b; }}
button.unsure {{ border-color: #d97706; color: #92400e; }}
button.unlabeled {{ border-color: #667085; color: #475467; }}
button.active {{ color: #fff; }}
button.good.active {{ background: #16a34a; }}
button.bad.active {{ background: #dc2626; }}
button.unsure.active {{ background: #d97706; }}
button.unlabeled.active {{ background: #667085; }}
input, select {{ border: 1px solid #d0d5dd; border-radius: 6px; padding: 5px 7px; width: 82px; }}
.filter-controls input {{ width: 100%; }}
.filter-controls select {{ width: 100%; }}
.category-filter-options {{ display: grid; gap: 5px; margin-top: 8px; padding: 8px; border: 1px solid #eaecf0; border-radius: 7px; background: #fff; }}
.category-filter-options label {{ display: flex; gap: 6px; align-items: center; color: #475467; font-size: 14px; }}
.category-filter-options input {{ width: auto; margin: 0; }}
.custom-upload-type {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0; }}
.custom-upload-type label {{ display: flex; gap: 6px; align-items: center; }}
.custom-upload-type input, #customRoiUploadFile {{ width: auto; }}
.auto-file-list {{ margin: 6px 0 0; padding-left: 18px; color: #475467; font-size: 13px; }}
.category-label-editor {{ display: grid; gap: 6px; margin-top: 8px; }}
.category-label-editor label {{ display: grid; grid-template-columns: minmax(90px, .35fr) minmax(180px, 1fr); gap: 8px; align-items: center; color: #475467; }}
.category-label-editor input {{ width: 100%; }}
canvas {{ width: 100%; display: block; background: #fff; border: 1px solid #d0d5dd; box-sizing: border-box; }}
#stackCanvas {{ height: 560px; cursor: grab; }}
#stackCanvas.dragging {{ cursor: grabbing; }}
#traceCanvas {{ height: 220px; cursor: grab; }}
#traceCanvas.dragging {{ cursor: grabbing; }}
#motionDriftCanvas {{ height: 470px; }}
#motionDistributionCanvas {{ height: 390px; }}
.oasis-diagnostics {{ width: 100%; max-width: 600px; }}
.oasis-diagnostics-header {{ display: flex; gap: 8px; align-items: center; justify-content: space-between; }}
.oasis-diagnostics-actions {{ display: flex; gap: 6px; align-items: center; }}
.oasis-diagnostics.minimized {{ display: none; }}


.oasis-diagnostics .note {{ margin-bottom: 6px; }}

.fov-side {{ flex: 1 1 0; min-width: 0; overflow-x: clip; display: flex; flex-direction: column; gap: 4px; overflow-y: auto; }}
.summary-header {{ display: flex; gap: 8px; align-items: center; justify-content: space-between; height: 28px; margin-bottom: 0; }}
.summary-header .title {{ margin-bottom: 0; }}
.summary-section {{ display: flex; flex-direction: column; min-height: 0; }}
.summary-grid {{ flex: 1 1 0; min-height: 0; min-width: 0; display: grid; grid-template-columns: 130px minmax(0, 1fr); grid-template-rows: minmax(0, min(43%, 196px)) minmax(0, 1fr); grid-template-areas: "bars rate" "trans trans"; gap: 6px 8px; align-items: stretch; }}
.summary-fraction {{ grid-area: bars; min-height: 0; }}
.head-actions {{ display: flex; gap: 6px; align-items: center; flex-wrap: nowrap; }}
.head-actions .info-button {{ font-size: 11px; padding: 1px 6px; white-space: nowrap; }}
.summary-fraction .canvas-box {{ flex: 1 1 0; width: 130px; min-height: 0; }}
.summary-rate .canvas-box {{ min-height: 0; }}
.summary-rate {{ grid-area: rate; min-width: 0; }}
.summary-transient {{ grid-area: trans; min-width: 0; }}
.transient-box > canvas[hidden] {{ display: none; }}
.canvas-box {{ position: relative; flex: 1 1 0; min-height: 60px; }}
.canvas-box > canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.gbu-plot-head {{ display: flex; flex-wrap: wrap; gap: 2px 6px; align-items: center; justify-content: space-between; margin: 2px 0 3px; min-width: 0; min-height: 21px; }}
.gbu-plot-title {{ white-space: nowrap; }}
.gbu-plot-head .gbu-plot-title {{ margin: 0; }}
.gbu-subtitle {{ font-size: 11px; color: #475467; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.gbu-subtitle .note {{ font-size: 11px; }}
.rate-toggle {{ display: inline-flex; }}
.rate-toggle button {{ font-size: 11px; padding: 1px 6px; border-radius: 0; white-space: nowrap; }}
.rate-toggle button:first-child {{ border-radius: 5px 0 0 5px; }}
.rate-toggle button:last-child {{ border-radius: 0 5px 5px 0; margin-left: -1px; }}
.rate-toggle button[aria-pressed="true"] {{ background: #eef4ff; border-color: #175cd3; color: #175cd3; font-weight: 600; position: relative; }}
.gbu-summary {{ width: 100%; }}
.gbu-plot-title {{ font-size: 12px; font-weight: 600; color: #344054; margin: 8px 0 3px; }}



.gbu-compact-row {{ display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr); gap: 10px; align-items: start; }}
.transient-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; align-items: start; }}
.transient-row.empty {{ display: none; }}
.transient-row .oasis-diagnostics {{ max-width: none; }}
.gbu-transient-panel.minimized {{ display: none; }}
.gbu-transient-panel .title {{ margin-bottom: 6px; }}
.oasis-panel {{ margin-top: 10px; }}
.oasis-toggle {{ display: flex; gap: 6px; align-items: center; font-weight: 700; }}
.oasis-toggle input {{ width: auto; }}
.oasis-threshold-row {{ display: flex; flex: 0 0 auto; gap: 6px; align-items: center; }}
.oasis-threshold-row input[type="range"] {{ width: 150px; padding: 0; }}
.oasis-threshold-row input[type="number"] {{ width: 78px; }}
.plots {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 8px; }}
.trace-sort {{ display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; margin: 6px 0 10px; padding: 8px 10px; background: #f8fafc; border: 1px solid #d0d5dd; border-radius: 6px; }}
.trace-sort label {{ font-size: 14px; color: #475467; }}
.trace-sort select {{ display: block; margin-top: 3px; min-width: 150px; width: auto; }}
.sort-metric-list {{ display: grid; grid-template-columns: minmax(240px, 1fr); gap: 6px; align-items: start; }}
.sort-metric-list label {{ display: flex; gap: 6px; align-items: center; margin: 0; }}
.sort-metric-list input {{ width: auto; margin: 0; }}
.sort-metric-group {{ font-weight: 700; color: #344054; grid-column: 1 / -1; margin-top: 4px; }}
#customSortMetrics {{ display: contents; }}
.sort-actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }}
.metric-formula {{ flex: 1 1 320px; }}
.metric-formula summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
.metric-table-wrap {{ max-height: 75vh; overflow: auto; border: 1px solid #d0d5dd; margin-top: 12px; }}
.metric-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
.metric-table th, .metric-table td {{ border: 1px solid #e5e7eb; padding: 4px 7px; text-align: right; white-space: nowrap; }}
.metric-table th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
.metric-table td:first-child, .metric-table td:nth-child(2), .metric-table td:last-child {{ text-align: left; }}
.metric-fail {{ background: rgba(248, 113, 113, .28); }}
.trace-loader {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 8px; }}
.trace-loader input {{ display: none; }}
.note {{ margin-top: 6px; color: #667085; font-size: 12px; }}
.docs-link {{ display: inline-block; color: #175cd3; font-size: 13px; font-weight: 600; text-decoration: none; }}
.docs-link:hover {{ text-decoration: underline; }}
@media (max-width: 1100px) {{ .review-main, .review-main.menu-collapsed, .fov-row, .fov-review, .grid, .controls {{ grid-template-columns: 1fr; }} .side-menu, .review-main.menu-collapsed .side-menu {{ position: static; width: auto; max-height: none; overflow: visible; }} .review-main.menu-collapsed .side-menu-toggle {{ writing-mode: horizontal-tb; min-height: 0; }} .head {{ display: block; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="head"><h1>{session_name} preprocessing QC</h1></div>
  <dialog id="morphologyDialog">
    <div class="dialog-header">
      <div class="dialog-title">ROI Metric Filters</div>
      <button id="closeMorphologyDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="dialog-section">
      <div class="filter-summary" id="filterSummary"></div>
      <div class="note">The loaded filter set previews pass/fail counts; labels change only when Apply Filters is clicked.</div>
      <div class="filter-controls metric-filter-controls">
        <label>Filter set name <select id="filterPreset"></select></label>
        <button id="resetFilter">Restore selected filter set</button>
        <input id="presetFile" type="file" accept=".json" style="display:none;">
        <button id="importPreset">Import filter set JSON</button>
      </div>
      <div class="source-heading">
        <button class="info-button" type="button" data-info-target="filterSetHelp" aria-expanded="false">Read more: filter sets</button>
        <button class="info-button" type="button" data-info-target="metricDistributionInfo" aria-expanded="false">Distribution Plot Info</button>
      </div>
      <div id="filterSetHelp" class="info-box" hidden>
        <p>A filter set is a named group of thresholds for the metrics below; choosing one fills them in, and any field can then be changed. An ROI passes only if it meets every threshold that is set, and a blank field is not used.</p>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#filter-sets" target="_blank" rel="noopener noreferrer">Filter sets in full</a></p>
      </div>
      <div id="metricDistributionInfo" class="info-box" hidden>
        <p>Each plot is a histogram of that metric across all ROIs, with solid green and red lines at thresholds you have entered and dashed gray lines at the starting value (in parentheses) of fields that are not used. Drag a green, red or gray line to set that threshold, with the counts below the min, between and above the max shown under the plot; drag elsewhere to move the purple line, which reports a value's percentile, and open <strong>Summary statistics</strong> for the quartiles and range.</p>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#metric-distributions" target="_blank" rel="noopener noreferrer">How the lines are placed</a></p>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Cell Identity Labels</span>
        <button class="info-button" type="button" data-info-target="cellTypeMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="cellTypeMetricSources" class="info-box" hidden>
        <p>Cell identity labels are cell-type or compartment labels, read from the session's outputs or uploaded with <strong>Upload Custom ROI Metrics/Cell Identity Labels</strong>. Check identities to keep only ROIs with those labels; with none checked, this filter is not used.</p>
        <p class="info-example-label">CSV (cell_identity_labels.csv)</p>
        <pre class="info-example">roi_index,cell_identity_code,cell_identity_label
0,1,inhibitory
1,-1,excitatory
2,1,inhibitory</pre>
        <p class="info-example-label">JSON (cell_identity_labels.json)</p>
        <pre class="info-example">[
  {{"roi_index": 0, "cell_identity_code": 1, "cell_identity_label": "inhibitory"}},
  {{"roi_index": 1, "cell_identity_code": -1, "cell_identity_label": "excitatory"}}
]</pre>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#cell-identity-labels" target="_blank" rel="noopener noreferrer">Cell identity labels and the roi_index column</a></p>
      </div>
      <div class="category-filter-options" id="cellTypeFilterOptions">
        <span class="note">No cell identity labels loaded.</span>
      </div>
      <div id="modelDerivedFilterSection">
      <div class="filter-subsection-title source-heading">
        <span>Model-Derived Metrics</span>
        <button class="info-button" type="button" data-info-target="modelDerivedMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="modelDerivedMetricSources" class="info-box" hidden>
        <p>ROI model probability is the classifier's p(good ROI), loaded from <code>roi_model_scores.h5</code>; the current model was trained on cerebellar dendrite ROIs only. The scores do not label ROIs unless the reviewer was generated with label initialization; use <strong>Assign Labels to Metric Ranges</strong> below to label from them.</p>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#roi-model-probability" target="_blank" rel="noopener noreferrer">ROI model probability in full</a></p>
      </div>
      <div class="filter-controls" id="modelScoreFilterControls">
        <label>ROI model probability min <span class="threshold-default" id="roiModelScoreProbabilityMinDefault"></span><input id="roiModelScoreProbabilityMin" type="number" min="0" max="1" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="roi_model_score_probability" data-min="roiModelScoreProbabilityMin" aria-label="ROI model probability distribution"></canvas></details>
      </div>
      <div class="filter-controls custom-metric-slot" data-section="model_derived"></div>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Uploaded ROI Metrics</span>
        <button class="info-button" type="button" data-info-target="customMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="customMetricSources" class="info-box" hidden>
        <p>Uploaded ROI metrics are your own per-ROI values, given with <code>--custom-roi-metrics</code> at generation or uploaded with <strong>Upload Custom ROI Metrics/Cell Identity Labels</strong>. Each column other than <code>roi_index</code> becomes a metric, and its <strong>Section</strong> menu moves it to any section of this menu and the Sort menu.</p>
        <p class="info-example-label">CSV (qc_metrics.csv)</p>
        <pre class="info-example">roi_index,roi_area_px,snr
0,112,4.2
1,86,1.8
2,140,3.1</pre>
        <p class="info-example-label">NumPy, command line only</p>
        <pre class="info-example">import numpy as np
# command line only (--custom-roi-metrics): one value per ROI, in original-index order
np.savez("qc_metrics.npz", roi_area_px=areas, snr=snr)</pre>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#uploaded-roi-metrics" target="_blank" rel="noopener noreferrer">Uploaded metrics and their sections</a></p>
      </div>
      <div class="filter-controls">
        <span id="customMetricStatus" class="note"></span>
      </div>
      <div id="customMetricFilterControls" class="filter-controls custom-metric-slot" data-section="uploaded"></div>
      <div class="filter-subsection-title source-heading">
        <span>Morphology Metrics</span>
        <button class="info-button" type="button" data-info-target="morphologyMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="morphologyMetricSources" class="info-box" hidden>
        <p>ROI area counts the mask's pixels, connectivity counts its separate groups of connected pixels, and aspect ratio and compactness are Suite2p's shape statistics, about 1 for a round ROI and higher for elongated or irregular ones. For masks from other software the generator computes them the same way, so binary masks give values close to but not identical with Suite2p's.</p>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#morphology-metrics" target="_blank" rel="noopener noreferrer">Morphology calculations</a> &middot; <a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields" target="_blank" rel="noopener noreferrer">Suite2p stat.npy field definitions</a></p>
      </div>
      <div class="filter-controls">
        <label>Aspect min <span class="threshold-default" id="aspectMinDefault"></span><input id="aspectMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Aspect max <span class="threshold-default" id="aspectMaxDefault"></span><input id="aspectMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="aspect" data-min="aspectMin" data-max="aspectMax" aria-label="Aspect ratio distribution"></canvas></details>
        <label>Compact min <span class="threshold-default" id="compactMinDefault"></span><input id="compactMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Compact max <span class="threshold-default" id="compactMaxDefault"></span><input id="compactMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="compact" data-min="compactMin" data-max="compactMax" aria-label="Compactness distribution"></canvas></details>
        <label>Connectivity max <span class="threshold-default" id="maxConnectDefault"></span><input id="maxConnect" type="number" min="0" step="1" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="connectivity" data-max="maxConnect" aria-label="Connectivity distribution"></canvas></details>
        <label>ROI area min (px) <span class="threshold-default" id="roiAreaMinDefault"></span><input id="roiAreaMin" type="number" min="0" step="1" placeholder="Not Used"></label>
        <label>ROI area max (px) <span class="threshold-default" id="roiAreaMaxDefault"></span><input id="roiAreaMax" type="number" min="0" step="1" placeholder="Not Used"></label>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="roi_area" data-min="roiAreaMin" data-max="roiAreaMax" aria-label="ROI area distribution"></canvas></details>
      </div>
      <div class="filter-controls custom-metric-slot" data-section="morphology"></div>
      <div class="filter-subsection-title source-heading">
        <span>Fluorescence Trace Metrics</span>
        <button class="info-button" type="button" data-info-target="fluorescenceMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="fluorescenceMetricSources" class="info-box" hidden>
        <p>Skew is Suite2p's trace skewness, SNR 95/50 is the 95th minus the 50th percentile of dF/F divided by residual noise, and the large-transient score is a CaImAn-style exceptional-event score. Larger values of all three indicate clearer transients.</p>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#fluorescence-trace-metrics" target="_blank" rel="noopener noreferrer">Trace metrics and their source code</a></p>
      </div>
      <div class="filter-controls">
        <label>Skew min <span class="threshold-default" id="skewMinDefault"></span><input id="skewMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Skew max <span class="threshold-default" id="skewMaxDefault"></span><input id="skewMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="skew" data-min="skewMin" data-max="skewMax" aria-label="Skew distribution"></canvas></details>
        <label>SNR: 95/50 percentile min <span class="threshold-default" id="eventSnrDefault"></span><input id="eventSnrMin" type="number" step="0.01" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="snr_95_50" data-min="eventSnrMin" aria-label="SNR 95/50 percentile distribution"></canvas></details>
        <label>SNR: CaImAn (large-transient score) min <span class="threshold-default" id="andreaPostdocSnrDefault"></span><input id="andreaPostdocSnrMin" type="number" step="0.01" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="andrea_postdoc_snr" data-min="andreaPostdocSnrMin" aria-label="CaImAn SNR distribution"></canvas></details>
      </div>
      <div class="filter-controls custom-metric-slot" data-section="fluorescence"></div>
      <div class="filter-subsection-title source-heading">
        <span>Inferred Spike Metrics</span>
        <button class="info-button" type="button" data-info-target="inferredSpikeMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="inferredSpikeMetricSources" class="info-box" hidden>
        <p>These metrics are computed from dF/F windows around inferred spikes above the threshold used when the reviewer was generated, from any ROI-by-frame inferred-spike array, whatever method produced it. Changing the threshold in the viewer updates only the selected ROI's display.</p>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#inferred-spike-metrics" target="_blank" rel="noopener noreferrer">Inferred spike metrics in full</a></p>
      </div>
      <div class="filter-controls">
        <label>Inferred spike SNR min <span class="threshold-default" id="oasisEventSnrDefault"></span><input id="oasisEventSnrMin" type="number" step="0.01" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="oasis_event_snr" data-min="oasisEventSnrMin" aria-label="Inferred spike SNR distribution"></canvas></details>
        <label>Inferred spike rise tau min (s) <span class="threshold-default" id="oasisRiseTauMinDefault"></span><input id="oasisRiseTauMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Inferred spike rise tau max (s) <span class="threshold-default" id="oasisRiseTauMaxDefault"></span><input id="oasisRiseTauMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="oasis_rise_tau_seconds" data-min="oasisRiseTauMin" data-max="oasisRiseTauMax" aria-label="Inferred spike rise tau distribution"></canvas></details>
        <label>Inferred spike decay tau min (s) <span class="threshold-default" id="oasisDecayTauMinDefault"></span><input id="oasisDecayTauMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Inferred spike decay tau max (s) <span class="threshold-default" id="oasisDecayTauMaxDefault"></span><input id="oasisDecayTauMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel" open><summary>Distribution</summary><canvas class="metric-histogram" data-metric="oasis_decay_tau_seconds" data-min="oasisDecayTauMin" data-max="oasisDecayTauMax" aria-label="Inferred spike decay tau distribution"></canvas></details>
      </div>
      <div class="filter-controls custom-metric-slot" data-section="inferred_spike"></div>
      <div class="filter-subsection-title">Save Filter Sets</div>
      <div class="source-heading">
        <span class="note">Save the current thresholds as a named filter set for reuse, or keep it in a copy of this HTML.</span>
        <button class="info-button" type="button" data-info-target="saveFiltersHelp" aria-expanded="false">Read more</button>
      </div>
      <div id="saveFiltersHelp" class="info-box" hidden>
        <p><strong>Save new filter set</strong> keeps the current thresholds under a name until the page closes, and <strong>Save filter set into HTML</strong> also downloads a copy of the reviewer with it, the labels and the sort order embedded. <strong>Import filter set JSON</strong> loads an exported or hand-written file like the one below, which can also be given at generation with <code>--filter-settings</code>.</p>
        <p class="info-example-label">Filter set JSON (strict_soma.json)</p>
        <pre class="info-example">{{
  "name": "strict soma",
  "filter": {{
    "skewMin": 0.5,
    "aspectMax": 1.5,
    "roiAreaMin": 40,
    "cellTypeFilter": ["1"],
    "customMetrics": {{"snr": {{"min": 2.0, "max": null}}}}
  }},
  "sort": {{"metrics": ["custom:snr"], "direction": "desc"}}
}}</pre>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#saving-and-importing-filter-sets" target="_blank" rel="noopener noreferrer">Rules for filter set files</a></p>
      </div>
      <div class="filter-controls">
        <label>New filter set name <input id="presetName" type="text" placeholder="my filter set"></label>
        <button id="savePreset">Save new filter set</button>
        <button id="savePresetHtml">Save filter set into HTML</button>
      </div>
    </div>
    <div class="dialog-actions">
      <button id="applyFilterToLabels">Apply Filters</button>
      <button id="closeMorphologyDialog" type="button">Close</button>
    </div>
  </dialog>
  <dialog id="sortDialog">
    <div class="dialog-header">
      <div class="dialog-title">Sort ROIs and dF/Fs</div>
      <button id="closeSortDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="source-heading">
      <button class="info-button" type="button" data-info-target="sortMetricSources" aria-expanded="false">Read more: metric sources</button>
    </div>
    <div id="sortMetricSources" class="info-box" hidden>
      <p>These are the same metrics as in the Filter menu, with the same definitions and sources. Cell identity code sorts by the integer code, which in legacy cell-type files is <code>-1</code> excitatory/non-red, <code>0</code> unsure and <code>1</code> inhibitory/red.</p>
      <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#sort-metrics" target="_blank" rel="noopener noreferrer">Sort metrics and their sources</a></p>
    </div>
    <div class="trace-sort">
      <div>
        <div class="note">Check one or more metrics. Multiple checked metrics are converted to 0-1 normalized values and summed with equal weight.</div>
        <div id="sortMetricList" class="sort-metric-list">
          <label><input type="checkbox" name="sortMetric" value="original" checked> <span class="roi-index-label">Original Suite2p index</span></label>
          <div class="sort-metric-group" id="modelDerivedSortGroup">Model-Derived Metrics</div>
          <label id="modelScoreSortOption"><input type="checkbox" name="sortMetric" value="roi_model_score_probability"> ROI model probability</label>
          <div class="custom-sort-slot" data-section="model_derived"></div>
          <div class="sort-metric-group">Morphology Metrics</div>
          <label><input type="checkbox" name="sortMetric" value="aspect"> Aspect ratio</label>
          <label><input type="checkbox" name="sortMetric" value="compact"> Compactness</label>
          <label><input type="checkbox" name="sortMetric" value="roi_area"> ROI area (px)</label>
          <label><input type="checkbox" name="sortMetric" value="connectivity"> Connectivity</label>
          <div class="custom-sort-slot" data-section="morphology"></div>
          <div class="sort-metric-group">Cell Identity Labels</div>
          <label><input type="checkbox" name="sortMetric" value="cell_type_code"> Cell identity code</label>
          <div class="sort-metric-group">Fluorescence Trace Metrics</div>
          <label><input type="checkbox" name="sortMetric" value="skew"> Skew</label>
          <label><input type="checkbox" name="sortMetric" value="snr_95_50"> SNR: 95/50 percentile</label>
          <label><input type="checkbox" name="sortMetric" value="andrea_postdoc_snr"> SNR: CaImAn (large-transient score)</label>
          <div class="custom-sort-slot" data-section="fluorescence"></div>
          <div class="sort-metric-group">Inferred Spike Metrics</div>
          <label><input type="checkbox" name="sortMetric" value="oasis_event_snr"> Inferred spike SNR</label>
          <label><input type="checkbox" name="sortMetric" value="oasis_rise_tau_seconds"> Inferred spike rise tau</label>
          <label><input type="checkbox" name="sortMetric" value="oasis_decay_tau_seconds"> Inferred spike decay tau</label>
          <div class="custom-sort-slot" data-section="inferred_spike"></div>
          <div id="customSortMetrics"></div>
        </div>
      </div>
      <div class="sort-actions">
        <label>Sort order
          <select id="sortDirection">
            <option value="desc">Highest first</option>
            <option value="asc" selected>Lowest first</option>
          </select>
        </label>
        <button id="applySort">Apply sort</button>
      </div>
    </div>
    <div class="dialog-actions"><button id="closeSortDialog" type="button">Close</button></div>
  </dialog>
  <dialog id="saveLabelsDialog">
    <div class="dialog-header">
      <div class="dialog-title">Save Labels</div>
      <div>
        <button id="saveLabelsInfo" class="info-button" type="button" aria-expanded="false" aria-controls="saveLabelsHelp">Read more</button>
        <button id="closeSaveLabelsDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
      </div>
    </div>
    <div id="saveLabelsHelp" class="info-box" hidden>
      <p><strong>Save current state into HTML</strong> downloads a copy that keeps the labels, uploaded metrics, saved filter sets and sort order; the CSV and NPY exports have one row or entry per ROI in original-index order. The labels NPY codes <code>NaN</code> as not labeled, <code>0</code> bad, <code>1</code> good and <code>2</code> unsure.</p>
      <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#saving-labels-and-exports" target="_blank" rel="noopener noreferrer">What each export contains</a></p>
    </div>
    <div class="save-options">
      <div class="save-option">
        <button id="saveHtmlWithLabels">Save current state into HTML</button>
        <span class="note">Use this to save the current state (labels, filter and sorting settings) of the .html to return to after closing the browser.</span>
      </div>
      <div class="save-option">
        <button id="saveMetricSpreadsheet">Export metric spreadsheet CSV</button>
        <span class="note">Exports one row per ROI with current labels, metrics, current filter failures, and reasons.</span>
      </div>
      <div class="save-option">
        <button id="saveManualLabels">Export labels NPY (roi_manual_labels.npy)</button>
        <span class="note">Exports a one-dimensional label array with one entry per ROI, in original-index order.</span>
      </div>
      <div class="save-option">
        <button id="saveCurrentFilters">Export current filter/sorting settings JSON</button>
        <span class="note">Exports the current thresholds, filter set name and sort order. Load them again with <strong>Import filter set JSON</strong> in the Filter menu, or when generating a reviewer with <code>--filter-settings</code> or a YAML config's <code>filter_settings</code> field. <a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#saving-and-importing-filter-sets" target="_blank" rel="noopener noreferrer">How to load filter/sorting settings</a></span>
      </div>
      <a class="docs-link" href="https://najafi-laboratory.github.io/2p_imaging/roi-reviewer-exports/#export-format-and-downstream-use" target="_blank" rel="noopener noreferrer">Output format details</a>
    </div>
    <div class="dialog-actions"><button id="closeSaveLabelsDialog" type="button">Close</button></div>
  </dialog>
  <dialog id="labelAllDialog">
    <div class="dialog-header">
      <div class="dialog-title">Bulk labeling menu</div>
      <button id="closeLabelAllDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="dialog-section">
      <div class="dialog-section-title">Label all visible ROIs as ...</div>
      <div class="note">This applies only to the ROIs currently visible in the reviewer.</div>
      <div class="bulk-label-controls bulk-label-row">
        <label>Label
          <select id="labelAllValue">
            <option value="1">Good</option>
            <option value="0">Bad</option>
            <option value="2">Unsure</option>
            <option value="-1">Not labeled</option>
          </select>
        </label>
        <button id="applyLabelAll" type="button">Apply</button>
      </div>
    </div>
    <div class="dialog-section">
        <div class="dialog-section-title source-heading">
          <span>Assign Labels to Metric Ranges</span>
          <button class="info-button" type="button" data-info-target="rangeLabelHelp" aria-expanded="false">Read more</button>
        </div>
        <div id="rangeLabelHelp" class="info-box" hidden>
          <p>Labels every ROI whose value of the chosen metric falls in one of the ranges, checked from the top with inclusive bounds; ROIs outside every range are left unchanged. <strong>Review labels</strong> sets Good, Bad, Unsure or Not labeled directly, while <strong>New variable</strong> stores each ROI's range number as a new uploaded metric.</p>
          <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#assigning-labels-to-metric-ranges" target="_blank" rel="noopener noreferrer">Range labels in full</a></p>
        </div>
        <div class="range-label-controls">
          <label>Metric <select id="rangeLabelMetric"></select></label>
          <label>Assign <select id="rangeLabelMode"><option value="gbu">Review labels</option><option value="variable">New variable</option></select></label>
          <label id="rangeLabelVariableNameLabel" hidden>Variable name <input id="rangeLabelVariableName" type="text" placeholder="e.g. score_class"></label>
          <label id="rangeLabelOnlyUnlabeledLabel" class="inline-check"><input id="rangeLabelOnlyUnlabeled" type="checkbox"> Only change ROIs that are not labeled</label>
        </div>
        <div id="rangeLabelRows" class="range-label-rows"></div>
        <div class="range-label-actions">
          <button id="addRangeLabelRow" type="button">Add range</button>
          <button id="fillModelRanges" type="button">Use model score thresholds</button>
          <button id="applyRangeLabels" type="button">Assign labels</button>
        </div>
        <div id="rangeLabelStatus" class="note"></div>
    </div>
    <div class="dialog-actions">
      <button id="closeLabelAllDialog" type="button">Close</button>
    </div>
  </dialog>
  <dialog id="customRoiUploadDialog">
    <div class="dialog-header">
      <div class="dialog-title">Upload Custom ROI Metrics/Cell Identity Labels</div>
      <button id="closeCustomRoiUploadDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="dialog-section">
      <div class="dialog-section-title">Currently Loaded Metric &amp; Label Files</div>
      <ul id="autoLoadedFileList" class="auto-file-list"></ul>
    </div>
    <div class="dialog-section">
      <div class="dialog-section-title">Upload File</div>
      <div class="custom-upload-type">
        <label><input type="radio" name="customUploadKind" value="numeric" checked> QC metric</label>
        <label><input type="radio" name="customUploadKind" value="categorical"> Cell identity label</label>
      </div>
      <input id="customRoiUploadFile" type="file" accept=".csv,.tsv,.txt,.json">
      <button id="applyCustomRoiUpload" type="button">Load selected file</button>
      <div id="customRoiUploadStatus" class="note"></div>
      <div class="info-box">
        <p>QC metric files add your own numeric per-ROI values, and cell identity files add an integer code, a name or both per ROI; both accept CSV, TSV, TXT and JSON. <code>roi_index</code> is the ROI's 0-based original index, and without it row <em>k</em> is ROI <em>k</em> and the file needs exactly one row per ROI.</p>
        <p class="info-example-label">CSV (qc_metrics.csv)</p>
        <pre class="info-example">roi_index,roi_area_px,snr
0,112,4.2
1,86,1.8
2,140,3.1</pre>
        <p class="info-example-label">CSV (cell_identity_labels.csv)</p>
        <pre class="info-example">roi_index,cell_identity_code,cell_identity_label
0,1,inhibitory
1,-1,excitatory
2,1,inhibitory</pre>
        <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#uploading-files" target="_blank" rel="noopener noreferrer">Upload formats in full</a></p>
      </div>
    </div>
    <div class="dialog-section" id="categoryLabelEditorSection" hidden>
      <div class="dialog-section-title">Cell identity labels</div>
      <div class="note">Edit the name for each unique cell identity value. These labels are used in the filter menu, sort readouts, and spreadsheet export.</div>
      <div id="categoryLabelEditor" class="category-label-editor"></div>
    </div>
    <div class="dialog-actions">
      <button id="closeCustomRoiUploadDialog" type="button">Close</button>
    </div>
  </dialog>
  <div class="review-main">
    <div class="viewer-column">
      <div class="fov-row">
        <div class="fov-review {fov_review_class}" id="fovReview">
          <div class="grid {fov_grid_class}" id="fovGrid">
            <button type="button" class="fov-resize-icon" id="fovResizeIcon" aria-label="Resize the field of view" title="Drag to resize the field of view; double-click to reset"><svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M9 2h5v5M14 2L8.5 7.5M7 14H2V9M2 14l5.5-5.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
            <div class="panel"><div class="panel-title-row"><div class="title">{channel1_title}</div></div><div class="imagewrap"><img id="green"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
            {red_panel}
          </div>
          <div class="fov-splitter" id="fovSplitter" role="separator" aria-orientation="vertical" title="Drag to resize the field of view; double-click to reset"></div>
          <div class="panel fov-side gbu-summary" id="summaryColumn">
            <div class="summary-header">
              <div class="title">Summary by QC label</div>
              <div class="oasis-diagnostics-actions"><button id="summaryHide" type="button" title="Hide the summary panels; show them again from the tab at the right">Hide</button><button class="info-button" type="button" data-info-target="gbuSummaryHelp" aria-expanded="false">Read more</button></div>
            </div>
            <div id="gbuSummaryHelp" class="info-box" hidden>
                <p>These plots group every ROI by its current label and regroup as you label, with ROIs that a filter labeled Bad counted as Bad. Transients are each ROI's spike-triggered average dF/F (mean &plusmn; s.e.m. across ROIs, windows beyond |dF/F| 20 left out), and firing rates are inferred spikes above threshold per second, drawn as a density per label.</p>
                <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#summary-of-good-bad-and-unsure-rois" target="_blank" rel="noopener noreferrer">How the summary is computed</a></p>
              </div>
            <div class="note" id="gbuSummaryNote"></div>
            <div class="summary-grid">
              <div class="summary-section summary-fraction">
                <div class="gbu-plot-head"><div class="gbu-plot-title">ROIs by label</div></div>
                <div class="canvas-box"><canvas id="gbuFractionCanvas"></canvas></div>
              </div>
                <div class="summary-section summary-rate">
                  <div class="gbu-plot-head">
                    <div class="gbu-plot-title" id="gbuRateTitle">Firing rate</div>
                    <div class="rate-toggle" role="group" aria-label="Firing rate view"><button type="button" data-rate-mode="label" aria-pressed="true">By label</button><button type="button" data-rate-mode="roi" aria-pressed="false">Selected ROI</button></div>
                  </div>
                  <div class="canvas-box"><canvas id="gbuRateCanvas"></canvas></div>
                </div>
                <div class="summary-section summary-transient">
                  <div class="gbu-plot-head">
                    <div class="gbu-plot-title" id="gbuTransientTitle">Average transient</div>
                    <div class="head-actions">
                      <div class="rate-toggle" role="group" aria-label="Average transient view"><button type="button" data-transient-mode="label" aria-pressed="true">By label</button><button type="button" data-transient-mode="roi" aria-pressed="false">Selected ROI</button></div>
                      <button class="info-button" type="button" data-info-target="oasisDiagnosticsHelp" aria-expanded="false">Read more</button>
                    </div>
                  </div>
                  <div id="oasisDiagnosticsHelp" class="info-box" hidden>
                    <p><strong>By label</strong> averages each labeled ROI&apos;s spike-triggered dF/F (mean &plusmn; s.e.m. across ROIs). <strong>Selected ROI</strong> averages that ROI&apos;s baseline-subtracted dF/F around every frame whose inferred-spike value is above the threshold, with an exponential decay fit (red) whose tau is the time to fall to 1/e of the peak.</p>
                    <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#inferred-spike-fit-panel" target="_blank" rel="noopener noreferrer">The fit panel in full</a></p>
                  </div>
                  <div class="gbu-subtitle" id="transientSubtitle">By label, mean &plusmn; s.e.m. across ROIs <span class="note" id="oasisDiagnosticsSummary"></span></div>
                  <div class="canvas-box transient-box">
                    <canvas id="gbuTransientCanvas"></canvas>
                    <canvas id="oasisTransientCanvas" hidden></canvas>
                  </div>
                </div>
            </div>
          </div>
          <button class="summary-show-tab" id="summaryShow" type="button">Show summary by QC label</button>
        </div>
      </div>

      <div class="panel trace-panel">
        <button type="button" class="load-info-icon" id="traceLoadInfo" aria-label="dF/F and inferred spike loading" aria-expanded="false" title="dF/F and inferred spike loading">i</button>
        <div class="trace-header-row">
          <div class="title" id="traceTitle">Selected ROI dF/F</div>
          <div class="trace-header-controls" id="traceOasisControls">
            <label class="oasis-toggle"><input id="showInferredSpikes" type="checkbox" checked> Show Inferred Spikes</label>
            <label>Amplitude threshold</label>
            <div class="oasis-threshold-row">
              <input id="oasisThresholdSlider" type="range" min="0" max="1" step="0.001" value="0.05">
              <input id="oasisThreshold" type="number" min="0" step="0.001" value="0.05">
            </div>
            <button id="resetOasisThreshold" type="button">Reset to ROI default</button>
            <button class="info-button" type="button" data-info-target="traceHelp" aria-expanded="false">Read more</button>
          </div>
        </div>
        
        <div class="trace-loader" id="traceLoader" hidden></div>
        <div id="traceHelp" class="info-box" hidden>
          <p>The inferred-spike array can hold any per-frame spike estimate (amplitudes, probabilities or 0/1 events, from any method), and frames above the threshold are drawn as red dots and used for the average transient and taus. dF/F and inferred spikes are float32 <code>.npy</code> arrays of shape (ROIs, frames) in original-index order that load automatically when this HTML is opened through a web server, named <code>SESSION_dff.npy</code> and <code>SESSION_oasis_spikes.npy</code> beside it.</p>
          <p class="info-source"><span id="spikeSourceText">The inferred-spike file does not record which method produced it.</span></p>
          <p class="info-example-label">Python</p>
          <pre class="info-example">import numpy as np
# one row per ROI in original-index order, one column per frame
np.save("SESSION_dff.npy", np.ascontiguousarray(dff, dtype=np.float32))            # (n_rois, n_frames)
np.save("SESSION_oasis_spikes.npy", np.ascontiguousarray(spikes, dtype=np.float32))  # same shape</pre>
          <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#inferred-spike-threshold" target="_blank" rel="noopener noreferrer">Inferred spikes and the threshold</a> &middot; <a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#loading-dff-and-inferred-spike-files" target="_blank" rel="noopener noreferrer">Loading dF/F and inferred spike files</a></p>
          <p id="spikeDocsSuite2p" hidden><a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/deconvolution/" target="_blank" rel="noopener noreferrer">Suite2p spike deconvolution / OASIS documentation</a></p>
          <p id="spikeDocsOasis" hidden><a class="docs-link" href="https://doi.org/10.1371/journal.pcbi.1005423" target="_blank" rel="noopener noreferrer">Original OASIS deconvolution paper</a></p>
        </div>
        <div class="load-info-pop" id="traceLoadNote" role="status" hidden>
          <div id="traceLoadStatus"></div>
          <p class="load-info-rule">Embedded or separate files: the generator embeds dF/F and inferred spikes in this HTML when ROIs &times; frames is at most {EMBEDDED_MAX_ROI_FRAMES:,} (about {EMBEDDED_DFF_BYTE_LIMIT // (1024 * 1024)} MB of encoded data per array). Larger sessions get <code>SESSION_dff.npy</code> and <code>SESSION_oasis_spikes.npy</code> beside the HTML instead, which load automatically only when the page is opened through a web server (an http:// or https:// address).</p>
          <p class="load-info-rule" id="traceLoadSession"></p>
        </div>
        <canvas id="traceCanvas"></canvas>
        <div class="note">Wheel or drag to zoom/pan time. Double-click to reset.</div>
        <div class="controls">
          <label>Start s <input id="timeStart" type="number" min="0" step="0.001" value="0"></label>
          <label>End s <input id="timeEnd" type="number" min="0" step="0.001" value="0"></label>
          <button id="reset">Reset zoom</button>
        </div>
      </div>
    </div>
    <aside class="side-menu">
      <button id="toggleSideMenu" class="side-menu-toggle" type="button" aria-expanded="true">Hide menu</button>
      <div class="control-column">
          <details class="panel label-controls menu-card" open>
            <summary>Upload (optional)</summary>
            <div class="menu-card-content">
              <button id="openCustomRoiUploadDialog" type="button">Upload Custom ROI Metrics/Cell Identity Labels</button>
            </div>
          </details>
          <details class="panel morphology-card menu-card" open>
            <summary>Filter</summary>
            <div class="menu-card-content">
            <div class="qc-header"><strong>ROI Metric Filters</strong><span id="targetStructureInline" class="qc-current"></span></div>
            <div id="filterSummaryInline" class="filter-summary"></div>
            <button id="openMorphologyDialog" type="button">Filter ROIs</button>
            </div>
          </details>
          <details class="panel sort-card menu-card" open>
            <summary>Sort</summary>
            <div class="menu-card-content">
              <div class="sort-header"><strong>Sorting</strong><span id="sortCurrent" class="sort-current"></span></div>
              <button id="openSortDialog" type="button">Sort filtered ROIs</button>
            </div>
          </details>
          <details class="panel label-controls menu-card" open>
            <summary>Labeler</summary>
            <div class="menu-card-content">
            <span id="labelCounts"></span>
            <strong class="labeler-keys">Left/right arrows select the previous/next visible ROI.</strong>
            <span class="note">Keyboard: G/B/U/N label.</span>
            <strong>Manually label filtered ROIs</strong>
            <label>Selected ROI (<span class="roi-index-label">Suite2p Original Index</span>) <input id="roiInput" type="number" min="0" value="0"> <span id="selectedSortPosition" class="note"></span></label>
            <button class="info-button" type="button" data-info-target="roiIndexHelp" aria-expanded="false">Read more</button>
            <div id="roiIndexHelp" class="info-box" hidden>
              <p><strong><span class="roi-index-label">Suite2p Original Index</span></strong> is the ROI's fixed 0-based identifier: ROI <em>k</em> is row <em>k</em> of every input and exported file and <em>k</em> in every <code>roi_index</code> column. Sort position counts from 1 within the ROIs currently shown and changes when you sort, filter or hide labels; type an original index in the box to jump to that ROI.</p>
              <p><a class="docs-link" href="https://najafi-laboratory.github.io/roi-reviewer-demo/interface-reference/#original-index-and-sort-position" target="_blank" rel="noopener noreferrer">Original index and sort position in full</a></p>
            </div>
            <details class="roi-details">
              <summary id="roiDetailsSummary">Selected ROI Details</summary>
              <div id="readout"></div>
            </details>
            <div class="button-row">
              <button id="markGood" class="good">Good (G)</button>
              <button id="markBad" class="bad">Bad (B)</button>
            </div>
            <div class="button-row">
              <button id="markUnsure" class="unsure">Unsure (U)</button>
              <button id="markUnlabeled" class="unlabeled">Not labeled (N)</button>
            </div>
            <button id="openLabelAllDialog" type="button">Bulk labeling menu</button>
            <div class="nav-row">
              <button id="previousRoi" class="nav-button" title="Previous visible ROI (Left arrow)">&#8592; Previous</button>
              <button id="nextRoi" class="nav-button" title="Next visible ROI (Right arrow)">Next &#8594;</button>
            </div>
            <strong>Show ROIs</strong>
            <details class="show-roi-menu" id="showRoiMenu">
              <summary id="showRoiSummary">All filtered ROIs</summary>
              <div class="show-roi-options">
                <label><input type="checkbox" class="roi-display-checkbox" value="good" checked> Good</label>
                <label><input type="checkbox" class="roi-display-checkbox" value="bad" checked> Bad</label>
                <label><input type="checkbox" class="roi-display-checkbox" value="unsure" checked> Unsure</label>
                <label><input type="checkbox" class="roi-display-checkbox" value="unlabeled" checked> Not labeled</label>
              </div>
            </details>
            </div>
          </details>
          <details class="panel label-controls menu-card" open>
            <summary>View and Export</summary>
            <div class="menu-card-content">
            <button id="openExclusions">View ROI metric spreadsheet</button>
            <button id="openSaveLabelsDialog" type="button">Save Labels</button>
            </div>
          </details>
        </div>
    </aside>
  </div>
  <div class="plots">
    <div class="panel">
      <div class="title" id="stackTitle">Stacked dF/F</div>
      <div class="controls">
        <strong>Stacked trace range</strong>
        <label>First ROI <input id="yStart" type="number" min="0" value="0"></label>
        <label>Last ROI <input id="yEnd" type="number" min="0" value="0"></label>
        <button id="showAllVisibleTraces" type="button">Show all visible traces</button>
      </div>
      <canvas id="stackCanvas"></canvas>
      <div class="note">Wheel to zoom time.</div>
    </div>
    <div class="panel">
      <div class="title">Motion correction shift per frame</div>
      <canvas id="motionDriftCanvas"></canvas>
      <div class="note">Rigid x/y shift per frame from Suite2p motion correction. These plots follow the current trace time window.</div>
      <div class="title" style="margin-top:8px;">Motion correction pixel shifts</div>
      <canvas id="motionDistributionCanvas"></canvas>
    </div>
  </div>
</div>
<script id="payload" type="application/json">{_payload_json(payload)}</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("payload").textContent);
if (!data.customRoiMetrics || typeof data.customRoiMetrics !== "object") {{
  data.customRoiMetrics = {{definitions: [], values: Array.from({{length: data.nRois}}, () => ({{}})), sources: []}};
}}
if (!Array.isArray(data.customRoiMetrics.definitions)) data.customRoiMetrics.definitions = [];
if (!Array.isArray(data.customRoiMetrics.values) || data.customRoiMetrics.values.length !== data.nRois) {{
  data.customRoiMetrics.values = Array.from({{length: data.nRois}}, (_, roi) => data.customRoiMetrics.values?.[roi] || {{}});
}}
if (!Array.isArray(data.customRoiMetrics.sources)) data.customRoiMetrics.sources = [];
data.categoryLabelMap = data.categoryLabelMap || {{"1": "inhibitory/red", "-1": "excitatory/non-red", "0": "unsure"}};
const ROI_INDEX_LABEL = data.roiIndexLabel || "Suite2p Original Index";
const ROI_INDEX_LABEL_LOWER = ROI_INDEX_LABEL.toLowerCase();
document.querySelectorAll(".roi-index-label").forEach(el => {{ el.textContent = ROI_INDEX_LABEL; }});
document.getElementById("green").src = data.green;
const redImage = document.getElementById("red");
if (redImage && data.redAvailable) redImage.src = data.red;
document.querySelectorAll(".imagewrap img").forEach(img => img.addEventListener("load", syncControlColumnHeight));
document.querySelectorAll(".overlay").forEach(svg => svg.setAttribute("viewBox", `0 0 ${{data.imageWidth}} ${{data.imageHeight}}`));
function targetStructureLabel(name) {{
  if (name === "all_rois") return "all_rois (no thresholds)";
  if (name === "neuron") return "soma";
  if (name === "cerebellum_lax") return "dendrite_relaxed";
  return name;
}}
document.getElementById("targetStructureInline").textContent = "";
document.getElementById("roiInput").max = Math.max(...data.suite2pIndices);
const sessionDurationSec = (data.nFrames - 1) / data.frameRate;
document.getElementById("timeStart").max = sessionDurationSec.toFixed(3);
document.getElementById("timeEnd").max = sessionDurationSec.toFixed(3);
document.getElementById("timeEnd").value = sessionDurationSec.toFixed(3);

function b64f32(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}
function b64f64(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float64Array(bytes.buffer);
}}
const motionX = data.motionAvailable && data.xoff ? b64f32(data.xoff) : null;
const motionY = data.motionAvailable && data.yoff ? b64f32(data.yoff) : null;
function parseNpy(arrayBuffer) {{
  const headerBytes = new Uint8Array(arrayBuffer);
  if (headerBytes.length < 10 || headerBytes[0] !== 0x93 || headerBytes[1] !== 0x4e || headerBytes[2] !== 0x55 || headerBytes[3] !== 0x4d || headerBytes[4] !== 0x50 || headerBytes[5] !== 0x59) {{
    throw new Error("Not a NumPy .npy file");
  }}
  const major = headerBytes[6];
  let offset = 0;
  if (major === 1) {{
    offset = 10 + new DataView(arrayBuffer).getUint16(8, true);
  }} else if (major === 2 || major === 3) {{
    offset = 12 + new DataView(arrayBuffer).getUint32(8, true);
  }} else {{
    throw new Error(`Unsupported .npy version: ${{major}}`);
  }}
  const headerText = new TextDecoder("latin1").decode(headerBytes.subarray(major === 1 ? 10 : 12, offset)).trim();
  const descrMatch = /'descr':\\s*'([^']+)'/.exec(headerText);
  const shapeMatch = /'shape':\\s*\\(([^\\)]*)\\)/.exec(headerText);
  const fortranMatch = /'fortran_order':\\s*(True|False)/.exec(headerText);
  if (!descrMatch || !shapeMatch || !fortranMatch) throw new Error("Could not parse .npy header");
  if (fortranMatch[1] !== "False") throw new Error("Fortran-ordered .npy files are not supported");
  const shape = shapeMatch[1].split(",").map(part => Number(part.trim())).filter(Number.isFinite);
  const descr = descrMatch[1];
  const dataOffset = offset;
  if (descr === "<f4" || descr === "|f4") {{
    return {{array: new Float32Array(arrayBuffer, dataOffset), shape}};
  }}
  if (descr === "<f8" || descr === "|f8") {{
    const source = new Float64Array(arrayBuffer, dataOffset);
    const target = new Float32Array(source.length);
    for (let i = 0; i < source.length; i++) target[i] = source[i];
    return {{array: target, shape}};
  }}
  throw new Error(`Unsupported .npy dtype: ${{descr}}`);
}}
let dff = null;
let oasisSpikes = null;
let oasisEventThreshold = Number(data.oasisEventThreshold ?? 0.05);
let showInferredSpikes = true;
let oasisSpikeMin = 0;
let oasisSpikeMax = 1;
let oasisDiagnosticsCache = new Map();
function configureOasisThresholdControls() {{
  const sp = spikeTrace(selected);
  if (sp) {{
    let minValue = Infinity;
    let maxValue = -Infinity;
    for (let i = 0; i < sp.length; i++) {{
      const value = sp[i];
      if (Number.isFinite(value)) {{
        minValue = Math.min(minValue, value);
        maxValue = Math.max(maxValue, value);
      }}
    }}
    if (!Number.isFinite(minValue)) minValue = 0;
    if (!Number.isFinite(maxValue) || maxValue <= minValue) {{
      maxValue = minValue + 0.001;
    }}
    oasisSpikeMin = minValue;
    oasisSpikeMax = Math.max(maxValue, oasisEventThreshold, minValue + 0.001);
  }}
  const slider = document.getElementById("oasisThresholdSlider");
  const box = document.getElementById("oasisThreshold");
  slider.min = oasisSpikeMin.toPrecision(6);
  slider.max = oasisSpikeMax.toPrecision(6);
  slider.value = oasisEventThreshold;
  box.min = oasisSpikeMin.toPrecision(6);
  box.max = oasisSpikeMax.toPrecision(6);
  box.value = oasisEventThreshold;
}}
function setOasisThreshold(value, redraw = true) {{
  const fallback = Number(data.oasisEventThreshold ?? 0.05);
  const next = Number.isFinite(value) ? value : fallback;
  oasisEventThreshold = Math.min(Math.max(next, oasisSpikeMin), oasisSpikeMax);
  configureOasisThresholdControls();
  if (redraw) drawSelectedTraceAndOasis();
}}
function selectedOasisDefaultThreshold() {{
  const value = dffMetric(selected, "oasis_optimal_threshold");
  return Number.isFinite(Number(value)) ? Number(value) : Number(data.oasisEventThreshold ?? 0.05);
}}
function setDffFromArrayBuffer(arrayBuffer, silent = false) {{
  const parsed = parseNpy(arrayBuffer);
  dff = parsed.array;
  if (parsed.shape.length >= 2) {{
    data.nRois = parsed.shape[0];
    data.nFrames = parsed.shape[1];
  }}
  oasisDiagnosticsCache.clear();
  invalidateGbuSummary();
  if (!silent) {{
    document.getElementById("traceLoadStatus").textContent = `Loaded dF/F file with ${{data.nRois}} ROIs x ${{data.nFrames}} frames.`;
  }} else {{
    updateSidecarLoadNote();
  }}
  draw();
}}
// Sidecars load by fetch() relative to this page, which browsers allow over http(s) but not
// for a page opened from disk (file://). The notes below say which case applies.
const openedFromDisk = window.location.protocol === "file:";
const SIDECAR_DISK_ADVICE = "This page was opened from disk (file://), and browsers do not let a page opened that way read files beside it. Open this page through a web server (an http:// or https:// address, for example python -m http.server run in this folder) to load them automatically.";
function setLoadInfoState(missing) {{
  const button = document.getElementById("traceLoadInfo");
  if (button) button.classList.toggle("warn", !!missing);
}}
function toggleLoadInfo(force) {{
  const button = document.getElementById("traceLoadInfo"), pop = document.getElementById("traceLoadNote");
  if (!button || !pop) return;
  const open = force === undefined ? pop.hidden : force;
  pop.hidden = !open;
  button.setAttribute("aria-expanded", String(!pop.hidden));
}}
function updateSidecarLoadNote() {{
  const note = document.getElementById("traceLoadStatus");
  const parts = [];
  if (data.dffStorageMode === "file") {{
    parts.push(dff ? "dF/F loaded" : "dF/F not loaded");
  }}
  if (data.oasisAvailable && data.oasisStorageMode === "file") {{
    parts.push(oasisSpikes ? "inferred spikes loaded" : "inferred spikes not loaded");
  }}
  if (parts.length) {{
    const missing = (data.dffStorageMode === "file" && !dff)
      || (data.oasisAvailable && data.oasisStorageMode === "file" && !oasisSpikes);
    const advice = !missing ? ""
      : openedFromDisk ? ` ${{SIDECAR_DISK_ADVICE}}`
      : ` They were not found beside this page: check that the .npy files are in the same folder as this HTML, named after the session (SESSION_dff.npy and SESSION_oasis_spikes.npy).`;
    const text = `${{parts.join("; ")}}.${{advice}}`;
    note.textContent = text.startsWith("dF/F") ? text : text.charAt(0).toUpperCase() + text.slice(1);
    setLoadInfoState(missing);
  }}
}}
function renderAutoLoadedFileList() {{
  // Lists only ROI metric and cell identity files given on the command line (--custom-roi-metrics)
  // or uploaded through this menu; traces, spikes and automatically detected session files are not listed.
  const list = document.getElementById("autoLoadedFileList");
  if (!list) return;
  const items = [];
  const originText = origin => origin === "upload" ? "uploaded in this menu" : "command line (--custom-roi-metrics)";
  for (const definition of customMetricDefinitions()) {{
    if (definition.origin !== "upload" && definition.origin !== "command line") continue;
    items.push(`Custom ROI metric "${{definition.label}}": ${{definition.source || "unnamed file"}} (${{originText(definition.origin)}})`);
  }}
  if (data.cellTypeLabelOrigin === "upload" && data.cellTypeLabelSource) items.push(`Cell identity labels: ${{data.cellTypeLabelSource}} (${{originText("upload")}})`);
  list.textContent = "";
  if (!items.length) {{
    const item = document.createElement("li");
    item.textContent = "None yet. Custom ROI metrics given with --custom-roi-metrics and files uploaded here are listed.";
    list.appendChild(item);
    return;
  }}
  for (const text of items) {{
    const item = document.createElement("li");
    item.textContent = text;
    list.appendChild(item);
  }}
}}
var sidecarsPending = 0;
async function loadSidecarByRelativeName(sidecarName, setter, label) {{
  if (!sidecarName) return false;
  sidecarsPending += 1;
  try {{
    const response = await fetch(sidecarName);
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const arrayBuffer = await response.arrayBuffer();
    setter(arrayBuffer, true);
    return true;
  }} catch (error) {{
    console.warn(`Could not auto-load ${{label}} sidecar ${{sidecarName}}:`, error);
    updateSidecarLoadNote();
    return false;
  }} finally {{
    sidecarsPending -= 1;
    if (typeof scheduleGbuSummaryDraw === "function") scheduleGbuSummaryDraw();
  }}
}}
if (data.dffStorageMode === "embedded" && data.dff) {{
  dff = b64f32(data.dff);
}}
{{
  // dF/F and inferred spikes come from the session (embedded, or .npy sidecars beside the HTML);
  // there are no file pickers, so a sidecar session must be opened through a web server.
  const note = document.getElementById("traceLoadStatus");
  const sessionLine = document.getElementById("traceLoadSession");
  if (sessionLine) {{
    const values = data.nRois * data.nFrames;
    const mode = data.dffStorageMode === "file" ? "separate .npy files" : "embedded in this HTML";
    sessionLine.textContent = `This session: ${{data.nRois.toLocaleString()}} ROIs × ${{data.nFrames.toLocaleString()}} frames = ${{values.toLocaleString()}}, so its arrays are ${{mode}}.`;
  }}
  const usesSidecars = data.dffStorageMode === "file" || data.oasisStorageMode === "file";
  note.textContent = usesSidecars
    ? (openedFromDisk
        ? `This session keeps its large arrays in .npy files beside this HTML. ${{SIDECAR_DISK_ADVICE}}`
        : `This session keeps its large arrays in .npy files beside this HTML. Loading them from this folder.`)
    : data.oasisAvailable
      ? ``
      : `No inferred spikes were found for this session, so spike overlays and fit metrics are not shown.`;
  if (!note.textContent) note.textContent = "dF/F and inferred spikes are embedded in this page.";
  setLoadInfoState(usesSidecars && openedFromDisk);
  document.getElementById("traceLoadInfo")?.addEventListener("click", event => {{ event.stopPropagation(); toggleLoadInfo(); }});
  document.addEventListener("click", event => {{ if (!event.target.closest("#traceLoadNote, #traceLoadInfo")) toggleLoadInfo(false); }});
  document.addEventListener("keydown", event => {{ if (event.key === "Escape") toggleLoadInfo(false); }});
}}
// The Suite2p and OASIS links are shown only when the inferred-spike file says it came from them
// (spikes.h5 attribute `method`, e.g. suite2p.extraction.dcnv.oasis). A replacement array loaded
// with the button carries no such record, so the links are hidden for it.
function updateSpikeSourceLinks() {{
  const uploaded = data.oasisStorageMode === "uploaded";
  const recorded = uploaded ? "" : String((data.oasisAttrs || {{}}).method || "").trim();
  const method = recorded.toLowerCase();
  const viaSuite2p = method.includes("suite2p"), viaOasis = method.includes("oasis");
  document.getElementById("spikeDocsSuite2p").hidden = !viaSuite2p;
  document.getElementById("spikeDocsOasis").hidden = !viaOasis;
  document.getElementById("spikeSourceText").textContent = uploaded
    ? "This array was loaded with the Load inferred spikes file button, so its source is not recorded."
    : viaSuite2p && viaOasis ? `This session's array was produced with Suite2p's OASIS deconvolution (recorded in the inferred-spike file as ${{recorded}}).`
    : viaOasis ? `This session's array was produced with OASIS deconvolution (recorded in the inferred-spike file as ${{recorded}}).`
    : recorded ? `The inferred-spike file records its method as ${{recorded}}.`
    : "The inferred-spike file does not record which method produced it.";
}}
updateSpikeSourceLinks();
function setOasisFromArrayBuffer(arrayBuffer, silent = false) {{
  const parsed = parseNpy(arrayBuffer);
  oasisSpikes = parsed.array;
  data.oasisAvailable = true;
  setOasisDiagnosticsVisible(true);
  data.oasisStorageMode = data.oasisStorageMode === "file" ? data.oasisStorageMode : "uploaded";
  updateSpikeSourceLinks();
  if (parsed.shape.length >= 2) {{
    data.nRois = parsed.shape[0];
    data.nFrames = parsed.shape[1];
  }}
  oasisDiagnosticsCache.clear();
  invalidateGbuSummary();
  document.getElementById("traceOasisControls").classList.add("visible");
  configureOasisThresholdControls();
  if (!silent) {{
    document.getElementById("traceLoadStatus").textContent = `Loaded inferred spikes file with ${{data.nRois}} ROIs x ${{data.nFrames}} frames.`;
  }} else {{
    updateSidecarLoadNote();
  }}
  draw();
}}
function updateCellTypeLabelStatus() {{
  updateCategoricalFilterOptions();
  renderCategoryLabelEditor();
  renderAutoLoadedFileList();
}}
function updateCategoricalFilterOptions() {{
  const container = document.getElementById("cellTypeFilterOptions");
  if (!container) return;
  const current = new Set(readCategoryFilter());
  container.textContent = "";
  const values = Array.from(new Set((data.cellTypeLabels || []).filter(value => value !== null && value !== undefined).map(value => String(value)))).sort((a, b) => Number(a) - Number(b));
  if (!values.length) {{
    const note = document.createElement("span");
    note.className = "note";
    note.textContent = "No cell identity labels loaded.";
    container.appendChild(note);
    return;
  }}
  const note = document.createElement("span");
  note.className = "note";
  note.textContent = "If no cell identities are checked, the cell identity filter is not used.";
  container.appendChild(note);
  for (const value of values) {{
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.className = "cell-type-filter-checkbox";
    input.value = value;
    input.checked = current.has(value);
    input.addEventListener("change", evaluateFilter);
    label.appendChild(input);
    label.appendChild(document.createTextNode(` ${{categoryLabelName(Number(value))}} (${{value}})`));
    container.appendChild(label);
  }}
  const notLoadedLabel = document.createElement("label");
  const notLoadedInput = document.createElement("input");
  notLoadedInput.type = "checkbox";
  notLoadedInput.className = "cell-type-filter-checkbox";
  notLoadedInput.value = "not_loaded";
  notLoadedInput.checked = current.has("not_loaded");
  notLoadedInput.addEventListener("change", evaluateFilter);
  notLoadedLabel.appendChild(notLoadedInput);
  notLoadedLabel.appendChild(document.createTextNode(" not loaded"));
  container.appendChild(notLoadedLabel);
}}
function readCategoryFilter() {{
  return Array.from(document.querySelectorAll(".cell-type-filter-checkbox:checked")).map(input => String(input.value));
}}
function writeCategoryFilter(values) {{
  const selected = new Set(Array.isArray(values) ? values.map(value => String(value)) : (values ? [String(values)] : []));
  document.querySelectorAll(".cell-type-filter-checkbox").forEach(input => {{
    input.checked = selected.has(String(input.value));
  }});
}}
function renderCategoryLabelEditor() {{
  const section = document.getElementById("categoryLabelEditorSection");
  const editor = document.getElementById("categoryLabelEditor");
  if (!section || !editor) return;
  const values = Array.from(new Set((data.cellTypeLabels || []).filter(value => value !== null && value !== undefined).map(value => String(value)))).sort((a, b) => Number(a) - Number(b));
  editor.textContent = "";
  section.hidden = values.length === 0;
  for (const value of values) {{
    const label = document.createElement("label");
    const span = document.createElement("span");
    span.textContent = `Value ${{value}}`;
    const input = document.createElement("input");
    input.type = "text";
    input.value = data.categoryLabelMap?.[String(value)] || `cell identity ${{value}}`;
    input.addEventListener("change", () => {{
      data.categoryLabelMap[String(value)] = input.value.trim() || `cell identity ${{value}}`;
      updateCategoricalFilterOptions();
      updateSortCurrent();
      draw();
    }});
    label.appendChild(span);
    label.appendChild(input);
    editor.appendChild(label);
  }}
}}
function parseDelimitedRows(text) {{
  const delimiter = text.includes("\\t") && !text.includes(",") ? "\\t" : ",";
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {{
    const ch = text[i];
    if (quoted) {{
      if (ch === '"' && text[i + 1] === '"') {{ field += '"'; i++; }}
      else if (ch === '"') quoted = false;
      else field += ch;
    }} else if (ch === '"') quoted = true;
    else if (ch === delimiter) {{ row.push(field); field = ""; }}
    else if (ch === "\\n" || ch === "\\r") {{
      if (ch === "\\r" && text[i + 1] === "\\n") i++;
      row.push(field); field = "";
      if (row.some(value => value.trim() !== "")) rows.push(row);
      row = [];
    }} else field += ch;
  }}
  row.push(field);
  if (row.some(value => value.trim() !== "")) rows.push(row);
  return rows;
}}
function parseCellTypeCode(code, label) {{
  const numeric = Number(String(code ?? "").trim());
  if (Number.isFinite(numeric)) return numeric;
  const text = String(label ?? code ?? "").trim().toLowerCase();
  if (["inhibitory", "inhibitory/red", "red", "inh"].includes(text)) return 1;
  if (["excitatory", "excitatory/non-red", "non-red", "nonred", "exc"].includes(text)) return -1;
  if (["unsure", "uncertain", "unknown", "gray", "grey"].includes(text)) return 0;
  return null;
}}
function customMetricKey(name) {{
  let key = String(name || "").trim().toLowerCase().replace(/[^0-9a-z_]+/g, "_").replace(/^_+|_+$/g, "");
  if (!key) key = "metric";
  if (/^[0-9]/.test(key)) key = `metric_${{key}}`;
  return key;
}}
function ensureUniqueCustomMetricKey(baseKey) {{
  const existing = new Set((data.customRoiMetrics.definitions || []).map(def => def.key));
  let key = baseKey;
  let suffix = 2;
  while (existing.has(key)) {{
    key = `${{baseKey}}_${{suffix}}`;
    suffix += 1;
  }}
  return key;
}}
function normalizeCustomMetricValue(value) {{
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}}
function addCustomMetricDefinition(label, values, sourceName, roiIndexValues = null) {{
  if (!Array.isArray(values)) values = Array.from(values || []);
  const metricValues = new Array(data.nRois).fill(null);
  if (roiIndexValues) {{
    const roiBySuite2p = new Map(data.suite2pIndices.map((value, roi) => [Number(value), roi]));
    for (let i = 0; i < values.length; i++) {{
      const roi = roiBySuite2p.get(Number(roiIndexValues[i]));
      if (roi === undefined) continue;
      metricValues[roi] = normalizeCustomMetricValue(values[i]);
    }}
  }} else {{
    if (values.length !== data.nRois) throw new Error(`Custom metric '${{label}}' has ${{values.length}} rows; expected ${{data.nRois}}.`);
    for (let roi = 0; roi < data.nRois; roi++) metricValues[roi] = normalizeCustomMetricValue(values[roi]);
  }}
  if (!metricValues.some(value => value !== null)) return;
  const key = ensureUniqueCustomMetricKey(customMetricKey(label));
  data.customRoiMetrics.definitions.push({{key, label: String(label), source: sourceName, origin: "upload"}});
  for (let roi = 0; roi < data.nRois; roi++) data.customRoiMetrics.values[roi][key] = metricValues[roi];
}}
const RETIRED_ROI_INDEX_COLUMNS = ["suite2p_index", "roi", "index"];
function roiIndexColumn(headers) {{
  // roi_index is the only accepted name; a former name is rejected instead of read as a metric.
  const lower = headers.map(value => String(value).trim().toLowerCase());
  const found = lower.indexOf("roi_index");
  if (found >= 0) return found;
  const retired = lower.findIndex(value => RETIRED_ROI_INDEX_COLUMNS.includes(value));
  if (retired >= 0) throw new Error(`The ROI index column must be named roi_index; rename the column "${{headers[retired]}}".`);
  return -1;
}}
function loadCustomMetricRows(rows, sourceName) {{
  if (rows.length < 2) throw new Error("Expected a header row and one row per ROI.");
  const headers = rows[0].map(value => value.trim());
  const indexCol = roiIndexColumn(headers);
  const roiIndexValues = indexCol >= 0 ? rows.slice(1).map(row => row[indexCol]) : null;
  for (let col = 0; col < headers.length; col++) {{
    if (col === indexCol) continue;
    const values = rows.slice(1).map(row => row[col]);
    if (!values.some(value => normalizeCustomMetricValue(value) !== null)) continue;
    addCustomMetricDefinition(headers[col], values, sourceName, roiIndexValues);
  }}
}}
function loadCustomMetricJson(text, sourceName) {{
  const raw = JSON.parse(text);
  const metrics = raw && raw.metrics ? raw.metrics : raw;
  if (Array.isArray(metrics)) {{
    if (!metrics.length || typeof metrics[0] !== "object") throw new Error("JSON row list must contain objects.");
    const headers = Array.from(new Set(metrics.flatMap(row => Object.keys(row))));
    const indexPosition = roiIndexColumn(headers);
    const indexHeader = indexPosition >= 0 ? headers[indexPosition] : null;
    const roiIndexValues = indexHeader ? metrics.map(row => row[indexHeader]) : null;
    for (const header of headers) {{
      if (header === indexHeader) continue;
      const values = metrics.map(row => row[header]);
      if (!values.some(value => normalizeCustomMetricValue(value) !== null)) continue;
      addCustomMetricDefinition(header, values, sourceName, roiIndexValues);
    }}
    return;
  }}
  if (!metrics || typeof metrics !== "object") throw new Error("JSON custom metrics must be an object or row list.");
  for (const [name, values] of Object.entries(metrics)) {{
    if (Array.isArray(values)) addCustomMetricDefinition(name, values, sourceName);
    else if (values && typeof values === "object" && Array.isArray(values.values)) {{
      roiIndexColumn(Object.keys(values));
      addCustomMetricDefinition(values.label || name, values.values, values.source || sourceName, values.roi_index || null);
    }}
  }}
}}
function loadCustomMetricText(text, sourceName) {{
  if (sourceName.toLowerCase().endsWith(".json")) loadCustomMetricJson(text, sourceName);
  else loadCustomMetricRows(parseDelimitedRows(text), sourceName);
  renderCustomMetricControls();
  renderCustomSortMetrics();
  updateMetricDefaults();
  evaluateFilter();
  updateSortMetricHandlers();
  renderAutoLoadedFileList();
}}
function categoryCodeFromLabel(label, labelToCode) {{
  const text = String(label ?? "").trim();
  if (!text) return null;
  const legacy = parseCellTypeCode("", text);
  if (legacy !== null) return legacy;
  const key = text.toLowerCase();
  if (!labelToCode.has(key)) {{
    let next = 1;
    const existing = new Set(Array.from(labelToCode.values()).map(value => Number(value)));
    while (existing.has(next) || existing.has(-next) || existing.has(0)) next++;
    labelToCode.set(key, next);
  }}
  return labelToCode.get(key);
}}
function loadCategoryRows(rowObjects, sourceName) {{
  if (!Array.isArray(rowObjects) || !rowObjects.length || typeof rowObjects[0] !== "object") {{
    throw new Error("JSON cell identity labels must be a list of row objects.");
  }}
  const headers = Array.from(new Set(rowObjects.flatMap(row => Object.keys(row))));
  const lowerHeaders = headers.map(value => value.toLowerCase());
  let codeHeader = null;
  for (const name of ["cell_identity_code", "category_code", "cell_type_code", "code"]) {{
    const found = lowerHeaders.indexOf(name);
    if (found >= 0) {{ codeHeader = headers[found]; break; }}
  }}
  let labelHeader = null;
  for (const name of ["cell_identity_label", "category_label", "cell_type_label", "label"]) {{
    const found = lowerHeaders.indexOf(name);
    if (found >= 0) {{ labelHeader = headers[found]; break; }}
  }}
  const indexPosition = roiIndexColumn(headers);
  const indexHeader = indexPosition >= 0 ? headers[indexPosition] : null;
  if (!codeHeader && !labelHeader) throw new Error("Expected cell_identity_code/cell_identity_label keys (or category_code/category_label, or legacy cell_type_code/cell_type_label).");
  const nextLabels = new Array(data.nRois).fill(null);
  const labelMap = {{}};
  const labelToCode = new Map();
  const roiBySuite2p = new Map(data.suite2pIndices.map((value, roi) => [Number(value), roi]));
  const parseObjectCode = (row) => {{
    let code = codeHeader ? parseCellTypeCode(row[codeHeader], labelHeader ? row[labelHeader] : "") : null;
    if (code === null && labelHeader) code = categoryCodeFromLabel(row[labelHeader], labelToCode);
    if (code !== null && labelHeader && String(row[labelHeader] ?? "").trim()) labelMap[String(code)] = String(row[labelHeader]).trim();
    return code;
  }};
  if (indexHeader) {{
    for (const row of rowObjects) {{
      const roi = roiBySuite2p.get(Number(row[indexHeader]));
      if (roi === undefined) continue;
      nextLabels[roi] = parseObjectCode(row);
    }}
  }} else {{
    if (rowObjects.length !== data.nRois) throw new Error(`Expected ${{data.nRois}} data rows, found ${{rowObjects.length}}.`);
    rowObjects.forEach((row, roi) => {{ nextLabels[roi] = parseObjectCode(row); }});
  }}
  data.cellTypeLabels = nextLabels;
  data.categoryLabelMap = {{...data.categoryLabelMap, ...labelMap}};
  data.cellTypeLabelSource = sourceName;
  renderCategoryLabelEditor();
}}
function loadCellTypeTableText(text, sourceName) {{
  if (sourceName.toLowerCase().endsWith(".json")) {{
    loadCategoryRows(JSON.parse(text), sourceName);
    updateCellTypeLabelStatus();
    evaluateFilter();
    updateSortCurrent();
    draw();
    return;
  }}
  const rows = parseDelimitedRows(text);
  if (rows.length < 2) throw new Error("Expected a header row and one row per ROI.");
  const headers = rows[0].map(value => value.trim().toLowerCase());
  let codeIndex = headers.indexOf("cell_identity_code");
  if (codeIndex < 0) codeIndex = headers.indexOf("category_code");
  if (codeIndex < 0) codeIndex = headers.indexOf("cell_type_code");
  let labelIndex = headers.indexOf("cell_identity_label");
  if (labelIndex < 0) labelIndex = headers.indexOf("category_label");
  if (labelIndex < 0) labelIndex = headers.indexOf("cell_type_label");
  const suite2pIndex = roiIndexColumn(headers);
  if (codeIndex < 0 && labelIndex < 0) throw new Error("Expected cell_identity_code/cell_identity_label columns (or category_code/category_label, or legacy cell_type_code/cell_type_label).");
  const nextLabels = new Array(data.nRois).fill(null);
  const labelMap = {{}};
  const labelToCode = new Map();
  const parseRowCode = (row) => {{
    let code = codeIndex >= 0 ? parseCellTypeCode(row[codeIndex], labelIndex >= 0 ? row[labelIndex] : "") : null;
    if (code === null && labelIndex >= 0) code = categoryCodeFromLabel(row[labelIndex], labelToCode);
    if (code !== null && labelIndex >= 0 && String(row[labelIndex] ?? "").trim()) labelMap[String(code)] = String(row[labelIndex]).trim();
    return code;
  }};
  if (suite2pIndex >= 0) {{
    const roiBySuite2p = new Map(data.suite2pIndices.map((value, roi) => [Number(value), roi]));
    for (const row of rows.slice(1)) {{
      const roi = roiBySuite2p.get(Number(row[suite2pIndex]));
      if (roi === undefined) continue;
      nextLabels[roi] = parseRowCode(row);
    }}
  }} else {{
    if (rows.length - 1 !== data.nRois) throw new Error(`Expected ${{data.nRois}} data rows, found ${{rows.length - 1}}.`);
    rows.slice(1).forEach((row, roi) => {{
      nextLabels[roi] = parseRowCode(row);
    }});
  }}
  data.cellTypeLabels = nextLabels;
  data.categoryLabelMap = {{...data.categoryLabelMap, ...labelMap}};
  data.cellTypeLabelSource = sourceName;
  updateCellTypeLabelStatus();
  evaluateFilter();
  updateSortCurrent();
  draw();
}}
const labels = new Int8Array(data.nRois);
for (let roi = 0; roi < data.nRois; roi++) {{
  labels[roi] = -1;
}}
if (Array.isArray(data.initialLabels) && data.initialLabels.length === data.nRois) {{
  for (let roi = 0; roi < data.nRois; roi++) {{
    const label = Number(data.initialLabels[roi]);
    if (label === -1 || label === 0 || label === 1 || label === 2) labels[roi] = label;
  }}
}}
const filterPass = new Uint8Array(data.nRois);
let customPresets = data.customMorphologyPresets && typeof data.customMorphologyPresets === "object" ? {{...data.customMorphologyPresets}} : {{}};
const defaultFilter = data.morphologyPresets.all_rois;
let selected = 0, x0 = 0, x1 = data.nFrames - 1, y0 = 0, y1 = 0, visibleRois = [];
let appliedSortMetrics = ["original"];
let appliedSortDirection = "asc";

function customMetricDefinitions() {{
  return data.customRoiMetrics?.definitions || [];
}}
function customMetricFilterIds(key) {{
  return {{min: `customMetric_${{key}}_min`, max: `customMetric_${{key}}_max`}};
}}
function customMetricValue(roi, key) {{
  const value = data.customRoiMetrics?.values?.[roi]?.[key];
  return value === null || value === undefined ? NaN : Number(value);
}}
const CUSTOM_METRIC_SECTIONS = {{
  uploaded: "Uploaded ROI Metrics",
  model_derived: "Model-Derived Metrics",
  morphology: "Morphology Metrics",
  fluorescence: "Fluorescence Trace Metrics",
  inferred_spike: "Inferred Spike Metrics",
}};
function customMetricSection(definition) {{
  return Object.prototype.hasOwnProperty.call(CUSTOM_METRIC_SECTIONS, definition?.section) ? definition.section : "uploaded";
}}
function customMetricSlot(section) {{
  return document.querySelector(`.custom-metric-slot[data-section="${{section}}"]`) || document.getElementById("customMetricFilterControls");
}}
function sessionHasModelScores() {{
  return (data.roiModelScores || []).some(entry => entry && Number.isFinite(Number(entry.probability)));
}}
function updateModelDerivedVisibility() {{
  // The model score controls appear only when the session has model scores; the section itself
  // also stays visible when an uploaded metric has been placed in it.
  const hasScores = sessionHasModelScores();
  const filterSlot = document.querySelector('.custom-metric-slot[data-section="model_derived"]');
  const sortSlot = document.querySelector('.custom-sort-slot[data-section="model_derived"]');
  const scoreRow = document.getElementById("modelScoreFilterControls");
  const sortOption = document.getElementById("modelScoreSortOption");
  if (scoreRow) scoreRow.hidden = !hasScores;
  if (sortOption) {{
    sortOption.hidden = !hasScores;
    if (!hasScores) sortOption.querySelector("input").checked = false;
  }}
  const filterSection = document.getElementById("modelDerivedFilterSection");
  if (filterSection) filterSection.hidden = !hasScores && !(filterSlot && filterSlot.children.length);
  const sortGroup = document.getElementById("modelDerivedSortGroup");
  if (sortGroup) sortGroup.hidden = !hasScores && !(sortSlot && sortSlot.children.length);
}}
function renderCustomMetricControls() {{
  const kept = readCustomMetricFiltersFromInputs();
  document.querySelectorAll(".custom-metric-slot").forEach(slot => {{ slot.textContent = ""; }});
  let moved = 0;
  for (const definition of customMetricDefinitions()) {{
    const section = customMetricSection(definition);
    if (section !== "uploaded") moved++;
    const controls = customMetricSlot(section);
    const ids = customMetricFilterIds(definition.key);
    const header = document.createElement("div");
    header.className = "custom-metric-header";
    const name = document.createElement("strong");
    name.textContent = definition.label;
    header.appendChild(name);
    const sectionLabel = document.createElement("label");
    sectionLabel.textContent = "Section";
    const sectionSelect = document.createElement("select");
    sectionSelect.className = "custom-metric-section";
    for (const [value, text] of Object.entries(CUSTOM_METRIC_SECTIONS)) {{
      const option = document.createElement("option");
      option.value = value; option.textContent = text;
      sectionSelect.appendChild(option);
    }}
    sectionSelect.value = section;
    sectionSelect.addEventListener("change", () => {{
      definition.section = sectionSelect.value;
      renderCustomMetricControls();
      renderCustomSortMetrics();
      updateSortMetricHandlers();
      updateMetricDefaults();
      drawMetricHistograms();
    }});
    sectionLabel.appendChild(sectionSelect);
    header.appendChild(sectionLabel);
    if (definition.categories) {{
      const legend = document.createElement("span");
      legend.className = "note";
      legend.textContent = Object.entries(definition.categories).map(([code, text]) => `${{code}} = ${{text}}`).join(", ");
      header.appendChild(legend);
    }}
    const makeBound = (id, text) => {{
      const label = document.createElement("label");
      label.textContent = `${{definition.label}} ${{text}} `;
      const reference = document.createElement("span");
      reference.className = "threshold-default";
      reference.id = `${{id}}Default`;
      label.appendChild(reference);
      const input = document.createElement("input");
      input.id = id; input.type = "number"; input.step = "0.01"; input.placeholder = "Not Used";
      const previous = kept[definition.key]?.[text];
      if (previous !== null && previous !== undefined && Number.isFinite(previous)) input.value = previous;
      input.addEventListener("input", () => {{ evaluateFilter(); drawMetricHistograms(); }});
      label.appendChild(input);
      return label;
    }};
    const details = document.createElement("details");
    details.className = "metric-histogram-panel";
    details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "Distribution";
    const canvas = document.createElement("canvas");
    canvas.className = "metric-histogram";
    canvas.dataset.metric = `custom:${{definition.key}}`;
    canvas.dataset.min = ids.min;
    canvas.dataset.max = ids.max;
    canvas.setAttribute("aria-label", `${{definition.label}} distribution`);
    details.appendChild(summary);
    details.appendChild(canvas);
    details.addEventListener("toggle", () => {{ if (details.open) drawMetricHistogram(canvas); }});
    controls.appendChild(header);
    controls.appendChild(makeBound(ids.min, "min"));
    controls.appendChild(makeBound(ids.max, "max"));
    controls.appendChild(details);
  }}
  const total = customMetricDefinitions().length;
  document.getElementById("customMetricStatus").textContent = total
    ? `${{total}} uploaded ROI metric(s) loaded${{moved ? `; ${{moved}} shown in other sections` : ""}}.`
    : "No uploaded ROI metrics loaded.";
  if (typeof refreshRangeMetricOptions === "function") refreshRangeMetricOptions();
  updateModelDerivedVisibility();
  arrangeMetricBlocks();
}}
function readCustomMetricFiltersFromInputs() {{
  const values = {{}};
  for (const definition of customMetricDefinitions()) {{
    const ids = customMetricFilterIds(definition.key);
    const read = id => {{
      const input = document.getElementById(id);
      if (!input || String(input.value).trim() === "") return null;
      const value = Number(input.value);
      return Number.isFinite(value) ? value : null;
    }};
    values[definition.key] = {{min: read(ids.min), max: read(ids.max)}};
  }}
  return values;
}}
function renderCustomSortMetrics() {{
  const checked = new Set(Array.from(document.querySelectorAll('input[name="sortMetric"]:checked')).map(input => input.value));
  document.querySelectorAll(".custom-sort-slot").forEach(slot => {{ slot.textContent = ""; }});
  const container = document.getElementById("customSortMetrics");
  container.textContent = "";
  const uploaded = customMetricDefinitions().filter(definition => customMetricSection(definition) === "uploaded");
  if (uploaded.length) {{
    const group = document.createElement("div");
    group.className = "sort-metric-group";
    group.textContent = "Uploaded ROI Metrics";
    container.appendChild(group);
  }}
  for (const definition of customMetricDefinitions()) {{
    const section = customMetricSection(definition);
    const target = section === "uploaded" ? container : (document.querySelector(`.custom-sort-slot[data-section="${{section}}"]`) || container);
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "sortMetric";
    input.value = `custom:${{definition.key}}`;
    input.checked = checked.has(input.value);
    label.appendChild(input);
    label.appendChild(document.createTextNode(` ${{definition.label}}`));
    target.appendChild(label);
  }}
  updateModelDerivedVisibility();
}}
function fit(canvas) {{
  const r = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * r));
  canvas.height = Math.max(1, Math.round(box.height * r));
}}
function syncControlColumnHeight() {{
  return;
}}
function trace(roi) {{
  if (!dff) return null;
  return dff.subarray(roi * data.nFrames, (roi + 1) * data.nFrames);
}}
function spikeTrace(roi) {{
  if (!oasisSpikes) return null;
  return oasisSpikes.subarray(roi * data.nFrames, (roi + 1) * data.nFrames);
}}
if (data.oasisAvailable) {{
  document.getElementById("traceOasisControls").classList.add("visible");
  if (data.oasisStorageMode === "embedded" && data.oasisSpikes) {{
    oasisSpikes = b64f32(data.oasisSpikes);
    configureOasisThresholdControls();
  }} else {{
    configureOasisThresholdControls();
  }}
}}
function val(roi, frame) {{
  if (!dff) return NaN;
  return dff[roi * data.nFrames + frame];
}}
function dffMetric(roi, key, fallbackKey = null) {{
  const metrics = data.dffMetrics[roi] || {{}};
  return dffMetricValue(metrics, key, fallbackKey);
}}
function dffMetricValue(metrics, key, fallbackKey = null) {{
  const value = metrics[key];
  if (value !== null && value !== undefined) return value;
  return fallbackKey ? metrics[fallbackKey] : undefined;
}}
function metricValue(roi, metric) {{
  if (metric && metric.startsWith("custom:")) return customMetricValue(roi, metric.slice("custom:".length));
  if (metric === "snr_95_50" || metric === "event_snr") return dffMetric(roi, "snr_95_50", "event_snr");
  if (metric === "andrea_postdoc_snr") return dffMetric(roi, "andrea_postdoc_snr");
  if (metric === "oasis_event_snr") return dffMetric(roi, "oasis_event_snr");
  if (metric === "oasis_rise_tau_seconds") return dffMetric(roi, "oasis_rise_tau_seconds");
  if (metric === "oasis_decay_tau_seconds") return dffMetric(roi, "oasis_decay_tau_seconds");
  if (metric === "cell_type_code") return data.cellTypeLabels?.[roi] ?? NaN;
  if (metric === "roi_model_score_probability") return data.roiModelScores?.[roi]?.probability ?? NaN;
  if (metric === "roi_area") return data.dffMetrics[roi].roi_area;
  if (metric === "connectivity") return data.morphology[roi].connect;
  if (metric === "skew") return data.morphology[roi].skew;
  if (metric === "aspect") return data.morphology[roi].aspect;
  if (metric === "compact") return data.morphology[roi].compact;
  if (metric === "original" || metric === "suite2p_index") return data.suite2pIndices[roi];
  return roi;
}}
function metricLabel(metric) {{
  if (metric && metric.startsWith("custom:")) {{
    const key = metric.slice("custom:".length);
    const definition = customMetricDefinitions().find(item => item.key === key);
    return definition ? definition.label : key;
  }}
  if (metric === "snr_95_50" || metric === "event_snr") return "SNR: 95/50 percentile";
  if (metric === "andrea_postdoc_snr") return "SNR: CaImAn (large-transient score)";
  if (metric === "oasis_event_snr") return "inferred spike SNR";
  if (metric === "oasis_rise_tau_seconds") return "inferred spike rise tau";
  if (metric === "oasis_decay_tau_seconds") return "inferred spike decay tau";
  if (metric === "cell_type_code") return "cell identity code";
  if (metric === "roi_model_score_probability") return "ROI model probability";
  if (metric === "roi_area") return "ROI area (px)";
  if (metric === "connectivity") return "Connectivity";
  if (metric === "skew") return "Skew";
  if (metric === "aspect") return "Aspect ratio";
  if (metric === "compact") return "Compactness";
  if (metric === "original" || metric === "suite2p_index") return ROI_INDEX_LABEL_LOWER;
  return metric.replace("_", " ");
}}
function sortDirectionText(direction = appliedSortDirection) {{
  return direction === "desc" ? "descending" : "ascending";
}}
function sortMetricText(metrics = appliedSortMetrics) {{
  if (!metrics.length) return metricLabel("original");
  if (metrics.length === 1) return metricLabel(metrics[0]);
  return metrics.map(metricLabel).join(" + ");
}}
function finiteMetricValues(metric) {{
  const values = [];
  for (let roi = 0; roi < data.nRois; roi++) {{
    const value = Number(metricValue(roi, metric));
    if (Number.isFinite(value)) values.push(value);
  }}
  return values;
}}
function percentile(values, q) {{
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}}
function mean(values) {{
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((total, value) => total + value, 0) / finite.length : NaN;
}}
const referenceStatistics = {{}};
const referenceStatisticLabels = {{}};
function referenceText(inputId) {{
  const value = referenceStatistics[inputId];
  if (!Number.isFinite(value)) return "";
  return `(${{referenceStatisticLabels[inputId] || "reference"}}: ${{fmt(value)}})`;
}}
function setReferenceStatistic(inputId, value, label) {{
  referenceStatistics[inputId] = Number.isFinite(value) ? value : NaN;
  if (label) referenceStatisticLabels[inputId] = label;
}}
function updateMetricDefaults() {{
  const skew = finiteMetricValues("skew");
  const aspect = finiteMetricValues("aspect");
  const compact = finiteMetricValues("compact");
  const connect = finiteMetricValues("connectivity");
  const roiArea = finiteMetricValues("roi_area");
  const snr9550 = finiteMetricValues("snr_95_50");
  const caiman = finiteMetricValues("andrea_postdoc_snr");
  const oasisSnr = finiteMetricValues("oasis_event_snr");
  const oasisRiseTau = finiteMetricValues("oasis_rise_tau_seconds");
  const oasisDecayTau = finiteMetricValues("oasis_decay_tau_seconds");
  const roiModelProbability = finiteMetricValues("roi_model_score_probability");
  // Lines and the values in parentheses use the loaded filter set where it sets a field,
  // and otherwise a starting statistic of this session's distribution.
  const selectedPreset = loadedFilterSet();
  const startFrom = (id, value, label) => {{
    const preset = selectedPreset[id];
    if (preset !== null && preset !== undefined && String(preset).trim() !== "" && Number.isFinite(Number(preset))) setReferenceStatistic(id, Number(preset), "filter set");
    else setReferenceStatistic(id, value, label);
  }};
  startFrom("skewMin", mean(skew), "mean");
  startFrom("skewMax", mean(skew), "mean");
  startFrom("aspectMin", mean(aspect), "mean");
  startFrom("aspectMax", mean(aspect), "mean");
  startFrom("compactMin", mean(compact), "mean");
  startFrom("compactMax", mean(compact), "mean");
  startFrom("maxConnect", percentile(connect, 0.75), "75th pct");
  startFrom("roiAreaMin", percentile(roiArea, 0.25), "25th pct");
  startFrom("roiAreaMax", percentile(roiArea, 0.75), "75th pct");
  startFrom("eventSnrMin", mean(snr9550), "mean");
  startFrom("andreaPostdocSnrMin", mean(caiman), "mean");
  startFrom("oasisEventSnrMin", mean(oasisSnr), "mean");
  startFrom("oasisRiseTauMin", percentile(oasisRiseTau, 0.25), "25th pct");
  startFrom("oasisRiseTauMax", percentile(oasisRiseTau, 0.75), "75th pct");
  startFrom("oasisDecayTauMin", percentile(oasisDecayTau, 0.25), "25th pct");
  startFrom("oasisDecayTauMax", percentile(oasisDecayTau, 0.75), "75th pct");
  startFrom("roiModelScoreProbabilityMin", mean(roiModelProbability), "mean");
  
  for (const definition of customMetricDefinitions()) {{
    const values = finiteMetricValues(`custom:${{definition.key}}`);
    const ids = customMetricFilterIds(definition.key);
    const presetBounds = selectedPreset.customMetrics?.[definition.key] || {{}};
    const presetMin = Number(presetBounds.min), presetMax = Number(presetBounds.max);
    if (presetBounds.min !== null && presetBounds.min !== undefined && Number.isFinite(presetMin)) setReferenceStatistic(ids.min, presetMin, "filter set");
    else setReferenceStatistic(ids.min, percentile(values, 0.25), "25th pct");
    if (presetBounds.max !== null && presetBounds.max !== undefined && Number.isFinite(presetMax)) setReferenceStatistic(ids.max, presetMax, "filter set");
    else setReferenceStatistic(ids.max, percentile(values, 0.75), "75th pct");
    const minDefault = document.getElementById(`${{ids.min}}Default`);
    const maxDefault = document.getElementById(`${{ids.max}}Default`);
    if (minDefault) minDefault.textContent = values.length ? referenceText(ids.min) : "";
    if (maxDefault) maxDefault.textContent = values.length ? referenceText(ids.max) : "";
  }}
  for (const id of ["skewMin", "skewMax", "aspectMin", "aspectMax", "compactMin", "compactMax"]) {{
    const element = document.getElementById(`${{id}}Default`);
    if (element) element.textContent = referenceText(id);
  }}
  document.getElementById("maxConnectDefault").textContent = referenceText("maxConnect");
  document.getElementById("roiAreaMinDefault").textContent = referenceText("roiAreaMin");
  document.getElementById("roiAreaMaxDefault").textContent = referenceText("roiAreaMax");
  document.getElementById("eventSnrDefault").textContent = referenceText("eventSnrMin");
  document.getElementById("andreaPostdocSnrDefault").textContent = referenceText("andreaPostdocSnrMin");
  document.getElementById("oasisEventSnrDefault").textContent = oasisSnr.length ? referenceText("oasisEventSnrMin") : "";
  document.getElementById("oasisRiseTauMinDefault").textContent = oasisRiseTau.length ? referenceText("oasisRiseTauMin") : "";
  document.getElementById("oasisRiseTauMaxDefault").textContent = oasisRiseTau.length ? referenceText("oasisRiseTauMax") : "";
  document.getElementById("oasisDecayTauMinDefault").textContent = oasisDecayTau.length ? referenceText("oasisDecayTauMin") : "";
  document.getElementById("oasisDecayTauMaxDefault").textContent = oasisDecayTau.length ? referenceText("oasisDecayTauMax") : "";
  document.getElementById("roiModelScoreProbabilityMinDefault").textContent = roiModelProbability.length ? referenceText("roiModelScoreProbabilityMin") : "";
  configureOasisFilterControls();
}}
function configureOasisFilterControls() {{
  const oasisMetricIds = [
    ["oasisEventSnrMin", "oasis_event_snr"],
    ["oasisRiseTauMin", "oasis_rise_tau_seconds"],
    ["oasisRiseTauMax", "oasis_rise_tau_seconds"],
    ["oasisDecayTauMin", "oasis_decay_tau_seconds"],
    ["oasisDecayTauMax", "oasis_decay_tau_seconds"],
  ];
  for (const [inputId, metric] of oasisMetricIds) {{
    const input = document.getElementById(inputId);
    if (!input) continue;
    const available = finiteMetricValues(metric).length > 0;
    input.disabled = !available;
    input.placeholder = available ? "Not Used" : "No inferred spikes file detected";
    if (!available) input.value = "";
  }}
}}
function loadedFilterSet() {{
  const select = document.getElementById("filterPreset");
  const [kind, name] = String(select?.value || "").split(":", 2);
  const preset = kind === "built-in" ? data.morphologyPresets[name] : kind === "custom" ? customPresets[name] : null;
  return preset && typeof preset === "object" ? preset : {{}};
}}
function loadedFilterSetName() {{
  const [, name] = String(document.getElementById("filterPreset")?.value || "").split(":", 2);
  return name || "";
}}
function updateSuite2pPresetThresholds(filter) {{
  const fallback = {{
    skewMin: mean(finiteMetricValues("skew")),
    skewMax: mean(finiteMetricValues("skew")),
    aspectMin: mean(finiteMetricValues("aspect")),
    aspectMax: mean(finiteMetricValues("aspect")),
    compactMin: mean(finiteMetricValues("compact")),
    compactMax: mean(finiteMetricValues("compact")),
  }};
  for (const id of ["skewMin","skewMax","aspectMin","aspectMax","compactMin","compactMax"]) {{
    const fromFilter = filter && filter[id] !== null && filter[id] !== undefined;
    setReferenceStatistic(id, fromFilter ? filter[id] : fallback[id], fromFilter ? "filter set" : "mean");
  }}
}}
// Draggable lines on each metric distribution. The min/max threshold lines (solid when entered,
// dashed gray when the field is not used) can be dragged to set that field's value; dragging
// anywhere else moves a purple value line that reports where a value sits in the session's
// distribution (percentile, ROIs below and at or above). Double-click resets the purple line to
// the median. A readout under the plot counts ROIs below the min, between, and above the max,
// and an expandable box lists summary statistics.
function histogramExtras(canvas) {{
  const holder = canvas.closest(".metric-histogram-panel") || canvas.parentElement;
  let readout = holder.querySelector(".histogram-probe-readout");
  if (!readout) {{
    const thresholds = document.createElement("div");
    thresholds.className = "histogram-threshold-readout";
    canvas.insertAdjacentElement("afterend", thresholds);
    readout = document.createElement("div");
    readout.className = "histogram-probe-readout";
    thresholds.insertAdjacentElement("afterend", readout);
    const stats = document.createElement("details");
    stats.className = "histogram-stats";
    stats.innerHTML = '<summary>Summary statistics</summary><table class="histogram-stats-table"></table>';
    readout.insertAdjacentElement("afterend", stats);
  }}
  return {{readout, thresholds: holder.querySelector(".histogram-threshold-readout"), table: holder.querySelector(".histogram-stats-table")}};
}}
function countBelow(sorted, value) {{
  let lo = 0, hi = sorted.length;
  while (lo < hi) {{ const mid = (lo + hi) >> 1; if (sorted[mid] < value) lo = mid + 1; else hi = mid; }}
  return lo;
}}
function countAtOrBelow(sorted, value) {{
  let lo = 0, hi = sorted.length;
  while (lo < hi) {{ const mid = (lo + hi) >> 1; if (sorted[mid] <= value) lo = mid + 1; else hi = mid; }}
  return lo;
}}
function roundToInputStep(input, value) {{
  const step = parseFloat(input.step);
  let v = value;
  if (Number.isFinite(step) && step > 0) {{
    const decimals = (String(input.step).split(".")[1] || "").length;
    v = Number((Math.round(v / step) * step).toFixed(decimals));
  }} else {{
    v = Number(v.toPrecision(4));
  }}
  const min = parseFloat(input.min);
  if (Number.isFinite(min)) v = Math.max(min, v);
  return v;
}}
function bindHistogramProbe(canvas) {{
  if (canvas._probeBound) return;
  canvas._probeBound = true;
  canvas.style.touchAction = "none";
  canvas.style.cursor = "crosshair";
  canvas.title = "Drag a min/max line to set that threshold; drag elsewhere to read off a value's percentile; double-click to reset the purple line";
  const canvasX = event => {{
    const rect = canvas.getBoundingClientRect();
    return (event.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
  }};
  const valueAt = x => {{
    const s = canvas._probeScale;
    const t = Math.min(1, Math.max(0, (x - s.pad) / s.plotW));
    return s.lo + t * (s.hi - s.lo);
  }};
  const nearestThreshold = x => {{
    const r = window.devicePixelRatio || 1;
    let best = null;
    for (const line of canvas._thresholdLines || []) {{
      const input = document.getElementById(line.inputId);
      if (!input || input.disabled) continue;
      const d = Math.abs(line.x - x);
      if (d <= 8 * r && (!best || d < best.d)) best = {{d, input}};
    }}
    return best ? best.input : null;
  }};
  const move = event => {{
    if (!canvas._probeScale) return;
    const x = canvasX(event);
    if (canvas._dragInput) {{
      canvas._dragInput.value = roundToInputStep(canvas._dragInput, valueAt(x));
      canvas._dragInput.dispatchEvent(new Event("input", {{bubbles: true}}));
    }} else {{
      canvas._probeValue = valueAt(x);
      drawMetricHistogram(canvas);
    }}
  }};
  canvas.addEventListener("pointerdown", event => {{
    if (!canvas._probeScale) return;
    canvas.setPointerCapture(event.pointerId);
    canvas._dragging = true;
    canvas._dragInput = nearestThreshold(canvasX(event));
    move(event);
  }});
  canvas.addEventListener("pointermove", event => {{
    if (canvas._dragging) {{ move(event); return; }}
    if (canvas._probeScale) canvas.style.cursor = nearestThreshold(canvasX(event)) ? "ew-resize" : "crosshair";
  }});
  const end = () => {{
    if (canvas._dragging && canvas._dragInput) canvas._dragInput.dispatchEvent(new Event("change", {{bubbles: true}}));
    canvas._dragging = false;
    canvas._dragInput = null;
  }};
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("dblclick", () => {{ canvas._probeValue = undefined; drawMetricHistogram(canvas); }});
}}
function drawHistogramProbe(canvas, ctx, values, scale) {{
  const {{readout, thresholds, table}} = histogramExtras(canvas);
  if (!values.length) {{ readout.textContent = ""; thresholds.textContent = ""; table.innerHTML = ""; canvas._probeScale = null; return; }}
  const sorted = values.slice().sort((a, b) => a - b);
  const n = sorted.length;
  if (canvas._probeMetric !== canvas.dataset.metric || !Number.isFinite(canvas._probeValue)) {{
    canvas._probeMetric = canvas.dataset.metric;
    canvas._probeValue = percentile(sorted, 0.5);
  }}
  canvas._probeScale = scale;
  const value = canvas._probeValue;
  const {{lo, hi, pad, plotW, plotTop, plotBottom, r}} = scale;
  const x = pad + Math.min(1, Math.max(0, (value - lo) / (hi - lo))) * plotW;
  ctx.save();
  ctx.strokeStyle = "#7c3aed"; ctx.fillStyle = "#7c3aed"; ctx.lineWidth = Math.max(1.5, 1.2 * r);
  ctx.beginPath(); ctx.moveTo(x, plotTop); ctx.lineTo(x, plotBottom); ctx.stroke();
  ctx.beginPath(); ctx.arc(x, plotBottom, 3.5 * r, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
  const below = countBelow(sorted, value), above = n - below;
  const pct = 100 * below / n;
  readout.textContent = `Purple line at ${{fmt(value)}}: percentile ${{pct.toFixed(1)}}, ${{below.toLocaleString()}} ROIs below and ${{above.toLocaleString()}} at or above`;
  // ROIs below the min, between, and above the max, for the thresholds currently entered
  const minValue = canvas.dataset.min ? filterValue(canvas.dataset.min) : NaN;
  const maxValue = canvas.dataset.max ? filterValue(canvas.dataset.max) : NaN;
  const hasMin = Number.isFinite(minValue), hasMax = Number.isFinite(maxValue);
  const nBelow = hasMin ? countBelow(sorted, minValue) : 0;
  const nAbove = hasMax ? n - countAtOrBelow(sorted, maxValue) : 0;
  const nBetween = Math.max(0, n - nBelow - nAbove);
  const num = v => v.toLocaleString();
  const ex = (text, v) => `${{text}}: <span class="excluded">${{num(v)}}</span>`;
  const kept = (text, v) => `${{text}}: <span class="kept">${{num(v)}}</span>`;
  thresholds.innerHTML = hasMin && hasMax
    ? `${{ex(`Below min ${{fmt(minValue)}}`, nBelow)}} &middot; ${{kept("between", nBetween)}} &middot; ${{ex(`above max ${{fmt(maxValue)}}`, nAbove)}} <span class="note">(of ${{num(n)}} ROIs)</span>`
    : hasMin ? `${{ex(`Below min ${{fmt(minValue)}}`, nBelow)}} &middot; ${{kept("at or above", nBetween)}} <span class="note">(of ${{num(n)}} ROIs)</span>`
    : hasMax ? `${{kept(`At or below max ${{fmt(maxValue)}}`, nBetween)}} &middot; ${{ex("above", nAbove)}} <span class="note">(of ${{num(n)}} ROIs)</span>`
    : `<span class="note">No threshold entered: type a value or drag a dashed line (${{num(n)}} ROIs with a value).</span>`;
  const rows = [["ROIs with a value", n.toLocaleString()], ["Minimum", fmt(sorted[0])], ["25th percentile", fmt(percentile(sorted, 0.25))],
                ["Median", fmt(percentile(sorted, 0.5))], ["75th percentile", fmt(percentile(sorted, 0.75))], ["Maximum", fmt(sorted[n - 1])]];
  table.innerHTML = rows.map(([k, v]) => `<tr><th scope="row">${{k}}</th><td>${{v}}</td></tr>`).join("");
}}
function drawMetricHistogram(canvas) {{
  fit(canvas);
  bindHistogramProbe(canvas);
  const ctx = canvas.getContext("2d");
  const metric = canvas.dataset.metric;
  const values = finiteMetricValues(metric);
  const w = canvas.width, h = canvas.height, pad = 24;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d0d5dd"; ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  if (!values.length) {{
    canvas._thresholdLines = [];
    drawHistogramProbe(canvas, ctx, values, null);
    return;
  }}
  const r = window.devicePixelRatio || 1;
  ctx.fillStyle = "#344054"; ctx.font = `${{11 * r}}px Arial`; ctx.textAlign = "left";
  ctx.fillText(metricLabel(metric), pad, 12 * r);
  let lo = percentile(values, 0.01), hi = percentile(values, 0.99);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {{ lo = Math.min(...values); hi = Math.max(...values); }}
  if (hi <= lo) hi = lo + 1;
  const bins = 28, counts = Array(bins).fill(0);
  values.forEach(value => {{
    const bin = Math.max(0, Math.min(bins - 1, Math.floor((value - lo) / (hi - lo) * bins)));
    counts[bin]++;
  }});
  const maxCount = Math.max(...counts, 1);
  const plotW = w - pad * 2, plotTop = 32 * r, plotBottom = h - pad * 1.45;
  const plotH = Math.max(1, plotBottom - plotTop);
  ctx.fillStyle = "#111827";
  counts.forEach((count, index) => {{
    const x = pad + index / bins * plotW;
    const bw = Math.max(1, plotW / bins - 1);
    const bh = count / maxCount * plotH;
    ctx.fillRect(x, plotBottom - bh, bw, bh);
  }});
  // Thresholds the user has entered are solid green (min) and red (max) lines labeled
  // "min"/"max". A field that is not used shows its starting value instead (filter set
  // value, mean or percentile, as in the parentheses beside the field) as a dashed gray
  // line labeled with that statistic, so it is never mistaken for an active threshold.
  // Values beyond the plotted range are pinned to the edge with an arrow.
  function lineSpec(inputId, bound) {{
    if (!inputId) return null;
    const activeValue = filterValue(inputId);
    if (Number.isFinite(activeValue)) {{
      return {{value: activeValue, active: true, bound, color: bound === "min" ? "#16a34a" : "#dc2626", text: `${{bound}} ${{fmt(activeValue)}}`}};
    }}
    const referenceValue = referenceStatistics[inputId];
    if (!Number.isFinite(referenceValue)) return null;
    const name = referenceStatisticLabels[inputId] || "reference";
    return {{value: referenceValue, active: false, bound, color: "#667085", text: `${{name}} ${{fmt(referenceValue)}}`}};
  }}
  let specs = [lineSpec(canvas.dataset.min, "min"), lineSpec(canvas.dataset.max, "max")].filter(Boolean);
  if (specs.length === 2 && !specs[0].active && !specs[1].active &&
      Math.abs(specs[0].value - specs[1].value) <= Math.max(1e-9, Math.abs(specs[0].value) * 1e-9)) {{
    specs = [specs[0]];  // the same starting value for both bounds: draw it once
  }}
  ctx.save();
  ctx.font = `${{10 * r}}px Arial`;
  const labelY = plotTop - 3 * r, edge = 2 * r;
  const placed = specs.map(spec => {{
    const raw = pad + (spec.value - lo) / (hi - lo) * plotW;
    const below = raw < pad, above = raw > pad + plotW;
    const x = Math.min(pad + plotW, Math.max(pad, raw));
    const text = below ? `\u2190 ${{spec.text}}` : above ? `${{spec.text}} \u2192` : spec.text;
    const width = ctx.measureText(text).width;
    let left = x - width / 2;
    if (below) left = x; else if (above) left = x - width;
    left = Math.min(w - edge - width, Math.max(edge, left));
    return {{...spec, x, text, width, left}};
  }});
  if (placed.length === 2) {{
    // keep the two labels apart: the lower value's label ends at its line, the higher one starts at its line
    const [a, b] = placed[0].x <= placed[1].x ? placed : [placed[1], placed[0]];
    if (a.left + a.width + 4 * r > b.left) {{
      a.left = Math.max(edge, Math.min(a.x, w - edge) - a.width - 2 * r);
      b.left = Math.min(w - edge - b.width, Math.max(edge, b.x + 2 * r));
      if (a.left + a.width + 4 * r > b.left) b.y = labelY + 11 * r;  // still crowded: drop the second label a row
    }}
  }}
  canvas._thresholdLines = placed.map(p => ({{x: p.x, inputId: p.bound === "min" ? canvas.dataset.min : canvas.dataset.max}}));
  for (const p of placed) {{
    ctx.strokeStyle = p.color; ctx.lineWidth = Math.max(p.active ? 2 : 1.5, (p.active ? 1.5 : 1.2) * r);
    ctx.setLineDash(p.active ? [] : [4 * r, 3 * r]);
    ctx.beginPath(); ctx.moveTo(p.x, plotTop); ctx.lineTo(p.x, plotBottom + 2); ctx.stroke();
    ctx.setLineDash([]);
    const y = p.y || labelY;
    if (p.y) {{ ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(p.left - 1, y - 9 * r, p.width + 2, 11 * r); }}
    ctx.fillStyle = p.color; ctx.textAlign = "left";
    ctx.fillText(p.text, p.left, y);
  }}
  ctx.restore();
  const binSize = (hi - lo) / bins;
  const yText = h - 7 * (window.devicePixelRatio || 1);
  ctx.fillStyle = "#475467"; ctx.font = `${{10 * (window.devicePixelRatio || 1)}}px Arial`;
  ctx.textAlign = "left"; ctx.fillText(fmt(lo), pad, yText);
  ctx.textAlign = "center"; ctx.fillText(`Bin ${{fmt(binSize)}}`, pad + plotW / 2, yText);
  ctx.textAlign = "right"; ctx.fillText(fmt(hi), pad + plotW, yText);
  drawHistogramProbe(canvas, ctx, values, {{lo, hi, pad, plotW, plotTop, plotBottom, r}});
}}
// Each metric's distribution sits above its own min/max fields: the histogram panel and the
// bound fields that precede it in the markup are wrapped into one block.
function arrangeMetricBlocks() {{
  document.querySelectorAll(".filter-controls").forEach(container => {{
    const panels = [...container.querySelectorAll(":scope > details.metric-histogram-panel")];
    if (!panels.length) return;
    for (const details of panels) {{
      const bounds = [];
      let header = null;
      let prev = details.previousElementSibling;
      while (prev && bounds.length < 2 && (prev.tagName === "LABEL" || (prev.tagName === "SPAN" && !prev.textContent.trim()))) {{
        bounds.unshift(prev);
        prev = prev.previousElementSibling;
      }}
      if (prev && prev.classList.contains("custom-metric-header")) header = prev;
      const block = document.createElement("div");
      block.className = "metric-block";
      (header || bounds[0] || details).before(block);
      if (header) block.appendChild(header);
      block.appendChild(details);
      const row = document.createElement("div");
      row.className = "metric-bounds";
      for (const element of bounds) {{
        if (element.tagName === "SPAN") element.remove(); else row.appendChild(element);
      }}
      block.appendChild(row);
    }}
    container.classList.add("metric-blocks");
  }});
}}
function drawMetricHistograms() {{
  document.querySelectorAll(".metric-histogram-panel[open] .metric-histogram").forEach(drawMetricHistogram);
}}
function sortVisibleRois(rois) {{
  const metrics = appliedSortMetrics.length ? appliedSortMetrics : ["original"];
  const direction = appliedSortDirection;
  const ranges = {{}};
  for (const metric of metrics) {{
    const values = rois.map(roi => Number(metricValue(roi, metric))).filter(Number.isFinite);
    const minValue = values.length ? Math.min(...values) : NaN;
    const maxValue = values.length ? Math.max(...values) : NaN;
    ranges[metric] = {{min: minValue, max: maxValue}};
  }}
  function normalizedMetricValue(roi, metric) {{
    const value = Number(metricValue(roi, metric));
    if (!Number.isFinite(value)) return NaN;
    const range = ranges[metric];
    if (!range || !Number.isFinite(range.min) || !Number.isFinite(range.max)) return NaN;
    if (range.max === range.min) return 0.5;
    return (value - range.min) / (range.max - range.min);
  }}
  function sortScore(roi) {{
    let total = 0, count = 0;
    for (const metric of metrics) {{
      const value = normalizedMetricValue(roi, metric);
      if (Number.isFinite(value)) {{ total += value; count++; }}
    }}
    return count ? total / count : NaN;
  }}
  return rois.slice().sort((a, b) => {{
    const av = sortScore(a);
    const bv = sortScore(b);
    const aFinite = Number.isFinite(av);
    const bFinite = Number.isFinite(bv);
    if (aFinite !== bFinite) return aFinite ? -1 : 1;
    if (!aFinite && !bFinite) return a - b;
    if (av === bv) return a - b;
    return direction === "asc" ? av - bv : bv - av;
  }});
}}
function roiFromSuite2pIndex(value) {{
  const suite2pIndex = Math.round(Number(value));
  const exact = data.suite2pIndices.indexOf(suite2pIndex);
  if (exact >= 0) return exact;
  return Math.max(0, Math.min(data.nRois - 1, suite2pIndex));
}}
function currentSortPositionText() {{
  // Sort position is 1-based within the ROIs currently shown; the original index (0-based) is shown separately.
  const total = visibleRois.length;
  const order = `${{sortMetricText()}}, ${{sortDirectionText()}}`;
  if (!visibleRois.includes(selected)) return `not in the current view (${{total}} shown, ROIs sorted by ${{order}})`;
  return `sort position ${{visibleRois.indexOf(selected) + 1}} of ${{total}} (ROIs sorted by ${{order}})`;
}}
function updateSortCurrent() {{
  document.getElementById("sortCurrent").textContent = `Current order: ${{sortMetricText()}}, ${{sortDirectionText()}}`;
  document.getElementById("selectedSortPosition").textContent = currentSortPositionText();
}}
function syncStackInputs() {{
  document.getElementById("yStart").value = y0;
  document.getElementById("yEnd").value = y1;
}}
function currentStackWindowSize() {{
  const start = Number(document.getElementById("yStart").value);
  const end = Number(document.getElementById("yEnd").value);
  if (Number.isFinite(start) && Number.isFinite(end) && end >= start) return Math.max(1, Math.round(end - start + 1));
  return Math.max(1, Math.round(y1 - y0 + 1));
}}
function ensureSelectedInStack() {{
  if (!visibleRois.length) return;
  const position = visibleRois.indexOf(selected);
  if (position < 0) return;
  const windowSize = Math.max(1, Math.min(visibleRois.length, currentStackWindowSize()));
  if (position >= y0 && position <= y1 && (y1 - y0 + 1) >= Math.min(windowSize, visibleRois.length)) {{
    syncStackInputs();
    return;
  }}
  let start = position;
  if (position > y1) start = position - windowSize + 1;
  start = Math.max(0, Math.min(start, visibleRois.length - windowSize));
  y0 = start;
  y1 = start + windowSize - 1;
  syncStackInputs();
}}
function selectedSortMetrics() {{
  const checked = Array.from(document.querySelectorAll('input[name="sortMetric"]:checked')).map(input => input.value);
  return checked.length ? checked : ["original"];
}}
function applySort() {{
  appliedSortMetrics = selectedSortMetrics();
  appliedSortDirection = document.getElementById("sortDirection").value;
  updateVisibleRois(true);
  updateSortCurrent();
  if (visibleRois.length) setSelected(visibleRois[0]);
}}
function syncTimeInputs() {{
  document.getElementById("timeStart").value = (x0 / data.frameRate).toFixed(3);
  document.getElementById("timeEnd").value = (x1 / data.frameRate).toFixed(3);
}}
function setTimeWindow(startSec, endSec) {{
  startSec = Number(startSec);
  endSec = Number(endSec);
  if (!Number.isFinite(startSec)) startSec = 0;
  if (!Number.isFinite(endSec)) endSec = sessionDurationSec;
  startSec = Math.max(0, Math.min(sessionDurationSec, startSec));
  endSec = Math.max(0, Math.min(sessionDurationSec, endSec));
  if (endSec <= startSec) endSec = Math.min(sessionDurationSec, startSec + 1 / data.frameRate);
  x0 = Math.max(0, Math.min(data.nFrames - 1, Math.round(startSec * data.frameRate)));
  x1 = Math.max(0, Math.min(data.nFrames - 1, Math.round(endSec * data.frameRate)));
  if (x1 <= x0) x1 = Math.min(data.nFrames - 1, x0 + 1);
  syncTimeInputs();
  draw();
}}
function setFrameWindow(startFrame, endFrame) {{
  x0 = Math.max(0, Math.min(data.nFrames - 1, startFrame));
  x1 = Math.max(0, Math.min(data.nFrames - 1, endFrame));
  if (x1 <= x0) x1 = Math.min(data.nFrames - 1, x0 + 1);
  syncTimeInputs();
  draw();
}}
function setSelected(roi) {{
  selected = Math.max(0, Math.min(data.nRois - 1, Math.round(roi)));
  ensureSelectedInStack();
  const metrics = data.morphology[selected];
  const dffMetrics = data.dffMetrics[selected];
  const snr9550 = dffMetric(selected, "snr_95_50", "event_snr");
  const postdocSnr = dffMetric(selected, "andrea_postdoc_snr");
  const suite2pRoi = data.suite2pIndices[selected];
  const cellType = cellTypeLabelName(data.cellTypeLabels?.[selected]);
  const roiModelScore = data.roiModelScores?.[selected] || null;
  const roiModelState = roiModelScore?.state === "gray" ? "unsure" : roiModelScore?.state;
  const roiModelText = roiModelScore
    ? ` | ROI model score ${{roiModelState}} p(good) ${{fmt(roiModelScore.probability)}}`
    : "";
  if (data.oasisAvailable) setOasisThreshold(selectedOasisDefaultThreshold(), false);
  document.getElementById("roiInput").value = suite2pRoi;
  document.getElementById("roiDetailsSummary").textContent = "Selected ROI Details";
  document.getElementById("readout").textContent = `cell identity ${{cellType}}${{roiModelText}} | area ${{fmt(dffMetrics.roi_area)}} px | skew ${{fmt(metrics.skew)}} connect ${{metrics.connect}} aspect ${{fmt(metrics.aspect)}} compact ${{fmt(metrics.compact)}} | SNR: 95/50 percentile ${{fmt(snr9550)}} | SNR: CaImAn (large-transient score) ${{fmt(postdocSnr)}} | inferred spike SNR ${{fmt(dffMetricValue(dffMetrics, "oasis_event_snr"))}} | rise tau ${{fmt(dffMetricValue(dffMetrics, "oasis_rise_tau_seconds"))}} s | decay tau ${{fmt(dffMetricValue(dffMetrics, "oasis_decay_tau_seconds"))}} s`;
  document.getElementById("traceTitle").textContent = `Selected ROI - ${{ROI_INDEX_LABEL}} ${{suite2pRoi}}, ${{currentSortPositionText()}}`;
  document.querySelectorAll(".roi").forEach(c => c.classList.toggle("selected", Number(c.dataset.roi) === selected));
  updateLabelControls();
  updateSortCurrent();
  draw();
}}
function roiMatchesDisplayMode(roi) {{
  const active = selectedRoiDisplayLabels();
  if (active.has("all")) return true;
  if (labels[roi] === 1) return active.has("good");
  if (labels[roi] === 0) return active.has("bad");
  if (labels[roi] === 2) return active.has("unsure");
  return active.has("unlabeled");
}}
function selectedRoiDisplayLabels() {{
  const checked = Array.from(document.querySelectorAll(".roi-display-checkbox:checked")).map(input => input.value);
  return new Set(checked.length ? checked : ["all"]);
}}
function updateRoiDisplaySummary() {{
  const summary = document.getElementById("showRoiSummary");
  const boxes = Array.from(document.querySelectorAll(".roi-display-checkbox"));
  const checked = boxes.filter(input => input.checked).map(input => input.value);
  if (!checked.length || checked.length === boxes.length) {{
    summary.textContent = "All filtered ROIs";
    return;
  }}
  const active = new Set(checked);
  const labelsText = [];
  if (active.has("good")) labelsText.push("good");
  if (active.has("bad")) labelsText.push("bad");
  if (active.has("unsure")) labelsText.push("unsure");
  if (active.has("unlabeled")) labelsText.push("not labeled");
  summary.textContent = labelsText.length ? labelsText.join(", ") : "All filtered ROIs";
}}
function functionalRoiVisible(roi) {{
  return Boolean(filterPass[roi] && roiMatchesDisplayMode(roi));
}}
function updateVisibleRois(resetRange = false) {{
  const rois = [];
  for (let roi = 0; roi < data.nRois; roi++) if (functionalRoiVisible(roi)) rois.push(roi);
  visibleRois = sortVisibleRois(rois);
  document.getElementById("yStart").max = Math.max(0, visibleRois.length - 1);
  document.getElementById("yEnd").max = Math.max(0, visibleRois.length - 1);
  if (resetRange) {{
    y0 = 0;
    y1 = Math.max(0, Math.min(19, visibleRois.length - 1));
  }} else {{
    y0 = Math.max(0, Math.min(y0, Math.max(0, visibleRois.length - 1)));
    y1 = Math.max(y0, Math.min(y1 || Math.min(19, Math.max(0, visibleRois.length - 1)), Math.max(0, visibleRois.length - 1)));
  }}
  syncStackInputs();
  document.querySelectorAll(".roi, .roi-hit").forEach(path => {{
    const roi = Number(path.dataset.roi);
    path.style.display = functionalRoiVisible(roi) ? "" : "none";
  }});
  const redOverlayMode = document.getElementById("redOverlayMode");
  if (redOverlayMode) setRedOverlayMode(redOverlayMode.value);
  if (!visibleRois.length) draw();
  else if (!visibleRois.includes(selected)) setSelected(visibleRois[0]);
  else {{ ensureSelectedInStack(); draw(); }}
}}
function updateLabelControls() {{
  const label = labels[selected];
  document.getElementById("markGood").classList.toggle("active", label === 1);
  document.getElementById("markBad").classList.toggle("active", label === 0);
  document.getElementById("markUnsure").classList.toggle("active", label === 2);
  document.getElementById("markUnlabeled").classList.toggle("active", label === -1);
  let good = 0, bad = 0, unsure = 0, unlabeled = 0;
  for (const value of labels) {{ if (value === 1) good++; else if (value === 0) bad++; else if (value === 2) unsure++; else unlabeled++; }}
  document.getElementById("labelCounts").textContent = `${{good}} good | ${{bad}} bad | ${{unsure}} unsure | ${{unlabeled}} not labeled`;
  scheduleGbuSummaryDraw();
}}
function labelName(label) {{
  if (label === 1) return "good";
  if (label === 0) return "bad";
  if (label === 2) return "unsure";
  return "not labeled";
}}
function cellTypeLabelName(label) {{
  if (label === null || label === undefined) return "not loaded";
  const numeric = Number(label);
  if (!Number.isFinite(numeric)) return "not loaded";
  const mapped = data.categoryLabelMap?.[String(numeric)];
  if (mapped) return mapped;
  if (numeric === 1) return "inhibitory/red";
  if (numeric === -1) return "excitatory/non-red";
  if (numeric === 0) return "unsure";
  return `cell identity ${{numeric}}`;
}}
function categoryLabelName(label) {{
  return cellTypeLabelName(label);
}}
function fmt(value) {{
  if (value === null || value === undefined) return "nan";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "nan";
}}
function filterValue(id) {{
  const element = document.getElementById(id);
  if (!element) return NaN;
  const raw = element.value;
  if (String(raw).trim() === "") return NaN;
  const value = Number(raw);
  return Number.isFinite(value) ? value : NaN;
}}
function filterString(id) {{
  const element = document.getElementById(id);
  if (!element) return "";
  return String(element.value || "").trim();
}}
function passesLower(value, lower) {{
  return !Number.isFinite(lower) || (Number.isFinite(value) && value >= lower);
}}
function passesUpper(value, upper) {{
  return !Number.isFinite(upper) || (Number.isFinite(value) && value <= upper);
}}
function readFilter() {{
  return {{
    skewMin: filterValue("skewMin"), skewMax: filterValue("skewMax"),
    maxConnect: filterValue("maxConnect"),
    aspectMin: filterValue("aspectMin"), aspectMax: filterValue("aspectMax"),
    compactMin: filterValue("compactMin"), compactMax: filterValue("compactMax"),
    roiAreaMin: filterValue("roiAreaMin"), roiAreaMax: filterValue("roiAreaMax"),
    eventSnrMin: filterValue("eventSnrMin"),
    andreaPostdocSnrMin: filterValue("andreaPostdocSnrMin"),
    oasisEventSnrMin: filterValue("oasisEventSnrMin"),
    oasisRiseTauMin: filterValue("oasisRiseTauMin"), oasisRiseTauMax: filterValue("oasisRiseTauMax"),
    oasisDecayTauMin: filterValue("oasisDecayTauMin"), oasisDecayTauMax: filterValue("oasisDecayTauMax"),
    cellTypeFilter: readCategoryFilter(),
    roiModelScoreProbabilityMin: filterValue("roiModelScoreProbabilityMin"),
    roiModelScoreProbabilityMax: filterValue("roiModelScoreProbabilityMax"),
    customMetrics: readCustomMetricFilters(),
  }};
}}
function readCustomMetricFilters() {{
  const filters = {{}};
  for (const definition of customMetricDefinitions()) {{
    const ids = customMetricFilterIds(definition.key);
    filters[definition.key] = {{min: filterValue(ids.min), max: filterValue(ids.max)}};
  }}
  return filters;
}}
function normalizeFilter(filter) {{
  const normalized = {{}};
  for (const key of ["skewMin","skewMax","maxConnect","aspectMin","aspectMax","compactMin","compactMax"]) {{
    if (filter[key] === null || filter[key] === undefined || String(filter[key]).trim() === "") {{
      normalized[key] = null;
      continue;
    }}
    const value = Number(filter[key]);
    normalized[key] = Number.isFinite(value) ? value : null;
  }}
  for (const key of ["eventSnrMin","eventSnrMax","andreaPostdocSnrMin","andreaPostdocSnrMax","roiAreaMin","roiAreaMax","oasisEventSnrMin","oasisRiseTauMin","oasisRiseTauMax","oasisDecayTauMin","oasisDecayTauMax","roiModelScoreProbabilityMin","roiModelScoreProbabilityMax"]) {{
    if (filter[key] === null || filter[key] === undefined || String(filter[key]).trim() === "") {{
      normalized[key] = null;
      continue;
    }}
    const value = Number(filter[key]);
    normalized[key] = Number.isFinite(value) ? value : null;
    if ((key.endsWith("Max")) && normalized[key] === 0) normalized[key] = null;
  }}
  const rawCellTypeFilter = Array.isArray(filter.cellTypeFilter)
    ? filter.cellTypeFilter
    : (filter.cellTypeFilter === null || filter.cellTypeFilter === undefined || String(filter.cellTypeFilter).trim() === "" ? [] : [filter.cellTypeFilter]);
  normalized.cellTypeFilter = rawCellTypeFilter.map(value => String(value)).filter(value => value.trim() !== "");
  normalized.customMetrics = {{}};
  const customMetrics = filter.customMetrics || {{}};
  for (const definition of customMetricDefinitions()) {{
    const raw = customMetrics[definition.key] || {{}};
    const min = raw.min === null || raw.min === undefined || String(raw.min).trim() === "" ? NaN : Number(raw.min);
    const max = raw.max === null || raw.max === undefined || String(raw.max).trim() === "" ? NaN : Number(raw.max);
    normalized.customMetrics[definition.key] = {{
      min: Number.isFinite(min) ? min : null,
      max: Number.isFinite(max) ? max : null,
    }};
  }}
  return normalized;
}}
function writeFilter(filter) {{
  const normalized = normalizeFilter(filter);
  for (const [id, value] of Object.entries(normalized)) {{
    const input = document.getElementById(id);
    if (input) input.value = value === null || value === undefined ? "" : value;
  }}
  for (const [key, bounds] of Object.entries(normalized.customMetrics || {{}})) {{
    const ids = customMetricFilterIds(key);
    const minInput = document.getElementById(ids.min);
    const maxInput = document.getElementById(ids.max);
    if (minInput) minInput.value = bounds.min === null || bounds.min === undefined ? "" : bounds.min;
    if (maxInput) maxInput.value = bounds.max === null || bounds.max === undefined ? "" : bounds.max;
  }}
  writeCategoryFilter(normalized.cellTypeFilter || []);
  evaluateFilter();
}}
function populatePresetSelect(selectedName = data.targetStructure) {{
  const select = document.getElementById("filterPreset");
  select.textContent = "";
  for (const name of Object.keys(data.morphologyPresets)) {{
    const option = document.createElement("option"); option.value = `built-in:${{name}}`; option.textContent = targetStructureLabel(name); select.appendChild(option);
  }}
  for (const name of Object.keys(customPresets).sort()) {{
    const option = document.createElement("option"); option.value = `custom:${{name}}`; option.textContent = `${{name}} (saved)`; select.appendChild(option);
  }}
  const desired = data.morphologyPresets[selectedName] ? `built-in:${{selectedName}}` : `custom:${{selectedName}}`;
  if ([...select.options].some(option => option.value === desired)) select.value = desired;
}}
function loadSelectedPreset() {{
  const [kind, name] = document.getElementById("filterPreset").value.split(":", 2);
  const preset = kind === "built-in" ? data.morphologyPresets[name] : customPresets[name];
  if (preset) {{
    updateMetricDefaults();
    writeFilter(preset);
  }}
}}
function currentPresetName() {{
  return document.getElementById("presetName").value.trim();
}}
function saveCurrentPresetToPage() {{
  const name = currentPresetName();
  if (!name) {{ alert("Enter a filter set name first."); return null; }}
  try {{
    customPresets[name] = normalizeFilter(readFilter());
  }} catch (error) {{
    alert(`Could not save the filter set: ${{error.message}}`);
    return null;
  }}
  populatePresetSelect(name);
  document.getElementById("filterPreset").value = `custom:${{name}}`;
  updateMetricDefaults();
  drawMetricHistograms();
  return name;
}}
function passesFilter(roi, metrics, filter) {{
  const dffMetrics = data.dffMetrics[roi];
  const snr9550 = dffMetric(roi, "snr_95_50", "event_snr");
  const postdocSnr = dffMetric(roi, "andrea_postdoc_snr");
  const oasisEventSnr = dffMetricValue(dffMetrics, "oasis_event_snr");
  const oasisRiseTau = dffMetricValue(dffMetrics, "oasis_rise_tau_seconds");
  const oasisDecayTau = dffMetricValue(dffMetrics, "oasis_decay_tau_seconds");
  const roiModelProbability = metricValue(roi, "roi_model_score_probability");
  const cellTypeFilter = Array.isArray(filter.cellTypeFilter) ? filter.cellTypeFilter : [];
  const cellTypeValue = data.cellTypeLabels?.[roi];
  const cellTypePass = !cellTypeFilter.length || cellTypeFilter.some(value =>
    value === "not_loaded" ? (cellTypeValue === null || cellTypeValue === undefined) : Number(cellTypeValue) === Number(value)
  );
  const customMetricPass = customMetricDefinitions().every(definition => {{
    const bounds = filter.customMetrics?.[definition.key] || {{}};
    const value = customMetricValue(roi, definition.key);
    return passesLower(value, bounds.min) && passesUpper(value, bounds.max);
  }});
  return (
    passesLower(metrics.skew, filter.skewMin) && passesUpper(metrics.skew, filter.skewMax) &&
    passesLower(metrics.aspect, filter.aspectMin) && passesUpper(metrics.aspect, filter.aspectMax) &&
    passesLower(metrics.compact, filter.compactMin) && passesUpper(metrics.compact, filter.compactMax) &&
    passesUpper(metrics.connect, filter.maxConnect) &&
    passesLower(dffMetrics.roi_area, filter.roiAreaMin) &&
    passesUpper(dffMetrics.roi_area, filter.roiAreaMax) &&
    passesLower(snr9550, filter.eventSnrMin) &&
    passesLower(postdocSnr, filter.andreaPostdocSnrMin) &&
    passesLower(oasisEventSnr, filter.oasisEventSnrMin) &&
    passesLower(oasisRiseTau, filter.oasisRiseTauMin) &&
    passesUpper(oasisRiseTau, filter.oasisRiseTauMax) &&
    passesLower(oasisDecayTau, filter.oasisDecayTauMin) &&
    passesUpper(oasisDecayTau, filter.oasisDecayTauMax) &&
    passesLower(roiModelProbability, filter.roiModelScoreProbabilityMin) &&
    passesUpper(roiModelProbability, filter.roiModelScoreProbabilityMax) &&
    cellTypePass &&
    customMetricPass
  );
}}
function morphologyReasons(metrics, dffMetrics, filter, roi = null) {{
  const reasons = [];
  const snr9550 = dffMetrics.snr_95_50 ?? dffMetrics.event_snr;
  const postdocSnr = dffMetrics.andrea_postdoc_snr;
  const oasisEventSnr = dffMetricValue(dffMetrics, "oasis_event_snr");
  const oasisRiseTau = dffMetricValue(dffMetrics, "oasis_rise_tau_seconds");
  const oasisDecayTau = dffMetricValue(dffMetrics, "oasis_decay_tau_seconds");
  const roiModelProbability = metricValue(roi, "roi_model_score_probability");
  const cellTypeFilter = Array.isArray(filter.cellTypeFilter) ? filter.cellTypeFilter : [];
  if (roi === null || roi === undefined) roi = data.dffMetrics.indexOf(dffMetrics);
  if (!passesLower(metrics.skew, filter.skewMin)) reasons.push(`skew ${{fmt(metrics.skew)}} below ${{filter.skewMin}}`);
  if (!passesUpper(metrics.skew, filter.skewMax)) reasons.push(`skew ${{fmt(metrics.skew)}} above ${{filter.skewMax}}`);
  if (!passesLower(metrics.aspect, filter.aspectMin)) reasons.push(`aspect_ratio ${{fmt(metrics.aspect)}} below ${{filter.aspectMin}}`);
  if (!passesUpper(metrics.aspect, filter.aspectMax)) reasons.push(`aspect_ratio ${{fmt(metrics.aspect)}} above ${{filter.aspectMax}}`);
  if (!passesLower(metrics.compact, filter.compactMin)) reasons.push(`compact ${{fmt(metrics.compact)}} below ${{filter.compactMin}}`);
  if (!passesUpper(metrics.compact, filter.compactMax)) reasons.push(`compact ${{fmt(metrics.compact)}} above ${{filter.compactMax}}`);
  if (!passesUpper(metrics.connect, filter.maxConnect)) reasons.push(`connectivity ${{metrics.connect}} exceeds ${{filter.maxConnect}}`);
  if (!passesLower(dffMetrics.roi_area, filter.roiAreaMin)) reasons.push(`ROI area ${{fmt(dffMetrics.roi_area)}} below ${{filter.roiAreaMin}}`);
  if (!passesUpper(dffMetrics.roi_area, filter.roiAreaMax)) reasons.push(`ROI area ${{fmt(dffMetrics.roi_area)}} above ${{filter.roiAreaMax}}`);
  if (!passesLower(snr9550, filter.eventSnrMin)) reasons.push(`SNR: 95/50 percentile ${{fmt(snr9550)}} below ${{filter.eventSnrMin}}`);
  if (!passesLower(postdocSnr, filter.andreaPostdocSnrMin)) reasons.push(`SNR: CaImAn (large-transient score) ${{fmt(postdocSnr)}} below ${{filter.andreaPostdocSnrMin}}`);
  if (!passesLower(oasisEventSnr, filter.oasisEventSnrMin)) reasons.push(`inferred spike SNR ${{fmt(oasisEventSnr)}} below ${{filter.oasisEventSnrMin}}`);
  if (!passesLower(oasisRiseTau, filter.oasisRiseTauMin)) reasons.push(`inferred spike rise tau ${{fmt(oasisRiseTau)}} below ${{filter.oasisRiseTauMin}}`);
  if (!passesUpper(oasisRiseTau, filter.oasisRiseTauMax)) reasons.push(`inferred spike rise tau ${{fmt(oasisRiseTau)}} above ${{filter.oasisRiseTauMax}}`);
  if (!passesLower(oasisDecayTau, filter.oasisDecayTauMin)) reasons.push(`inferred spike decay tau ${{fmt(oasisDecayTau)}} below ${{filter.oasisDecayTauMin}}`);
  if (!passesUpper(oasisDecayTau, filter.oasisDecayTauMax)) reasons.push(`inferred spike decay tau ${{fmt(oasisDecayTau)}} above ${{filter.oasisDecayTauMax}}`);
  if (!passesLower(roiModelProbability, filter.roiModelScoreProbabilityMin)) reasons.push(`ROI model probability ${{fmt(roiModelProbability)}} below ${{filter.roiModelScoreProbabilityMin}}`);
  if (!passesUpper(roiModelProbability, filter.roiModelScoreProbabilityMax)) reasons.push(`ROI model probability ${{fmt(roiModelProbability)}} above ${{filter.roiModelScoreProbabilityMax}}`);
  if (cellTypeFilter.length) {{
    const cellTypeValue = data.cellTypeLabels?.[roi];
    const passesCellType = cellTypeFilter.some(value =>
      value === "not_loaded" ? (cellTypeValue === null || cellTypeValue === undefined) : Number(cellTypeValue) === Number(value)
    );
    if (!passesCellType) {{
      const selectedLabels = cellTypeFilter.map(value => value === "not_loaded" ? "not loaded" : cellTypeLabelName(Number(value))).join(", ");
      reasons.push(`cell identity ${{cellTypeLabelName(cellTypeValue)}} does not match selected cell identities (${{selectedLabels}})`);
    }}
  }}
  for (const definition of customMetricDefinitions()) {{
    const bounds = filter.customMetrics?.[definition.key] || {{}};
    const value = customMetricValue(roi, definition.key);
    const min = bounds.min, max = bounds.max;
    if (!passesLower(value, min)) reasons.push(`${{definition.label}} ${{fmt(value)}} below ${{min}}`);
    if (!passesUpper(value, max)) reasons.push(`${{definition.label}} ${{fmt(value)}} above ${{max}}`);
  }}
  return reasons;
}}
function evaluateFilter() {{
  const filter = readFilter();
  let pass = 0;
  for (let roi = 0; roi < data.nRois; roi++) {{
    filterPass[roi] = passesFilter(roi, data.morphology[roi], filter) ? 1 : 0;
    pass += filterPass[roi];
  }}
  const summary = `${{pass}} / ${{data.nRois}} original Suite2p ROIs pass the current ROI metric filters.`;
  document.getElementById("filterSummary").textContent = summary;
  document.getElementById("filterSummaryInline").textContent = summary;
  drawMetricHistograms();
  draw();
}}
function resetFilter() {{
  loadSelectedPreset();
}}
function applyFilterToLabels() {{
  const current = selected;
  for (let roi = 0; roi < data.nRois; roi++) {{
    labels[roi] = filterPass[roi] ? -1 : 0;
  }}
  updateVisibleRois(true);
  if (visibleRois.includes(current)) setSelected(current);
}}
function setLabel(label) {{
  labels[selected] = label;
  updateLabelControls();
  updateVisibleRois();
}}
function moveVisible(direction) {{
  if (!visibleRois.length) return;
  const currentIndex = visibleRois.indexOf(selected);
  const origin = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = (origin + direction + visibleRois.length) % visibleRois.length;
  setSelected(visibleRois[nextIndex]);
}}
function appendFunctionalRois(svg) {{
  data.rois.forEach(r => {{
    const hitPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    hitPath.setAttribute("d", r.hitPath || r.path);
    hitPath.dataset.roi = r.roi; hitPath.classList.add("roi-hit", "functional-overlay-item"); hitPath.addEventListener("click", () => setSelected(r.roi)); svg.appendChild(hitPath);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", r.path);
    path.dataset.roi = r.roi; path.classList.add("roi", "functional-overlay-item"); svg.appendChild(path);
  }});
}}
function appendCellposeRois(svg) {{
  (data.cellposeRois || []).forEach(r => {{
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", r.path);
    path.dataset.cellposeRoi = r.label;
    path.classList.add("cellpose-roi", "cellpose-overlay-item");
    path.style.display = "none";
    svg.appendChild(path);
  }});
}}
function setRedOverlayMode(mode) {{
  const redOverlay = document.querySelector(".red-overlay");
  if (!redOverlay) return;
  const showCellpose = mode === "cellpose" && data.cellposeMaskAvailable;
  redOverlay.querySelectorAll(".functional-overlay-item").forEach(path => {{
    const roi = Number(path.dataset.roi);
    path.style.display = (!showCellpose && functionalRoiVisible(roi)) ? "" : "none";
  }});
  redOverlay.querySelectorAll(".cellpose-overlay-item").forEach(path => {{ path.style.display = showCellpose ? "" : "none"; }});
}}
function makeOverlays() {{
  document.querySelectorAll(".overlay").forEach(svg => {{
    appendFunctionalRois(svg);
    if (svg.classList.contains("red-overlay")) appendCellposeRois(svg);
  }});
  const redOverlayMode = document.getElementById("redOverlayMode");
  if (redOverlayMode) {{
    redOverlayMode.disabled = !data.cellposeMaskAvailable;
    redOverlayMode.title = data.cellposeMaskAvailable ? "" : "Cellpose anatomical masks were not available in the generated HTML.";
    redOverlayMode.addEventListener("change", event => setRedOverlayMode(event.target.value));
    setRedOverlayMode(redOverlayMode.value);
  }}
}}
const fovZoom = {{ scale: 1, x: 0, y: 0, dragging: false, lastX: 0, lastY: 0 }};
function applyFovTransform() {{
  const transform = `translate(${{fovZoom.x}}px, ${{fovZoom.y}}px) scale(${{fovZoom.scale}})`;
  document.querySelectorAll(".fov-review .imagewrap img, .fov-review .imagewrap svg").forEach(el => {{
    el.style.transform = transform;
  }});
}}
function clampFovPan() {{
  const wrap = document.querySelector(".fov-review .imagewrap");
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  const minX = rect.width * (1 - fovZoom.scale);
  const minY = rect.height * (1 - fovZoom.scale);
  fovZoom.x = Math.min(0, Math.max(minX, fovZoom.x));
  fovZoom.y = Math.min(0, Math.max(minY, fovZoom.y));
}}
function resetFovZoom() {{
  fovZoom.scale = 1; fovZoom.x = 0; fovZoom.y = 0; applyFovTransform();
}}
function setupFovZoom() {{
  document.querySelectorAll(".fov-review .imagewrap").forEach(wrap => {{
    wrap.addEventListener("wheel", event => {{
      event.preventDefault();
      const rect = wrap.getBoundingClientRect();
      const oldScale = fovZoom.scale;
      const nextScale = Math.min(8, Math.max(1, oldScale * (event.deltaY < 0 ? 1.18 : 1 / 1.18)));
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      fovZoom.x = px - (px - fovZoom.x) * (nextScale / oldScale);
      fovZoom.y = py - (py - fovZoom.y) * (nextScale / oldScale);
      fovZoom.scale = nextScale;
      clampFovPan();
      applyFovTransform();
    }}, {{passive: false}});
    wrap.addEventListener("mousedown", event => {{
      if (event.button !== 0) return;
      fovZoom.dragging = true; fovZoom.lastX = event.clientX; fovZoom.lastY = event.clientY;
      wrap.classList.add("panning");
    }});
    wrap.addEventListener("dblclick", resetFovZoom);
  }});
  window.addEventListener("mousemove", event => {{
    if (!fovZoom.dragging) return;
    fovZoom.x += event.clientX - fovZoom.lastX;
    fovZoom.y += event.clientY - fovZoom.lastY;
    fovZoom.lastX = event.clientX; fovZoom.lastY = event.clientY;
    clampFovPan();
    applyFovTransform();
  }});
  window.addEventListener("mouseup", () => {{
    if (!fovZoom.dragging) return;
    fovZoom.dragging = false;
    document.querySelectorAll(".fov-review .imagewrap").forEach(wrap => wrap.classList.remove("panning"));
  }});
}}
setupFovZoom();
function drawAxes(ctx, w, h, l, t, pw, ph, xLabel, yLabel, options = {{}}) {{
  // Axis labels start uppercase, except the dF/F notation.
  const cap = s => (s && !/^dF/.test(s)) ? s.charAt(0).toUpperCase() + s.slice(1) : s;
  xLabel = cap(xLabel); yLabel = cap(yLabel);
  ctx.strokeStyle = "#d0d5dd"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(l,t); ctx.lineTo(l,t+ph); ctx.lineTo(l+pw,t+ph); ctx.stroke();
  const xLabelY = options.xLabelY ?? h - 8;
  const yLabelX = options.yLabelX ?? 14;
  ctx.fillStyle = "#475467"; ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText(xLabel, l + pw / 2, xLabelY);
  ctx.save(); ctx.translate(yLabelX, t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
}}
function tickStep(seconds) {{
  const options = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600];
  for (const step of options) if (seconds / step <= 8) return step;
  return 1200;
}}
function timeLabel(seconds, majorStep) {{
  if (majorStep < 1) return `${{Math.round(seconds * 1000)}} ms`;
  if (seconds < 60) return `${{seconds.toFixed(majorStep < 2 ? 1 : 0)}} s`;
  return `${{(seconds / 60).toFixed(1)}} min`;
}}
function numericTickStep(span, target = 5) {{
  if (!Number.isFinite(span) || span <= 0) return 1;
  const raw = span / Math.max(1, target);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const scaled = raw / pow;
  const mult = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return mult * pow;
}}
function numericLabel(value, step) {{
  if (!Number.isFinite(value)) return "";
  const decimals = Math.max(0, Math.min(4, Math.ceil(-Math.log10(Math.abs(step || 1))) + 1));
  return Math.abs(value) >= 1000 ? value.toExponential(1) : value.toFixed(decimals);
}}
function drawNumericTicks(ctx, l, t, pw, ph, xMin, xMax, yMin, yMax, xTicks = 5, yTicks = 4) {{
  ctx.save();
  ctx.strokeStyle = "#e5e7eb";
  ctx.fillStyle = "#475467";
  ctx.font = `${{10 * (window.devicePixelRatio || 1)}}px Arial`;
  ctx.lineWidth = 1;
  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  const xStep = numericTickStep(xMax - xMin, xTicks);
  for (let value = Math.ceil(xMin / xStep) * xStep; value <= xMax + xStep * 0.5; value += xStep) {{
    const x = l + (value - xMin) / (xMax - xMin) * pw;
    ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
    ctx.fillText(numericLabel(value, xStep), x, t + ph + 8);
  }}
  const yStep = numericTickStep(yMax - yMin, yTicks);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let value = Math.ceil(yMin / yStep) * yStep; value <= yMax + yStep * 0.5; value += yStep) {{
    const y = t + (1 - (value - yMin) / (yMax - yMin)) * ph;
    ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + pw, y); ctx.stroke();
    ctx.fillText(numericLabel(value, yStep), l - 8, y);
  }}
  ctx.restore();
}}
function drawTimeGrid(ctx, l, t, pw, ph) {{
  const startSec = x0 / data.frameRate, endSec = x1 / data.frameRate, spanSec = Math.max(1e-9, endSec - startSec);
  const major = tickStep(spanSec), minor = major >= 1 ? major / 10 : major / 5;
  function xOfSec(sec) {{ return l + (sec - startSec) / spanSec * pw; }}
  ctx.save();
  ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
  if (spanSec <= 20) {{
    ctx.strokeStyle = "#f1f5f9"; ctx.lineWidth = 1;
    for (let sec = Math.ceil(startSec / minor) * minor; sec <= endSec; sec += minor) {{
      const x = xOfSec(sec); ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
    }}
  }}
  ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 1;
  const labelTicks = [];
  for (let sec = Math.ceil(startSec / major) * major; sec <= endSec; sec += major) {{
    const x = xOfSec(sec); ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
    labelTicks.push([x, timeLabel(sec, major)]);
  }}
  ctx.restore();
  ctx.fillStyle = "#475467"; ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.textBaseline = "top";
  labelTicks.forEach(([x, label]) => ctx.fillText(label, x, t + ph + 8));
}}
function colorForRoi(roi) {{
  const palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#2563eb", "#dc2626", "#059669", "#ca8a04", "#7c3aed", "#0891b2", "#be123c", "#4d7c0f", "#c2410c", "#4338ca"];
  return palette[roi % palette.length];
}}
function drawStack() {{
  const firstRow = Math.max(0, Math.floor(y0));
  const lastRow = Math.min(visibleRois.length - 1, Math.ceil(y1));
  const selectedPosition = visibleRois.includes(selected) ? visibleRois.indexOf(selected) + 1 : 0;
  const selectedWhere = selectedPosition
    ? `selected ROI at sort position ${{selectedPosition}} of ${{visibleRois.length}}`
    : `selected ROI not in the current view`;
  document.getElementById("stackTitle").textContent = `Stacked raw dF/F, ${{selectedWhere}}, sorted by ${{sortMetricText()}}, ${{sortDirectionText()}}`;
  const canvas = document.getElementById("stackCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 62, r = 16, t = 14, b = 56, pw = w-l-r, ph = h-t-b;
  ctx.clearRect(0,0,w,h); ctx.fillStyle = "#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)","Current Sort Order");
  if (!visibleRois.length) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText("No ROIs match the current metric and label display filters.", l + pw / 2, t + ph / 2);
    return;
  }}
  if (!dff) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText("Load dF/F and inferred spikes files to view stacked traces (see (i) for auto loading info)", l + pw / 2, t + ph / 2);
    return;
  }}
  const ys = Math.max(0, Math.floor(y0)), ye = Math.min(visibleRois.length - 1, Math.ceil(y1));
  const xs = Math.max(0, Math.floor(x0)), xe = Math.min(data.nFrames - 1, Math.ceil(x1));
  drawTimeGrid(ctx, l, t, pw, ph);
  const count = Math.max(1, ye - ys + 1), rowH = ph / (count + 1), pixelCount = Math.max(1, Math.floor(pw));
  let amplitudes = [];
  for (let rowIndex = ys; rowIndex <= ye; rowIndex++) {{
    const roi = visibleRois[rowIndex];
    const tr = trace(roi);
    let sum = 0, n = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ sum += v; n++; }} }}
    const center = n ? sum / n : 0;
    let dev = 0, dn = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ dev += Math.abs(v - center); dn++; }} }}
    amplitudes.push(Math.max(dn ? dev / dn : 0, 0.2));
  }}
  const sortedAmp = amplitudes.slice().sort((a,b) => a-b);
  const typicalAmp = Math.max(sortedAmp[Math.floor(sortedAmp.length / 2)] || 1, 0.2);
  const scale = (rowH * 0.24) / typicalAmp;
  const rowLabels = [];
  ctx.save(); ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
  for (let rowIndex = ys; rowIndex <= ye; rowIndex++) {{
    const roi = visibleRois[rowIndex], tr = trace(roi), row = rowIndex - ys, baseline = t + rowH * (row + 1), color = colorForRoi(roi);
    rowLabels.push({{roi, rowIndex, baseline, color}});
    ctx.strokeStyle = roi === selected ? "#111827" : color;
    ctx.lineWidth = roi === selected ? Math.max(1.8, 1.8 * (window.devicePixelRatio || 1)) : Math.max(0.8, window.devicePixelRatio || 1);
    ctx.beginPath();
    let first = true;
    let sum = 0, n = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ sum += v; n++; }} }}
    const center = n ? sum / n : 0;
    const framesPerPixel = (xe - xs + 1) / pixelCount;
    if (framesPerPixel <= 1.2) {{
      for (let f = xs; f <= xe; f++) {{
        const x = l + (f - x0) / (x1 - x0) * pw;
        const y = baseline - (tr[f] - center) * scale;
        if (first) {{ ctx.moveTo(x, y); first = false; }} else ctx.lineTo(x, y);
      }}
    }} else {{
      for (let px = 0; px < pixelCount; px++) {{
        const f0 = Math.max(xs, Math.floor(xs + px * framesPerPixel));
        const f1 = Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel));
        let minV = Infinity, maxV = -Infinity;
        for (let f = f0; f <= f1; f++) {{
          const v = tr[f];
          if (Number.isFinite(v)) {{ minV = Math.min(minV, v); maxV = Math.max(maxV, v); }}
        }}
        if (!Number.isFinite(minV)) continue;
        const x = l + px;
        ctx.moveTo(x, baseline - (minV - center) * scale);
        ctx.lineTo(x, baseline - (maxV - center) * scale);
      }}
    }}
    ctx.stroke();
  }}
  ctx.restore();
  if (count <= 80) {{
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
    for (const rowLabel of rowLabels) {{
      ctx.fillStyle = rowLabel.roi === selected ? "#111827" : rowLabel.color;
      ctx.fillText(String(rowLabel.rowIndex + 1), l - 8, rowLabel.baseline);
    }}
  }}
}}
function drawTrace() {{
  const canvas = document.getElementById("traceCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w=canvas.width,h=canvas.height,l=62,r=16,t=14,b=56,pw=w-l-r,ph=h-t-b, tr=trace(selected);
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)","dF/F"); drawTimeGrid(ctx,l,t,pw,ph);
  if (!tr) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText("Load dF/F and inferred spikes files to view for selected ROI (see (i) for auto loading info)", l + pw / 2, t + ph / 2);
    return;
  }}
  const xs=Math.max(0,Math.floor(x0)), xe=Math.min(data.nFrames-1,Math.ceil(x1)); let lo=Infinity,hi=-Infinity;
  for (let f=xs; f<=xe; f++) {{ const v=tr[f]; if (Number.isFinite(v)) {{ lo=Math.min(lo,v); hi=Math.max(hi,v); }} }}
  if (!Number.isFinite(lo) || hi<=lo) {{ lo=-1; hi=1; }} const pad=(hi-lo)*.08||1; lo-=pad; hi+=pad;
  const xOf=f=>l+(f-x0)/(x1-x0)*pw, yOf=v=>t+(1-(v-lo)/(hi-lo))*ph;
  ctx.strokeStyle="#1d4ed8"; ctx.lineWidth=Math.max(1,window.devicePixelRatio||1); ctx.beginPath();
  const pixelCount = Math.max(1, Math.floor(pw)), framesPerPixel = (xe - xs + 1) / pixelCount;
  if (framesPerPixel <= 1.2) {{
    let first=true; for (let f=xs; f<=xe; f++) {{ const x=xOf(f), y=yOf(tr[f]); if (first) {{ ctx.moveTo(x,y); first=false; }} else ctx.lineTo(x,y); }}
  }} else {{
    for (let px=0; px<pixelCount; px++) {{
      const f0=Math.max(xs, Math.floor(xs + px * framesPerPixel)), f1=Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel));
      let minV=Infinity, maxV=-Infinity;
      for (let f=f0; f<=f1; f++) {{ const v=tr[f]; if (Number.isFinite(v)) {{ minV=Math.min(minV,v); maxV=Math.max(maxV,v); }} }}
      if (!Number.isFinite(minV)) continue;
      const x=l+px; ctx.moveTo(x, yOf(minV)); ctx.lineTo(x, yOf(maxV));
    }}
  }}
  ctx.stroke();
  const sp = showInferredSpikes ? spikeTrace(selected) : null;
  if (sp) {{
    ctx.save();
    ctx.fillStyle = "rgba(220,38,38,.9)";
    const radius = Math.max(2.4, 2.4 * (window.devicePixelRatio || 1));
    for (let f=xs; f<=xe; f++) {{
      const s = sp[f];
      if (!Number.isFinite(s) || s <= oasisEventThreshold) continue;
      const x = xOf(f);
      const y = yOf(tr[f]);
      if (!Number.isFinite(y)) continue;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();
  }}
  ctx.fillStyle="#475467"; ctx.textAlign="right"; ctx.textBaseline="middle"; for (let i=0; i<=4; i++) {{ const v=lo+i/4*(hi-lo); ctx.fillText(v.toFixed(2), l-8, yOf(v)); }}
}}
function motionFiniteValues(values) {{
  if (!values) return [];
  const out = [];
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) out.push(values[i]);
  return out;
}}
function drawMotionDrift() {{
  const canvas = document.getElementById("motionDriftCanvas");
  fit(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 62, r = 18, t = 18, b = 56;
  const gap = 30 * (window.devicePixelRatio || 1);
  const panelH = (h - t - b - gap) / 2;
  const pw = w - l - r;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  if (!motionX || !motionY) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center";
    ctx.fillText("Motion shifts not available for this session.", l + pw / 2, h / 2);
    return;
  }}
  const n = Math.min(motionX.length, motionY.length, data.nFrames);
  const xs = Math.max(0, Math.min(n - 1, Math.floor(x0)));
  const xe = Math.max(xs, Math.min(n - 1, Math.ceil(x1)));
  function drawPanel(values, top, title, color, showXLabel = false) {{
    const ph = panelH;
    let lo = Infinity, hi = -Infinity;
    for (let f = xs; f <= xe; f++) {{
      const value = values[f];
      if (Number.isFinite(value)) {{ lo = Math.min(lo, value); hi = Math.max(hi, value); }}
    }}
    if (!Number.isFinite(lo) || hi <= lo) {{ lo = -1; hi = 1; }}
    const pad = Math.max((hi - lo) * 0.08, 0.5); lo -= pad; hi += pad;
    const xOf = f => l + (f - x0) / (x1 - x0) * pw;
    const yOf = v => top + (1 - (v - lo) / (hi - lo)) * ph;
    drawAxes(ctx, w, h, l, top, pw, ph, showXLabel ? "time (s)" : "", "shift (px)", {{ xLabelY: top + ph + 34 * (window.devicePixelRatio || 1) }});
    drawTimeGrid(ctx, l, top, pw, ph);
    const zeroY = yOf(0);
    if (zeroY >= top && zeroY <= top + ph) {{
      ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(l, zeroY); ctx.lineTo(l + pw, zeroY); ctx.stroke();
    }}
    ctx.save();
    ctx.beginPath(); ctx.rect(l, top, pw, ph); ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
    ctx.beginPath();
    let first = true;
    const pixelCount = Math.max(1, Math.floor(pw));
    const framesPerPixel = (xe - xs + 1) / pixelCount;
    if (framesPerPixel <= 1.2) {{
      for (let f = xs; f <= xe; f++) {{
        const value = values[f];
        if (!Number.isFinite(value)) continue;
        const x = xOf(f), y = yOf(value);
        if (first) {{ ctx.moveTo(x, y); first = false; }} else ctx.lineTo(x, y);
      }}
    }} else {{
      const meanPoints = [];
      for (let px = 0; px < pixelCount; px++) {{
        const f0 = Math.max(xs, Math.floor(xs + px * framesPerPixel));
        const f1 = Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel));
        let minV = Infinity, maxV = -Infinity, sumV = 0, nV = 0;
        for (let f = f0; f <= f1; f++) {{
          const value = values[f];
          if (Number.isFinite(value)) {{ minV = Math.min(minV, value); maxV = Math.max(maxV, value); sumV += value; nV++; }}
        }}
        if (!Number.isFinite(minV)) continue;
        const x = l + px;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.18;
        ctx.beginPath();
        ctx.moveTo(x, yOf(minV)); ctx.lineTo(x, yOf(maxV));
        ctx.stroke();
        ctx.restore();
        meanPoints.push([x, yOf(sumV / nV)]);
      }}
      ctx.beginPath();
      for (let i = 0; i < meanPoints.length; i++) {{
        const [x, y] = meanPoints[i];
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
    }}
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Arial`;
    ctx.fillText(title, l + 8, top + 8);
    ctx.fillStyle = "#475467"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i <= 3; i++) {{
      const value = lo + i / 3 * (hi - lo);
      ctx.fillText(value.toFixed(1), l - 8, yOf(value));
    }}
  }}
  drawPanel(motionX, t, "X shift per frame", "#2563eb", false);
  drawPanel(motionY, t + panelH + gap, "Y shift per frame", "#dc2626", true);
}}
function drawMotionDistribution() {{
  const canvas = document.getElementById("motionDistributionCanvas");
  fit(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 74, r = 18, t = 30, b = 74, pw = w - l - r, ph = h - t - b;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  const xValues = motionFiniteValues(motionX);
  const yValues = motionFiniteValues(motionY);
  if (!xValues.length || !yValues.length) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center";
    ctx.fillText("Motion shifts not available for this session.", l + pw / 2, t + ph / 2);
    return;
  }}
  const gap = 46 * (window.devicePixelRatio || 1);
  const panelW = (pw - 2 * gap) / 3;
  const absXValues = xValues.map(Math.abs);
  const absYValues = yValues.map(Math.abs);
  function drawPanelAxes(left, width, xLabel, yLabel) {{
    ctx.strokeStyle = "#d0d5dd";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, t);
    ctx.lineTo(left, t + ph);
    ctx.lineTo(left + width, t + ph);
    ctx.stroke();
    ctx.fillStyle = "#475467";
    ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(xLabel, left + width / 2, h - 26);
    ctx.save();
    ctx.translate(left - 41, t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }}
  function panelLimits(valuesA, valuesB, absolute = false) {{
    const all = valuesA.concat(valuesB).filter(Number.isFinite).sort((a, b) => a - b);
    if (!all.length) return absolute ? [0, 1] : [-1, 1];
    let lo = absolute ? 0 : all[0];
    let hi = all[all.length - 1];
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return absolute ? [0, 1] : [-1, 1];
    if (hi <= lo) hi = lo + 1;
    const pad = Math.max((hi - lo) * 0.08, 0.5);
    if (absolute) {{
      hi += pad;
    }} else {{
      lo -= pad;
      hi += pad;
    }}
    return [lo, hi];
  }}
  function histogram(values, lo, hi, bins) {{
    const counts = new Array(bins).fill(0);
    const step = (hi - lo) / bins;
    for (const value of values) {{
      if (!Number.isFinite(value) || value < lo || value > hi) continue;
      const index = Math.min(bins - 1, Math.max(0, Math.floor((value - lo) / step)));
      counts[index] += 1;
    }}
    const total = counts.reduce((sum, count) => sum + count, 0);
    return counts.map(count => total ? count / total : 0);
  }}
  function drawHistogramPanel(left, width, title, valuesA, valuesB, labelA, labelB) {{
    const [lo, hi] = panelLimits(valuesA, valuesB, false);
    const bins = 64;
    const histA = histogram(valuesA, lo, hi, bins);
    const histB = histogram(valuesB, lo, hi, bins);
    const yMax = Math.max(...histA, ...histB, 1e-9) * 1.12;
    const xOf = value => left + (value - lo) / (hi - lo) * width;
    const yOf = value => t + (1 - value / yMax) * ph;
    drawPanelAxes(left, width, "Shift (px)", "Fraction / bin");
    drawNumericTicks(ctx, left, t, width, ph, lo, hi, 0, yMax, 7, 4);
    ctx.fillStyle = "#475467"; ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(title, left + width / 2, 5);
    function drawBars(hist, color, label, yLabelOffset, offsetFraction) {{
      const binWidth = width / bins;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.34;
      for (let i = 0; i < hist.length; i++) {{
        const x0b = xOf(lo + i / bins * (hi - lo));
        const y = yOf(hist[i]);
        ctx.fillRect(x0b + offsetFraction * binWidth, y, binWidth * 0.48, t + ph - y);
      }}
      ctx.globalAlpha = 1;
      ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(label, left + 8, t + yLabelOffset);
    }}
    drawBars(histA, "#2563eb", labelA, 24, 0.02);
    drawBars(histB, "#dc2626", labelB, 42, 0.50);
  }}
  function drawCdfPanel(left, width, title, valuesA, valuesB, labelA, labelB, absolute = false) {{
    const [lo, hi] = panelLimits(valuesA, valuesB, absolute);
    const xOf = value => left + (value - lo) / (hi - lo) * width;
    const yOf = value => t + (1 - value) * ph;
    drawPanelAxes(left, width, "Shift (px)", "Cumulative fraction");
    drawNumericTicks(ctx, left, t, width, ph, lo, hi, 0, 1, 7, 4);
    ctx.fillStyle = "#475467"; ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(title, left + width / 2, 5);
    function drawCdf(values, color, label, yLabelOffset) {{
      const sorted = values.slice().filter(value => Number.isFinite(value) && value >= lo && value <= hi).sort((a, b) => a - b);
      if (!sorted.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1.4, 1.4 * (window.devicePixelRatio || 1));
      ctx.beginPath();
      for (let i = 0; i < sorted.length; i++) {{
        const fraction = sorted.length === 1 ? 1 : i / (sorted.length - 1);
        const x = xOf(sorted[i]), y = yOf(fraction);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(label, left + 8, t + yLabelOffset);
    }}
    drawCdf(valuesA, "#2563eb", labelA, 24);
    drawCdf(valuesB, "#dc2626", labelB, 42);
  }}
  drawHistogramPanel(l, panelW, "Signed shift histogram", xValues, yValues, "X shift", "Y shift");
  drawCdfPanel(l + panelW + gap, panelW, "Signed shift CDF", xValues, yValues, "X shift", "Y shift", false);
  drawCdfPanel(l + 2 * (panelW + gap), panelW, "Absolute shift CDF", absXValues, absYValues, "|X shift|", "|Y shift|", true);
}}
function diagnosticMean(values) {{
  let sum = 0, n = 0;
  for (const value of values) if (Number.isFinite(value)) {{ sum += value; n++; }}
  return n ? sum / n : NaN;
}}
function diagnosticMedian(values) {{
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}}
function selectedOasisDiagnostics(roi, threshold) {{
  const key = `${{roi}}:${{Number(threshold).toPrecision(8)}}`;
  if (oasisDiagnosticsCache.has(key)) return oasisDiagnosticsCache.get(key);
  const tr = trace(roi), sp = spikeTrace(roi);
  if (!tr || !sp) return null;
  const preFrames = Math.max(3, Math.round(data.frameRate * 0.5));
  const postFrames = Math.max(6, Math.round(data.frameRate * 2.0));
  const minSeparation = Math.max(1, Math.round(data.frameRate * 0.25));
  const events = [];
  let last = -Infinity;
  for (let f = preFrames; f < data.nFrames - postFrames; f++) {{
    if (!Number.isFinite(sp[f]) || sp[f] <= threshold) continue;
    if (f - last < minSeparation) continue;
    events.push(f);
    last = f;
  }}
  if (!events.length) {{
    const empty = {{events, waveform: [], times: [], expModel: [], tau: NaN}};
    oasisDiagnosticsCache.set(key, empty);
    return empty;
  }}
  const len = preFrames + postFrames + 1;
  const sums = new Float64Array(len);
  const counts = new Uint32Array(len);
  for (const eventFrame of events) {{
    const baselineValues = [];
    for (let f = eventFrame - preFrames; f < eventFrame; f++) {{
      const value = tr[f];
      if (Number.isFinite(value)) baselineValues.push(value);
    }}
    const baseline = Number.isFinite(diagnosticMedian(baselineValues)) ? diagnosticMedian(baselineValues) : 0;
    for (let offset = -preFrames; offset <= postFrames; offset++) {{
      const value = tr[eventFrame + offset];
      if (!Number.isFinite(value)) continue;
      const index = offset + preFrames;
      sums[index] += value - baseline;
      counts[index] += 1;
    }}
  }}
  const waveform = [];
  const times = [];
  for (let i = 0; i < len; i++) {{
    waveform.push(counts[i] ? sums[i] / counts[i] : NaN);
    times.push((i - preFrames) / data.frameRate);
  }}
  let peakIndex = preFrames;
  let peak = -Infinity;
  for (let i = preFrames; i < waveform.length; i++) {{
    const value = waveform[i];
    if (Number.isFinite(value) && value > peak) {{ peak = value; peakIndex = i; }}
  }}
  const baselineTail = diagnosticMedian(waveform.slice(Math.max(0, waveform.length - Math.max(3, Math.round(data.frameRate * 0.25)))));
  const amplitude = peak - (Number.isFinite(baselineTail) ? baselineTail : 0);
  let tau = NaN;
  if (Number.isFinite(amplitude) && amplitude > 0) {{
    const target = (Number.isFinite(baselineTail) ? baselineTail : 0) + amplitude / Math.E;
    for (let i = peakIndex; i < waveform.length; i++) {{
      const value = waveform[i];
      if (!Number.isFinite(value)) continue;
      if (value <= target) {{
        tau = Math.max(0, times[i] - times[peakIndex]);
        break;
      }}
    }}
  }}
  const expModel = waveform.map((_, i) => {{
    if (!Number.isFinite(amplitude) || amplitude <= 0 || i < peakIndex) return NaN;
    if (!Number.isFinite(tau) || tau <= 0) return NaN;
    return (Number.isFinite(baselineTail) ? baselineTail : 0) + amplitude * Math.exp(-(times[i] - times[peakIndex]) / tau);
  }});
  const diagnostics = {{events, waveform, times, expModel, tau}};
  oasisDiagnosticsCache.set(key, diagnostics);
  return diagnostics;
}}
function drawLineSeries(ctx, series, xOf, yOf, color, lineWidth = 1.5) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(lineWidth, lineWidth * (window.devicePixelRatio || 1));
  ctx.beginPath();
  let first = true;
  for (const point of series) {{
    const x = Array.isArray(point) ? point[0] : point.x;
    const y = Array.isArray(point) ? point[1] : point.y;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (first) {{ ctx.moveTo(xOf(x), yOf(y)); first = false; }} else ctx.lineTo(xOf(x), yOf(y));
  }}
  ctx.stroke();
}}
function drawOasisDiagnostics() {{
  // Selected ROI's spike-triggered average transient and its exponential decay model,
  // shown in the summary column when its average-transient toggle is on Selected ROI.
  if (document.getElementById("fovReview")?.classList.contains("summary-hidden") || gbuTransientMode !== "roi") return;
  const summary = document.getElementById("oasisDiagnosticsSummary");
  const c = gbuCanvas("oasisTransientCanvas");
  if (!summary || !c) return;
  if (!data.oasisAvailable || !dff || !oasisSpikes) {{
    summary.textContent = "";
    gbuEmptyPlot(c, "Time from inferred spike (s)", "dF/F", "Load dF/F and inferred spikes files to view the selected ROI's average transient");
    return;
  }}
  const diagnostics = selectedOasisDiagnostics(selected, oasisEventThreshold);
  if (!diagnostics || !diagnostics.events.length) {{
    summary.textContent = "";
    gbuEmptyPlot(c, "Time from inferred spike (s)", "dF/F", `No inferred spikes above ${{fmt(oasisEventThreshold)}} for this ROI`);
    return;
  }}
  summary.textContent = `(${{diagnostics.events.length}} spikes above ${{fmt(oasisEventThreshold)}})`;
  const ctx = c.ctx, r = c.r, w = c.w, h = c.h;
  const l = 46 * r, rr = 8 * r, t = 8 * r, b = 32 * r, pw = w - l - rr, ph = h - t - b;
  const finiteWave = diagnostics.waveform.filter(Number.isFinite);
  const finiteModel = diagnostics.expModel.filter(Number.isFinite);
  const xMin = diagnostics.times[0], xMax = diagnostics.times[diagnostics.times.length - 1];
  let yMin = Math.min(...finiteWave, ...finiteModel), yMax = Math.max(...finiteWave, ...finiteModel);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax <= yMin) {{ yMin = -1; yMax = 1; }}
  const pad = (yMax - yMin) * 0.12 || 1; yMin -= pad; yMax += pad;
  const xOf = x => l + (x - xMin) / (xMax - xMin) * pw;
  const yOf = y => t + (1 - (y - yMin) / (yMax - yMin)) * ph;
  drawAxes(ctx, w, h, l, t, pw, ph, "Time from inferred spike (s)", "dF/F", {{ xLabelY: h - 4 * r, yLabelX: 10 * r }});
  drawNumericTicks(ctx, l, t, pw, ph, xMin, xMax, yMin, yMax, 5, 4);
  drawLineSeries(ctx, diagnostics.times.map((time, i) => [time, diagnostics.waveform[i]]), xOf, yOf, "#1d4ed8", 1.6);
  drawLineSeries(ctx, diagnostics.times.map((time, i) => [time, diagnostics.expModel[i]]), xOf, yOf, "#dc2626", 1.4);
  ctx.font = `${{11 * r}}px Arial`; ctx.textAlign = "right"; ctx.textBaseline = "top";
  ctx.fillStyle = "#1d4ed8"; ctx.fillText("ROI average", l + pw - 4 * r, t + 4 * r);
  ctx.fillStyle = "#dc2626"; ctx.fillText(`Exp. decay, tau ${{fmt(diagnostics.tau)}} s`, l + pw - 4 * r, t + 18 * r);
}}
function setOasisDiagnosticsVisible(show) {{
  // The fit plot now lives in the summary column, which is shown or hidden as a whole.
  requestAnimationFrame(draw);
}}
function setGbuSummaryVisible(show) {{
  const column = document.querySelector(".viewer-column");
  if (!column) return;
  column.classList.toggle("summary-hidden", !show);
  document.getElementById("fovReview")?.classList.toggle("summary-hidden", !show);
  layoutFovRow();
  if (show) scheduleGbuSummaryDraw();
}}
function updateFovSideLayout() {{ layoutFovRow(); }}
// ---- Field of view, dF/F trace and summary column sizing ----
// The summary column runs beside both the field of view and the dF/F trace. The splitter between
// them sets how wide the left side (field of view and trace) is; the handle under the field of
// view sets the size of its square images. Defaults fit the field of view and the trace in the
// window; double-clicking a handle returns it to its default, and dragged sizes are remembered.
var FOV_WIDTH_KEY = "roiReviewer.fovGridWidth";
function storedSize(key) {{ try {{ const v = Number(localStorage.getItem(key)); return v > 0 ? v : null; }} catch (error) {{ return null; }} }}
function storeSize(key, value) {{ try {{ if (value) localStorage.setItem(key, String(Math.round(value))); else localStorage.removeItem(key); }} catch (error) {{}} }}
var fovGridWidthOverride = storedSize(FOV_WIDTH_KEY);
function fovChannelCount() {{ return Math.max(1, document.querySelectorAll("#fovGrid > .panel").length); }}
function summaryHidden() {{ return document.querySelector(".viewer-column")?.classList.contains("summary-hidden"); }}
function fovWidthLimits() {{
  const total = document.querySelector(".viewer-column").clientWidth;
  const max = Math.max(200, total - (summaryHidden() ? 40 : 12 + 260));
  return {{ min: Math.min(max, 180 * fovChannelCount()), max }};
}}
function defaultFovGridWidth() {{
  // the sizes used before resizing was added, reduced only as far as needed to leave the summary column 380 px
  const preferred = fovChannelCount() > 1 ? 720 : 520;
  if (summaryHidden()) return preferred;
  return Math.max(180 * fovChannelCount(), Math.min(preferred, document.querySelector(".viewer-column").clientWidth - 12 - 380));
}}
function layoutFovRow() {{
  // The summary column sits beside the field of view and stops at its bottom edge; the dF/F
  // trace spans the full width underneath. The field of view keeps square images, so its width
  // sets the row's height. The splitter and the corner icon both resize it.
  const grid = document.getElementById("fovGrid"), side = document.getElementById("summaryColumn");
  if (!grid) return;
  const lim = fovWidthLimits();
  grid.style.width = `${{Math.max(lim.min, Math.min(lim.max, fovGridWidthOverride || defaultFovGridWidth()))}}px`;
  if (side) side.style.height = `${{grid.offsetHeight}}px`;
  requestAnimationFrame(draw);
}}
function setGbuSummaryVisible(show) {{
  const column = document.querySelector(".viewer-column");
  if (!column) return;
  column.classList.toggle("summary-hidden", !show);
  document.getElementById("fovReview")?.classList.toggle("summary-hidden", !show);
  layoutFovRow();
  if (show) scheduleGbuSummaryDraw();
}}
function updateFovSideLayout() {{ layoutFovRow(); }}
function bindFovResizer(handle, axis) {{
  if (!handle) return;
  let start = null;
  handle.addEventListener("pointerdown", event => {{
    start = {{ x: event.clientX, y: event.clientY, fov: document.getElementById("fovGrid").getBoundingClientRect().width }};
    handle.setPointerCapture(event.pointerId); handle.classList.add("dragging"); event.preventDefault();
  }});
  handle.addEventListener("pointermove", event => {{
    if (!start) return;
    const dx = event.clientX - start.x, dy = start.y - event.clientY;
    // splitter: horizontal only; corner icon: moving right widens, moving up by d enlarges each square image by d
    const delta = axis === "x" || Math.abs(dx) >= Math.abs(dy) ? dx : dy * fovChannelCount();
    const lim = fovWidthLimits();
    fovGridWidthOverride = Math.max(lim.min, Math.min(lim.max, start.fov + delta));
    layoutFovRow();
  }});
  const end = () => {{
    if (!start) return;
    start = null; handle.classList.remove("dragging");
    storeSize(FOV_WIDTH_KEY, fovGridWidthOverride);
  }};
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  handle.addEventListener("dblclick", () => {{ fovGridWidthOverride = null; storeSize(FOV_WIDTH_KEY, null); layoutFovRow(); }});
}}

// ---- Summary of good/bad/unsure ROIs ----
// Per-ROI average transients and firing rates are computed once, in slices so the page stays
// responsive, whenever dF/F and inferred spikes are (re)loaded; label changes only regroup them.
// `var` (not let/const) so label-change hooks that run during start-up never hit an uninitialized binding.
var GBU_GROUPS = [
  {{ code: 1, name: "Good", color: "#16a34a" }},
  {{ code: 0, name: "Bad", color: "#dc2626" }},
  {{ code: 2, name: "Unsure", color: "#d97706" }},
];
var GBU_UNLABELED = {{ code: -1, name: "Not labeled", color: "#667085" }};
var gbuRoiStats = null;
var gbuGeneration = 0;
var gbuProgress = 0;
var gbuDrawPending = false;
var gbuRateMode = "label";  // "label": rate distribution per label; "roi": selected ROI over time
function gbuDefaultThreshold(roi) {{
  const value = dffMetric(roi, "oasis_optimal_threshold");
  return Number.isFinite(Number(value)) ? Number(value) : Number(data.oasisEventThreshold ?? 0.05);
}}
var GBU_MAX_ABS_DFF = 20;  // same plausibility rule as multi_session_qc_summary (max_abs_dff)
function gbuRoiSummary(roi) {{
  // Events and windows follow the selected-ROI fit panel; windows whose baseline-subtracted dF/F
  // exceeds GBU_MAX_ABS_DFF in magnitude are left out of the average but still count as events.
  const threshold = gbuDefaultThreshold(roi);
  const d = selectedOasisDiagnostics(roi, threshold);
  if (roi !== selected) oasisDiagnosticsCache.delete(`${{roi}}:${{Number(threshold).toPrecision(8)}}`);
  if (!d) return null;
  const tr = trace(roi);
  const preFrames = Math.max(3, Math.round(data.frameRate * 0.5));
  const postFrames = Math.max(6, Math.round(data.frameRate * 2.0));
  const len = preFrames + postFrames + 1;
  const seconds = Math.max(1, data.nFrames - preFrames - postFrames) / data.frameRate;
  const sums = new Float64Array(len), counts = new Uint32Array(len), win = new Float64Array(len);
  let kept = 0, excluded = 0;
  for (const eventFrame of d.events) {{
    const baselineValues = [];
    for (let f = eventFrame - preFrames; f < eventFrame; f++) if (Number.isFinite(tr[f])) baselineValues.push(tr[f]);
    const median = diagnosticMedian(baselineValues);
    const baseline = Number.isFinite(median) ? median : 0;
    let plausible = true;
    for (let offset = -preFrames; offset <= postFrames; offset++) {{
      const value = tr[eventFrame + offset] - baseline;
      win[offset + preFrames] = value;
      if (Number.isFinite(value) && Math.abs(value) > GBU_MAX_ABS_DFF) {{ plausible = false; break; }}
    }}
    if (!plausible) {{ excluded += 1; continue; }}
    kept += 1;
    for (let i = 0; i < len; i++) if (Number.isFinite(win[i])) {{ sums[i] += win[i]; counts[i] += 1; }}
  }}
  let waveform = null;
  if (kept) {{
    waveform = new Float32Array(len);
    for (let i = 0; i < len; i++) waveform[i] = counts[i] ? sums[i] / counts[i] : NaN;
    if (!waveform.every(Number.isFinite)) waveform = null;
  }}
  const times = Array.from({{length: len}}, (_, i) => (i - preFrames) / data.frameRate);
  return {{ rate: d.events.length / seconds, waveform, times, kept, excluded }};
}}
function gbuExclusionText() {{
  if (!Array.isArray(gbuRoiStats)) return "";
  let windows = 0, rois = 0;
  for (let roi = 0; roi < gbuRoiStats.length; roi++) {{
    const s = gbuRoiStats[roi];
    if (!s || !s.excluded || !(labels[roi] === 0 || labels[roi] === 1 || labels[roi] === 2)) continue;
    windows += s.excluded;
    if (!s.kept) rois += 1;
  }}
  if (!windows) return "";
  return ` Left out of the average transients: ${{windows.toLocaleString()}} event window${{windows === 1 ? "" : "s"}} reaching |dF/F| > ${{GBU_MAX_ABS_DFF}}`
    + (rois ? `, which leaves ${{rois}} ROI${{rois === 1 ? "" : "s"}} with no transient.` : ".");
}}
function invalidateGbuSummary() {{ startGbuSummary(); }}
function startGbuSummary() {{
  const generation = ++gbuGeneration;
  gbuRoiStats = null; gbuProgress = 0;
  if (!data.oasisAvailable || !dff || !oasisSpikes) {{ scheduleGbuSummaryDraw(); return; }}
  const stats = new Array(data.nRois).fill(null);
  let roi = 0;
  const step = () => {{
    if (generation !== gbuGeneration) return;
    const started = performance.now();
    while (roi < data.nRois && performance.now() - started < 25) {{ stats[roi] = gbuRoiSummary(roi); roi += 1; }}
    gbuProgress = roi / Math.max(1, data.nRois);
    if (roi >= data.nRois) gbuRoiStats = stats;
    scheduleGbuSummaryDraw();
    if (roi < data.nRois) setTimeout(step, 0);
  }};
  setTimeout(step, 0);
}}
function scheduleGbuSummaryDraw() {{
  if (gbuDrawPending) return;
  gbuDrawPending = true;
  requestAnimationFrame(() => {{ gbuDrawPending = false; drawGbuSummary(); }});
}}
function gbuPercentile(sorted, q) {{
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}}
function gbuCanvas(id) {{
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  fit(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  return {{ canvas, ctx, w: canvas.width, h: canvas.height, r: window.devicePixelRatio || 1 }};
}}
function gbuMessage(c, text) {{
  c.ctx.fillStyle = "#475467"; c.ctx.font = `${{12 * c.r}}px Arial`; c.ctx.textAlign = "center"; c.ctx.textBaseline = "middle";
  c.ctx.fillText(text, c.w / 2, c.h / 2);
}}
function gbuYTicks(ctx, l, t, ph, lo, hi, n, r, digits) {{
  ctx.fillStyle = "#475467"; ctx.font = `${{10 * r}}px Arial`; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= n; i++) {{
    const v = lo + (hi - lo) * i / n;
    ctx.fillText(v.toFixed(digits), l - 5 * r, t + (1 - i / n) * ph);
  }}
}}
function gbuKde(values, grid) {{
  // Gaussian KDE with Silverman's bandwidth, reflected at zero (rates cannot be negative). values must be sorted.
  const n = values.length;
  if (n < 2) return null;
  let mean = 0;
  for (const v of values) mean += v;
  mean /= n;
  let ss = 0;
  for (const v of values) ss += (v - mean) * (v - mean);
  const sd = Math.sqrt(ss / (n - 1));
  const iqr = gbuPercentile(values, 0.75) - gbuPercentile(values, 0.25);
  const spread = iqr > 0 ? Math.min(sd, iqr / 1.34) : sd;
  const bw = 0.9 * spread * Math.pow(n, -0.2);
  if (!(bw > 0)) return null;
  const norm = 1 / (n * bw * Math.sqrt(2 * Math.PI));
  return grid.map(x => {{
    let s = 0;
    for (const v of values) {{
      const a = (x - v) / bw, r = (x + v) / bw;
      s += Math.exp(-0.5 * a * a) + Math.exp(-0.5 * r * r);
    }}
    return s * norm;
  }});
}}
function drawSelectedRoiRate(rc) {{
  // Firing rate of the selected ROI in time bins across the session, from inferred spikes above the
  // current amplitude threshold (the dashed line is its mean rate over the whole session).
  if (!data.oasisAvailable || !dff || !oasisSpikes) {{
    gbuEmptyPlot(rc, "Time (s)", "Events/s", "Load inferred spikes file to view the selected ROI's firing rate over time");
    return;
  }}
  const d = selectedOasisDiagnostics(selected, oasisEventThreshold);
  const events = d ? d.events : [];
  const seconds = Math.max(1 / data.frameRate, data.nFrames / data.frameRate);
  const bin = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200].find(s => seconds / s <= 60) || 1800;
  const nb = Math.max(1, Math.ceil(seconds / bin));
  const counts = new Array(nb).fill(0);
  for (const frame of events) counts[Math.min(nb - 1, Math.floor(frame / data.frameRate / bin))] += 1;
  const rates = counts.map((count, k) => count / Math.max(1e-9, Math.min(bin, seconds - k * bin)));
  const meanRate = events.length / seconds;
  const minutes = seconds > 600, unit = minutes ? 60 : 1;
  let yMax = Math.max(meanRate, ...rates) * 1.35;
  if (!(yMax > 0)) yMax = 1;
  const ctx = rc.ctx, r = rc.r;
  const l = 46 * r, rr = 8 * r, t = 8 * r, b = 32 * r, pw = rc.w - l - rr, ph = rc.h - t - b;
  const xMax = seconds / unit;
  const xOf = x => l + x / xMax * pw, yOf = y => t + (1 - y / yMax) * ph;
  drawAxes(ctx, rc.w, rc.h, l, t, pw, ph, `Time (${{minutes ? "min" : "s"}})`, "Events/s", {{ xLabelY: rc.h - 4 * r, yLabelX: 10 * r }});
  drawNumericTicks(ctx, l, t, pw, ph, 0, xMax, 0, yMax, 5, 3);
  ctx.save();
  ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 1.5 * r; ctx.beginPath();
  rates.forEach((rate, k) => {{
    const x0 = xOf(k * bin / unit), x1 = xOf(Math.min(seconds, (k + 1) * bin) / unit), y = yOf(rate);
    if (k === 0) ctx.moveTo(x0, y); else ctx.lineTo(x0, y);
    ctx.lineTo(x1, y);
  }});
  ctx.stroke();
  ctx.setLineDash([4 * r, 3 * r]); ctx.strokeStyle = "#344054"; ctx.lineWidth = 1 * r;
  ctx.beginPath(); ctx.moveTo(l, yOf(meanRate)); ctx.lineTo(l + pw, yOf(meanRate)); ctx.stroke();
  ctx.restore();
  ctx.font = `${{11 * r}}px Arial`; ctx.textAlign = "right"; ctx.textBaseline = "top";
  ctx.fillStyle = "#7c3aed";
  ctx.fillText(`${{ROI_INDEX_LABEL}} ${{data.suite2pIndices[selected]}}, ${{bin}} s bins`, l + pw - 4 * r, t + 4 * r);
  ctx.fillStyle = "#344054";
  ctx.fillText(`Mean ${{meanRate.toPrecision(3)}}/s (${{events.length.toLocaleString()}} spikes above ${{fmt(oasisEventThreshold)}})`, l + pw - 4 * r, t + 18 * r);
}}
function gbuEmptyPlot(c, xLabel, yLabel, text) {{
  // Axes drawn before the data exist, so the plot's place and scale are visible, with a centered
  // gray message (wrapped to the plot width) saying what is needed to fill it.
  const r = c.r, l = 46 * r, rr = 8 * r, t = 8 * r, b = 32 * r, pw = c.w - l - rr, ph = c.h - t - b;
  drawAxes(c.ctx, c.w, c.h, l, t, pw, ph, xLabel, yLabel, {{ xLabelY: c.h - 4 * r, yLabelX: 10 * r }});
  const ctx = c.ctx;
  ctx.fillStyle = "#667085"; ctx.font = `${{11 * r}}px Arial`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const lines = [];
  let line = "";
  for (const word of String(text).split(" ")) {{
    const next = line ? `${{line}} ${{word}}` : word;
    if (ctx.measureText(next).width > pw - 16 * r && line) {{ lines.push(line); line = word; }} else line = next;
  }}
  if (line) lines.push(line);
  const lh = 14 * r, y0 = t + ph / 2 - (lines.length - 1) * lh / 2;
  lines.forEach((s, i) => ctx.fillText(s, l + pw / 2, y0 + i * lh));
}}
var gbuTransientMode = "label";  // "label": mean transient per label; "roi": selected ROI with its decay fit
function drawGbuSummary() {{
  if (document.getElementById("fovReview")?.classList.contains("summary-hidden")) return;  // hidden: redrawn when shown again
  const counts = new Map([[1, 0], [0, 0], [2, 0], [-1, 0]]);
  for (const value of labels) {{
    const key = value === 1 || value === 0 || value === 2 ? value : -1;
    counts.set(key, counts.get(key) + 1);
  }}
  const total = Math.max(1, labels.length);
  const f = gbuCanvas("gbuFractionCanvas");
  if (f) {{
    // vertical bars at a fixed size: one per label, percentage and count above, name below
    const groups = GBU_GROUPS.concat([GBU_UNLABELED]);
    const ctx = f.ctx, r = f.r;
    const l = 4 * r, rr = 4 * r, t = 28 * r, b = 30 * r, pw = f.w - l - rr, ph = f.h - t - b;
    const slot = pw / groups.length, bw = slot * 0.62;
    ctx.strokeStyle = "#d0d5dd"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(l, t + ph); ctx.lineTo(l + pw, t + ph); ctx.stroke();
    groups.forEach((g, i) => {{
      const n = counts.get(g.code), frac = n / total;
      const cx = l + (i + 0.5) * slot, h = frac * ph, y = t + ph - h;
      ctx.fillStyle = g.color; ctx.fillRect(cx - bw / 2, y, bw, h);
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillStyle = "#344054"; ctx.font = `${{9.5 * r}}px Arial`;
      ctx.fillText(`${{(100 * frac).toFixed(0)}}%`, cx, y - 12 * r);
      ctx.fillStyle = "#667085"; ctx.fillText(`(${{n}})`, cx, y - 2 * r);
      ctx.textBaseline = "top"; ctx.fillStyle = "#475467";
      const words = g.name.split(" ");
      words.forEach((word, k) => ctx.fillText(word, cx, t + ph + 4 * r + k * 11 * r));
    }});
  }}
  const note = document.getElementById("gbuSummaryNote");
  const tc = gbuTransientMode === "label" ? gbuCanvas("gbuTransientCanvas") : null;
  if (gbuTransientMode === "roi") drawOasisDiagnostics();
  const rc = gbuCanvas("gbuRateCanvas");
  if (rc && gbuRateMode === "roi") drawSelectedRoiRate(rc);
  const spikesMissing = !data.oasisAvailable || !dff || !oasisSpikes;
  const ready = Array.isArray(gbuRoiStats);
  if (note) {{
    note.textContent = spikesMissing
      ? `${{labels.length}} ROIs: ${{counts.get(1)}} good, ${{counts.get(0)}} bad, ${{counts.get(2)}} unsure, ${{counts.get(-1)}} not labeled.`
      : ready ? `${{labels.length}} ROIs: ${{counts.get(1)}} good, ${{counts.get(0)}} bad, ${{counts.get(2)}} unsure, ${{counts.get(-1)}} not labeled.` + gbuExclusionText()
      : `Computing per-ROI transients and firing rates: ${{Math.round(100 * gbuProgress)}}%`;
  }}
  if (!ready) {{
    const loading = sidecarsPending > 0 && window.location.protocol !== "file:";
    if (tc) gbuEmptyPlot(tc, "Time from inferred spike (s)", "dF/F", spikesMissing ? (loading ? "Loading dF/F and inferred spikes..." : "Load dF/F and inferred spikes files to view average transients of ROIs by label") : "Computing...");
    if (rc && gbuRateMode !== "roi") gbuEmptyPlot(rc, "Firing rate (events/s)", "Density", spikesMissing ? (loading ? "Loading inferred spikes..." : "Load inferred spikes file to view firing rate distribution of ROIs by label") : "Computing...");
    return;
  }}
  if (tc) {{
    const first = gbuRoiStats.find(s => s && s.waveform);
    const times = first ? first.times : null;
    const series = [];
    let yMin = Infinity, yMax = -Infinity;
    if (times) {{
      for (const g of GBU_GROUPS) {{
        const rows = [];
        for (let roi = 0; roi < gbuRoiStats.length; roi++) {{
          const s = gbuRoiStats[roi];
          if (s && s.waveform && labels[roi] === g.code) rows.push(s.waveform);
        }}
        if (!rows.length) continue;
        const mean = new Array(times.length), sem = new Array(times.length);
        for (let i = 0; i < times.length; i++) {{
          let sum = 0;
          for (const row of rows) sum += row[i];
          const m = sum / rows.length;
          let ss = 0;
          for (const row of rows) ss += (row[i] - m) * (row[i] - m);
          mean[i] = m;
          sem[i] = rows.length > 1 ? Math.sqrt(ss / (rows.length - 1)) / Math.sqrt(rows.length) : 0;
          yMin = Math.min(yMin, m - sem[i]); yMax = Math.max(yMax, m + sem[i]);
        }}
        series.push({{ g, n: rows.length, mean, sem }});
      }}
    }}
    if (!series.length) {{
      const anyLabeled = counts.get(1) + counts.get(0) + counts.get(2) > 0;
      gbuEmptyPlot(tc, "Time from inferred spike (s)", "dF/F", anyLabeled ? "No good, bad or unsure ROI has an inferred spike above its threshold"
                                : "Label ROIs as good, bad or unsure to see their average transients");
    }} else {{
      const pad = (yMax - yMin) * 0.1 || 0.1; yMin -= pad; yMax += pad;
      const l = 46 * tc.r, rr = 8 * tc.r, t = 8 * tc.r, b = 32 * tc.r, pw = tc.w - l - rr, ph = tc.h - t - b;
      const xMin = times[0], xMax = times[times.length - 1];
      const xOf = x => l + (x - xMin) / (xMax - xMin) * pw;
      const yOf = y => t + (1 - (y - yMin) / (yMax - yMin)) * ph;
      drawAxes(tc.ctx, tc.w, tc.h, l, t, pw, ph, "Time from inferred spike (s)", "dF/F", {{ xLabelY: tc.h - 4 * tc.r, yLabelX: 10 * tc.r }});
      drawNumericTicks(tc.ctx, l, t, pw, ph, xMin, xMax, yMin, yMax, 5, 4);
      series.forEach((s, k) => {{
        tc.ctx.save();
        tc.ctx.globalAlpha = 0.22; tc.ctx.fillStyle = s.g.color; tc.ctx.beginPath();
        times.forEach((x, i) => {{ const px = xOf(x), py = yOf(s.mean[i] + s.sem[i]); if (i) tc.ctx.lineTo(px, py); else tc.ctx.moveTo(px, py); }});
        for (let i = times.length - 1; i >= 0; i--) tc.ctx.lineTo(xOf(times[i]), yOf(s.mean[i] - s.sem[i]));
        tc.ctx.closePath(); tc.ctx.fill();
        tc.ctx.restore();
        drawLineSeries(tc.ctx, times.map((x, i) => [x, s.mean[i]]), xOf, yOf, s.g.color, 1.6);
        tc.ctx.fillStyle = s.g.color; tc.ctx.font = `${{11 * tc.r}}px Arial`; tc.ctx.textAlign = "right"; tc.ctx.textBaseline = "top";
        tc.ctx.fillText(`${{s.g.name}} (n = ${{s.n}} ROI${{s.n === 1 ? "" : "s"}})`, l + pw - 4 * tc.r, t + 4 * tc.r + k * 14 * tc.r);
      }});
    }}
  }}
  if (rc && gbuRateMode !== "roi") {{
    const groups = [];
    const pooled = [];
    for (const g of GBU_GROUPS) {{
      const values = [];
      for (let roi = 0; roi < gbuRoiStats.length; roi++) {{
        const s = gbuRoiStats[roi];
        if (s && labels[roi] === g.code && Number.isFinite(s.rate)) values.push(s.rate);
      }}
      values.sort((a, b) => a - b);
      for (const v of values) pooled.push(v);
      groups.push({{ g, values }});
    }}
    pooled.sort((a, b) => a - b);
    let hi = pooled.length ? gbuPercentile(pooled, 0.995) * 1.1 : 1;
    if (!(hi > 0)) hi = pooled.length ? Math.max(pooled[pooled.length - 1], 1e-3) * 1.1 : 1;
    {{
      const raw = hi / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      hi = 4 * mag * [1, 2, 2.5, 5, 10].find(m => m * mag >= raw);
    }}
    const grid = Array.from({{ length: 200 }}, (_, i) => hi * i / 199);
    const curves = groups.map(({{ g, values }}) => ({{ g, values, density: gbuKde(values, grid), median: gbuPercentile(values, 0.5) }}));
    let yMax = 0;
    for (const cv of curves) if (cv.density) for (const d of cv.density) yMax = Math.max(yMax, d);
    yMax = yMax > 0 ? yMax * 1.5 : 1;  // headroom above the curves for the legend
    const l = 46 * rc.r, rr = 8 * rc.r, t = 8 * rc.r, b = 32 * rc.r, pw = rc.w - l - rr, ph = rc.h - t - b;
    const xOf = x => l + x / hi * pw;
    const yOf = y => t + (1 - y / yMax) * ph;
    drawAxes(rc.ctx, rc.w, rc.h, l, t, pw, ph, "Firing rate (events/s)", "Density", {{ xLabelY: rc.h - 4 * rc.r, yLabelX: 10 * rc.r }});
    drawNumericTicks(rc.ctx, l, t, pw, ph, 0, hi, 0, yMax, 4, 3);
    if (!counts.get(1) && !counts.get(0) && !counts.get(2)) {{
      rc.ctx.fillStyle = "#475467"; rc.ctx.font = `${{11 * rc.r}}px Arial`; rc.ctx.textAlign = "center"; rc.ctx.textBaseline = "middle";
      rc.ctx.fillText("Label ROIs as good, bad or unsure to see their firing rates", l + pw / 2, t + ph / 2);
      return;
    }}
    curves.forEach((cv, k) => {{
      const ctx = rc.ctx, color = cv.g.color;
      if (cv.density) {{
        ctx.save();
        ctx.globalAlpha = 0.15; ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(xOf(0), yOf(0));
        grid.forEach((x, i) => ctx.lineTo(xOf(x), yOf(cv.density[i])));
        ctx.lineTo(xOf(hi), yOf(0)); ctx.closePath(); ctx.fill();
        ctx.restore();
        drawLineSeries(ctx, grid.map((x, i) => [x, cv.density[i]]), xOf, yOf, color, 1.6);
      }}
      if (Number.isFinite(cv.median) && cv.median <= hi) {{
        ctx.save();
        ctx.strokeStyle = color; ctx.lineWidth = (cv.density ? 1 : 1.6) * rc.r; ctx.setLineDash(cv.density ? [4 * rc.r, 3 * rc.r] : []);
        // median drawn from the axis up to its own curve, so it never crosses the legend
        let top = yMax / 1.5;
        if (cv.density) {{ const pos = cv.median / hi * (grid.length - 1), i0 = Math.floor(pos), i1 = Math.min(grid.length - 1, i0 + 1);
          top = cv.density[i0] + (cv.density[i1] - cv.density[i0]) * (pos - i0); }}
        ctx.beginPath(); ctx.moveTo(xOf(cv.median), t + ph); ctx.lineTo(xOf(cv.median), yOf(top)); ctx.stroke();
        ctx.restore();
      }}
      const n = cv.values.length;
      const text = n ? `${{cv.g.name}} (n = ${{n}}, median ${{cv.median.toPrecision(3)}}/s)` : `${{cv.g.name}} (n = 0)`;
      ctx.fillStyle = color; ctx.font = `${{11 * rc.r}}px Arial`; ctx.textAlign = "right"; ctx.textBaseline = "top";
      ctx.fillText(text, l + pw - 4 * rc.r, t + 4 * rc.r + k * 14 * rc.r);
    }});
  }}
}}
function drawSelectedTraceAndOasis() {{ drawTrace(); drawOasisDiagnostics(); }}
function draw() {{ drawSelectedTraceAndOasis(); drawStack(); drawMotionDrift(); drawMotionDistribution(); scheduleGbuSummaryDraw(); }}
function reset() {{ x0=0; x1=data.nFrames-1; y0=0; y1=Math.min(19, visibleRois.length-1); document.getElementById("yStart").value=0; document.getElementById("yEnd").value=y1; syncTimeInputs(); draw(); }}
function showAllVisibleTraces() {{
  y0 = 0;
  y1 = Math.max(0, visibleRois.length - 1);
  document.getElementById("yStart").value = y0;
  document.getElementById("yEnd").value = y1;
  draw();
}}
function closeDialog(dialog) {{
  if (typeof dialog.close === "function") dialog.close();
  else dialog.removeAttribute("open");
}}
document.getElementById("roiInput").addEventListener("change", e => setSelected(roiFromSuite2pIndex(e.target.value)));
document.getElementById("timeStart").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("timeEnd").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("yStart").addEventListener("change", e => {{ y0=Number(e.target.value); draw(); }});
document.getElementById("yEnd").addEventListener("change", e => {{ y1=Number(e.target.value); draw(); }});
document.getElementById("showAllVisibleTraces").addEventListener("click", showAllVisibleTraces);
document.getElementById("reset").addEventListener("click", reset);
document.getElementById("markGood").addEventListener("click", () => setLabel(1));
document.getElementById("markBad").addEventListener("click", () => setLabel(0));
document.getElementById("markUnsure").addEventListener("click", () => setLabel(2));
document.getElementById("markUnlabeled").addEventListener("click", () => setLabel(-1));
const labelAllDialog = document.getElementById("labelAllDialog");
document.getElementById("openLabelAllDialog").addEventListener("click", () => {{
  if (typeof labelAllDialog.showModal === "function") labelAllDialog.showModal();
  else labelAllDialog.setAttribute("open", "");
}});
document.getElementById("closeLabelAllDialog").addEventListener("click", () => {{
  closeDialog(labelAllDialog);
}});
document.getElementById("closeLabelAllDialogTop").addEventListener("click", () => closeDialog(labelAllDialog));
document.getElementById("applyLabelAll").addEventListener("click", () => {{
  const label = Number(document.getElementById("labelAllValue").value);
  for (const roi of visibleRois) {{
    labels[roi] = label;
  }}
  updateLabelControls();
  updateVisibleRois();
  closeDialog(labelAllDialog);
}});
document.getElementById("previousRoi").addEventListener("click", () => moveVisible(-1));
document.getElementById("nextRoi").addEventListener("click", () => moveVisible(1));
document.querySelectorAll(".roi-display-checkbox").forEach(input => {{
  input.addEventListener("change", () => {{
    updateRoiDisplaySummary();
    updateVisibleRois(true);
  }});
}});
document.getElementById("applySort").addEventListener("click", applySort);
function updateSortMetricHandlers() {{
  document.querySelectorAll('input[name="sortMetric"]').forEach(input => {{
    if (input.dataset.handlerAttached === "1") return;
    input.dataset.handlerAttached = "1";
    input.addEventListener("change", event => {{
    const original = document.querySelector('input[name="sortMetric"][value="original"]');
    const metrics = Array.from(document.querySelectorAll('input[name="sortMetric"]'));
    if (event.target.value === "original" && event.target.checked) {{
      metrics.forEach(metric => {{ if (metric.value !== "original") metric.checked = false; }});
    }} else if (event.target.checked && original) {{
      original.checked = false;
    }}
    if (!metrics.some(metric => metric.checked) && original) original.checked = true;
  }});
  }});
}}
updateSortMetricHandlers();
const sortDialog = document.getElementById("sortDialog");
document.getElementById("openSortDialog").addEventListener("click", () => {{
  if (typeof sortDialog.showModal === "function") sortDialog.showModal();
  else sortDialog.setAttribute("open", "");
}});
document.getElementById("closeSortDialog").addEventListener("click", () => {{
  closeDialog(sortDialog);
}});
document.getElementById("closeSortDialogTop").addEventListener("click", () => closeDialog(sortDialog));
const customRoiUploadFileInput = document.getElementById("customRoiUploadFile");
const customRoiUploadDialog = document.getElementById("customRoiUploadDialog");
document.getElementById("openCustomRoiUploadDialog").addEventListener("click", () => {{
  renderAutoLoadedFileList();
  renderCategoryLabelEditor();
  document.getElementById("customRoiUploadStatus").textContent = "";
  if (typeof customRoiUploadDialog.showModal === "function") customRoiUploadDialog.showModal();
  else customRoiUploadDialog.setAttribute("open", "");
}});
document.getElementById("closeCustomRoiUploadDialog").addEventListener("click", () => closeDialog(customRoiUploadDialog));
document.getElementById("closeCustomRoiUploadDialogTop").addEventListener("click", () => closeDialog(customRoiUploadDialog));
document.getElementById("applyCustomRoiUpload").addEventListener("click", () => {{
  const file = customRoiUploadFileInput.files && customRoiUploadFileInput.files[0];
  const status = document.getElementById("customRoiUploadStatus");
  if (!file) {{
    status.textContent = "Choose a CSV, TSV, TXT, or JSON file first.";
    return;
  }}
  const kind = document.querySelector('input[name="customUploadKind"]:checked')?.value || "numeric";
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      if (kind === "categorical") {{
        loadCellTypeTableText(String(reader.result || ""), file.name);
        data.cellTypeLabelOrigin = "upload";
        renderAutoLoadedFileList();
        status.textContent = `Loaded cell identity labels from ${{file.name}}.`;
      }} else {{
        loadCustomMetricText(String(reader.result || ""), file.name);
        status.textContent = `Loaded QC metric(s) from ${{file.name}}.`;
      }}
    }} catch (error) {{
      status.textContent = `Could not load ${{kind === "categorical" ? "cell identity labels" : "QC metric"}}: ${{error.message}}`;
    }}
  }};
  reader.readAsText(file);
}});
document.getElementById("showInferredSpikes").addEventListener("change", event => {{
  showInferredSpikes = event.target.checked;
  draw();
}});
document.getElementById("oasisThresholdSlider").addEventListener("input", event => {{
  setOasisThreshold(Number(event.target.value));
}});
document.getElementById("oasisThreshold").addEventListener("input", event => {{
  setOasisThreshold(Number(event.target.value));
}});
document.getElementById("resetOasisThreshold").addEventListener("click", () => {{
  setOasisThreshold(selectedOasisDefaultThreshold());
}});
function csvEscape(value) {{
  const text = String(value ?? "");
  return /[",\\n]/.test(text) ? `"${{text.replaceAll('"', '""')}}"` : text;
}}
function metricSpreadsheetRows() {{
  const filter = readFilter();
  return data.morphology.map((metrics, roi) => {{
    const dffMetrics = data.dffMetrics[roi];
    const snr9550 = dffMetric(roi, "snr_95_50", "event_snr");
    const postdocSnr = dffMetric(roi, "andrea_postdoc_snr");
    const oasisEventSnr = dffMetricValue(dffMetrics, "oasis_event_snr");
    const oasisRiseTau = dffMetricValue(dffMetrics, "oasis_rise_tau_seconds");
    const oasisDecayTau = dffMetricValue(dffMetrics, "oasis_decay_tau_seconds");
    const cellTypeCode = data.cellTypeLabels?.[roi] ?? null;
    const roiModelScore = data.roiModelScores?.[roi] || null;
    const roiModelScoreState = roiModelScore?.state === "gray" ? "unsure" : roiModelScore?.state ?? "";
    const customMetrics = Object.fromEntries(customMetricDefinitions().map(definition => [definition.key, customMetricValue(roi, definition.key)]));
    const customMetricFail = Object.fromEntries(customMetricDefinitions().map(definition => {{
      const bounds = filter.customMetrics?.[definition.key] || {{}};
      const value = customMetricValue(roi, definition.key);
      return [definition.key, !(passesLower(value, bounds.min) && passesUpper(value, bounds.max))];
    }}));
    const reasons = morphologyReasons(metrics, dffMetrics, filter, roi);
    if (labels[roi] === 0) reasons.push("manual/current label: bad");
    else if (labels[roi] === 2) reasons.push("manual/current label: unsure");
    else if (labels[roi] === -1 && reasons.length === 0) reasons.push("not labeled");
    return {{
      suite2p: data.suite2pIndices[roi],
      label: labelName(labels[roi]),
      categoryLabel: cellTypeLabelName(cellTypeCode),
      categoryCode: cellTypeCode,
      roiModelScoreState,
      roiModelScoreProbability: roiModelScore?.probability ?? null,
      footprint: metrics.footprint,
      skew: metrics.skew,
      aspect: metrics.aspect,
      compact: metrics.compact,
      connectivity: metrics.connect,
      roiArea: dffMetrics.roi_area,
      snr9550,
      postdocSnr,
      oasisEventSnr,
      oasisRiseTau,
      oasisDecayTau,
      customMetrics,
      fail: {{
        skew: !(passesLower(metrics.skew, filter.skewMin) && passesUpper(metrics.skew, filter.skewMax)),
        aspect: !(passesLower(metrics.aspect, filter.aspectMin) && passesUpper(metrics.aspect, filter.aspectMax)),
        compact: !(passesLower(metrics.compact, filter.compactMin) && passesUpper(metrics.compact, filter.compactMax)),
        connectivity: !passesUpper(metrics.connect, filter.maxConnect),
        roiArea: !(passesLower(dffMetrics.roi_area, filter.roiAreaMin) && passesUpper(dffMetrics.roi_area, filter.roiAreaMax)),
        snr9550: !passesLower(snr9550, filter.eventSnrMin),
        postdocSnr: !passesLower(postdocSnr, filter.andreaPostdocSnrMin),
        oasisEventSnr: !passesLower(oasisEventSnr, filter.oasisEventSnrMin),
        oasisRiseTau: !(passesLower(oasisRiseTau, filter.oasisRiseTauMin) && passesUpper(oasisRiseTau, filter.oasisRiseTauMax)),
        oasisDecayTau: !(passesLower(oasisDecayTau, filter.oasisDecayTauMin) && passesUpper(oasisDecayTau, filter.oasisDecayTauMax)),
        roiModelScoreProbability: !(passesLower(roiModelScore?.probability, filter.roiModelScoreProbabilityMin) && passesUpper(roiModelScore?.probability, filter.roiModelScoreProbabilityMax)),
        customMetrics: customMetricFail,
      }},
      reason: reasons.join("; ") || "included",
    }};
  }});
}}
function metricSpreadsheetCsv(rowsData = metricSpreadsheetRows()) {{
  const customHeaders = customMetricDefinitions().flatMap(definition => definition.categories ? [`custom_${{definition.key}}`, `custom_${{definition.key}}_label`] : [`custom_${{definition.key}}`]);
  const csvHeader = ["roi_index","manual_label","cell_identity_label","cell_identity_code","roi_model_score_state","roi_model_score_probability","footprint","skew","aspect_ratio","compact","connectivity","roi_area_px","snr_95_50","caiman_exceptional_event_snr","inferred_spike_snr","inferred_spike_rise_tau_seconds","inferred_spike_decay_tau_seconds", ...customHeaders, "reason"];
  const csvRows = rowsData.map(row => [
    row.suite2p, row.label, row.categoryLabel, row.categoryCode ?? "", row.roiModelScoreState, fmt(row.roiModelScoreProbability), fmt(row.footprint), fmt(row.skew), fmt(row.aspect), fmt(row.compact),
    row.connectivity, fmt(row.roiArea), fmt(row.snr9550), fmt(row.postdocSnr),
    fmt(row.oasisEventSnr), fmt(row.oasisRiseTau), fmt(row.oasisDecayTau),
    ...customMetricDefinitions().flatMap(definition => definition.categories
      ? [fmt(row.customMetrics[definition.key]), customCategoryName(definition, row.customMetrics[definition.key])]
      : [fmt(row.customMetrics[definition.key])]),
    row.reason,
  ].map(csvEscape).join(","));
  return [csvHeader.join(","), ...csvRows].join("\\n") + "\\n";
}}
function openMetricSpreadsheet() {{
  function td(value, failed = false) {{
    return `<td${{failed ? ' class="metric-fail"' : ""}}>${{value}}</td>`;
  }}
  function labelTd(label) {{
    const cls = label === "good" ? "label-good" : label === "bad" ? "label-bad" : label === "unsure" ? "label-unsure" : "";
    return `<td${{cls ? ` class="${{cls}}"` : ""}}>${{label}}</td>`;
  }}
  const rowsData = metricSpreadsheetRows();
  const customHeaders = customMetricDefinitions().map(definition => `<th>custom_${{csvEscape(definition.key)}}</th>` + (definition.categories ? `<th>custom_${{csvEscape(definition.key)}}_label</th>` : "")).join("");
  const rows = rowsData.map(row => `<tr>${{
    td(row.suite2p) +
    labelTd(row.label) +
    td(row.categoryLabel) +
    td(row.categoryCode ?? "") +
    td(row.roiModelScoreState) +
    td(fmt(row.roiModelScoreProbability), row.fail.roiModelScoreProbability) +
    td(fmt(row.footprint), row.fail.footprint) +
    td(fmt(row.skew), row.fail.skew) +
    td(fmt(row.aspect), row.fail.aspect) +
    td(fmt(row.compact), row.fail.compact) +
    td(row.connectivity, row.fail.connectivity) +
    td(fmt(row.roiArea), row.fail.roiArea) +
    td(fmt(row.snr9550), row.fail.snr9550) +
    td(fmt(row.postdocSnr), row.fail.postdocSnr) +
    td(fmt(row.oasisEventSnr), row.fail.oasisEventSnr) +
    td(fmt(row.oasisRiseTau), row.fail.oasisRiseTau) +
    td(fmt(row.oasisDecayTau), row.fail.oasisDecayTau) +
    customMetricDefinitions().map(definition => td(fmt(row.customMetrics[definition.key]), row.fail.customMetrics[definition.key]) + (definition.categories ? td(customCategoryName(definition, row.customMetrics[definition.key])) : "")).join("") +
    td(row.reason)
  }}</tr>`).join("");
  const csv = metricSpreadsheetCsv(rowsData);
  const win = window.open("", "_blank");
  win.document.write(`<!doctype html><title>${{data.session}} ROI metrics</title><style>body{{font-family:Arial,sans-serif;margin:20px}}button{{margin:8px 0 12px;padding:6px 10px}}.metric-table-wrap{{max-height:80vh;overflow:auto;border:1px solid #d0d5dd}}.metric-table{{border-collapse:collapse;width:100%;font-size:12px}}.metric-table th,.metric-table td{{border:1px solid #e5e7eb;padding:4px 7px;text-align:right;white-space:nowrap}}.metric-table th{{position:sticky;top:0;background:#f8fafc;z-index:1}}.metric-table td:nth-child(1),.metric-table td:nth-child(2),.metric-table td:nth-child(3),.metric-table td:nth-child(4),.metric-table td:nth-child(5),.metric-table td:last-child{{text-align:left}}.metric-fail,.label-bad{{background:rgba(248,113,113,.28)}}.label-good{{background:rgba(34,197,94,.28)}}.label-unsure{{background:rgba(250,204,21,.28)}}</style><h1>${{data.session}} ROI metric spreadsheet</h1><p>filter_set: ${{loadedFilterSetName() || "none"}}</p><p>cell_identity_label_source: ${{data.cellTypeLabelSource || "not loaded"}}</p><p>roi_model_score_label_source: ${{data.roiModelLabelSource || "not loaded"}}</p><p>roi_model_score_source: ${{data.roiModelScoreSource || "not loaded"}}</p><button id="downloadCsv">Download CSV</button><div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>roi_index</th><th>manual_label</th><th>cell_identity_label</th><th>cell_identity_code</th><th>roi_model_score_state</th><th>roi_model_score_probability</th><th>footprint</th><th>skew</th><th>aspect_ratio</th><th>compact</th><th>connectivity</th><th>roi_area_px</th><th>snr_95_50</th><th>caiman_exceptional_event_snr</th><th>inferred_spike_snr</th><th>inferred_spike_rise_tau_seconds</th><th>inferred_spike_decay_tau_seconds</th>${{customHeaders}}<th>reason</th></tr></thead><tbody>${{rows}}</tbody></table></div><script>const csv = ${{JSON.stringify(csv)}}; document.getElementById("downloadCsv").addEventListener("click", () => {{ const blob = new Blob([csv], {{type: "text/csv"}}); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "${{data.session}}_roi_metric_spreadsheet.csv"; a.click(); URL.revokeObjectURL(a.href); }});<\\/script>`);
  win.document.close();
}}
function customCategoryName(definition, value) {{
  if (!definition.categories || value === null || value === undefined || !Number.isFinite(Number(value))) return "";
  return definition.categories[String(Number(value))] ?? "";
}}
function saveMetricSpreadsheet() {{
  downloadBlob(new Blob([metricSpreadsheetCsv()], {{type: "text/csv"}}), `${{data.session}}_roi_metric_spreadsheet.csv`);
}}
function currentSortSettings() {{
  return {{metrics: Array.from(appliedSortMetrics), direction: appliedSortDirection}};
}}
function applySortSettings(sort) {{
  if (!sort || typeof sort !== "object") return false;
  const wanted = Array.isArray(sort.metrics) && sort.metrics.length ? sort.metrics.map(String) : ["original"];
  const inputs = Array.from(document.querySelectorAll('input[name="sortMetric"]'));
  const known = new Set(inputs.map(input => input.value));
  const usable = wanted.filter(metric => known.has(metric));
  inputs.forEach(input => {{ input.checked = usable.length ? usable.includes(input.value) : input.value === "original"; }});
  document.getElementById("sortDirection").value = sort.direction === "desc" ? "desc" : "asc";
  return true;
}}
function saveCurrentFilters() {{
  const name = currentPresetName() || loadedFilterSetName() || `${{data.session}} filter set`;
  const payload = {{
    format: "roi_reviewer_filter_sort_settings_v1",
    name,
    session: data.session,
    saved_at: new Date().toISOString(),
    filter: normalizeFilter(readFilter()),
    sort: currentSortSettings(),
  }};
  downloadBlob(new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{type: "application/json"}}), `${{data.session}}_filter_sort_settings.json`);
}}
document.getElementById("openExclusions").addEventListener("click", openMetricSpreadsheet);
function npyBlob(values, rows) {{
  const encoder = new TextEncoder();
  let header = `{{'descr': '<f8', 'fortran_order': False, 'shape': (${{rows}},), }}`;
  const preambleLength = 10;
  const padding = (16 - ((preambleLength + header.length + 1) % 16)) % 16;
  header += " ".repeat(padding) + "\\n";
  const headerBytes = encoder.encode(header);
  const buffer = new ArrayBuffer(preambleLength + headerBytes.length + values.byteLength);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  new DataView(buffer).setUint16(8, headerBytes.length, true);
  bytes.set(headerBytes, preambleLength);
  bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), preambleLength + headerBytes.length);
  return new Blob([buffer], {{type: "application/octet-stream"}});
}}
function downloadBlob(blob, filename) {{
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}}
function reviewedLabelArray() {{
  const rows = data.suite2pRoiCount;
  const values = new Float64Array(rows);
  values.fill(NaN);
  for (let roi = 0; roi < data.nRois; roi++) {{
    const suite2pRoi = data.suite2pIndices[roi];
    const label = labels[roi];
    values[suite2pRoi] = label === -1 ? NaN : label;
  }}
  return values;
}}
async function saveManualLabels() {{
  const reviewedLabels = reviewedLabelArray();
  const blob = npyBlob(reviewedLabels, data.suite2pRoiCount);
  if ("showSaveFilePicker" in window) {{
    try {{
      const handle = await window.showSaveFilePicker({{
        suggestedName: "roi_manual_labels.npy",
        types: [{{description: "NumPy array", accept: {{"application/octet-stream": [".npy"]}}}}],
      }});
      const writable = await handle.createWritable();
      await writable.write(blob); await writable.close();
      return;
    }} catch (error) {{
      if (error.name === "AbortError") return;
      console.warn("Direct save failed; downloading instead", error);
    }}
  }}
  downloadBlob(blob, "roi_manual_labels.npy");
}}
async function saveHtmlWithLabels() {{
  const payloadScript = document.getElementById("payload");
  const savedPayload = {{
    ...data,
    initialLabels: Array.from(labels),
    initialMorphologyFilter: readFilter(),
    customMorphologyPresets: customPresets,
    initialFilterSet: loadedFilterSetName() || null,
    initialSort: currentSortSettings(),
    savedAt: new Date().toISOString(),
  }};
  const filename = `${{data.session}}_interactive_fov_roi_dff.html`;
  const htmlClone = document.documentElement.cloneNode(true);
  htmlClone.querySelector("#payload").textContent = JSON.stringify(savedPayload).replace(/</g, "\\u003c");
  htmlClone.querySelectorAll("dialog").forEach(dialog => {{
    dialog.removeAttribute("open");
  }});
  htmlClone.querySelectorAll(".info-box").forEach(box => {{
    box.setAttribute("hidden", "");
  }});
  htmlClone.querySelectorAll("[aria-expanded]").forEach(element => {{
    element.setAttribute("aria-expanded", "false");
  }});
  const blob = new Blob(["<!doctype html>\\n" + htmlClone.outerHTML], {{type: "text/html"}});
  if ("showSaveFilePicker" in window) {{
    try {{
      const handle = await window.showSaveFilePicker({{
        suggestedName: filename,
        types: [{{description: "HTML", accept: {{"text/html": [".html"]}}}}],
      }});
      const writable = await handle.createWritable();
      await writable.write(blob); await writable.close();
      return;
    }} catch (error) {{
      if (error.name === "AbortError") return;
      console.warn("Direct HTML save failed; downloading instead", error);
    }}
  }}
  downloadBlob(blob, filename);
}}
function importPresetObject(payload, fallbackName = "") {{
  // Older exports carry no name; fall back to the file name so they still import.
  const name = String(payload.name || payload.preset_name || payload.filter_set || fallbackName || "").trim();
  const filter = payload.filter || payload.morphology_filter || payload;
  if (!name) throw new Error("the filter set JSON has no name.");
  customPresets[name] = normalizeFilter(filter);
  populatePresetSelect(name);
  document.getElementById("filterPreset").value = `custom:${{name}}`;
  document.getElementById("presetName").value = name;
  updateMetricDefaults();
  writeFilter(customPresets[name]);
  if (applySortSettings(payload.sort)) applySort();
}}
document.getElementById("saveManualLabels").addEventListener("click", saveManualLabels);
document.getElementById("saveHtmlWithLabels").addEventListener("click", saveHtmlWithLabels);
document.getElementById("saveMetricSpreadsheet").addEventListener("click", saveMetricSpreadsheet);
document.getElementById("saveCurrentFilters").addEventListener("click", saveCurrentFilters);
const saveLabelsDialog = document.getElementById("saveLabelsDialog");
const saveLabelsHelp = document.getElementById("saveLabelsHelp");
const saveLabelsInfo = document.getElementById("saveLabelsInfo");
document.getElementById("openSaveLabelsDialog").addEventListener("click", () => {{
  if (typeof saveLabelsDialog.showModal === "function") saveLabelsDialog.showModal();
  else saveLabelsDialog.setAttribute("open", "");
}});
document.getElementById("closeSaveLabelsDialog").addEventListener("click", () => {{
  closeDialog(saveLabelsDialog);
}});
document.getElementById("closeSaveLabelsDialogTop").addEventListener("click", () => closeDialog(saveLabelsDialog));
saveLabelsInfo.addEventListener("click", () => {{
  const shouldShow = saveLabelsHelp.hasAttribute("hidden");
  saveLabelsHelp.toggleAttribute("hidden", !shouldShow);
  saveLabelsInfo.setAttribute("aria-expanded", String(shouldShow));
}});
updateModelDerivedVisibility();
document.querySelectorAll("[data-info-target]").forEach(button => {{
  button.addEventListener("click", () => {{
    const target = document.getElementById(button.dataset.infoTarget);
    if (!target) return;
    const shouldShow = target.hasAttribute("hidden");
    target.toggleAttribute("hidden", !shouldShow);
    button.setAttribute("aria-expanded", String(shouldShow));
  }});
}});
const presetFileInput = document.getElementById("presetFile");
document.getElementById("importPreset").addEventListener("click", () => presetFileInput.click());
presetFileInput.addEventListener("change", () => {{
  const file = presetFileInput.files && presetFileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      importPresetObject(JSON.parse(reader.result), file.name.replace(/\\.json$/i, ""));
    }} catch (error) {{
      alert(`Could not import the filter set: ${{error.message}}`);
    }}
  }};
  reader.readAsText(file);
}});
document.getElementById("resetFilter").addEventListener("click", resetFilter);
document.getElementById("applyFilterToLabels").addEventListener("click", applyFilterToLabels);
const morphologyDialog = document.getElementById("morphologyDialog");
document.getElementById("openMorphologyDialog").addEventListener("click", () => {{
  if (typeof morphologyDialog.showModal === "function") morphologyDialog.showModal();
  else morphologyDialog.setAttribute("open", "");
  requestAnimationFrame(drawMetricHistograms);
}});
document.getElementById("closeMorphologyDialog").addEventListener("click", () => {{
  closeDialog(morphologyDialog);
}});
document.getElementById("closeMorphologyDialogTop").addEventListener("click", () => closeDialog(morphologyDialog));
document.getElementById("filterPreset").addEventListener("change", loadSelectedPreset);
document.getElementById("savePreset").addEventListener("click", () => {{
  saveCurrentPresetToPage();
}});
document.getElementById("savePresetHtml").addEventListener("click", () => {{
  if (saveCurrentPresetToPage()) saveHtmlWithLabels();
}});
document.getElementById("toggleSideMenu").addEventListener("click", () => {{
  const review = document.querySelector(".review-main");
  const collapsed = !review.classList.contains("menu-collapsed");
  review.classList.toggle("menu-collapsed", collapsed);
  const button = document.getElementById("toggleSideMenu");
  button.textContent = collapsed ? "Show menu" : "Hide menu";
  button.setAttribute("aria-expanded", String(!collapsed));
  requestAnimationFrame(draw);
}});
startGbuSummary();
document.getElementById("summaryHide")?.addEventListener("click", () => setGbuSummaryVisible(false));
document.getElementById("summaryShow")?.addEventListener("click", () => setGbuSummaryVisible(true));
document.querySelectorAll("[data-rate-mode]").forEach(button => {{
  button.addEventListener("click", () => {{
    gbuRateMode = button.dataset.rateMode;
    document.querySelectorAll("[data-rate-mode]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
    scheduleGbuSummaryDraw();
  }});
}});
document.querySelectorAll("[data-transient-mode]").forEach(button => {{
  button.addEventListener("click", () => {{
    gbuTransientMode = button.dataset.transientMode;
    document.querySelectorAll("[data-transient-mode]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
    const roi = gbuTransientMode === "roi";
    document.getElementById("gbuTransientCanvas").hidden = roi;
    document.getElementById("oasisTransientCanvas").hidden = !roi;
    const sub = document.getElementById("transientSubtitle");
    sub.firstChild.textContent = roi ? "Selected ROI, with exponential decay fit " : "By label, mean \u00b1 s.e.m. across ROIs ";
    scheduleGbuSummaryDraw();
  }});
}});
bindFovResizer(document.getElementById("fovSplitter"), "x");
bindFovResizer(document.getElementById("fovResizeIcon"), "corner");
layoutFovRow();
if (document.readyState !== "complete") window.addEventListener("load", layoutFovRow);
["skewMin","skewMax","maxConnect","aspectMin","aspectMax","compactMin","compactMax","roiAreaMin","roiAreaMax","eventSnrMin","andreaPostdocSnrMin","oasisEventSnrMin","oasisRiseTauMin","oasisRiseTauMax","oasisDecayTauMin","oasisDecayTauMax","roiModelScoreProbabilityMin","roiModelScoreProbabilityMax"].forEach(id => {{
  const element = document.getElementById(id);
  if (!element) return;
  element.addEventListener("change", evaluateFilter);
  element.addEventListener("input", drawMetricHistograms);
}});
arrangeMetricBlocks();
document.querySelectorAll(".metric-histogram-panel").forEach(panel => {{
  panel.addEventListener("toggle", () => {{
    if (!panel.open) return;
    requestAnimationFrame(() => {{
      const canvas = panel.querySelector(".metric-histogram");
      if (canvas) drawMetricHistogram(canvas);
    }});
  }});
}});
window.addEventListener("keydown", event => {{
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key.toLowerCase() === "g") setLabel(1);
  else if (event.key.toLowerCase() === "b") setLabel(0);
  else if (event.key.toLowerCase() === "u") setLabel(2);
  else if (event.key.toLowerCase() === "n") setLabel(-1);
  else if (event.key === "ArrowLeft") moveVisible(-1);
  else if (event.key === "ArrowRight") moveVisible(1);
}});
document.getElementById("stackCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
let dragging=false, sx=0, start0=0, start1=0;
let stackDragging=false, stackSx=0, stackStart0=0, stackStart1=0;
document.getElementById("traceCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
document.getElementById("traceCanvas").addEventListener("mousedown", e => {{ dragging=true; sx=e.clientX; start0=x0; start1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!dragging) return; const rect=document.getElementById("traceCanvas").getBoundingClientRect(), shift=-(e.clientX-sx)/rect.width*(start1-start0); setFrameWindow(start0+shift, start1+shift); }});
window.addEventListener("mouseup", () => {{ dragging=false; document.getElementById("traceCanvas").classList.remove("dragging"); }});
document.getElementById("traceCanvas").addEventListener("dblclick", reset);
document.getElementById("stackCanvas").addEventListener("mousedown", e => {{ stackDragging=true; stackSx=e.clientX; stackStart0=x0; stackStart1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!stackDragging) return; const rect=document.getElementById("stackCanvas").getBoundingClientRect(), shift=-(e.clientX-stackSx)/rect.width*(stackStart1-stackStart0); setFrameWindow(stackStart0+shift, stackStart1+shift); }});
window.addEventListener("mouseup", () => {{ stackDragging=false; document.getElementById("stackCanvas").classList.remove("dragging"); }});
document.getElementById("stackCanvas").addEventListener("dblclick", reset);
let motionDragging=false, motionSx=0, motionStart0=0, motionStart1=0;
document.getElementById("motionDriftCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
document.getElementById("motionDriftCanvas").addEventListener("mousedown", e => {{ motionDragging=true; motionSx=e.clientX; motionStart0=x0; motionStart1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!motionDragging) return; const rect=document.getElementById("motionDriftCanvas").getBoundingClientRect(), shift=-(e.clientX-motionSx)/rect.width*(motionStart1-motionStart0); setFrameWindow(motionStart0+shift, motionStart1+shift); }});
window.addEventListener("mouseup", () => {{ motionDragging=false; document.getElementById("motionDriftCanvas").classList.remove("dragging"); }});
document.getElementById("motionDriftCanvas").addEventListener("dblclick", reset);
window.addEventListener("resize", () => {{ syncControlColumnHeight(); layoutFovRow(); }});
const RANGE_LABEL_BUILT_IN = [
  "roi_model_score_probability", "aspect", "compact", "connectivity", "roi_area", "skew",
  "snr_95_50", "andrea_postdoc_snr", "oasis_event_snr", "oasis_rise_tau_seconds", "oasis_decay_tau_seconds",
];
function rangeLabelMetrics() {{
  const metrics = RANGE_LABEL_BUILT_IN.filter(metric => finiteMetricValues(metric).length);
  for (const definition of customMetricDefinitions()) metrics.push(`custom:${{definition.key}}`);
  return metrics;
}}
function refreshRangeMetricOptions() {{
  const select = document.getElementById("rangeLabelMetric");
  if (!select) return;
  const current = select.value;
  select.textContent = "";
  for (const metric of rangeLabelMetrics()) {{
    const option = document.createElement("option");
    option.value = metric;
    option.textContent = metricLabel(metric);
    select.appendChild(option);
  }}
  if ([...select.options].some(option => option.value === current)) select.value = current;
  document.getElementById("fillModelRanges").hidden = !finiteMetricValues("roi_model_score_probability").length;
  updateRangeLabelPreview();
}}
function rangeLabelMode() {{
  return document.getElementById("rangeLabelMode").value;
}}
function addRangeLabelRow(min = "", max = "", target = "") {{
  const rows = document.getElementById("rangeLabelRows");
  const row = document.createElement("div");
  row.className = "range-label-row";
  const field = (text, cls, value) => {{
    const label = document.createElement("label");
    label.textContent = text;
    const input = document.createElement("input");
    input.type = "number"; input.step = "any"; input.placeholder = "Not Used"; input.className = cls;
    input.value = value === null || value === undefined ? "" : value;
    input.addEventListener("input", updateRangeLabelPreview);
    label.appendChild(input);
    return label;
  }};
  row.appendChild(field("Min", "range-min", min));
  row.appendChild(field("Max", "range-max", max));
  const targetLabel = document.createElement("label");
  targetLabel.className = "range-target-label";
  row.appendChild(targetLabel);
  row.dataset.target = target;
  const count = document.createElement("span");
  count.className = "range-count";
  row.appendChild(count);
  const remove = document.createElement("button");
  remove.type = "button"; remove.textContent = "Remove"; remove.className = "range-remove";
  remove.addEventListener("click", () => {{ row.remove(); updateRangeLabelPreview(); }});
  row.appendChild(remove);
  rows.appendChild(row);
  renderRangeTargets();
  updateRangeLabelPreview();
}}
function renderRangeTargets() {{
  const mode = rangeLabelMode();
  document.querySelectorAll("#rangeLabelRows .range-label-row").forEach((row, index) => {{
    const holder = row.querySelector(".range-target-label");
    const previous = holder.querySelector("select, input");
    const kept = previous ? previous.value : row.dataset.target;
    holder.textContent = mode === "gbu" ? "Label" : "Range name";
    let control;
    if (mode === "gbu") {{
      control = document.createElement("select");
      for (const [value, text] of [["1", "Good"], ["0", "Bad"], ["2", "Unsure"], ["-1", "Not labeled"]]) {{
        const option = document.createElement("option");
        option.value = value; option.textContent = text;
        control.appendChild(option);
      }}
      if (["1", "0", "2", "-1"].includes(String(kept))) control.value = String(kept);
    }} else {{
      control = document.createElement("input");
      control.type = "text";
      control.placeholder = `range ${{index + 1}}`;
      control.value = ["1", "0", "2", "-1"].includes(String(kept)) ? "" : (kept || "");
    }}
    control.className = "range-target";
    control.addEventListener("change", () => {{ row.dataset.target = control.value; }});
    row.dataset.target = control.value;
    holder.appendChild(control);
  }});
  document.getElementById("rangeLabelVariableNameLabel").hidden = mode !== "variable";
  document.getElementById("rangeLabelOnlyUnlabeledLabel").hidden = mode !== "gbu";
  document.getElementById("applyRangeLabels").textContent = mode === "gbu" ? "Assign labels" : "Create variable";
}}
function readRangeLabelRows() {{
  const read = input => {{
    if (!input || String(input.value).trim() === "") return null;
    const value = Number(input.value);
    return Number.isFinite(value) ? value : NaN;
  }};
  return Array.from(document.querySelectorAll("#rangeLabelRows .range-label-row")).map((row, index) => ({{
    row,
    index,
    min: read(row.querySelector(".range-min")),
    max: read(row.querySelector(".range-max")),
    target: row.querySelector(".range-target")?.value ?? "",
  }}));
}}
function matchRangeRow(value, ranges) {{
  if (!Number.isFinite(value)) return -1;
  for (let k = 0; k < ranges.length; k++) {{
    const range = ranges[k];
    if (range.min === null && range.max === null) continue;
    if (range.min !== null && !(value >= range.min)) continue;
    if (range.max !== null && !(value <= range.max)) continue;
    return k;
  }}
  return -1;
}}
function updateRangeLabelPreview() {{
  const metric = document.getElementById("rangeLabelMetric")?.value;
  const ranges = readRangeLabelRows();
  const counts = new Array(ranges.length).fill(0);
  let outside = 0, missing = 0;
  if (metric) {{
    for (let roi = 0; roi < data.nRois; roi++) {{
      const value = Number(metricValue(roi, metric));
      if (!Number.isFinite(value)) {{ missing++; continue; }}
      const k = matchRangeRow(value, ranges);
      if (k < 0) outside++; else counts[k]++;
    }}
  }}
  ranges.forEach((range, k) => {{
    const invalid = (range.min === null && range.max === null) || Number.isNaN(range.min) || Number.isNaN(range.max)
      || (range.min !== null && range.max !== null && range.min > range.max);
    range.row.querySelector(".range-count").textContent = invalid ? "set a min and/or max" : `${{counts[k].toLocaleString()}} ROIs`;
  }});
  const status = document.getElementById("rangeLabelStatus");
  if (status && !status.dataset.sticky) {{
    status.textContent = ranges.length
      ? `${{outside.toLocaleString()}} ROIs fall outside every range${{missing ? ` and ${{missing.toLocaleString()}} have no value for this metric` : ""}}; they will be left unchanged.`
      : "Add a range to start.";
  }}
  if (status) delete status.dataset.sticky;
}}
function fillModelScoreRanges() {{
  const scored = (data.roiModelScores || []).find(entry => entry && Number.isFinite(Number(entry.good_threshold)) && Number.isFinite(Number(entry.bad_threshold)));
  const good = scored ? Number(scored.good_threshold) : 0.85;
  const bad = scored ? Number(scored.bad_threshold) : 0.15;
  document.getElementById("rangeLabelMetric").value = "roi_model_score_probability";
  document.getElementById("rangeLabelMode").value = "gbu";
  document.getElementById("rangeLabelRows").textContent = "";
  addRangeLabelRow(good, "", "1");
  addRangeLabelRow("", bad, "0");
  addRangeLabelRow(bad, good, "2");
}}
function applyRangeLabels() {{
  const metric = document.getElementById("rangeLabelMetric").value;
  const ranges = readRangeLabelRows();
  const status = document.getElementById("rangeLabelStatus");
  const say = text => {{ status.textContent = text; status.dataset.sticky = "1"; }};
  if (!metric) return say("No numeric metric is available to assign from.");
  if (!ranges.length) return say("Add at least one range first.");
  for (const range of ranges) {{
    if ((range.min === null && range.max === null) || Number.isNaN(range.min) || Number.isNaN(range.max)) return say(`Range ${{range.index + 1}} needs a numeric min and/or max.`);
    if (range.min !== null && range.max !== null && range.min > range.max) return say(`Range ${{range.index + 1}} has its min above its max.`);
  }}
  const mode = rangeLabelMode();
  if (mode === "gbu") {{
    const onlyUnlabeled = document.getElementById("rangeLabelOnlyUnlabeled").checked;
    const counts = {{1: 0, 0: 0, 2: 0, "-1": 0}};
    let outside = 0, skipped = 0;
    for (let roi = 0; roi < data.nRois; roi++) {{
      const k = matchRangeRow(Number(metricValue(roi, metric)), ranges);
      if (k < 0) {{ outside++; continue; }}
      if (onlyUnlabeled && labels[roi] !== -1) {{ skipped++; continue; }}
      const label = Number(ranges[k].target);
      labels[roi] = label;
      counts[label]++;
    }}
    updateLabelControls();
    updateVisibleRois(true);
    say(`Labeled ${{counts[1].toLocaleString()}} good, ${{counts[0].toLocaleString()}} bad, ${{counts[2].toLocaleString()}} unsure and ${{counts[-1].toLocaleString()}} not labeled from ${{metricLabel(metric)}}; ${{outside.toLocaleString()}} ROIs were outside every range or had no value${{skipped ? `, and ${{skipped.toLocaleString()}} already labeled ROIs were left as they were` : ""}}.`);
    return;
  }}
  const name = document.getElementById("rangeLabelVariableName").value.trim();
  if (!name) return say("Enter a name for the new variable first.");
  if (!data.customRoiMetrics || !Array.isArray(data.customRoiMetrics.values)) {{
    data.customRoiMetrics = {{definitions: [], values: Array.from({{length: data.nRois}}, () => ({{}})), sources: []}};
  }}
  const key = ensureUniqueCustomMetricKey(customMetricKey(name));
  const categories = {{}};
  ranges.forEach((range, k) => {{ categories[String(k + 1)] = String(range.target || "").trim() || `range ${{k + 1}}`; }});
  const tally = new Array(ranges.length).fill(0);
  for (let roi = 0; roi < data.nRois; roi++) {{
    const k = matchRangeRow(Number(metricValue(roi, metric)), ranges);
    data.customRoiMetrics.values[roi][key] = k < 0 ? null : k + 1;
    if (k >= 0) tally[k]++;
  }}
  data.customRoiMetrics.definitions.push({{key, label: name, source: `ranges of ${{metricLabel(metric)}}`, origin: "ranges", section: "uploaded", categories}});
  renderCustomMetricControls();
  renderCustomSortMetrics();
  updateSortMetricHandlers();
  updateMetricDefaults();
  evaluateFilter();
  say(`Created "${{name}}" under Uploaded ROI Metrics: ${{ranges.map((range, k) => `${{k + 1}} = ${{categories[String(k + 1)]}} (${{tally[k].toLocaleString()}} ROIs)`).join(", ")}}.`);
}}
function initRangeLabels() {{
  document.getElementById("rangeLabelMetric").addEventListener("change", updateRangeLabelPreview);
  document.getElementById("rangeLabelMode").addEventListener("change", renderRangeTargets);
  document.getElementById("addRangeLabelRow").addEventListener("click", () => addRangeLabelRow());
  document.getElementById("fillModelRanges").addEventListener("click", fillModelScoreRanges);
  document.getElementById("applyRangeLabels").addEventListener("click", applyRangeLabels);
  refreshRangeMetricOptions();
  if (finiteMetricValues("roi_model_score_probability").length) fillModelScoreRanges();
  else addRangeLabelRow();
}}
makeOverlays(); syncTimeInputs(); renderCustomMetricControls(); renderCustomSortMetrics(); updateSortMetricHandlers(); updateMetricDefaults(); updateCellTypeLabelStatus(); updateRoiDisplaySummary();
const initialFilterSet = data.initialFilterSet && (customPresets[data.initialFilterSet] || data.morphologyPresets[data.initialFilterSet]) ? data.initialFilterSet : "all_rois";
populatePresetSelect(initialFilterSet); resetFilter();
if (data.initialMorphologyFilter) writeFilter(data.initialMorphologyFilter);
initRangeLabels();
applySortSettings(data.initialSort);
applySort(); syncControlColumnHeight(); setSelected(visibleRois[0]);
if (data.dffStorageMode === "file" && data.dffSidecarName) {{
  loadSidecarByRelativeName(data.dffSidecarName, setDffFromArrayBuffer, "dF/F");
}}
if (data.oasisAvailable && data.oasisStorageMode === "file" && data.oasisSidecarName) {{
  loadSidecarByRelativeName(data.oasisSidecarName, setOasisFromArrayBuffer, "inferred spikes");
}}
requestAnimationFrame(() => {{ syncControlColumnHeight(); draw(); }});
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def create_preprocessing_summary(
    session_data_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    pdf_name: str | None = None,
    html_name: str | None = None,
    target_structure: str | None = None,
    input_layout: str = "auto",
    initialize_labels_from_roi_model_scores: bool = False,
    custom_roi_metrics: list[str | os.PathLike[str]] | None = None,
    custom_roi_metric_sections: list[str] | None = None,
    filter_settings: str | os.PathLike[str] | None = None,
    html_only: bool = False,
) -> tuple[Path, Path]:
    input_path = Path(os.path.abspath(os.path.expanduser(os.fspath(session_data_path))))
    session_dir, resolved_layout = _resolve_summary_session_and_layout(input_path, input_layout)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else session_dir
    loaded_filter_settings = _load_filter_settings(filter_settings) if filter_settings else None
    globals()["_LAST_BUILD_WAS_HTML_ONLY"] = False
    if html_only:
        existing_html = out_dir / (html_name or HTML_NAME_TEMPLATE.format(session_name=session_dir.name))
        existing_pdf = out_dir / (pdf_name or PDF_NAME_TEMPLATE.format(session_name=session_dir.name))
        if existing_html.exists():
            _rebuild_reviewer_html(
                existing_html,
                input_layout_name=resolved_layout.name,
                target_structure=target_structure,
                custom_roi_metrics=custom_roi_metrics,
                custom_roi_metric_sections=custom_roi_metric_sections,
                filter_settings=loaded_filter_settings,
            )
            globals()["_LAST_BUILD_WAS_HTML_ONLY"] = True
            return existing_pdf, existing_html
        print(f"--html-only: no existing reviewer at {existing_html}; generating everything instead.")
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = _load_ops(session_dir)
    _log_summary_input_layout(resolved_layout, input_layout)
    qc_parameters = _load_qc_parameters(session_dir)
    pipeline_parameters = _load_pipeline_parameters(session_dir)
    xoff, yoff = _load_offsets(session_dir, ops)
    fluo_probe = np.load(resolved_layout.fluo_path, allow_pickle=False, mmap_mode="r")

    mean_green = _load_layout_image(session_dir, resolved_layout, "mean_func")
    if mean_green is None:
        mean_green = np.asarray(ops["meanImg"]) if "meanImg" in ops else None
    elif "meanImg" in ops and np.asarray(mean_green).shape[:2] != np.asarray(ops["meanImg"]).shape[:2]:
        mean_green = np.asarray(ops.get("meanImg"))
    suite2p_stat = _load_layout_stat(resolved_layout, traces=fluo_probe, mean_green=mean_green)
    if mean_green is None:
        mean_green = np.zeros(_stat_image_shape(suite2p_stat), dtype=np.float32)
    max_green = _load_layout_image(session_dir, resolved_layout, "max_func")
    if max_green is None:
        max_green = np.asarray(ops.get("max_proj", ops.get("maxImg", mean_green)))
    elif np.asarray(max_green).shape[:2] != np.asarray(mean_green).shape[:2]:
        max_green = np.asarray(ops.get("max_proj", ops.get("maxImg", mean_green)))
        if np.asarray(max_green).shape[:2] != np.asarray(mean_green).shape[:2]:
            max_green = mean_green
    mean_red = _load_layout_image(session_dir, resolved_layout, "mean_anat")
    red_available = mean_red is not None
    if (
        (mean_red is None or np.asarray(mean_red).shape[:2] != np.asarray(mean_green).shape[:2])
        and ops.get("nchannels", 1) != 1
        and "meanImg_chan2" in ops
    ):
        mean_red = np.asarray(ops.get("meanImg_chan2"))
        red_available = True
    max_red = _load_layout_image(session_dir, resolved_layout, "max_anat")
    if max_red is None and red_available:
        max_red = np.asarray(ops.get("meanImg_chan2_corrected", mean_red))
    elif max_red is not None and red_available and np.asarray(max_red).shape[:2] != np.asarray(mean_green).shape[:2]:
        max_red = np.asarray(ops.get("meanImg_chan2_corrected", mean_red))
        if np.asarray(max_red).shape[:2] != np.asarray(mean_green).shape[:2]:
            max_red = mean_red
    if not red_available:
        mean_red = None
        max_red = None
    cellpose_mask = (
        _load_masks_h5_label_mask(session_dir, "masks_anat", np.asarray(mean_green).shape[:2], ops)
        if red_available
        else None
    )

    # Frame rate: Suite2p's ops["fs"]; for external masks, "frame_rate" (Hz) in
    # external_roi_metadata.json; otherwise 30 Hz.
    external_frame_rate = (
        _load_external_metadata(resolved_layout.input_dir).get("frame_rate")
        if resolved_layout.name == "external_rois"
        else None
    )
    if "fs" in ops:
        frame_rate = float(ops["fs"])
    elif external_frame_rate is not None:
        frame_rate = float(external_frame_rate)
    else:
        frame_rate = 30.0
    dff_label = resolved_layout.dff_label
    n_rois = min(int(fluo_probe.shape[0]), len(suite2p_stat))
    n_frames = int(fluo_probe.shape[1])
    stat = suite2p_stat[:n_rois]
    suite2p_indices = np.arange(n_rois, dtype=np.int64)
    estimated_embedded_dff_bytes = _estimate_embedded_dff_bytes(n_rois, n_frames)
    dff_storage_mode = "embedded" if estimated_embedded_dff_bytes <= EMBEDDED_DFF_BYTE_LIMIT else "file"
    dff_sidecar_name = None
    dff_sidecar_path = None
    if dff_storage_mode == "file":
        dff_sidecar_name = f"{session_dir.name}_dff.npy"
        dff_sidecar_path = out_dir / dff_sidecar_name
    dff, dff_metrics = _build_dff_and_metrics(
        session_dir,
        ops,
        resolved_layout,
        n_rois,
        n_frames,
        frame_rate,
        dff_storage_mode,
        dff_sidecar_path,
    )
    dff_metrics = _add_roi_area_metrics(dff_metrics, stat)
    oasis_spikes, oasis_attrs = _load_oasis_spikes(session_dir, n_rois, n_frames, resolved_layout)
    oasis_storage_mode = "none"
    oasis_sidecar_name = None
    if oasis_spikes is not None:
        dff_metrics = _add_oasis_event_metrics(
            dff_metrics,
            dff,
            dff_sidecar_path,
            oasis_spikes,
            float(oasis_attrs.get("event_threshold", 0.05)),
            frame_rate,
        )
        oasis_storage_mode = "embedded" if estimated_embedded_dff_bytes <= EMBEDDED_DFF_BYTE_LIMIT else "file"
        if oasis_storage_mode == "file":
            oasis_sidecar_name = f"{session_dir.name}_oasis_spikes.npy"
            oasis_sidecar_path = out_dir / oasis_sidecar_name
            existing_oasis_sidecar = _load_matching_sidecar(
                oasis_sidecar_path,
                tuple(oasis_spikes.shape),
                label="inferred spikes",
            )
            if existing_oasis_sidecar is None:
                np.save(oasis_sidecar_path, oasis_spikes)
    mask = _stat_to_mask(stat, np.asarray(mean_green).shape[:2])
    iscell_path = resolved_layout.iscell_path
    iscell = load_iscell(iscell_path, n_rois)
    if not iscell_path.exists():
        iscell[:, :] = 1.0
    suite2p_fingerprint = suite2p_stat_fingerprint(stat)
    morphology_metrics = roi_morphology_metrics(stat)
    cell_type_labels, cell_type_label_source = _load_cell_type_labels(session_dir, n_rois, stat)
    roi_model_scores_labels, roi_model_score_label_source = _load_roi_label_h5(session_dir, n_rois, stat)
    roi_model_scores, roi_model_score_source = _load_roi_model_scores(session_dir, n_rois)
    custom_roi_metric_payload = _load_custom_roi_metrics(custom_roi_metrics, n_rois, suite2p_indices)
    for definition in custom_roi_metric_payload.get("definitions", []):
        definition.setdefault("origin", "command line")
    _assign_custom_metric_sections(custom_roi_metric_payload, custom_roi_metric_sections)
    target_structure = target_structure or _target_structure(pipeline_parameters, qc_parameters)
    preset_exclusion_reasons = (
        morphology_exclusion_reasons(morphology_metrics, qc_parameters)
        if qc_parameters is not None
        else [[] for _ in range(n_rois)]
    )
    pdf_name = pdf_name or PDF_NAME_TEMPLATE.format(session_name=session_dir.name)
    html_name = html_name or HTML_NAME_TEMPLATE.format(session_name=session_dir.name)
    pdf_path = out_dir / pdf_name
    html_path = out_dir / html_name
    _write_summary_pdf(
        pdf_path,
        session_name=session_dir.name,
        n_rois=n_rois,
        mean_green=mean_green,
        max_green=max_green,
        mean_red=mean_red,
        max_red=max_red,
        mask=mask,
        xoff=xoff,
        yoff=yoff,
        frame_rate=frame_rate,
    )
    _write_html(
        html_path,
        input_layout_name=resolved_layout.name,
        session_name=session_dir.name,
        mean_green=mean_green,
        mean_red=mean_red,
        cellpose_mask=cellpose_mask,
        mask=mask,
        stat=stat,
        suite2p_indices=suite2p_indices,
        iscell=iscell,
        suite2p_fingerprint=suite2p_fingerprint,
        morphology_metrics=morphology_metrics,
        cell_type_labels=cell_type_labels,
        cell_type_label_source=cell_type_label_source,
        roi_model_scores_labels=roi_model_scores_labels,
        roi_model_score_label_source=roi_model_score_label_source,
        roi_model_scores=roi_model_scores,
        roi_model_score_source=roi_model_score_source,
        custom_roi_metrics=custom_roi_metric_payload,
        preset_exclusion_reasons=preset_exclusion_reasons,
        qc_parameters=qc_parameters,
        target_structure=target_structure,
        n_rois=n_rois,
        n_frames=n_frames,
        dff_label=dff_label,
        frame_rate=frame_rate,
        dff_metrics=dff_metrics,
        dff=dff,
        dff_storage_mode=dff_storage_mode,
        dff_sidecar_name=dff_sidecar_name,
        estimated_embedded_dff_bytes=estimated_embedded_dff_bytes,
        xoff=xoff,
        yoff=yoff,
        oasis_spikes=oasis_spikes,
        oasis_attrs=oasis_attrs,
        oasis_storage_mode=oasis_storage_mode,
        oasis_sidecar_name=oasis_sidecar_name,
        initialize_labels_from_roi_model_scores=initialize_labels_from_roi_model_scores,
        filter_settings=loaded_filter_settings,
    )
    return pdf_path, html_path


def create_processing_summary(
    session_data_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    pdf_name: str | None = None,
    html_name: str | None = None,
    target_structure: str | None = None,
    input_layout: str = "auto",
    initialize_labels_from_roi_model_scores: bool = False,
    custom_roi_metrics: list[str | os.PathLike[str]] | None = None,
    custom_roi_metric_sections: list[str] | None = None,
    filter_settings: str | os.PathLike[str] | None = None,
    html_only: bool = False,
) -> tuple[Path, Path]:
    """Create the processing summary PDF and interactive HTML outputs.

    With ``html_only=True`` and an existing reviewer in the output directory, only the
    HTML is rebuilt, from the data already embedded in it; nothing is recomputed and the
    PDF and sidecar files are left as they are.
    """

    return create_preprocessing_summary(
        session_data_path,
        output_dir=output_dir,
        pdf_name=pdf_name,
        html_name=html_name,
        target_structure=target_structure,
        input_layout=input_layout,
        initialize_labels_from_roi_model_scores=initialize_labels_from_roi_model_scores,
        custom_roi_metrics=custom_roi_metrics,
        custom_roi_metric_sections=custom_roi_metric_sections,
        filter_settings=filter_settings,
        html_only=html_only,
    )


def run(ops: dict[str, Any], output_dir: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
    return create_processing_summary(ops["save_path0"], output_dir=output_dir)


def archive_summary_outputs(output_folder: Path, archive_path: Path | None = None) -> Path:
    """Create a zip archive with ``output_folder`` as the top-level folder."""

    output_folder = Path(output_folder).expanduser().resolve()
    if not output_folder.is_dir():
        raise FileNotFoundError(f"Summary output folder does not exist: {output_folder}")
    archive_path = (
        Path(archive_path).expanduser().resolve()
        if archive_path is not None
        else output_folder.parent / f"{output_folder.name}_preprocessing_summary.zip"
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output_folder.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, arcname=path.relative_to(output_folder.parent))
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create processing PDF and interactive HTML outputs.")
    parser.add_argument(
        "session_data_path",
        help=(
            "Processed session root or a concrete data directory such as "
            "suite2p/plane0, suite2p, qc_results, or manual_qc_results."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: resolved session root.")
    parser.add_argument("--pdf-name", default=None)
    parser.add_argument("--html-name", default=None)
    parser.add_argument(
        "--archive-output",
        action="store_true",
        help=(
            "Write outputs into a subfolder named after the resolved session and create a compressed zip archive "
            "containing that folder. This preserves HTML sidecar-file relative paths after extraction."
        ),
    )
    parser.add_argument(
        "--archive-name",
        default=None,
        help=(
            "Optional zip filename or path used with --archive-output. Default: "
            "<session_name>_preprocessing_summary.zip in --output-dir."
        ),
    )
    parser.add_argument(
        "--input-layout",
        choices=["auto", "suite2p", "qc_results", "manual_qc_results", "external_rois"],
        default="auto",
        help=(
            "Input file layout for ROI/stat/trace arrays. 'suite2p' uses "
            "suite2p/plane0/stat.npy, F.npy, Fneu.npy, and iscell.npy. "
            "'qc_results' and 'manual_qc_results' map stat.npy, fluo.npy, "
            "neuropil.npy, and optional iscell.npy from those filtered-output "
            "directories. 'external_rois' accepts stat.npy, a dense ROI label "
            "mask, or a sparse ROI spatial-component matrix with matching "
            "F.npy/fluo.npy traces and optional Fneu.npy/neuropil.npy. "
            "If session_data_path already points to a concrete "
            "layout directory, that directory is used directly. Default: "
            "auto-detect Suite2p before filtered layouts from a session root."
        ),
    )
    parser.add_argument(
        "--filter-set",
        dest="filter_set",
        choices=SUMMARY_TARGET_STRUCTURES,
        default=None,
        help=(
            "Built-in filter set (ROI metric thresholds) the reviewer's distributions start from; "
            "default: the filter set matching the session's pipeline target structure."
        ),
    )
    # Former name of --filter-set, kept so existing pipeline calls and scripts keep working.
    parser.add_argument("--target-structure", dest="filter_set", choices=SUMMARY_TARGET_STRUCTURES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--initialize-labels-from-roi-model-scores",
        "--initialize-labels-from-roi-qc",
        dest="initialize_labels_from_roi_model_scores",
        action="store_true",
        help=(
            "Initialize reviewer labels from ROI_label.h5 / roi_model_scores.h5. "
            "By default, all ROIs open as not labeled and model score outputs are loaded only as metrics."
        ),
    )
    parser.add_argument(
        "--roi-index-label",
        default=None,
        help=(
            "Label used for the original ROI index in the reviewer, e.g. \"Original Index\". "
            f"Default: {ROI_INDEX_LABEL}."
        ),
    )
    parser.add_argument(
        "--custom-roi-metrics",
        action="append",
        default=[],
        help=(
            "Path to ROI-level custom metrics to embed in the reviewer. Can be repeated. "
            "Supported formats: CSV/TSV/JSON with one numeric value per ROI per metric, "
            "or NPY/NPZ arrays shaped (n_rois,) or (n_rois, n_metrics). CSV/TSV/JSON may "
            "include a roi_index column (the 0-based original ROI index) to map rows to ROIs; "
            "otherwise row k is ROI k."
        ),
    )
    parser.add_argument(
        "--custom-roi-metric-section",
        action="append",
        default=[],
        metavar="NAME=SECTION",
        help=(
            "Show a custom ROI metric under a given section of the Filter menu (and the matching Sort group) "
            f"instead of 'Uploaded ROI Metrics'. NAME is the metric's column name; SECTION is one of "
            f"{', '.join(CUSTOM_METRIC_SECTIONS)}. Can be repeated."
        ),
    )
    parser.add_argument(
        "--filter-settings",
        default=None,
        help=(
            "Filter/sorting settings to open the reviewer with: a JSON file exported with "
            "'Export current filter/sorting settings JSON' (or the same keys in JSON/YAML). "
            "Its thresholds are added as a saved filter set and loaded, and its sort order is applied."
        ),
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help=(
            "Rebuild only the reviewer HTML from the data embedded in the existing reviewer in the output "
            "directory, without recomputing metrics, sidecars or the PDF. Labels and settings saved in that "
            "HTML are kept. Falls back to a full build when no reviewer exists yet."
        ),
    )
    args = parser.parse_args()
    if args.roi_index_label:
        globals()["ROI_INDEX_LABEL"] = args.roi_index_label
    output_dir = args.output_dir
    archive_path = None
    if args.archive_output:
        input_path = Path(os.path.abspath(os.path.expanduser(os.fspath(args.session_data_path))))
        session_dir, _ = _resolve_summary_session_and_layout(input_path, args.input_layout)
        base_output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session_dir
        output_dir = base_output_dir / session_dir.name
        if args.archive_name:
            raw_archive_path = Path(args.archive_name).expanduser()
            archive_path = raw_archive_path if raw_archive_path.is_absolute() else base_output_dir / raw_archive_path
        else:
            archive_path = base_output_dir / f"{session_dir.name}_preprocessing_summary.zip"
    pdf_path, html_path = create_processing_summary(
        args.session_data_path,
        output_dir=output_dir,
        pdf_name=args.pdf_name,
        html_name=args.html_name,
        target_structure=args.filter_set,
        input_layout=args.input_layout,
        initialize_labels_from_roi_model_scores=args.initialize_labels_from_roi_model_scores,
        custom_roi_metrics=args.custom_roi_metrics,
        custom_roi_metric_sections=args.custom_roi_metric_section,
        filter_settings=args.filter_settings,
        html_only=args.html_only,
    )
    if _LAST_BUILD_WAS_HTML_ONLY:
        print(f"Rebuilt the reviewer from its embedded data; kept the existing PDF and sidecars: {pdf_path.parent}")
    else:
        print(f"Saved processing PDF: {pdf_path}")
    print(f"Saved interactive HTML: {html_path}")
    if args.archive_output:
        archived = archive_summary_outputs(Path(output_dir), archive_path=archive_path)
        print(f"Saved compressed summary archive: {archived}")


if __name__ == "__main__":
    main()
