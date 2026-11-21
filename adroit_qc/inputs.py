"""Write a session directory in the documented input layout from Python arrays.

Every reviewer input other than a Suite2p ``suite2p/plane0`` folder is a file in one
directory, the *documented layout* (docs: Preparing your inputs). These helpers write
each file with the name, shape, dtype and container the generator reads, so arrays
from any segmentation, trace-extraction or spike-inference software can be reviewed:

    from adroit_qc import inputs

    session = inputs.write_session(
        "data/my_session",
        masks=label_image,            # (Ly, Lx) ints, 0 = background, ROI k has label k + 1
        F=raw_traces,                 # (n_rois, n_frames)
        Fneu=neuropil_traces,         # optional
        spikes=deconvolved,           # optional, any method, same shape as F
        mean_func=mean_image,         # optional
        frame_rate=30.0,
    )
    inputs.validate(session)          # prints the report; returns it
    inputs.build_reviewer(session, output_dir="outputs/my_session")

Row ``i`` of every per-ROI array must describe the ROI whose mask has label ``i + 1``
(or column ``i`` of a sparse component matrix). Nothing can check that pairing for you;
spot-check a known active ROI in the reviewer.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from scipy import sparse

__all__ = [
    "write_session",
    "write_label_mask",
    "write_spatial_components",
    "write_traces",
    "write_dff",
    "write_spikes",
    "write_images",
    "write_motion_offsets",
    "write_cell_types",
    "write_iscell",
    "write_classifier_scores",
    "write_metadata",
    "write_metric_table",
    "validate",
    "build_reviewer",
]



def _dir(path: Any) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _trace_matrix(values: Any, label: str, n_rois: int | None = None) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be 2-D (n_rois, n_frames); got shape {arr.shape}")
    if n_rois is not None and arr.shape[0] != n_rois:
        if arr.shape[1] == n_rois:
            raise ValueError(f"{label} has shape {arr.shape}; it looks frames-by-ROIs. Pass {label}.T")
        raise ValueError(f"{label} has {arr.shape[0]} rows; expected one per ROI ({n_rois})")
    return np.ascontiguousarray(arr, dtype=np.float32)


def write_label_mask(out_dir: Any, masks: Any) -> Path:
    """Write ``roi_mask.npy``: a 2-D integer label image.

    0 (or negative) is background. The distinct positive labels, sorted ascending, are
    trace rows 0, 1, 2, ...; with labels 1..N that is "label k + 1 is row k". When trace
    rows follow some other order, pass ``row_order`` to :func:`write_session`.
    Returns the number of ROIs.
    """
    mask = np.asarray(masks)
    if mask.ndim != 2:
        raise ValueError(f"masks must be a 2-D label image; got shape {mask.shape}")
    mask = mask.astype(np.int32, copy=False)
    labels = np.unique(mask[mask > 0])
    if labels.size == 0:
        raise ValueError("masks contain no ROI (no positive labels)")
    np.save(_dir(out_dir) / "roi_mask.npy", mask)
    return int(labels.size)


def write_spatial_components(
    out_dir: Any,
    components: Any,
    image_shape: Sequence[int],
    flatten_order: str = "F",
) -> int:
    """Write ``spatial_components.npz``: a sparse (n_pixels, n_rois) weight matrix.

    Column ``k`` holds ROI ``k``'s pixel weights, as CaImAn's ``estimates.A``.
    ``image_shape`` is (Ly, Lx); ``flatten_order`` is how pixels were flattened:
    ``"F"`` (column-major, CaImAn's convention) or ``"C"`` (row-major). Both are stored
    in ``external_roi_metadata.json``. Returns the number of ROIs.
    """
    matrix = sparse.csc_matrix(components)
    ly, lx = (int(v) for v in image_shape)
    if matrix.shape[0] != ly * lx:
        if matrix.shape[1] == ly * lx:
            raise ValueError(f"components have shape {matrix.shape}; expected (pixels, ROIs). Pass components.T")
        raise ValueError(f"components have {matrix.shape[0]} rows; image_shape {ly}x{lx} has {ly * lx} pixels")
    if flatten_order not in ("F", "C"):
        raise ValueError("flatten_order must be 'F' or 'C'")
    sparse.save_npz(_dir(out_dir) / "spatial_components.npz", matrix)
    write_metadata(out_dir, image_shape=(ly, lx), flatten_order=flatten_order)
    return int(matrix.shape[1])


def write_traces(out_dir: Any, F: Any, Fneu: Any = None, n_rois: int | None = None) -> tuple[int, int]:
    """Write ``F.npy`` (raw fluorescence) and, if given, ``Fneu.npy`` (neuropil).

    Both are (n_rois, n_frames) float32, C-ordered. Without ``Fneu`` the generator
    subtracts nothing. dF/F is computed as Suite2p does,
    ``(F - c * Fneu - baseline) / baseline`` with ``c`` = 0.7 (or the metadata's
    ``neuropil_coefficient``) and a Gaussian-filtered baseline, unless ``dff.h5`` is
    supplied (:func:`write_dff`). Returns (n_rois, n_frames).
    """
    out = _dir(out_dir)
    raw = _trace_matrix(F, "F", n_rois)
    np.save(out / "F.npy", raw)
    if Fneu is not None:
        neu = _trace_matrix(Fneu, "Fneu", raw.shape[0])
        if neu.shape != raw.shape:
            raise ValueError(f"Fneu shape {neu.shape} differs from F {raw.shape}")
        np.save(out / "Fneu.npy", neu)
    return raw.shape


def write_dff(out_dir: Any, dff: Any, n_rois: int | None = None) -> Path:
    """Write ``dff.h5`` (dataset ``dff``): a precomputed (n_rois, n_frames) dF/F.

    When present it is displayed and measured as given instead of the dF/F computed
    from ``F.npy``. ``F.npy`` is still required (it fixes the ROI and frame counts and
    feeds the classifier inputs), and the shapes must match.
    """
    path = _dir(out_dir) / "dff.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("dff", data=_trace_matrix(dff, "dff", n_rois))
    return path


def write_spikes(out_dir: Any, spikes: Any, n_rois: int | None = None, event_threshold: float | None = None) -> Path:
    """Write ``spikes.h5`` (dataset ``spikes``): inferred spikes from any method.

    Same shape as the traces. Values can be amplitudes, rates, probabilities, counts or
    0/1 events; the reviewer's spike threshold applies to them in their own units.
    ``event_threshold`` sets the threshold the reviewer opens with (default 0.05).
    """
    path = _dir(out_dir) / "spikes.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("spikes", data=_trace_matrix(spikes, "spikes", n_rois))
        if event_threshold is not None:
            handle.attrs["event_threshold"] = float(event_threshold)
    return path


def write_images(
    out_dir: Any,
    mean_func: Any = None,
    mean_anat: Any = None,
    max_func: Any = None,
    max_anat: Any = None,
    image_shape: Sequence[int] | None = None,
) -> list[Path]:
    """Write the field-of-view images: ``mean_func.npy``, ``mean_anat.npy``,
    ``max_func.npy``, ``max_anat.npy``. Each is a 2-D (Ly, Lx) float image in the
    masks' pixel coordinates; all are optional (without ``mean_func`` the outlines are
    drawn on a blank background)."""
    out = _dir(out_dir)
    written = []
    for name, image in (("mean_func", mean_func), ("mean_anat", mean_anat), ("max_func", max_func), ("max_anat", max_anat)):
        if image is None:
            continue
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2-D (Ly, Lx); got {arr.shape}")
        if image_shape is not None and tuple(arr.shape) != tuple(int(v) for v in image_shape):
            raise ValueError(f"{name} shape {arr.shape} does not match the masks {tuple(image_shape)}")
        np.save(out / f"{name}.npy", np.ascontiguousarray(arr))
        written.append(out / f"{name}.npy")
    return written


def write_motion_offsets(out_dir: Any, xoff: Any, yoff: Any) -> Path:
    """Write ``move_offset.h5`` with datasets ``xoff`` and ``yoff``: the rigid shift of
    each frame in pixels, one value per frame, in trace frame order."""
    x = np.asarray(xoff, dtype=np.float32).reshape(-1)
    y = np.asarray(yoff, dtype=np.float32).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"xoff has {x.size} values and yoff {y.size}")
    path = _dir(out_dir) / "move_offset.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("xoff", data=x)
        handle.create_dataset("yoff", data=y)
    return path


def write_cell_types(out_dir: Any, codes: Any, n_rois: int | None = None) -> Path:
    """Write ``roi_cell_type_labels.npy``: one float per ROI, ``1`` inhibitory (red),
    ``-1`` excitatory (non-red), ``0`` unsure, ``NaN`` unknown. For any other
    categories (cell classes, compartments, layers) use :func:`write_metric_table`
    with a ``cell_identity`` categorical instead."""
    arr = np.asarray(codes, dtype=np.float32).reshape(-1)
    if n_rois is not None and arr.size != n_rois:
        raise ValueError(f"{arr.size} cell type codes for {n_rois} ROIs")
    finite = arr[np.isfinite(arr)]
    if not np.isin(finite, [-1, 0, 1]).all():
        raise ValueError("cell type codes must be -1, 0, 1 or NaN; use write_metric_table for other categories")
    path = _dir(out_dir) / "roi_cell_type_labels.npy"
    np.save(path, arr)
    return path


def write_iscell(out_dir: Any, accepted: Any, probability: Any = None) -> Path:
    """Write ``iscell.npy``: (n_rois, 2), an upstream accept (1) / reject (0) decision and
    its probability, as Suite2p stores it. Shown for reference; it does not filter."""
    acc = np.asarray(accepted, dtype=np.float32).reshape(-1)
    prob = acc if probability is None else np.asarray(probability, dtype=np.float32).reshape(-1)
    path = _dir(out_dir) / "iscell.npy"
    np.save(path, np.stack([acc, prob], axis=1))
    return path


def write_classifier_scores(
    out_dir: Any,
    scores: Any,
    good_threshold: float,
    bad_threshold: float,
    *,
    states: Sequence[str] | None = None,
    roi: Any = None,
    model_source: str = "",
    target_structure: str = "",
) -> Path:
    """Write ``roi_model_scores.h5`` from any classifier.

    ``scores`` are 0-1 values, higher for ROIs more likely good. ``state`` per ROI is
    ``good`` (score >= good_threshold), ``bad`` (<= bad_threshold) or ``gray``, computed
    here unless ``states`` is given. ``roi`` gives each row's original ROI index when the
    rows are not in ROI order. ``model_source`` is free-text provenance; do not put
    absolute paths with user names in files you will share.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not (0 <= bad_threshold < good_threshold <= 1):
        raise ValueError("need 0 <= bad_threshold < good_threshold <= 1")
    if states is None:
        states = np.where(s >= good_threshold, "good", np.where(s <= bad_threshold, "bad", "gray"))
    states = np.asarray(states, dtype=object)
    if states.size != s.size:
        raise ValueError(f"{states.size} states for {s.size} scores")
    path = _dir(out_dir) / "roi_model_scores.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("classifier_score", data=s)
        handle.create_dataset("state", data=states.astype(str).astype(object), dtype=h5py.string_dtype("utf-8"))
        handle.create_dataset("roi", data=np.arange(s.size) if roi is None else np.asarray(roi, dtype=np.int64))
        handle.attrs["good_threshold"] = float(good_threshold)
        handle.attrs["bad_threshold"] = float(bad_threshold)
        handle.attrs["model_source"] = str(model_source)
        handle.attrs["target_structure"] = str(target_structure)
    return path


def write_metadata(out_dir: Any, **fields: Any) -> Path:
    """Merge fields into ``external_roi_metadata.json``.

    Recognized fields: ``image_shape`` [Ly, Lx] (needed for sparse components without a
    mean image), ``flatten_order`` ("F" or "C"), ``frame_rate`` (Hz; default 30),
    ``diameter`` (expected ROI diameter in pixels, one number or [dy, dx]; used by the
    shape statistics), ``neuropil_coefficient`` (default 0.7). ``None`` values are skipped.
    """
    known = {"image_shape", "flatten_order", "frame_rate", "diameter", "neuropil_coefficient"}
    unknown = set(fields) - known
    if unknown:
        raise ValueError(f"unknown metadata fields {sorted(unknown)}; known: {sorted(known)}")
    path = _dir(out_dir) / "external_roi_metadata.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for key, value in fields.items():
        if value is None:
            continue
        data[key] = [int(v) for v in value] if key == "image_shape" else (value.tolist() if hasattr(value, "tolist") else value)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def write_metric_table(
    path: Any,
    metrics: Mapping[str, Any] | None = None,
    categorical: Mapping[str, Any] | None = None,
    *,
    category_names: Mapping[str, Mapping[int, str]] | None = None,
    roi_index: Any = None,
) -> Path:
    """Write a per-ROI table for **Upload Custom ROI Metrics/Cell Identity Labels** or
    ``--custom-roi-metrics``.

    ``metrics`` maps column names to numeric per-ROI values (NaN = missing).
    ``categorical`` maps a variable NAME to integer codes per ROI, written as
    ``NAME_code``; ``category_names[NAME]`` maps codes to names, written as
    ``NAME_label``. A categorical named ``cell_identity`` becomes the reviewer's cell
    type/compartment variable. ``roi_index`` (0-based original indices) lets the table
    cover a subset of ROIs in any order; without it row k is ROI k.
    """
    metrics = dict(metrics or {})
    categorical = dict(categorical or {})
    category_names = dict(category_names or {})
    columns: dict[str, list[Any]] = {}
    lengths = set()
    if roi_index is not None:
        columns["roi_index"] = [int(v) for v in np.asarray(roi_index).reshape(-1)]
        lengths.add(len(columns["roi_index"]))
    for name, codes in categorical.items():
        arr = np.asarray(codes).reshape(-1)
        columns[f"{name}_code"] = ["" if (isinstance(v, float) and np.isnan(v)) else int(v) for v in arr.tolist()]
        lengths.add(arr.size)
        if name in category_names:
            lookup = {int(k): str(v) for k, v in category_names[name].items()}
            columns[f"{name}_label"] = [lookup.get(v, "") if v != "" else "" for v in columns[f"{name}_code"]]
    for name, values in metrics.items():
        if name == "roi_index" or name.endswith(("_code", "_label")):
            raise ValueError(f"metric name {name!r} is reserved; pass categories through categorical=")
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        columns[name] = ["" if not np.isfinite(v) else repr(float(v)) for v in arr]
        lengths.add(arr.size)
    if len(lengths) != 1:
        raise ValueError(f"columns have different lengths: { {k: len(v) for k, v in columns.items()} }")
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        writer.writerows(zip(*columns.values()))
    return out


def write_session(
    out_dir: Any,
    *,
    F: Any,
    masks: Any = None,
    spatial_components: Any = None,
    image_shape: Sequence[int] | None = None,
    flatten_order: str = "F",
    row_order: Any = None,
    Fneu: Any = None,
    dff: Any = None,
    spikes: Any = None,
    event_threshold: float | None = None,
    mean_func: Any = None,
    mean_anat: Any = None,
    max_func: Any = None,
    max_anat: Any = None,
    xoff: Any = None,
    yoff: Any = None,
    cell_types: Any = None,
    iscell: Any = None,
    classifier_scores: Any = None,
    good_threshold: float = 0.8,
    bad_threshold: float = 0.2,
    frame_rate: float | None = None,
    diameter: Any = None,
    neuropil_coefficient: float | None = None,
) -> Path:
    """Write a complete session directory in the documented layout and return its path.

    Give exactly one of ``masks`` (2-D label image) or ``spatial_components`` (sparse
    pixels x ROIs, with ``image_shape``). ``row_order`` (optional, for label images)
    lists, for each trace row, the mask label it belongs to; the mask is renumbered to
    match (labels not listed become background). Every other argument is optional and maps to one file; see the individual
    ``write_*`` functions.
    """
    if (masks is None) == (spatial_components is None):
        raise ValueError("give exactly one of masks= or spatial_components=")
    out = _dir(out_dir)
    if masks is not None:
        mask = np.asarray(masks).astype(np.int32, copy=False)
        if row_order is not None:
            order = np.asarray(row_order, dtype=np.int64).reshape(-1)
            relabeled = np.zeros_like(mask)
            for row, label in enumerate(order):
                relabeled[mask == label] = row + 1
            mask = relabeled
        n_rois = write_label_mask(out, mask)
        shape = mask.shape
        write_metadata(out, image_shape=shape)
    else:
        if image_shape is None:
            raise ValueError("spatial_components needs image_shape=(Ly, Lx)")
        n_rois = write_spatial_components(out, spatial_components, image_shape, flatten_order)
        shape = tuple(int(v) for v in image_shape)
    write_traces(out, F, Fneu, n_rois=n_rois)
    if dff is not None:
        write_dff(out, dff, n_rois=n_rois)
    if spikes is not None:
        write_spikes(out, spikes, n_rois=n_rois, event_threshold=event_threshold)
    write_images(out, mean_func, mean_anat, max_func, max_anat, image_shape=shape)
    if xoff is not None or yoff is not None:
        write_motion_offsets(out, xoff, yoff)
    if cell_types is not None:
        write_cell_types(out, cell_types, n_rois=n_rois)
    if iscell is not None:
        write_iscell(out, iscell)
    if classifier_scores is not None:
        write_classifier_scores(out, classifier_scores, good_threshold, bad_threshold)
    write_metadata(out, frame_rate=frame_rate, diameter=diameter, neuropil_coefficient=neuropil_coefficient)
    return out


def validate(session_dir: Any, quiet: bool = False):
    """Check a directory against the input contract; print and return the report.
    ``report.errors`` is empty when the directory can be built."""
    from adroit_qc.validate import validate as _validate

    report = _validate(Path(session_dir))
    if not quiet:
        print(session_dir)
        print(report.render())
    return report


def build_reviewer(session_dir: Any, output_dir: Any = None, **options: Any):
    """Build the reviewer for one session; returns (pdf_path, html_path).

    A thin wrapper around :func:`adroit_qc.summary.create_processing_summary`; every
    keyword is a build setting (``input_layout``, ``target_structure`` (filter set),
    ``initialize_labels_from_roi_model_scores``, ``custom_roi_metrics``,
    ``custom_roi_metric_sections``, ``filter_settings``, ``title``, ``force_embed``,
    ``embed_limit_mb``, ``html_only``, ``html_name``, ``pdf_name``).
    """
    from adroit_qc.summary import create_processing_summary

    return create_processing_summary(session_dir, output_dir=output_dir, **options)
