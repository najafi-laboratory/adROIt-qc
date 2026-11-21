"""Shared fixtures: small synthetic sessions in every documented input layout.

Ground truth is known by construction (6 ROIs, 64 x 64 pixels, 600 frames at 30 Hz),
so the tests can check that each input reaches the reviewer unchanged and in the right
ROI order, and that each output can be read back.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
N_ROIS, N_FRAMES, SHAPE, FS = 6, 600, (64, 64), 30.0
# ROI k: a filled rectangle; areas differ so ROI order can be checked through ROI area.
BOXES = [(4, 10, 4, 12), (4, 14, 30, 36), (20, 26, 8, 20), (24, 34, 40, 50), (44, 48, 6, 30), (40, 58, 52, 58)]


def label_mask() -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.int32)
    for k, (y0, y1, x0, x1) in enumerate(BOXES):
        mask[y0:y1, x0:x1] = k + 1
    return mask


def areas() -> list[int]:
    return [int((y1 - y0) * (x1 - x0)) for y0, y1, x0, x1 in BOXES]


def traces(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spikes = (rng.random((N_ROIS, N_FRAMES)) < 0.02).astype(np.float32) * rng.uniform(0.5, 2.0, (N_ROIS, N_FRAMES)).astype(np.float32)
    kernel = np.exp(-np.arange(60) / (0.5 * FS)).astype(np.float32)
    calcium = np.stack([np.convolve(s, kernel)[:N_FRAMES] for s in spikes])
    fluo = (200.0 + 80.0 * calcium + rng.normal(0, 3, (N_ROIS, N_FRAMES))).astype(np.float32)
    neuropil = (100.0 + rng.normal(0, 2, (N_ROIS, N_FRAMES))).astype(np.float32)
    return fluo, neuropil, spikes


def write_spikes(path: Path, spikes: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("spikes", data=spikes)


def stat_from_mask(mask: np.ndarray) -> np.ndarray:
    stat = []
    for k in range(N_ROIS):
        ypix, xpix = np.nonzero(mask == k + 1)
        stat.append({"ypix": ypix, "xpix": xpix, "lam": np.ones(ypix.size, dtype=np.float32), "npix": int(ypix.size),
                     "med": [float(np.median(ypix)), float(np.median(xpix))], "skew": 1.0, "compact": 1.1,
                     "aspect_ratio": 1.2, "footprint": 1.0, "radius": 3.0, "overlap": np.zeros(ypix.size, dtype=bool)})
    return np.array(stat, dtype=object)


@pytest.fixture
def suite2p_session(tmp_path: Path) -> Path:
    s = tmp_path / "s2p_session"; plane = s / "suite2p" / "plane0"; plane.mkdir(parents=True)
    mask = label_mask(); fluo, neuropil, spikes = traces()
    np.save(plane / "stat.npy", stat_from_mask(mask), allow_pickle=True)
    np.save(plane / "F.npy", fluo); np.save(plane / "Fneu.npy", neuropil); np.save(plane / "spks.npy", spikes)
    np.save(plane / "iscell.npy", np.column_stack([np.ones(N_ROIS), np.linspace(0.5, 1, N_ROIS)]).astype(np.float32))
    rng = np.random.default_rng(1)
    ops = {"meanImg": rng.random(SHAPE).astype(np.float32), "fs": FS, "nframes": N_FRAMES, "Ly": SHAPE[0], "Lx": SHAPE[1],
           "xoff": np.linspace(-2, 2, N_FRAMES), "yoff": np.linspace(1, -1, N_FRAMES), "neucoeff": 0.7, "tau": 0.5}
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    write_spikes(s / "spikes.h5", spikes)
    return s


def _external(tmp_path: Path, name: str, geometry: str) -> Path:
    s = tmp_path / name; s.mkdir()
    mask = label_mask(); fluo, neuropil, spikes = traces()
    if geometry == "dense":
        np.save(s / "roi_mask.npy", mask)
    else:
        cols = [np.ravel(mask == k + 1, order="F").astype(np.float32) for k in range(N_ROIS)]
        sparse.save_npz(s / "spatial_components.npz", sparse.csc_matrix(np.stack(cols, axis=1)))
    np.save(s / "F.npy", fluo)
    write_spikes(s / "spikes.h5", spikes)
    np.save(s / "mean_func.npy", np.random.default_rng(2).random(SHAPE).astype(np.float32))
    (s / "external_roi_metadata.json").write_text(json.dumps({"image_shape": list(SHAPE), "flatten_order": "F", "frame_rate": FS, "diameter": [6, 6]}))
    return s


@pytest.fixture
def dense_session(tmp_path: Path) -> Path:
    return _external(tmp_path, "dense_session", "dense")


@pytest.fixture
def sparse_session(tmp_path: Path) -> Path:
    return _external(tmp_path, "sparse_session", "sparse")


def build(session: Path, out: Path, **kwargs) -> tuple[Path, dict]:
    from adroit_qc import summary as ps
    kwargs.setdefault("input_layout", "suite2p" if (session / "suite2p").exists() else "external_rois")
    _, html = ps.create_preprocessing_summary(session, out, **kwargs)
    return Path(html), payload_of(Path(html))


def payload_of(html: Path) -> dict:
    text = html.read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, re.S)
    assert m, "payload script missing"

    def reject(constant: str):
        raise ValueError(f"non-standard JSON constant {constant} in payload")

    return json.loads(m.group(1), parse_constant=reject)


def decode_f32(b64: str, shape: tuple[int, ...]) -> np.ndarray:
    import base64
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
