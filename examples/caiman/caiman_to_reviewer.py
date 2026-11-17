#!/usr/bin/env python3
"""Convert a CaImAn CNMF result into the reviewer's external-ROI input layout.

    python examples/caiman/caiman_to_reviewer.py \
        --cnmf  caiman_out/cnmf_results.hdf5 \
        --movie caiman_out/memmap_d1_512_d2_512_d3_1_order_C_frames_16380.mmap \
        --shifts caiman_out/rigid_shifts.npy \
        --frame-rate 30 \
        --out   reviewer_inputs/PPC_caiman

What is written (all in --out):

    spatial_components.npz      CaImAn's A (pixels x components, Fortran pixel order), unchanged
    external_roi_metadata.json  image_shape, flatten_order "F", frame_rate, diameter
    F.npy                       raw fluorescence per component: the footprint-weighted mean of the
                                motion-corrected movie (components x frames), so the generator
                                computes dF/F the same way it does for Suite2p's F.npy
    spikes.h5                   dataset "spikes": CaImAn's deconvolved activity S (components x frames)
    mean_func.npy               mean of the motion-corrected movie
    max_func.npy                CaImAn's local correlation image, when the result holds one
    move_offset.h5              datasets "xoff", "yoff": rigid shift per frame in pixels (optional)
    iscell.npy                  (components, 2): CaImAn's accept/reject and its SNR-based probability
    caiman_metrics.csv          roi_index plus CaImAn's own component metrics, ready for
                                --custom-roi-metrics or the reviewer's upload dialog:
                                caiman_snr, caiman_r_value, caiman_cnn_prob (when computed) and the
                                categorical caiman_status_code / caiman_status_label (1 accepted, 0 rejected)

Component k (column k of A, row k of C and S) becomes reviewer ROI k, so roi_index k in every
exported file refers back to CaImAn component k. Rejected components are kept and marked in
iscell.npy and caiman_metrics.csv, so they can be reviewed rather than silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def load_cnmf(path: Path) -> dict:
    """Read the parts of a CaImAn CNMF .hdf5 that the reviewer needs, without importing CaImAn."""
    with h5py.File(path, "r") as f:
        est = f["estimates"]
        a = est["A"]
        A = sparse.csc_matrix((a["data"][()], a["indices"][()], a["indptr"][()]), shape=tuple(a["shape"][()]))
        out = {"A": A, "C": np.asarray(est["C"]), "S": np.asarray(est["S"]) if "S" in est else None}
        for key in ("idx_components", "SNR_comp", "r_values", "cnn_preds", "YrA"):
            if key in est and est[key].shape not in ((), None) and est[key].dtype != object:
                out[key] = np.asarray(est[key])
        dims = f["dims"][()] if "dims" in f else est["dims"][()]
        out["dims"] = tuple(int(v) for v in np.ravel(dims)[:2])
        out["Cn"] = np.asarray(f["Cn"]) if "Cn" in f and f["Cn"].shape else None
        params = f.get("params")
        out["gSig"] = None
        if params is not None and "init" in params and "gSig" in params["init"]:
            out["gSig"] = [float(v) for v in np.ravel(params["init"]["gSig"][()])]
    return out


def open_memmap(path: Path) -> tuple[np.memmap, tuple[int, int], int]:
    """Open a CaImAn C-order memmap (pixels x frames, pixel order F) from its file name."""
    parts = path.stem.split("_")
    d1 = int(parts[parts.index("d1") + 1]); d2 = int(parts[parts.index("d2") + 1])
    T = int(parts[parts.index("frames") + 1])
    order = parts[parts.index("order") + 1]
    Yr = np.memmap(path, mode="r", dtype=np.float32, shape=(d1 * d2, T), order=order)
    return Yr, (d1, d2), T


def footprint_traces(A: sparse.csc_matrix, Yr: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Weighted mean of the movie inside each footprint: (A / sum A)^T Y, computed in frame chunks."""
    weights = sparse.csc_matrix(A.multiply(1.0 / np.maximum(np.asarray(A.sum(axis=0)).ravel(), 1e-12)))
    T = Yr.shape[1]
    out = np.empty((A.shape[1], T), dtype=np.float32)
    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        out[:, start:stop] = (weights.T @ np.asarray(Yr[:, start:stop])).astype(np.float32)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnmf", type=Path, required=True, help="CaImAn CNMF result saved with cnm.save(...)")
    p.add_argument("--movie", type=Path, required=True, help="motion-corrected C-order CaImAn memmap (.mmap)")
    p.add_argument("--shifts", type=Path, default=None, help="optional (frames, 2) rigid shifts .npy, (y, x) in pixels")
    p.add_argument("--frame-rate", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)

    res = load_cnmf(args.cnmf)
    A, C, S = res["A"], res["C"], res["S"]
    n, T = C.shape
    Yr, dims, T_movie = open_memmap(args.movie)
    if dims != res["dims"]:
        raise SystemExit(f"movie is {dims} but the CNMF result is {res['dims']}")
    if T_movie != T:
        raise SystemExit(f"movie has {T_movie} frames but C has {T}")
    if A.shape != (dims[0] * dims[1], n):
        raise SystemExit(f"A has shape {A.shape}; expected ({dims[0] * dims[1]}, {n})")

    sparse.save_npz(out / "spatial_components.npz", A.tocsc())
    diameter = [2.0 * g for g in res["gSig"]] if res["gSig"] else None
    meta = {"image_shape": list(dims), "flatten_order": "F", "frame_rate": args.frame_rate, "source": "CaImAn CNMF"}
    if diameter:
        meta["diameter"] = diameter
    (out / "external_roi_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    np.save(out / "F.npy", footprint_traces(A, Yr))
    with h5py.File(out / "spikes.h5", "w") as h5:
        h5.create_dataset("spikes", data=np.asarray(S if S is not None else np.zeros_like(C), dtype=np.float32))
        h5.attrs["method"] = "CaImAn CNMF deconvolution (OASIS)"
    mean = np.zeros(dims[0] * dims[1], dtype=np.float64)
    for start in range(0, T, 2000):
        mean += np.asarray(Yr[:, start:start + 2000], dtype=np.float64).sum(axis=1)
    np.save(out / "mean_func.npy", (mean / T).reshape(dims, order="F").astype(np.float32))
    if res.get("Cn") is not None and res["Cn"].shape == dims:
        np.save(out / "max_func.npy", res["Cn"].astype(np.float32))
    if args.shifts is not None:
        shifts = np.load(args.shifts)
        if shifts.shape[0] != T:
            raise SystemExit(f"shifts have {shifts.shape[0]} frames; expected {T}")
        with h5py.File(out / "move_offset.h5", "w") as h5:
            h5.create_dataset("yoff", data=shifts[:, 0].astype(np.float32))
            h5.create_dataset("xoff", data=shifts[:, 1].astype(np.float32))

    accepted = np.zeros(n, dtype=bool)
    if "idx_components" in res:
        accepted[np.asarray(res["idx_components"], dtype=int)] = True
    else:
        accepted[:] = True
    snr = res.get("SNR_comp", np.full(n, np.nan))
    prob = 1.0 / (1.0 + np.exp(-(np.nan_to_num(snr, nan=0.0) - 2.0)))  # monotone in SNR, 0.5 at CaImAn's default min_SNR 2
    np.save(out / "iscell.npy", np.stack([accepted.astype(np.float32), prob.astype(np.float32)], axis=1))
    cols = {"caiman_snr": snr, "caiman_r_value": res.get("r_values")}
    if res.get("cnn_preds") is not None and np.size(res["cnn_preds"]) == n:
        cols["caiman_cnn_prob"] = res["cnn_preds"]
    with (out / "caiman_metrics.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        names = [k for k, v in cols.items() if v is not None]
        w.writerow(["roi_index", *names, "caiman_status_code", "caiman_status_label"])
        for k in range(n):
            w.writerow([k, *[f"{float(cols[c][k]):.6g}" for c in names], int(accepted[k]), "accepted" if accepted[k] else "rejected"])
    print(f"wrote {n} components x {T} frames to {out} ({int(accepted.sum())} accepted by CaImAn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
