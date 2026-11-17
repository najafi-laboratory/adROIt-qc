#!/usr/bin/env python3
"""Run CaImAn motion correction and CNMF on part of a Bruker two-photon session.

This is an example of adapting another tool's output, file formats and quality metrics to the
reviewer's inputs, not a comprehensive or optimized CaImAn analysis: parameters are near CaImAn's
defaults and untuned, and no neuropil is estimated. Tune CaImAn for your own data.

Example used for the reviewer docs (PPC session, functional channel 2):

    python examples/caiman/run_caiman.py \
        --tiff-dir /path/to/VTYH01_PPC_20250212_1451ShortLong-374 \
        --channel Ch2 --n-files 4 --frame-rate 30 --out caiman_out

Each Bruker OME-TIFF file is read on its own (is_ome=False), so the companion OME metadata
does not make tifffile load the whole session. Output in --out: the motion-corrected C-order
memmap, rigid_shifts.npy (frames x 2, y and x in pixels), cnmf_results.hdf5 and run_params.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import tifffile


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiff-dir", type=Path, required=True)
    p.add_argument("--channel", default="Ch2")
    p.add_argument("--n-files", type=int, default=4)
    p.add_argument("--frame-rate", type=float, default=30.0)
    p.add_argument("--gsig", type=float, default=5.0, help="half-size of a soma in pixels")
    p.add_argument("--decay-time", type=float, default=0.4)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = p.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    files = sorted(glob.glob(str(args.tiff_dir / f"*_{args.channel}_*.ome.tif")))[: args.n_files]
    if not files:
        raise SystemExit(f"no *_{args.channel}_*.ome.tif files in {args.tiff_dir}")
    stacks = []
    for path in files:
        with tifffile.TiffFile(path, is_ome=False) as tif:
            stacks.append(np.stack([page.asarray() for page in tif.pages]))
        print(f"read {Path(path).name}: {stacks[-1].shape}", flush=True)
    movie = np.concatenate(stacks).astype(np.float32); del stacks
    raw_h5 = out / "raw_movie.h5"
    with h5py.File(raw_h5, "w") as h5:
        h5.create_dataset("mov", data=movie, chunks=(500, movie.shape[1], movie.shape[2]))
    n_frames = movie.shape[0]; del movie
    print(f"{n_frames} frames from {len(files)} files ({time.time() - t0:.0f} s)", flush=True)

    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import cnmf, params as params_mod

    _, dview, n_proc = cm.cluster.setup_cluster(backend="multiprocessing", n_processes=args.processes, single_thread=False)
    opts = params_mod.CNMFParams(params_dict={
        "fnames": [str(raw_h5)], "var_name_hdf5": "mov", "fr": args.frame_rate, "decay_time": args.decay_time,
        "pw_rigid": False, "max_shifts": (20, 20), "border_nan": "copy",
        "p": 1, "nb": 2, "rf": 25, "stride": 12, "K": 6, "gSig": (args.gsig, args.gsig),
        "method_init": "greedy_roi", "rolling_sum": True, "merge_thr": 0.85, "ssub": 2, "tsub": 2,
        "only_init": True, "min_SNR": 2.0, "rval_thr": 0.85, "use_cnn": False,
    })
    mc = MotionCorrect([str(raw_h5)], dview=dview, var_name_hdf5="mov", **opts.get_group("motion"))
    mc.motion_correct(save_movie=True)
    shifts = np.asarray(mc.shifts_rig, dtype=np.float32)  # (frames, 2): y, x
    np.save(out / "rigid_shifts.npy", shifts)
    print(f"motion corrected ({time.time() - t0:.0f} s); max |shift| {np.abs(shifts).max():.1f} px", flush=True)
    fname_new = cm.save_memmap(mc.mmap_file, base_name=str(out / "memmap_"), order="C", border_to_0=0, dview=dview)
    # Recent CaImAn writes memmaps under CAIMAN_DATA/temp whatever base_name says: move the C-order
    # movie next to the other outputs and drop the F-order motion-corrected copy.
    if Path(fname_new).parent.resolve() != out.resolve():
        moved = out / Path(fname_new).name
        shutil.move(fname_new, moved)
        fname_new = str(moved)
    for path in (mc.mmap_file if isinstance(mc.mmap_file, (list, tuple)) else [mc.mmap_file]):
        if path and Path(path).exists() and Path(path).resolve() != Path(fname_new).resolve():
            Path(path).unlink()
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")
    cm.stop_server(dview=dview)
    _, dview, n_proc = cm.cluster.setup_cluster(backend="multiprocessing", n_processes=args.processes, single_thread=False)
    cnm = cnmf.CNMF(n_proc, params=opts, dview=dview)
    # Recent CaImAn versions fit in place and return None; older ones return the object.
    fitted = cnm.fit(images)
    cnm = fitted if fitted is not None else cnm
    refitted = cnm.refit(images, dview=dview)
    cnm = refitted if refitted is not None else cnm
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    try:
        cn = cm.local_correlations(images[::10].transpose(1, 2, 0), swap_dim=False)
        cnm.estimates.Cn = np.nan_to_num(cn).astype(np.float32)
    except Exception as exc:  # the correlation image is only a display aid
        print("correlation image skipped:", exc)
    cnm.save(str(out / "cnmf_results.hdf5"))
    cm.stop_server(dview=dview)
    n = cnm.estimates.A.shape[1]; acc = len(cnm.estimates.idx_components)
    run = {"files": [Path(f).name for f in files], "n_frames": int(T), "dims": list(map(int, dims)), "n_components": int(n),
           "n_accepted": int(acc), "memmap": Path(fname_new).name, "seconds": round(time.time() - t0),
           "caiman_version": cm.__version__, "params": {k: str(v) for k, v in {"gSig": args.gsig, "decay_time": args.decay_time, "fr": args.frame_rate, "K": 6, "rf": 25, "stride": 12, "min_SNR": 2.0, "rval_thr": 0.85}.items()}}
    (out / "run_params.json").write_text(json.dumps(run, indent=2) + "\n")
    raw_h5.unlink()
    for path in out.glob("raw_movie*_els__*.mmap"):
        path.unlink()
    print(json.dumps(run), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
