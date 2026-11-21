#!/usr/bin/env python3
"""Build every tutorial example reviewer from the example sessions.

Each numbered step is one input route; run them all, or one with --only N. Every step
writes into OUT/<step name>/ and prints the reviewer it wrote. Steps 2 and 6 also show
how to write the documented input layout from your own arrays with adroit_qc.inputs.

    python examples/tutorial/build_examples.py --examples /path/to/adroit_qc_examples --out tutorial_out

The example folder (--examples, or $ADROIT_QC_EXAMPLES) holds:

    sessions/ppc_suite2p/VTYH01_PPC_20250212_1451ShortLong/        Suite2p output, cell-type codes, spikes.h5
    sessions/ppc_masks/VTYH01_PPC_20250212_1451ShortLong/          same session in the documented layout
    sessions/crbl_suite2p/E5LG_CRBL_crux1_20250520_EBC-334/        Suite2p output, classifier scores, spikes.h5
    sessions/crbl_masks/E5LG_CRBL_crux1_20250520_EBC-334/          same session in the documented layout
    sessions/caiman_ppc/VTYH01_PPC_20250212_1451ShortLong_caiman/  CaImAn CNMF result, converted
    extras/VTYH01_PPC_20250212_1451ShortLong_cell_identity_labels.csv

On the Georgia Tech PACE cluster it is
/storage/project/r-fnajafi3-0/shared/2P_Imaging/adroit_qc_examples (laboratory members only).
[PLACEHOLDER: Zenodo record DOI for the example sessions; the same folder layout will be
downloadable there.]
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

from adroit_qc import inputs

PPC = "VTYH01_PPC_20250212_1451ShortLong"
CRBL = "E5LG_CRBL_crux1_20250520_EBC-334"
DEFAULT_EXAMPLES = "/storage/project/r-fnajafi3-0/shared/2P_Imaging/adroit_qc_examples"


def step1_suite2p(ex: Path, out: Path) -> Path:
    """A Suite2p session as it is: point the generator at the session root."""
    _, html = inputs.build_reviewer(
        ex / "sessions/ppc_suite2p" / PPC,
        output_dir=out / "1_suite2p",
        input_layout="suite2p",
        target_structure="soma",          # the filter set the reviewer opens on
        title="PPC session from Suite2p",
    )
    return html


def suite2p_arrays(session: Path) -> dict:
    """Read a Suite2p plane into plain arrays, as any other pipeline would hand them over."""
    plane = session / "suite2p" / "plane0"
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    masks = np.zeros((ly, lx), dtype=np.int32)   # dense label image: ROI k has label k + 1
    for k, roi in enumerate(stat):               # Suite2p ROIs can overlap; a later ROI wins a shared pixel
        masks[roi["ypix"], roi["xpix"]] = k + 1
    max_proj = ops.get("max_proj")
    arrays = dict(
        masks=masks,
        F=np.load(plane / "F.npy"),
        Fneu=np.load(plane / "Fneu.npy"),
        mean_func=ops.get("meanImg"),
        mean_anat=ops.get("meanImg_chan2"),
        max_func=max_proj if max_proj is not None and max_proj.shape == (ly, lx) else None,  # Suite2p crops max_proj
        xoff=ops.get("xoff"),
        yoff=ops.get("yoff"),
        frame_rate=float(ops.get("fs", 30.0)),
    )
    if (plane / "iscell.npy").exists():
        arrays["iscell"] = np.load(plane / "iscell.npy")[:, 0]
    if (session / "spikes.h5").exists():
        with h5py.File(session / "spikes.h5", "r") as handle:
            arrays["spikes"] = handle["spikes"][()]
    if (session / "roi_cell_type_labels.npy").exists():
        arrays["cell_types"] = np.load(session / "roi_cell_type_labels.npy")
    return arrays


def step2_arrays_to_layout(ex: Path, out: Path) -> Path:
    """Arrays from any software -> the documented layout -> validate -> build."""
    arrays = suite2p_arrays(ex / "sessions/ppc_suite2p" / PPC)
    session = inputs.write_session(out / "2_arrays" / "inputs" / PPC, **arrays)
    report = inputs.validate(session)
    if report.errors:
        raise SystemExit("validation failed; see the report above")
    _, html = inputs.build_reviewer(session, output_dir=out / "2_arrays" / "reviewer",
                                    input_layout="external_rois", target_structure="all_rois")
    return html


def step3_masks_with_cell_identity(ex: Path, out: Path) -> Path:
    """A session already in the documented layout, plus a categorical cell-identity table."""
    _, html = inputs.build_reviewer(
        ex / "sessions/ppc_masks" / PPC,
        output_dir=out / "3_masks_cell_identity",
        input_layout="external_rois",
        target_structure="all_rois",
        custom_roi_metrics=[ex / "extras" / f"{PPC}_cell_identity_labels.csv"],
    )
    return html


def step4_classifier_scores(ex: Path, out: Path) -> Path:
    """A session with classifier scores (roi_model_scores.h5): open with labels from the scores."""
    _, html = inputs.build_reviewer(
        ex / "sessions/crbl_suite2p" / CRBL,
        output_dir=out / "4_classifier_scores",
        input_layout="suite2p",
        target_structure="dendrite",
        initialize_labels_from_roi_model_scores=True,
    )
    return html


def step5_caiman(ex: Path, out: Path) -> Path:
    """CaImAn output converted by examples/caiman/caiman_to_reviewer.py, with CaImAn's own metrics."""
    session = ex / "sessions/caiman_ppc" / f"{PPC}_caiman"
    _, html = inputs.build_reviewer(
        session,
        output_dir=out / "5_caiman",
        input_layout="external_rois",
        target_structure="all_rois",
        custom_roi_metrics=[session / "caiman_metrics.csv"],
        custom_roi_metric_sections=["caiman_snr=model_derived", "caiman_r_value=model_derived"],
    )
    return html


def step6_own_dff_and_metrics(ex: Path, out: Path) -> Path:
    """Your own dF/F (dff.h5), classifier scores and per-ROI metrics, written with adroit_qc.inputs."""
    src = ex / "sessions/ppc_masks" / PPC
    session = out / "6_own_dff" / "inputs" / PPC
    if session.exists():
        shutil.rmtree(session)
    shutil.copytree(src, session)
    F = np.load(session / "F.npy").astype(np.float64)
    Fneu = np.load(session / "Fneu.npy").astype(np.float64)
    # Your own dF/F: here a neuropil coefficient of 0.8 and a constant 10th-percentile baseline.
    signal = F - 0.8 * Fneu
    baseline = np.percentile(signal, 10, axis=1, keepdims=True)
    inputs.write_dff(session, (signal - baseline) / np.abs(baseline))
    # Scores from any classifier, 0-1, higher = more likely good. This toy score is NOT a model.
    sd_ratio = F.std(axis=1) / Fneu.std(axis=1)
    inputs.write_classifier_scores(session, 1 / (1 + np.exp(-(sd_ratio - 1.5))), good_threshold=0.8,
                                   bad_threshold=0.2, model_source="toy score from the F/Fneu SD ratio")
    # Numeric metrics and a categorical variable, in the upload-table format.
    table = inputs.write_metric_table(
        session / "my_metrics.csv",
        metrics={"f_to_neuropil_sd_ratio": sd_ratio},
        categorical={"example_group": np.arange(F.shape[0]) % 3},
        category_names={"example_group": {0: "group A", 1: "group B", 2: "group C"}},
    )
    inputs.validate(session)
    _, html = inputs.build_reviewer(session, output_dir=out / "6_own_dff" / "reviewer",
                                    input_layout="external_rois", target_structure="all_rois",
                                    custom_roi_metrics=[table], force_embed=False)
    return html


STEPS = [step1_suite2p, step2_arrays_to_layout, step3_masks_with_cell_identity,
         step4_classifier_scores, step5_caiman, step6_own_dff_and_metrics]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--examples", type=Path, default=Path(os.environ.get("ADROIT_QC_EXAMPLES", DEFAULT_EXAMPLES)))
    p.add_argument("--out", type=Path, default=Path("tutorial_out"))
    p.add_argument("--only", type=int, choices=range(1, len(STEPS) + 1), action="append",
                   help="run only this step (repeatable)")
    args = p.parse_args()
    if not (args.examples / "sessions").is_dir():
        raise SystemExit(f"{args.examples} has no sessions/ folder; pass --examples or set ADROIT_QC_EXAMPLES")
    for number, step in enumerate(STEPS, start=1):
        if args.only and number not in args.only:
            continue
        print(f"\n== step {number}: {step.__doc__.splitlines()[0]}")
        print("   reviewer:", step(args.examples, args.out))


if __name__ == "__main__":
    main()
