"""adroit_qc.inputs writes every documented input; each one reaches the reviewer unchanged."""
from __future__ import annotations

import csv
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from conftest import FS, N_FRAMES, N_ROIS, SHAPE, REPO, areas, build, decode_f32, label_mask, traces


def _everything(tmp_path, **extra):
    from adroit_qc import inputs
    fluo, neuropil, spikes = traces()
    rng = np.random.default_rng(3)
    return inputs.write_session(
        tmp_path / "helper_session",
        masks=label_mask(), F=fluo, Fneu=neuropil, spikes=spikes, event_threshold=0.3,
        mean_func=rng.random(SHAPE), mean_anat=rng.random(SHAPE),
        xoff=np.linspace(-1, 1, N_FRAMES), yoff=np.zeros(N_FRAMES),
        cell_types=[1, -1, 0, np.nan, 1, -1],
        classifier_scores=np.linspace(0.05, 0.95, N_ROIS), good_threshold=0.8, bad_threshold=0.2,
        frame_rate=FS, diameter=[6, 6], **extra,
    ), fluo, neuropil, spikes


def test_write_session_validates_and_builds(tmp_path):
    from adroit_qc import inputs
    session, *_ = _everything(tmp_path)
    report = inputs.validate(session, quiet=True)
    assert report.errors == []
    html, payload = build(session, tmp_path / "out")
    assert (payload["nRois"], payload["nFrames"], payload["frameRate"]) == (N_ROIS, N_FRAMES, FS)
    assert [int(r["npix"]) for r in sorted(payload["rois"], key=lambda r: r["roi"])] == areas()
    got = [s["probability"] for s in payload["roiModelScores"]]
    np.testing.assert_allclose(got, np.linspace(0.05, 0.95, N_ROIS), atol=1e-6)
    assert payload["oasisAvailable"] and payload["motionAvailable"] and payload["motionFrameCount"] == N_FRAMES
    np.testing.assert_allclose(decode_f32(payload["xoff"], (N_FRAMES,)), np.linspace(-1, 1, N_FRAMES), atol=1e-6)


def test_cli_wrapper_and_module_agree(tmp_path):
    session, *_ = _everything(tmp_path)
    a = subprocess.run([sys.executable, "scripts/validate_inputs.py", str(session)], cwd=REPO, capture_output=True, text=True)
    b = subprocess.run([sys.executable, "-m", "adroit_qc.validate", str(session)], cwd=REPO, capture_output=True, text=True)
    assert a.returncode == b.returncode == 0 and a.stdout == b.stdout


def test_supplied_dff_is_used_as_given(tmp_path):
    from adroit_qc import inputs
    fluo, neuropil, _ = traces()
    supplied = np.random.default_rng(4).normal(0, 1, (N_ROIS, N_FRAMES)).astype(np.float32)
    session = inputs.write_session(tmp_path / "s", masks=label_mask(), F=fluo, Fneu=neuropil, dff=supplied, frame_rate=FS)
    _, payload = build(session, tmp_path / "embedded", force_embed=True)
    np.testing.assert_array_equal(decode_f32(payload["dff"], (N_ROIS, N_FRAMES)), supplied)
    build(session, tmp_path / "sidecar", force_embed=False)
    sidecar = np.load(tmp_path / "sidecar" / "s_dff.npy")
    np.testing.assert_array_equal(sidecar, supplied)


def test_supplied_dff_shape_mismatch_is_an_error(tmp_path):
    from adroit_qc import inputs
    fluo, _, _ = traces()
    session = inputs.write_session(tmp_path / "s", masks=label_mask(), F=fluo)
    inputs.write_dff(session, np.zeros((N_ROIS, N_FRAMES - 5)))
    with pytest.raises(ValueError, match="dff"):
        build(session, tmp_path / "out")


def test_neuropil_coefficient_from_metadata(tmp_path):
    from adroit_qc import inputs
    fluo, neuropil, _ = traces()
    outs = {}
    for c in (0.0, 0.7):
        session = inputs.write_session(tmp_path / f"s{c}", masks=label_mask(), F=fluo, Fneu=neuropil, neuropil_coefficient=c)
        build(session, tmp_path / f"o{c}", force_embed=False)
        outs[c] = np.load(tmp_path / f"o{c}" / f"s{c}_dff.npy")
    assert not np.allclose(outs[0.0], outs[0.7], equal_nan=True)


def test_sparse_components_helper(tmp_path):
    from adroit_qc import inputs
    mask = label_mask(); fluo, _, _ = traces()
    cols = [np.ravel(mask == k + 1, order="C").astype(np.float32) for k in range(N_ROIS)]
    session = inputs.write_session(tmp_path / "sp", spatial_components=sparse.csc_matrix(np.stack(cols, axis=1)),
                                   image_shape=SHAPE, flatten_order="C", F=fluo)
    assert inputs.validate(session, quiet=True).errors == []
    _, payload = build(session, tmp_path / "out")
    assert [int(r["npix"]) for r in payload["rois"]] == areas()


def test_row_order_renumbers_masks(tmp_path):
    from adroit_qc import inputs
    mask = label_mask() * 10  # labels 10..60 in mask order
    fluo, _, _ = traces()
    order = [60, 50, 40, 30, 20, 10]  # trace row 0 belongs to label 60
    session = inputs.write_session(tmp_path / "ro", masks=mask, row_order=order, F=fluo)
    _, payload = build(session, tmp_path / "out")
    assert [int(r["npix"]) for r in payload["rois"]] == areas()[::-1]


def test_metric_table_round_trip(tmp_path, dense_session):
    from adroit_qc import inputs
    table = inputs.write_metric_table(
        tmp_path / "metrics.csv",
        metrics={"snr_custom": np.arange(N_ROIS) / 2.0},
        categorical={"cell_identity": [1, 2, 1, 2, 1, 2]},
        category_names={"cell_identity": {1: "PV", 2: "SST"}},
    )
    rows = list(csv.DictReader(table.open()))
    assert rows[1] == {"cell_identity_code": "2", "cell_identity_label": "SST", "snr_custom": "0.5"}
    _, payload = build(dense_session, tmp_path / "out", custom_roi_metrics=[table])
    assert "snr_custom" in json.dumps(payload)


@pytest.mark.parametrize("bad", [
    dict(masks=np.ones((4, 4, 2))),
    dict(masks=label_mask(), F=np.zeros((N_FRAMES, N_ROIS))),
])
def test_helpers_reject_wrong_shapes(tmp_path, bad):
    from adroit_qc import inputs
    kwargs = dict(F=traces()[0]); kwargs.update(bad)
    with pytest.raises(ValueError):
        inputs.write_session(tmp_path / "bad", **kwargs)


def test_label_gaps_follow_sorted_order(tmp_path):
    from adroit_qc import inputs
    mask = label_mask() * 7  # labels 7, 14, ... keep their order
    session = inputs.write_session(tmp_path / "gaps", masks=mask, F=traces()[0])
    _, payload = build(session, tmp_path / "out")
    assert [int(r["npix"]) for r in sorted(payload["rois"], key=lambda r: r["roi"])] == areas()


def test_old_package_name_still_works():
    r = subprocess.run([sys.executable, "-c", "import roi_reviewer.summary as s, adroit_qc.summary as a; assert s is a; "
                        "from roi_reviewer import roi_metrics; print('ok')"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
    r = subprocess.run([sys.executable, "-m", "roi_reviewer.summary", "--help"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "--force-embed" in r.stdout
