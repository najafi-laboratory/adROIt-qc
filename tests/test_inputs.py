"""Every documented input reaches the reviewer unchanged and in ROI order."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from conftest import FS, N_FRAMES, N_ROIS, areas, build, decode_f32, label_mask, run_script


def roi_areas(payload: dict) -> list[int]:
    return [int(r["npix"]) for r in sorted(payload["rois"], key=lambda r: r["roi"])]


def test_suite2p_layout(suite2p_session, tmp_path):
    _, p = build(suite2p_session, tmp_path / "out")
    assert (p["nRois"], p["nFrames"]) == (N_ROIS, N_FRAMES)
    assert p["frameRate"] == FS
    assert roi_areas(p) == areas()
    assert p["oasisAvailable"]
    ops = np.load(suite2p_session / "suite2p/plane0/ops.npy", allow_pickle=True).item()
    np.testing.assert_allclose(decode_f32(p["xoff"], (N_FRAMES,)), ops["xoff"], atol=1e-5)


@pytest.mark.parametrize("fixture", ["dense_session", "sparse_session"])
def test_external_masks_keep_roi_order(fixture, request, tmp_path):
    session = request.getfixturevalue(fixture)
    _, p = build(session, tmp_path / "out")
    assert p["inputLayout"] == "external_rois"
    assert roi_areas(p) == areas(), "mask label k+1 / matrix column k must be ROI k"
    assert p["frameRate"] == FS


def test_sparse_flatten_order_is_honored(sparse_session, tmp_path):
    meta = json.loads((sparse_session / "external_roi_metadata.json").read_text())
    meta["flatten_order"] = "C"   # columns were written in F order, so C must change the geometry
    (sparse_session / "external_roi_metadata.json").write_text(json.dumps(meta))
    _, p = build(sparse_session, tmp_path / "out")
    outlines_c = [r["path"] for r in p["rois"]]
    meta["flatten_order"] = "F"
    (sparse_session / "external_roi_metadata.json").write_text(json.dumps(meta))
    _, q = build(sparse_session, tmp_path / "out2")
    assert outlines_c != [r["path"] for r in q["rois"]], "flatten_order must change where pixels land"


def test_move_offset_h5_overrides_ops(suite2p_session, tmp_path):
    x = np.full(N_FRAMES, 3.5, dtype=np.float32)
    with h5py.File(suite2p_session / "move_offset.h5", "w") as h5:
        h5.create_dataset("xoff", data=x); h5.create_dataset("yoff", data=-x)
    _, p = build(suite2p_session, tmp_path / "out")
    np.testing.assert_allclose(decode_f32(p["xoff"], (N_FRAMES,)), x)
    np.testing.assert_allclose(decode_f32(p["yoff"], (N_FRAMES,)), -x)


def test_external_motion_offsets(dense_session, tmp_path):
    with h5py.File(dense_session / "move_offset.h5", "w") as h5:
        h5.create_dataset("xoff", data=np.arange(N_FRAMES, dtype=np.float32) / 100)
        h5.create_dataset("yoff", data=np.zeros(N_FRAMES, dtype=np.float32))
    _, p = build(dense_session, tmp_path / "out")
    assert p["motionAvailable"] and p["motionFrameCount"] == N_FRAMES


def test_spikes_shape_and_values(dense_session, tmp_path):
    _, p = build(dense_session, tmp_path / "out", force_embed=True)
    with h5py.File(dense_session / "spikes.h5") as h5:
        np.testing.assert_array_equal(decode_f32(p["oasisSpikes"], (N_ROIS, N_FRAMES)), h5["spikes"][()])


def test_roi_model_scores(dense_session, tmp_path):
    prob = np.linspace(0.05, 0.95, N_ROIS).astype(np.float32)
    with h5py.File(dense_session / "roi_model_scores.h5", "w") as h5:
        h5.create_dataset("probability", data=prob)
        h5.create_dataset("state", data=np.array(["bad", "gray", "gray", "gray", "good", "good"], dtype="S4"))
    _, p = build(dense_session, tmp_path / "out")
    got = [s["probability"] for s in p["roiModelScores"]] if isinstance(p["roiModelScores"], list) else p["roiModelScores"]["probability"]
    np.testing.assert_allclose(got, prob, atol=1e-6)


# ---------- uploaded metrics and categorical labels (command line) ----------

def load_custom(tmp_path, text, name="m.csv"):
    from adroit_qc import summary as ps
    path = tmp_path / name; path.write_text(text)
    return ps._load_custom_roi_metrics([str(path)], N_ROIS, np.arange(N_ROIS))


@pytest.mark.parametrize("name,sep", [("m.csv", ","), ("m.tsv", "\t"), ("m.txt", "\t"), ("m2.txt", ",")])
def test_numeric_metrics_all_delimiters(tmp_path, name, sep):
    rows = ["roi_index" + sep + "snr"] + [f"{k}{sep}{k / 2}" for k in range(N_ROIS)]
    p = load_custom(tmp_path, "\n".join(rows) + "\n", name)
    assert [v["snr"] for v in p["values"]] == [k / 2 for k in range(N_ROIS)]


def test_roi_index_maps_rows_and_missing_rows_are_blank(tmp_path):
    p = load_custom(tmp_path, "roi_index,snr\n4,9\n1,3\n")
    assert [v.get("snr") for v in p["values"]] == [None, 3.0, None, None, 9.0, None]


def test_row_count_must_match_without_roi_index(tmp_path):
    with pytest.raises(Exception):
        load_custom(tmp_path, "snr\n1\n2\n")


@pytest.mark.parametrize("header", ["suite2p_index", "roi", "index"])
def test_retired_index_columns_are_rejected(tmp_path, header):
    with pytest.raises(Exception):
        load_custom(tmp_path, f"{header},snr\n0,1\n")


@pytest.mark.parametrize("name", ["m.json", "m.npy", "m.npz"])
def test_non_table_formats_are_rejected(tmp_path, name):
    with pytest.raises(Exception):
        load_custom(tmp_path, "{}", name)


def test_categorical_variables_and_cell_identity(tmp_path):
    p = load_custom(tmp_path, "roi_index,cell_identity_code,cell_identity_label,layer_code,depth\n0,1,PV,2,100\n1,2,SST,3,120\n3,1,PV,2,90\n")
    assert p["cellIdentity"]["codes"] == [1, 2, None, 1, None, None]
    assert p["cellIdentity"]["names"] == {"1": "PV", "2": "SST"}
    layer = next(d for d in p["definitions"] if d["key"] == "layer")
    assert layer["kind"] == "categorical" and layer["needsNames"] and set(layer["categories"]) == {"2", "3"}
    assert any(d["key"] == "depth" and d.get("kind") != "categorical" for d in p["definitions"])


@pytest.mark.parametrize("text", ["roi_index,layer_label\n0,L2\n", "roi_index,layer_code\n0,1.5\n", "roi_index,layer_code\n0,two\n"])
def test_bad_categorical_columns_are_errors(tmp_path, text):
    with pytest.raises(ValueError):
        load_custom(tmp_path, text)


def test_custom_metrics_reach_the_payload(dense_session, tmp_path):
    f = tmp_path / "labels.csv"
    f.write_text("roi_index,cell_identity_code,cell_identity_label,snr\n" + "".join(f"{k},{k % 2},{'A' if k % 2 == 0 else 'B'},{k}\n" for k in range(N_ROIS)))
    _, p = build(dense_session, tmp_path / "out", custom_roi_metrics=[str(f)])
    assert p["customRoiMetrics"]["cellIdentity"]["codes"] == [k % 2 for k in range(N_ROIS)]
    assert [v["snr"] for v in p["customRoiMetrics"]["values"]] == list(map(float, range(N_ROIS)))


def test_filter_settings_file(dense_session, tmp_path):
    fs = tmp_path / "set.json"
    fs.write_text(json.dumps({"name": "strict", "filter": {"roiAreaMin": 40}}))
    _, p = build(dense_session, tmp_path / "out", filter_settings=str(fs))
    assert p["initialMorphologyFilter"]["roiAreaMin"] == 40


# ---------- validator ----------

def test_validator_accepts_good_sessions(dense_session, sparse_session):
    for s in (dense_session, sparse_session):
        r = run_script("scripts/validate_inputs.py", str(s))
        assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("breakage", ["stray_spikes_npy", "short_traces", "truncated_npy", "mask_label_gap"])
def test_validator_rejects_broken_sessions(dense_session, breakage):
    s = dense_session
    if breakage == "stray_spikes_npy":
        (s / "spikes.h5").unlink(); np.save(s / "spikes.npy", np.zeros((N_ROIS, N_FRAMES), np.float32))
    elif breakage == "short_traces":
        np.save(s / "F.npy", np.zeros((N_ROIS - 1, N_FRAMES), np.float32))
    elif breakage == "truncated_npy":
        data = (s / "F.npy").read_bytes(); (s / "F.npy").write_bytes(data[:40])
    elif breakage == "mask_label_gap":
        m = label_mask(); m[m == 3] = 0; np.save(s / "roi_mask.npy", m)
    r = run_script("scripts/validate_inputs.py", str(s))
    assert r.returncode == 1, f"{breakage} was not reported:\n{r.stdout}"
    assert "Traceback" not in r.stdout + r.stderr
