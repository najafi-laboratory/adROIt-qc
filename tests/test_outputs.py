"""Every output can be read back and matches what went in."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from conftest import N_FRAMES, N_ROIS, build, decode_f32, payload_of, run_script


def test_payload_is_strict_json_for_external_masks(dense_session, tmp_path):
    # footprint is undefined for external masks; it must be written as null, never NaN
    _, p = build(dense_session, tmp_path / "out")
    assert all(m.get("footprint") is None or np.isfinite(m["footprint"]) for m in p["morphology"])


def test_default_embeds_small_sessions(dense_session, tmp_path):
    html, p = build(dense_session, tmp_path / "out")
    assert p["dffStorageMode"] == "embedded" and p["dff"]
    assert not list(html.parent.glob("*_dff.npy"))


def test_force_embed_false_writes_float32_sidecars(dense_session, tmp_path):
    html, p = build(dense_session, tmp_path / "out", force_embed=False)
    session = p["session"]
    dff = np.load(html.parent / f"{session}_dff.npy"); spk = np.load(html.parent / f"{session}_inferred_spikes.npy")
    for arr in (dff, spk):
        assert arr.dtype == np.float32 and arr.shape == (N_ROIS, N_FRAMES) and arr.flags.c_contiguous
    with h5py.File(dense_session / "spikes.h5") as h5:
        np.testing.assert_array_equal(spk, h5["spikes"][()])
    assert p["dff"] is None and p["dffSidecarName"] == f"{session}_dff.npy"


def test_force_embed_true_and_limit(dense_session, tmp_path):
    _, p = build(dense_session, tmp_path / "a", force_embed=True)
    assert p["dffStorageMode"] == "embedded"
    _, q = build(dense_session, tmp_path / "b", embed_limit_mb=2.01)
    assert q["embedLimitBytes"] > 0


def test_embed_limit_must_exceed_two_mb():
    r = run_script("-m", "roi_reviewer.summary", "x", "--embed-limit-mb", "1")
    assert r.returncode == 2 and "embed" in r.stderr.lower()


@pytest.mark.parametrize("value,ok", [("true", True), ("FALSE", True), ("1", True), ("no", True), ("maybe", False)])
def test_force_embed_values(value, ok):
    from roi_reviewer import summary as ps
    if ok:
        assert ps._parse_bool_flag(value) in (True, False)
    else:
        with pytest.raises(Exception):
            ps._parse_bool_flag(value)


def test_html_only_round_trip_is_lossless(dense_session, tmp_path):
    out = tmp_path / "out"
    html, p = build(dense_session, out, force_embed=False)
    name = p["dffSidecarName"]; before = (out / name).read_bytes()
    _, q = build(dense_session, out, html_only=True, force_embed=True)
    assert q["dffStorageMode"] == "embedded"
    (out / name).unlink()
    _, r = build(dense_session, out, html_only=True, force_embed=False)
    assert (out / name).read_bytes() == before


def test_html_only_keeps_data_and_title(dense_session, tmp_path):
    out = tmp_path / "out"
    _, p = build(dense_session, out, title="A <b>&</b> test")
    _, q = build(dense_session, out, html_only=True)
    assert q["morphology"] == p["morphology"] and q["dffMetrics"] == p["dffMetrics"]
    text = next(out.glob("*.html")).read_text()
    assert "<title>A &lt;b&gt;&amp;&lt;/b&gt; test</title>" in text


def test_label_export_round_trip(suite2p_session, tmp_path):
    from roi_reviewer import roi_metrics as rm
    stat = np.load(suite2p_session / "suite2p/plane0/stat.npy", allow_pickle=True)
    export = {"suite2p_roi_count": N_ROIS, "suite2p_stat_fingerprint": rm.suite2p_stat_fingerprint(stat),
              "labels": [{"suite2p_roi": k, "label": [1, 0, 2, None, 1, 0][k]} for k in range(N_ROIS)]}
    f = tmp_path / "labels.json"; f.write_text(json.dumps(export))
    out = rm.apply_label_export(f, suite2p_session / "suite2p/plane0", output_path=tmp_path / "roi_manual_labels.npy")
    got = np.load(out)
    np.testing.assert_array_equal(np.nan_to_num(got, nan=-1), [1, 0, 2, -1, 1, 0])


def test_label_export_rejects_other_sessions(suite2p_session, tmp_path):
    from roi_reviewer import roi_metrics as rm
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"suite2p_roi_count": N_ROIS + 1, "labels": []}))
    with pytest.raises(ValueError):
        rm.apply_label_export(f, suite2p_session / "suite2p/plane0", output_path=tmp_path / "x.npy")


# ---------- converters ----------

def test_make_standard_inputs_output_validates_and_builds(tmp_path):
    from conftest import label_mask, traces
    src = tmp_path / "seg"; src.mkdir()
    fluo, _, spikes = traces()
    np.save(src / "masks.npy", label_mask()); np.save(src / "traces.npy", fluo.T); np.save(src / "spikes.npy", spikes)
    np.save(src / "fov.npy", np.ones((64, 64), np.float32))
    out = tmp_path / "session"
    r = run_script("scripts/make_standard_inputs.py", "--masks", str(src / "masks.npy"), "--traces", str(src / "traces.npy"),
                   "--spikes", str(src / "spikes.npy"), "--fov", str(src / "fov.npy"), "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_script("scripts/validate_inputs.py", str(out)).returncode == 0
    _, p = build(out, tmp_path / "rev")
    assert p["nRois"] == N_ROIS


def test_caiman_converter(tmp_path):
    """A CaImAn-shaped result (A in F order, C, S, idx_components) converts, validates and builds."""
    from scipy import sparse
    from conftest import label_mask, traces
    d1, d2 = 64, 64
    mask = label_mask(); fluo, _, spikes = traces()
    A = sparse.csc_matrix(np.stack([np.ravel(mask == k + 1, order="F").astype(np.float32) for k in range(N_ROIS)], axis=1))
    res = tmp_path / "cnmf_results.hdf5"
    with h5py.File(res, "w") as f:
        est = f.create_group("estimates"); a = est.create_group("A")
        for key, val in (("data", A.data), ("indices", A.indices), ("indptr", A.indptr), ("shape", np.array(A.shape))):
            a.create_dataset(key, data=val)
        est.create_dataset("C", data=fluo - 200); est.create_dataset("S", data=spikes)
        est.create_dataset("idx_components", data=np.array([0, 1, 2, 4])); est.create_dataset("SNR_comp", data=np.linspace(1, 6, N_ROIS))
        est.create_dataset("r_values", data=np.linspace(0.5, 0.99, N_ROIS)); est.create_dataset("dims", data=np.array([d1, d2]))
    movie = np.zeros((d1 * d2, N_FRAMES), dtype=np.float32)
    for k in range(N_ROIS):
        movie[np.ravel(mask == k + 1, order="F")] = fluo[k]
    mm = tmp_path / f"memmap__d1_{d1}_d2_{d2}_d3_1_order_C_frames_{N_FRAMES}.mmap"
    np.memmap(mm, mode="w+", dtype=np.float32, shape=movie.shape, order="C")[:] = movie
    np.save(tmp_path / "shifts.npy", np.zeros((N_FRAMES, 2), np.float32))
    out = tmp_path / "caiman_session"
    r = run_script("examples/caiman/caiman_to_reviewer.py", "--cnmf", str(res), "--movie", str(mm), "--shifts", str(tmp_path / "shifts.npy"),
                   "--frame-rate", "30", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    np.testing.assert_allclose(np.load(out / "F.npy"), fluo, rtol=1e-5)
    assert np.load(out / "iscell.npy")[:, 0].tolist() == [1, 1, 1, 0, 1, 0]
    assert run_script("scripts/validate_inputs.py", str(out)).returncode == 0
    _, p = build(out, tmp_path / "rev", custom_roi_metrics=[str(out / "caiman_metrics.csv")])
    assert [int(r["npix"]) for r in sorted(p["rois"], key=lambda r: r["roi"])] == [int((mask == k + 1).sum()) for k in range(N_ROIS)]
    status = next(d for d in p["customRoiMetrics"]["definitions"] if d["key"] == "caiman_status")
    assert status["kind"] == "categorical" and status["categories"] == {"0": "rejected", "1": "accepted"}
