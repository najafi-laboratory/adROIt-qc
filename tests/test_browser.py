"""The reviewer page itself: opens without script errors, and its uploads and exports work.

Skipped unless Playwright and its Chromium are installed:
    pip install playwright && python -m playwright install chromium
"""
from __future__ import annotations

import csv
import io

import pytest

from conftest import N_ROIS, build

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def page_of(dense_session, tmp_path):
    html, _ = build(dense_session, tmp_path / "out")
    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:  # browser binary not installed
            pytest.skip(f"Chromium unavailable: {exc}")
        page = browser.new_context(accept_downloads=True, viewport={"width": 1500, "height": 1000}).new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(html.as_uri()); page.wait_for_timeout(1500)
        yield page, errors, tmp_path
        browser.close()


def test_page_opens_without_errors(page_of):
    page, errors, _ = page_of
    assert page.evaluate("data.nRois") == N_ROIS
    assert not errors


def test_every_info_icon_opens_bulleted_help(page_of):
    page, errors, _ = page_of
    targets = page.evaluate("[...document.querySelectorAll('.info-button[data-info-target]')].filter(b => b.offsetParent && !b.closest('dialog')).map(b => b.dataset.infoTarget)")
    assert targets
    for t in targets:
        page.click(f".info-button[data-info-target={t}]:visible")
        assert page.evaluate(f"!document.getElementById('{t}').hidden && document.getElementById('{t}').querySelectorAll('li').length > 0"), t
        page.click(f".info-button[data-info-target={t}]:visible")
    assert page.evaluate("[...document.querySelectorAll('details.menu-card')].every(d => d.open)")
    assert page.evaluate("[...document.querySelectorAll('button')].every(b => !/Read more/.test(b.textContent))")
    assert not errors


def test_categorical_upload_filter_and_export(page_of):
    page, errors, tmp = page_of
    f = tmp / "labels.csv"
    f.write_text("roi_index,cell_identity_code,cell_identity_label,zone_code,depth\n" + "".join(f"{k},{1 + k % 2},{'PV' if k % 2 == 0 else 'SST'},{k % 3},{k * 10}\n" for k in range(N_ROIS)))
    page.click("#openCustomRoiUploadDialog"); page.set_input_files("#customRoiUploadFile", str(f)); page.click("#applyCustomRoiUpload"); page.wait_for_timeout(500)
    status = page.evaluate("document.getElementById('customRoiUploadStatus').textContent")
    assert "categorical" in status and "cell identity" in status and "name each code" in status
    assert page.evaluate("cellTypeLabelName(data.cellTypeLabels[1])") == "SST"
    page.evaluate("closeDialog(document.getElementById('customRoiUploadDialog'))")
    page.click("#openMorphologyDialog"); page.wait_for_timeout(300)
    page.check(".custom-category-filter[data-key=zone][value='0']")
    passing = page.evaluate("Array.from({length: data.nRois}, (_, r) => r).filter(r => passesFilter(r, data.morphology[r], readFilter())).length")
    assert passing == len([k for k in range(N_ROIS) if k % 3 == 0])
    cols = page.evaluate("spreadsheetColumns().map(c => c.key)")
    assert {"custom_zone", "custom_zone_label", "custom_depth", "cell_identity_label"} <= set(cols)
    assert not errors


def test_spreadsheet_csv_follows_column_choice(page_of):
    page, errors, _ = page_of
    page.evaluate("setSpreadsheetExcluded(spreadsheetColumns().map(c => c.key).filter(k => !['manual_QC_label', 'roi_area_px'].includes(k)))")
    text = page.evaluate("metricSpreadsheetCsv()")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["roi_index", "manual_QC_label", "roi_area_px"]
    assert len(rows) == N_ROIS + 1 and rows[1][1] == "not labeled"
    page.evaluate("setSpreadsheetExcluded([])")
    assert not errors
