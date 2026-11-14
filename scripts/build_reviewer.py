#!/usr/bin/env python3
"""Build an interactive reviewer from a configuration file.

    python scripts/build_reviewer.py config/my_session.yaml
    python scripts/build_reviewer.py config/my_session.yaml --dry-run

This is a thin wrapper: it resolves and checks the paths in the config, runs
the inputs through validate_inputs.py, then invokes the generator. The point of
the wrapper is that the command which produced a given reviewer is recorded in
the output directory as run.json, so an example can be regenerated exactly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

LAYOUTS = {"auto", "suite2p", "external_rois", "qc_results", "manual_qc_results"}
REPO_ROOT = Path(__file__).resolve().parent.parent


# Every key a config may carry. Each one other than name and notes maps to
# exactly one generator flag in build_command(); see docs/cli-mapping.md.
KNOWN_KEYS = {
    "name", "notes", "session_path", "output_dir", "input_layout",
    "roi_index_label", "filter_set", "target_structure", "initialize_labels_from_model_scores",
    "custom_roi_metrics", "pdf_name", "html_name", "archive_output", "archive_name",
    "custom_roi_metric_sections", "filter_settings", "html_only", "title", "force_embed", "embed_limit_mb",
}
METRIC_SECTIONS = {"uploaded", "model_derived", "morphology", "fluorescence", "inferred_spike"}
FILTER_SETS = {"all_rois", "soma", "dendrite", "dendrite_relaxed"}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    unknown = sorted(set(cfg) - KNOWN_KEYS)
    if unknown:
        # A misspelled key would otherwise be dropped without a word and the
        # reviewer built without the setting the user thought they had made.
        raise SystemExit(f"{path}: unrecognized field(s): {', '.join(unknown)}; "
                         f"known fields are {', '.join(sorted(KNOWN_KEYS))}")
    missing = [k for k in ("name", "session_path", "output_dir") if not cfg.get(k)]
    if missing:
        raise SystemExit(f"{path}: missing required field(s): {', '.join(missing)}")
    layout = cfg.get("input_layout", "auto")
    if layout not in LAYOUTS:
        raise SystemExit(f"{path}: input_layout must be one of {sorted(LAYOUTS)}, got {layout!r}")
    if "target_structure" in cfg:
        # former name of filter_set
        if "filter_set" in cfg:
            raise SystemExit(f"{path}: give filter_set only; target_structure is its former name")
        cfg["filter_set"] = cfg.pop("target_structure")
    filter_set = cfg.get("filter_set")
    if filter_set and filter_set not in FILTER_SETS:
        raise SystemExit(f"{path}: filter_set must be one of {sorted(FILTER_SETS)} "
                         f"or null, got {filter_set!r}")
    metrics = cfg.get("custom_roi_metrics") or []
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics):
        raise SystemExit(f"{path}: custom_roi_metrics must be a path or a list of paths")
    cfg["custom_roi_metrics"] = metrics
    sections = cfg.get("custom_roi_metric_sections") or {}
    if not isinstance(sections, dict):
        raise SystemExit(f"{path}: custom_roi_metric_sections must map metric names to sections")
    bad = {name: sec for name, sec in sections.items() if sec not in METRIC_SECTIONS}
    if bad:
        raise SystemExit(f"{path}: custom_roi_metric_sections has unknown section(s) {bad}; "
                         f"use one of {sorted(METRIC_SECTIONS)}")
    cfg["custom_roi_metric_sections"] = sections
    settings = cfg.get("filter_settings")
    if settings is not None and not isinstance(settings, str):
        raise SystemExit(f"{path}: filter_settings must be the path of an exported settings JSON (or YAML) file")
    return cfg


def resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_command(cfg: dict, session: Path, output_dir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "roi_reviewer.summary", str(session),
           "--output-dir", str(output_dir)]
    if cfg.get("input_layout", "auto") != "auto":
        cmd += ["--input-layout", cfg["input_layout"]]
    if cfg.get("roi_index_label"):
        cmd += ["--roi-index-label", cfg["roi_index_label"]]
    if cfg.get("filter_set"):
        cmd += ["--filter-set", cfg["filter_set"]]
    if cfg.get("initialize_labels_from_model_scores"):
        cmd += ["--initialize-labels-from-roi-model-scores"]
    for metric_path in cfg.get("custom_roi_metrics") or []:
        cmd += ["--custom-roi-metrics", str(resolve(metric_path))]
    for metric_name, section in (cfg.get("custom_roi_metric_sections") or {}).items():
        cmd += ["--custom-roi-metric-section", f"{metric_name}={section}"]
    if cfg.get("filter_settings"):
        cmd += ["--filter-settings", str(resolve(cfg["filter_settings"]))]
    if cfg.get("html_only"):
        cmd += ["--html-only"]
    if cfg.get("title"):
        cmd += ["--title", str(cfg["title"])]
    if cfg.get("force_embed") is not None:
        if not isinstance(cfg["force_embed"], bool):
            raise SystemExit("force_embed must be true, false or null")
        cmd += ["--force-embed", "true" if cfg["force_embed"] else "false"]
    if cfg.get("embed_limit_mb") is not None:
        cmd += ["--embed-limit-mb", str(cfg["embed_limit_mb"])]
    if cfg.get("pdf_name"):
        cmd += ["--pdf-name", cfg["pdf_name"]]
    if cfg.get("html_name"):
        cmd += ["--html-name", cfg["html_name"]]
    if cfg.get("archive_output"):
        cmd += ["--archive-output"]
    if cfg.get("archive_name"):
        cmd += ["--archive-name", cfg["archive_name"]]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the generator command and exit without running it")
    parser.add_argument("--skip-validation", action="store_true",
                        help="do not run validate_inputs.py first")
    args = parser.parse_args()

    cfg = load_config(args.config)
    session = resolve(cfg["session_path"])
    output_dir = resolve(cfg["output_dir"])
    cmd = build_command(cfg, session, output_dir)

    print(f"config      {args.config}")
    print(f"session     {session}")
    print(f"output      {output_dir}")
    print(f"command     {shlex.join(cmd)}")

    if args.dry_run:
        return 0

    if not session.is_dir():
        raise SystemExit(
            f"session directory not found: {session}\n"
            "For the bundled demos, fetch the data first:\n"
            "  python scripts/fetch_demo_data.py --all"
        )

    if not args.skip_validation and cfg.get("input_layout") == "external_rois":
        print("\nvalidating inputs")
        validation = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "validate_inputs.py"), str(session)],
            check=False,
        )
        if validation.returncode != 0:
            raise SystemExit("input validation failed; fix the errors above or pass --skip-validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nrunning generator")
    result = subprocess.run(cmd, check=False)

    (output_dir / "run.json").write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_file": str(args.config),
        "config": cfg,
        "command": cmd,
        "exit_code": result.returncode,
    }, indent=2) + "\n")

    if result.returncode != 0:
        raise SystemExit(f"generator exited {result.returncode}")
    print(f"\nwrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
