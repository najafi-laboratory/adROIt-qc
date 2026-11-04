# Quickstart

## Install

```bash
git clone https://github.com/najafi-laboratory/roi-reviewer-demo
cd roi-reviewer-demo
pip install -e .
```

Or, with conda:

```bash
conda env create -f environment.yml
conda activate roi-reviewer-demo
pip install -e .
```

That is the whole dependency story: numpy, scipy, h5py, matplotlib, Pillow and PyYAML.
Confirm the generator is importable:

```bash
python -m roi_reviewer.summary --help
```

## Build a reviewer from a Suite2p session

Copy the template, point it at your session, and build:

```bash
cp config/template.yaml config/my_session.yaml
# edit session_path and output_dir
python scripts/build_reviewer.py config/my_session.yaml
```

Use `--dry-run` to resolve and print the paths without generating anything, which is
the fastest way to confirm a configuration is correct:

```bash
python scripts/build_reviewer.py config/my_session.yaml --dry-run
```

The build writes `run.json` beside the outputs, recording the resolved input paths and
the options in force, so a reviewer file can be traced back to what produced it.

## Build from another pipeline's output

If your segmentation did not come from Suite2p, convert it into the documented layout
first. The converter takes a dense label mask plus a trace matrix, transposes traces
that arrive frames-by-ROIs, and compacts mask labels that are not a gap-free run:

```bash
python scripts/make_standard_inputs.py \
    --masks  segmentation/cell_masks.npy \
    --traces segmentation/traces.npy \
    --spikes segmentation/deconvolved.npy \
    --fov    segmentation/mean_image.npy \
    --out    data/my_session

python scripts/validate_inputs.py data/my_session
```

Then set `input_layout: external_rois` in your configuration and build as above.
[Preparing Your Inputs](inputs.md) documents every file the layout accepts.

## Open the result

Open `outputs/my_session/<name>_interactive_fov_roi_dff.html` in any browser.

!!! warning "Keep the sidecars with the HTML"
    For a large session the dF/F and inferred-spike arrays are written beside the HTML
    as `<name>_dff.npy` and `<name>_oasis_spikes.npy`. Move or email the whole
    directory, not the HTML alone. If the arrays are missing, the reviewer still opens
    and the trace panels report that they are not loaded.

### Whether the sidecars load automatically

That depends on how the page is opened, not on the reviewer:

| How you open the HTML | Address starts with | Sidecars |
|:---|:---|:---|
| Through a web server, for example `python -m http.server` run in the folder | `https://` or `http://` | Load automatically, as long as they sit in the same folder as the HTML |
| By double-clicking it, or from any downloaded, copied or mounted folder | `file://` | Not loaded. Browsers do not let a page opened from disk read files beside it, so select them with **Load dF/F file** and **Load inferred spikes file** |

To serve a folder on your own machine, so both arrays load by themselves:

```bash
cd outputs/my_session
python -m http.server 8000
# then open http://localhost:8000/<name>_interactive_fov_roi_dff.html
```

The note under the two load buttons says which case applies. Sessions small enough to
embed their arrays have no sidecars and open the same way from anywhere.
