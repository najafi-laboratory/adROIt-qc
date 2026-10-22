# Demo sessions

Two configurations under `config/demo/` build reviewers from laboratory recordings,
one per input path worth demonstrating.

| Config | Demonstrates |
|---|---|
| `demo_cerebellum_dendrite.yaml` | Suite2p input path; dendrite morphology preset |
| `demo_ppc_multi_indicator.yaml` | anatomical second channel; excitatory/inhibitory cell-type labels; sidecar array storage |

```yaml
name: demo_ppc_multi_indicator
session_path: data/demo/ppc_multi_indicator
input_layout: suite2p
output_dir: outputs/demo_ppc_multi_indicator
roi_index_label: "Original Index"
target_structure: soma
```

!!! note "Everything here is deletable"
    `config/demo/` and `scripts/demo_assemble_ppc_session.sh` are laboratory-specific.
    Delete both for a clean start; nothing else in the repository depends on them.

## Try them without installing anything

Both demo reviewers are archived with their trace arrays, so you can explore the
interface before building anything yourself.

| Session | What it shows | Record |
|:---|:---|:---|
| Posterior parietal cortex, 225 ROIs | Two channels, excitatory/inhibitory cell-type labels; the session in every screenshot on this site | [https://sandbox.zenodo.org/records/620871](https://sandbox.zenodo.org/records/620871) |
| Cerebellar dendrite, 2,702 ROIs | Classifier scores; set **ROI model score probability min** to 0.8 and press **Apply Filters** to keep 279 | [https://sandbox.zenodo.org/records/620873](https://sandbox.zenodo.org/records/620873) |

1. Download the three files from a record into one folder.
2. Open the reviewer in one of two ways:
    * **Served, so both arrays load automatically.** Run `python -m http.server 8000` in
      that folder and open `http://localhost:8000/<name>_interactive_fov_roi_dff.html`.
    * **From disk.** Double-click the `.html` file, then click **Load dF/F file** and
      choose the `_dff.npy` file, and **Load inferred spikes file** and choose the
      `_oasis_spikes.npy` file. A page opened from disk is not allowed to read the files
      beside it, so they have to be selected by hand.

[Whether the sidecars load automatically](quickstart.md#whether-the-sidecars-load-automatically)
explains the difference.

!!! note "Sandbox records"
    These are test records on the Zenodo sandbox. They will move to zenodo.org, with
    permanent DOIs, when the accompanying paper is published.

## Getting the data

The recordings are published separately from the code. `scripts/fetch_demo_data.py`
downloads them into `data/demo/` once the archive record is set at the top of that
script.

## Assembling the PPC session

The posterior parietal cortex session is not one directory. Its Suite2p outputs, its
inferred spikes and its cell-type labels live in three places, and the labels do not
exist as a file at all until they are derived from the red-channel call.
`scripts/demo_assemble_ppc_session.sh` records that assembly:

```bash
bash scripts/demo_assemble_ppc_session.sh
python scripts/build_reviewer.py config/demo/demo_ppc_multi_indicator.yaml
```

It links the Suite2p outputs and the spikes file into one session directory, derives
per-ROI cell-type codes from `redcell.npy` following the documented coding (1 for
inhibitory/red, −1 for excitatory/non-red), prints the resulting label counts, and
names the build command to run next.

This script assumes the laboratory's storage layout and will not run elsewhere. It is
in the repository as a worked example of the general problem it solves: a processed
session scattered across several locations, with one input that has to be computed
rather than copied.

## Storage mode

The two demos differ in how the trace arrays are stored, which is deliberate. Below the
embedding threshold the arrays go inside the HTML and the file is self-contained on its
own. Above it they are written beside the HTML as `.npy` sidecars, which the page reads in
full when it opens if it is served over a web address, and which you select with the load
buttons if it is opened from disk. The PPC session, at 225 ROIs over 107,135 frames, lands in sidecar
mode, so it also serves as the example of that path.
