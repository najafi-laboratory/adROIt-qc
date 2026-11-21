# adROIt-qc

This repository builds an interactive, self-contained reviewer for region-of-interest
quality control in two-photon calcium imaging. You point it at a processed session and
it writes a single HTML file that embeds the field-of-view images, the per-ROI masks,
the per-ROI metrics and the review interface itself. A reviewer opens that file in a
browser with no installed software, no server and no network connection, labels ROIs,
and saves the reviewed state back into another self-contained copy.

<div class="screenshot screenshot-tall" markdown>
![The reviewer interface: field of view, selected-ROI trace with inferred-spike overlay, stacked traces, motion correction, and the control menu.](img/full-layout.png)
</div>

## What this repository is for

adROIt-qc is open source; issues, forks and pull requests are welcome on
[GitHub](https://github.com/najafi-laboratory/adROIt-qc). The generator here reads a documented input layout,
so output from any segmentation or deconvolution software can be reviewed, not only
Suite2p. Use it to

* review a session of your own, by writing one configuration file;
* convert output from another pipeline into the documented layout;
* check a converted directory before building, so a silent mismatch is caught early;
* read a complete example of each supported input path.

!!! note "Self-contained"
    `pip install -e .` installs everything the generator needs. No laboratory
    package or cluster environment is required.

## How it fits together

```
your segmentation output
        |
        |  adroit_qc.inputs / scripts/make_standard_inputs.py   (optional converters)
        v
documented input layout  ---->  scripts/validate_inputs.py   (check before building)
        |
        |  scripts/build_reviewer.py config/my_session.yaml
        v
outputs/my_session/
    <name>_interactive_fov_roi_dff.html      the reviewer
    <name>_processing_summary.pdf            static summary
    <name>_dff.npy, <name>_inferred_spikes.npy  sidecar arrays for large sessions
    run.json                                 resolved paths and options
```

For sessions above the embedding threshold the trace arrays are written beside the
HTML as sidecars rather than embedded in it. The HTML and its sidecars must travel
together. They load automatically when the page is opened through a web address, and
are selected with **Load dF/F** and **Load inferred spikes** when it is opened from disk; see
[Whether the sidecars load automatically](quickstart.md#whether-the-sidecars-load-automatically).

## Where to start

* [Demo Sessions](demo-sessions.md#try-them-without-installing-anything) links two
  example reviewers you can download and open in a browser right away.
* [Quickstart](quickstart.md) installs the package and builds a reviewer.
* [Tutorial](tutorial.md) goes from installation to reviewers built from every kind of
  input, by hand and from one script.
* [Inputs, settings and helpers](reference.md) lists every input file, build setting and
  helper function on one page.
* [Preparing Your Inputs](inputs.md) is the input contract, including the one
  correspondence no tool can check for you.
* [Build Settings](cli-mapping.md) documents every configuration field and flag.
* [Interface Tour](interface.md) walks through the reviewer's panels, and the
  [Interface Reference](interface-reference.md) holds the full text behind each **(i)**.
