# calibration_studies

Analysis scripts for the CMS W boson mass measurement calibration chain.

## Structure

### `lineshape/`

QED radiation and resonance lineshape studies used to model the Z/J/ψ/Υ invariant
mass distributions in the momentum scale calibration.

- `constants.py` — physics constants (masses, widths, α_QED)
- `functions.py` — radiator kernel and BW⊗AP convolution via FFT
- `ioutils.py` — HDF5 helpers for loading histograms and scaling by cross section
- `drell_yan_xsec.py` — Drell-Yan cross section calculations
- `plot_radiation_kernel.py` — comparison of radiator kernels for Z, Υ, J/ψ
- `analytic_gen_distributions.py` — generator-level distributions (Gaussian, BW, convolutions)
- `postfsr_gen_distributions.py` — post-FSR generator distributions vs. MC

Run from within `lineshape/` so that bare `import constants` etc. resolve correctly.

### `alcareco_validation/`

Validation scripts for the `TkAlKsToPiPi` and `TkAlLambdaToProtonPi` ALCARECOs
developed for track-based alignment using V0 decays.

- `extract_v0_kinematics.py` — FWLite (python2/el7) script to extract V0 kinematics
  from ALCARECO ROOT files into text files for plotting
- `plot_ks_mass.py` — KS→π⁺π⁻ invariant mass peak
- `plot_lambda_mass.py` — Λ⁰→pπ⁻ invariant mass peak
- `plot_v0_kinematics.py` — full kinematic distributions: pT, η, φ, flight lengths,
  daughter tracks, candidates per event
- `plot_lam_daughters.py` — Λ⁰ daughter analysis: p/p̄ vs π∓ pT and η, Armenteros-Podolanski plot

All plotting scripts use the WRemnants `wums.plot_tools` infrastructure and output
CMS Preliminary figures as PDF and PNG.

Data paths at the top of each plotting script point to the smoke-test extraction in
`/tmp/`; update them to point to your ALCARECO output files.

### `production/`

Large CVH refit productions on submit slurm. Each production is a filelist, a
chunk list, a submit/status/resume trio and a `STATE.md` recording the exact
configuration and why each option is what it is.

- `jpsimc_20M_260905` — 21.7M events of UL16 `JPsiToMuMu_Pt8toInf` MC ALCARECO
  through the two-track J/ψ CVH fit, with the global-correction gradients
  (factored Hessian) and the in-maker resolution-CF exponents on at once.
  **Read `production/STATE.md` before using the output**: `doRes` shifts the
  global parameter indices relative to the 2016 data productions, and the fit
  runs with the SIM's own field (`useDefaultField`), not the 3D TOSCA grid.
- `truncated_inputs_260905.txt` — the 43 zombie files found in this sample's
  repack. Size does not identify them; re-scan with `scan_events.py` before
  using any other slice.

```bash
cd production
./status.sh              # complete/running/pending, events, projected finish
./resume.sh --list-only  # which chunks are missing
./resume.sh              # resubmit only those
```

### Reading a production: `resolution/prodfiles.py`

From 2026-09-06 the CVH makers run `numberOfThreads=4`, so **a task is four
files**, `task_XXXX/globalcor_0.root .. globalcor_3.root`. An event never
splits across streams and the candidate content is bit-identical to a
single-thread run after sorting on (run, lumi, event), so the four files simply
concatenate — but a reader that names `globalcor_0.root` takes a quarter of the
statistics and says nothing.

Every reader here lists its inputs through `resolution/prodfiles.py` (shell
twin `prodfiles.sh`) rather than `sorted(glob.glob(...))`:

* the stream index is **widened**, so an existing `--files
  '.../task_*/globalcor_0.root'` reads the whole task;
* `--files` also takes a production or task **directory**, or an
  **`@list.txt`** of explicit inputs;
* **`--ntasks` caps TASKS, not files** — a cap on the file list would take 1/N
  of the intended tasks;
* a task is used only if its `.complete` sentinel is there, no stream file is
  empty, and it has as many streams as the sentinel declares; otherwise it is
  skipped **whole** and counted;
* the `runtree` parameter map (13 MB, byte-identical in every stream) is read
  once per task via `runtree_file()`, which returns the first EXISTING stream
  rather than assuming stream 0.

```bash
python3 resolution/prodfiles.py "$PROD" --stats      # what a spec resolves to
python3 resolution/prodfiles.py "$PROD" --runtrees   # one file per task
```

`production/PRODUCTION_NEXT.md` sec. 10 has the full API and the validation
table.

## License
MIT License.
