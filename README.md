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

## License
MIT License.
