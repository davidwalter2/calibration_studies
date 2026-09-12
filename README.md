# calibration_studies

Analysis scripts for the CMS muon momentum-scale calibration chain (CVH refit →
resolution and material model → unbinned mass likelihood → Z mass / width).

## Directory map

| directory | what is in it |
|---|---|
| `production/` | the large CVH refit productions — filelists, chunk lists, submit/status/resume machinery, and **`PRODUCTIONS.md`**, the reference for every sample |
| `slurm/` | generic per-file slurm array submission for any parameterisable `cmsRun` config |
| `transfer/` | grid → ceph dataset transfers with per-file verification and atomic publish |
| `resolution/` | the resolution / material workstream: CF exponents, track-resolution fits, the clean-propagation test (`cleanprop/`), private gun samples (`simprod/`), and `prodfiles.py` (see below) |
| `zchannel/` | the Z → μμ channel of the unbinned mass likelihood: FSR kernel, lineshape, datacards, generator-level closure |
| `fullscale/` | the full-statistics joint fits driven through rabbit |
| `engaging/` | running the offline TensorFlow / rabbit fits on MIT Engaging GPUs |
| `global_corrections/` | solving the per-candidate gradients for the global parameters (B-field modes, material groups, alignment) |
| `lineshape/` | QED radiation and resonance lineshape modelling for Z / J/ψ / Υ |
| `alcareco_validation/` | validation of the `TkAlKsToPiPi` / `TkAlLambdaToProtonPi` V0 ALCARECOs |
| `kinkfinder/`, `pixelhits/`, `module_level_corrections/` | targeted studies: decay-in-flight kinks, pixel hit quality, per-module corrections |
| `sin2thetaW_running/`, `slides/` | the running-sin²θ_W figure; talk sources |
| `env/`, `env_tf/`, `setup_env.sh` | the python environments (numpy-2 venv; the TF/rabbit image shim) |

Scripts that `import constants` and friends bare (e.g. in `lineshape/`) must be
run from inside their own directory.

**`lineshape/`** — `constants.py` (masses, widths, α_QED), `functions.py` (the
radiator kernel and the BW ⊗ AP convolution via FFT), `ioutils.py` (HDF5
histogram loading and cross-section scaling), `drell_yan_xsec.py`; the
`plot_lineshape*.py` / `plot_radiation_kernel.py` figures; `analytic_gen_
distributions.py` and `postfsr_gen_distributions.py` for generator-level
distributions; `zwidth_sensitivity.py` for the Γ_Z projection; `alpha_running/`
and `results/` for the running-α inputs and outputs.

**`alcareco_validation/`** — `extract_v0_kinematics.py` (FWLite / el7, ALCARECO
→ text) feeding `plot_{ks,lambda}_mass.py`, `plot_v0_kinematics.py` and
`plot_lam_daughters.py` (Armenteros-Podolanski); plus the CVH-variant drivers
(`run_cvh_variants*.sh`, `plot_cvh_variants.py`), `plot_dedx_vs_p.py` and the
track-pair-mass pair. Data paths sit at the top of each plotting script — point
them at your own extraction. Figures use the WRemnants `wums.plot_tools`
infrastructure and come out as CMS-Preliminary PDF + PNG.

## Productions

`production/PRODUCTIONS.md` is the single reference: which samples exist, what
each contains, the ceph paths, the configuration option by option, the cost and
memory numbers, how to re-run on HTCondor or slurm, and the defect list. The two
samples the analysis uses:

| tag | what | events | candidates | ceph |
|---|---|---:|---:|---|
| `jpsimc_20M_260906_v2` | UL16 J/ψ MC ALCARECO, two-track CVH | 21 750 740 | 21 678 062 | `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2/` |
| `dymc_8p5M_260906_v2` | UL16 DY MiniAODv2, two-track Z | 8 502 597 | 3 799 624 | `/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2/` |

Two things to know before using the output: `doRes` shifts the global parameter
indices relative to the 2016 data productions (so they cannot be pooled without
re-mapping through `runtree`), and the fit ran with the SIM's own field
(`useDefaultField`), not the 3D TOSCA grid.

## Reading a production: `resolution/prodfiles.py`

The CVH makers run `numberOfThreads=4`, so **a task is four files**,
`task_XXXX/globalcor_0.root … globalcor_3.root`. An event never splits across
streams and the candidate content is bit-identical to a single-thread run after
sorting on `(run, lumi, event)`, so the four files simply concatenate — but a
reader that names `globalcor_0.root` takes a quarter of the statistics and says
nothing.

Every reader here lists its inputs through `resolution/prodfiles.py` (shell twin
`prodfiles.sh`) rather than `sorted(glob.glob(...))`:

```python
task_dirs(base)                        # sorted task directories
stream_files(task_dir)                 # that task's streams, in STREAM order
task_complete(task_dir) / task_reason  # usable?  or why not
runtree_file(task_dir)                 # ONE file: the FIRST EXISTING stream
single_file(task_dir)                  # the one stream of a 1-thread output
iter_files(base, max_tasks)            # every stream of the first N usable tasks
resolve(spec, max_tasks)               # DROP-IN for sorted(glob(spec))[:ntasks]
last_stats()                           # ntasks_used / skipped / why
```

* the stream index is **widened**, so an existing
  `--files '.../task_*/globalcor_0.root'` reads the whole task;
* `--files` also takes a production or task **directory**, or an
  **`@list.txt`** of explicit inputs (how the sharded wrappers hand a worker a
  subset without a symlink farm). A directory spec auto-detects the output stem
  (`globalcor` vs `globalcor_resclosure`) and refuses a directory holding both;
* **`--ntasks` caps TASKS, not files** (likewise `--nfiles`, `--max-files`,
  `NTASKS=`);
* a task is used only if its `.complete` sentinel is there, no stream file is
  empty, and it has as many streams as the sentinel declares — otherwise it is
  skipped **whole** and counted. A tree with no sentinels at all (a hand-made
  smoke, a staged shard directory) keeps working: the requirement turns itself
  off, with a warning;
* the `runtree` parameter map (13 MB, byte-identical in every stream) is read
  once per task via `runtree_file()`, never as a concatenation;
* `pf_clean_incomplete` removes **every** stream of an incomplete task — a
  cleanup that deleted stream 0 only would leave streams 1–3 for a widened glob
  to swallow.

```bash
python3 resolution/prodfiles.py "$PROD" --stats      # what a spec resolves to
python3 resolution/prodfiles.py "$PROD" --runtrees   # one file per task
```

The shell twin `prodfiles.sh` delegates to the same module, so drivers cannot
drift from readers: `pf_files`, `pf_task_dirs`, `pf_runtrees`, `pf_stream_files`,
`pf_task_complete`, `pf_clean_incomplete`, `pf_stats`.

## Figures

Plot output goes to `~/public_html/ZMass/cvh/<YYMMDD>_<tag>/`, one file per
panel, with `index.php` copied from
`~/public_html/ZMass/cvh/260814_cleanprop/index.php` into every new directory.
Some scripts still default `--outpath` to the older `~/public_html/cvh/…`; pass
`--outpath` explicitly.

## License
MIT License.
