# Ideal-geometry MC closure re-productions (2026-09-17) — running state

Four HTCondor productions on the CMS global pool, submitted from submit82.
They reproduce the v2 closure samples on the **ideal tracker geometry**, plus a
matched aligned control of each leg.

## Why the geometry changes

The MC's Geant4 tracker is the IDEAL one; `106X_mcRun2_asymptotic_v17` carries
the realistic-misalignment payload `TrackerAlignment_2016_ultralegacymc_v1`, so
refitting the aligned geometry displaces every reconstructed hit from the
position the particle actually crossed. Nothing in the phase-2 fits floats
alignment, so that displacement can only be absorbed by field or material.
AN-21-131's nominal MC was ideal geometry. David, 2026-09-17: the MC closure
samples move to the ideal geometry.

## The four productions

| tag | leg | geometry | tasks | events | cluster | dir |
|---|---|---|---:|---:|---|---|
| `dymc_8p5M_260917_ideal` | Z (DY MiniAODv2) | IDEAL | 380 | 8 502 597 | 3804891 | `condor_dymc_ideal/` |
| `dymc_8p5M_260917_alignctl` | Z | aligned | 40 (0–39) | 885 378 | 3804892 | `condor_dymc_alignctl/` |
| `jpsimc_20M_260917_ideal` | J/psi ALCARECO | IDEAL | 600 (0–599) | 7 967 454 | 3804893 | `condor_jpsimc_ideal/` |
| `jpsimc_20M_260917_alignctl` | J/psi | aligned | 60 (0–59) | 790 051 | 3804894 | `condor_jpsimc_alignctl/` |

Output root `/ceph/submit/data/user/d/david_w/ZMass/cvh/<tag>/`.

## Configuration

`condor_{dymc,jpsimc}_v2/` verbatim — same chunk lists (so `task_NNNN` is the
same event range as in `dymc_8p5M_260906_v2` / `jpsimc_20M_260906_v2`), same
4 threads / `request_memory = 5000` MB / `request_disk` 4 000 000 (DY) and
6 000 000 (J/psi) KB, same site list and node fences, same `.complete`
sentinel, same size-verified xrdcp stage-out, same J/psi `submit50–55` input
door rule. Changed:

* `useIdealGeometry=True` (the two `_ideal` tags), `False` (the two `_alignctl`);
* the Z leg adds `bsConstraint=True exportBsResidual=True exportVtxResidual=True`
  (the current DY practice on this tip; the v2 configs had none of the three);
* `doVtxConstraint` is no longer forced to `False` — the v2 scripts passed that
  override explicitly and it is gone, so the maker default (ON) applies;
* one `NTASKS_USED` knob in each config caps `submit`/`status`/`resume` to the
  first N tasks of the shared chunk list, for the two subset productions.

CMSSW: `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2` @ `7b54ce096b27`,
built and clean. **One payload tarball** (md5 `a315135efb36583e1e4537dd651113c5`)
was built once and copied into all four output trees, so every task of all four
runs a bit-identical binary.

## Operating them

```bash
cd calibration_studies/production/condor_dymc_ideal      # or the other three
./status_dymc_v2.sh [--slow]
./resume_dymc_v2.sh --dry-run ; ./resume_dymc_v2.sh
```

The J/psi dirs use `status_jpsimc_v2.sh` / `resume_jpsimc_v2.sh`, and the
subset productions need their `NTASKS_USED` (it is the config default, so no
env var is required).

## The overlay is x86-64-v3; the base release is x86-64-v2

The first five failures of the DY submission were SIGILL (rc 132) inside the
OVERLAY's own `pluginFWCoreServicesPlugins.so`, in
`InitRootHandlers::ThreadTracker::on_scheduler_entry` — i.e. during service
construction, before any event. All five were at **Wisconsin**, on five
different nodes, so it is a site CPU generation and not a black-hole node.

`objdump` on the area's libraries finds `vfmadd*`, `shlx`, `lzcnt`, `andn` —
**x86-64-v3**. The cvmfs base release ships `scram_x86-64-v2` variants with none
of those. Any pre-Haswell execute node therefore kills the job the moment a
payload `.so` is loaded. Two consequences:

* `hep.wisc.edu` is fenced, in all four configs and live via `condor_qedit` on
  the DY clusters. Watch for further sites and fence them the same way;
* the 34 `edm::StreamSchedule::fillWorkers` SIGILLs that
  `PRODUCTIONS.md` §8.3 records as "black-hole nodes" are very probably the same
  thing. The permanent fix is to build the overlay at the release's own target
  rather than the build host's; that is a rebuild and is out of scope here.

## Next commands

Monitor, then integrity, the spot-check, and the four pairs caches.

```bash
for d in condor_dymc_ideal condor_dymc_alignctl condor_jpsimc_ideal condor_jpsimc_alignctl; do
  ( cd calibration_studies/production/$d && ./status_*_v2.sh ); done
```

## Pairs caches to build when the productions finish

| cache | command |
|---|---|
| `fullscale/runs/zpairs_dyideal_full.npz` | `cf_inmaker.py pairs --files <dymc_8p5M_260917_ideal> --ntasks 0 --cache … --jac-parmtypes 14 15 --mass-window 91.1876 60` |
| `fullscale/runs/zpairs_dyalignctl.npz` | same, on `dymc_8p5M_260917_alignctl` |
| `fullscale/runs/jpairs_ideal_n600.npz` | `cf_inmaker.py pairs --files <jpsimc_20M_260917_ideal> --ntasks 600 --cache … --jac-parmtypes 14 15` |
| `fullscale/runs/jpairs_alignctl_n60.npz` | same, `--ntasks 60`, on `jpsimc_20M_260917_alignctl` |

Big caches live on ceph under
`/ceph/submit/data/user/d/david_w/ZMass/cvh/pairs_260917/` with symlinks under
`fullscale/runs/` (the /work quota).

## Status

* 2026-09-17 15:42 — DY ideal (3804891, 380) and DY alignctl (3804892, 40)
  submitted, all idle.
* 2026-09-17 15:52 — J/psi ideal (3804893, 600) and J/psi alignctl
  (3804894, 60) submitted. 1080 tasks in flight across the four.
* 2026-09-17 17:00 — 1019 of the 1080 running, none finished yet. Peak
  `MemoryUsage` over the DY cluster is 2906 MB against `request_memory = 5000`,
  so the v2 sizing stands and must not be raised.

### Node fences added during the run

All applied to the live clusters with `condor_qedit` AND written into the four
configs, so a resume carries them:

| fence | rc | what |
|---|---|---|
| `hep.wisc.edu` | 132 | SIGILL, 5 nodes — site CPU generation (see above) |
| `compute-21-23.ultralight.org`, `compute-12n-5.ultralight.org` | 132 | same, two more Caltech nodes |
| `s1wn17.pi.infn.it` | 132 | same, one Pisa node |
| `node38-4.wn.iihe.ac.be` | 139 | SIGSEGV, 9 of 9 — black-hole node |

`resume_*.sh` re-drives anything left without a sentinel; it refuses to queue an
index that is already in the queue.

## What the first finished task says (J/psi `task_0463`, 959 events)

| | v2 | 260917 ideal |
|---|---|---|
| attempted / succeeded | 955 / 955 | 955 / 955 |
| `Jpsikin_mass` (pre-refit) | — | **bit-identical**, all 955 |
| `Jpsi_mass` (refitted) | — | median −0.031 MeV, rms 9.93 MeV (rel. −1.0e-5 / 3.1e-3) |
| tree payload | 16.12 MB | 16.55 MB (243 branches vs 228) |
| `runtree` on disk | 13.1 MB | 4.55 MB |
| file size | 29.2 MB | 21.1 MB |
| `nParms` median | 246 | 247 |
| `ndof` = `nRank` median | 28 | 30 |

* **the parameter map is bit-identical** — `iidx`, `parmtype`, `rawdetid`,
  `subdet`, `layer`, `stereo`, `glued` and `xi` all agree over the 126 452
  entries, so these files pool with the v2 productions through the same
  `runtree`;
* the runtree's GEOMETRY columns differ, as they must: module centres move by
  up to 1.29 mm with an rms of 16.6 um (`dx`), and `b0`/`bz`/`bradial` move with
  them. The ideal lattice compresses far better, which is the whole file-size
  difference — the tree payload is slightly LARGER;
* the 260917 schema is a SUPERSET of v2's: 15 extra branches
  (`Mu*gen_{pdgId,idx,motherIdx,motherPdgId,isPrompt,fromHardProcess}`,
  `Jpsigen_sameDecay` from the per-leg gen match, and `cfmass_grp_vqms`,
  `cfmass_grp_vqio`). No branch was lost;
* `ndof` rises 28 -> 30 because `doVtxConstraint` is now ON.

## The Z leg costs 2.8x the v2 volume

DY `task_0262` (2349 events, 1031 candidates) against its v2 twin: the tree
payload goes from **68.8 to 190.6 kB/candidate**, and the file gains 110
branches and loses none. Where it goes, per candidate:

| block | kB/cand | switch |
|---|---:|---|
| `cfbs_grp_{ms,ioni_re,ioni_im,rad_re,rad_im}` + `resinfbsv` | **69.4** | `exportBsResidual` |
| `cfvtx_grp_*` + `resinfvtxv` + `resinfv` | **38.3** | `exportVtxResidual` |
| `hessfactorv` | 29.2 (v2: 25.0) | the vertex constraint's extra rows |
| `cfmass_grp_*` | 30.8 | unchanged from v2 |

Two CF residual terms each carry a FULL per-material-group exponent block, and
the beam-line term carries TWO residuals. Projected volume: **~730 GB** for the
380-task Z leg (v2: 267 GB), ~80 GB for the 40-task control, ~515 GB and ~55 GB
for the two J/psi legs — **~1.4 TB in all**, against 32 TB free under the ceph
user quota.

`request_disk` was raised **4 -> 6 GB** on the two Z clusters (`condor_qedit`
and in the configs): the largest chunk (30 929 events) now stages out 2.15 GB
and peaks near 2.5 GB of scratch against the old 3.81 GiB request.

`exportVtxResidual` is the one switch not named in the brief. It is what the two
current DY runs on this tip use (`resolution/vtxres/run_prod_tail.sh`,
`run_prod_beam3.sh`), and it costs 38.3 kB/candidate — ~120 GB over the Z leg.
Drop it from `condor_dymc_ideal/config_dymc_v2.sh` if that is not wanted.

The **pairs caches are not affected**: `cf_inmaker.py pairs` is run without
`--groups`, so none of the per-group blocks is read into them.
