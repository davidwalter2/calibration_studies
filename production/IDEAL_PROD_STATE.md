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

## ...and about 4x the CPU — but NOT for that reason

Same chunk, `task_0262`: the Geant4e propagator is called **84 660** times in
the 260917 run against **22 967** in v2 — 333 calls per candidate against 87,
i.e. **3.7x**. The cause is **not** the extra CF terms. It is the
reference-EDM defect below: every candidate runs to the 10-iteration cap
instead of stopping at ⟨niter⟩ ≈ 2.7-3.2, and 10/2.67 = 3.75 accounts for the
whole factor. Wall clock on that chunk went 504 s (v2, DESY) to 2919 s (IIHE),
the rest being the node.

Realised turnaround: the Z leg's completed tasks ran a median **8.0 h** (p95
12.7 h, max 17.9 h) against v2's ~1-1.6 h; the J/psi leg finished its 600 tasks
in **15.3 h**. The storage factor (190.6 vs 68.8 kB/candidate) IS the CF
residual blocks and is unrelated.

## Spot check, J/psi `task_0000` (12 185 events)

Three pairwise comparisons on the SAME task separate the two changes. Keys are
`(run, lumi, event, Muplustrk_pt, Muminustrk_pt)`; `production/spotcheck_geometry.py`.

| comparison | what changes | common | `Jpsikin_mass` | rel. median `Jpsi_mass` | rel. rms |
|---|---|---:|---|---:|---:|
| v2 -> alignctl | the maker DEFAULTS only | 12 138 | bit-identical | **−1.9e-7** | **4.90e-3** |
| alignctl -> ideal | the GEOMETRY only | 12 138 | bit-identical | **+1.7e-6** | **5.43e-4** |
| v2 -> ideal | both | 12 137 | bit-identical | +9.8e-7 | 4.72e-3 |

* `Jpsikin_mass`, the pre-refit kinematic-fit mass, is **bit-identical in every
  comparison** — same input, same pair finding, so any difference is the refit;
* the geometry moves the refitted mass by a relative median of **1.7e-6** with a
  **5.4e-4** per-candidate rms. The median is small because the misalignment is
  a random per-module displacement that largely averages out over a track;
* the DEFAULT change (`doVtxConstraint` ON) is the LARGER of the two in spread,
  4.9e-3 per candidate, at essentially zero median — it is a different fit, not
  a different scale;
* candidates: v2 12 147, alignctl 12 139, ideal 12 138. The 9 lost to v2 are
  `skipped[leghits<8]` — `minLegHits = 8` is a newer default that the v2
  production did not have (0.074 % of candidates). The one further loss in the
  ideal leg is a single `fail[prop]`.

## These four samples carry two CVH defects fixed one commit later

The area moved to **`3d4c926ff461`** on 2026-09-18, ONE commit after the
`7b54ce096b27` these productions' payload was built from. Two of its three
fixes bear on this sample.

**1. The MS within-step correlation sign** (`Geant4ePropagator::PropagateErrorMSC`).
`res(1,4)` was written `-S3` in the (lambda, yt) projection while the (phi, xt)
one had `+S3`; both curvilinear levers are positive, so the negative sign was
wrong. Consequence in these files: the **offset variance of a leg in the
non-bending projection is low by a median 6.0 %** (p05 26.5 %, worst 52 %). The
process-noise matrix is therefore wrong in exactly the projection a resolution
study cares about, and everything derived from `resinfvarv` inherits it.

**2. The reference EDM under the vertex constraint.** The reference-block EDM
was taken on the full 10x10 vertex-PCA block of `covstate`, which is singular
once index 6 is frozen — and `doVtxConstraint` is ON by default. Measured in
these productions, `task_0000` / `task_0262`, against their v2 twins:

| | `<niter>` | at the 10-iteration cap | `edmvalref` NaN | `edmval` NaN |
|---|---:|---:|---:|---:|
| `jpsimc_20M_260906_v2` | 3.16 | 1.3 % | 0 % | 0 % |
| `jpsimc_20M_260917_alignctl` | **10.00** | **100 %** | **100 %** | 0 % |
| `jpsimc_20M_260917_ideal` | **10.00** | **100 %** | **100 %** | 0 % |
| `dymc_8p5M_260906_v2` | 2.67 | 0.8 % | 0 % | 0 % |
| `dymc_8p5M_260917_ideal` | **10.00** | **100 %** | **100 %** | 0 % |

`edmval`, the main fit's EDM, is finite everywhere — the fit itself converges
and then keeps taking Gauss-Newton steps to the cap. That is what the 3.7x
propagator count and the 8 h median task are. The spot check bounds what it
does to the answer: v2 -> aligned control, where the geometry is identical and
only the defaults change, moves `Jpsi_mass` by a relative median of
**−1.9e-7** with a 4.9e-3 per-candidate rms — no scale shift, a different fit
per candidate.

**3.** The third fix (step records emptied under `doMassConstraint`) does not
apply: these productions run `doMassConstraint=False` and
`exportStepRecords=False`.

The payload was deliberately pinned before the fix, so the four samples are
internally consistent and comparable with each other. They are NOT comparable
with anything built at `3d4c926ff461` or later, and defect 1 is a physics
error in the process noise. **David decides whether to repeat.**

## Spot check, Z `task_0001` (22 120 events)

`task_0000` of the Z ideal leg was still running, so the Z spot check is on
`task_0001` — the same chunk in all three productions.

| comparison | what changes | common | `Jpsikin_mass` | rel. median | sigma68 | \|r\|>1 % | max \|r\| |
|---|---|---:|---|---:|---:|---:|---:|
| alignctl -> ideal | the GEOMETRY only | 9 642 | bit-identical | **+1.20e-5** | **5.92e-4** | 6 (0.06 %) | 0.063 |
| v2 -> alignctl | the maker DEFAULTS only | 9 642 | — | +4.88e-5 | 4.13e-3 | **828 (8.59 %)** | **83** |
| v2 -> ideal | both | 9 641 | — | −1.51e-5 | 4.18e-3 | 834 (8.65 %) | 83 |

**The geometry is the clean, small effect**: a relative median of `+1.20e-5`,
i.e. **+1.09 MeV on a 91.1876 GeV Z**, with a 5.9e-4 per-candidate width and
0.06 % beyond 1 %. That is the number this production was made to measure.

**The maker-default change is the disruptive one**: 8.6 % of Z candidates move
by more than 1 % and 61 by more than 10 %, up to a factor 83. That is the
10-iteration runaway of the reference-EDM defect, not the vertex constraint as
such — a candidate that keeps taking Gauss-Newton steps after convergence can
walk a long way. It is the strongest argument for repeating these productions
against `3d4c926ff461`.

Candidates: v2 9 846, alignctl 9 643, ideal 9 644. The 204 lost to v2 are
`skipped[leghits<8] = 211` (the `minLegHits = 8` default v2 did not have);
failures fall 11 -> 3/2 and `clamped[step]` 152 -> 32.

## Final state

All four productions complete (380/380, 600/600, 40/40, 60/60), integrity clean;
the four pairs caches are built — see `PRODUCTIONS.md` §9 for the numbers.
Nothing is running. Next step is David's: whether to repeat against
`3d4c926ff461`, then the cards and fits.
