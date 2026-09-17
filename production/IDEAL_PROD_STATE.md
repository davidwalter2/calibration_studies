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
