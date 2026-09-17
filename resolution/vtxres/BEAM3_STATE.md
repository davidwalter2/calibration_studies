# BEAM3 — the luminous region as a floated 3x3 covariance

Resumable state for the `--beam3` study.  The physics results are in
`STATE.md` section 14.20; this file is what is needed to pick the work up:
what runs where, what is still in flight, and the exact next command.

## Status

| piece | state |
|---|---|
| the term (`rabbit`, `MaterialCFTerm` beam3 block) | DONE, committed, pushed to the PR branch |
| `tests/test_beam3.py` | 11 tests, all pass |
| the extraction, the card builder, the gates, the reference, the reports, the plots | DONE, committed |
| the closure at 8 000 candidates (`dy_bs_final`) | DONE — STATE.md 14.20 |
| the injections, the frozen-`rho` control, the sandwich | DONE — STATE.md 14.20 |
| figures | `~/public_html/ZMass/cvh/260917_beam3/` |
| the two productions for the high-statistics closure | **IN FLIGHT** (below) |
| the closure at ~1.2e5 candidates | **NOT DONE** — waiting on the productions |

## The productions in flight

| tag | slurm job | geometry | files x events | expected |
|---|---|---|---|---|
| `dy_beam3_7b54ce096b27` | 6447928 | aligned (`useIdealGeometry=False`) | 80 x 3500 | ~4.3 h/task, 40 at a time |
| `dy_beam3_ideal_7b54ce096b27` | 6450187 | **IDEAL** (`useIdealGeometry=True`) | 80 x 3500 | starts as the first array drains |

Both under
`/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/`,
both from `production/filelist_dymc_beam3_260917.txt` (lines 7-86 of the 8.5 M
DY list), both built from dev2 @ `7b54ce096b27`, everything else the
`condor_dymc_v2` configuration plus `exportVtxResidual=True`,
`bsConstraint=True`, `exportBsResidual=True`.  The two legs are therefore the
SAME EVENTS and differ only in the tracker geometry, which makes the
comparison same-candidate.

The aligned leg was submitted before the instruction to move the MC closure
samples to the ideal geometry and was left to finish; `run_prod_beam3.sh` now
defaults to `GEOM=True`, so the ideal leg is the one to quote and the aligned
leg is the cross-check of what the geometry does to the beam parameters.
Neither affects the gen vertices, which is where the closure reference comes
from.

**Expected size**: 80 x 3500 events x 0.434 candidates/event ~ **1.2e5
candidates** per leg, ~22 GB each.  That is what the tilt closure needs:
per candidate the beam pull measures `dxdz` to ~6.4e-4, so
`sigma(dxdz) = 6.4e-4/sqrt(N)` is 2e-6 at 1e5 against a record-vs-simulation
offset of 6e-6 -- a 3 sigma test, where the 10 254-candidate sample gives
6.3e-6 and cannot separate them.

## Resume, in order

```bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
BL=/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline
PROD=dy_beam3_ideal_7b54ce096b27        # or dy_beam3_7b54ce096b27
RUNS=$BL/runs_beam3_ideal               # or runs_beam3_aligned

ssh submit50 'squeue -u david_w'
./mark_complete.sh 6450187 $BL/$PROD      # 6447928 for the aligned leg

PROD=$PROD R=$RUNS ./run_all_beam3.sh extract     # ~20 min x 4 functionals
PROD=$PROD R=$RUNS ./run_all_beam3.sh gates
PROD=$PROD R=$RUNS ./run_all_beam3.sh genref
PROD=$PROD R=$RUNS MAXN=40000 ./run_all_beam3.sh cards
PROD=$PROD R=$RUNS ./run_all_beam3.sh fits
PROD=$PROD R=$RUNS ./run_all_beam3.sh report
PROD=$PROD R=$RUNS ./run_all_beam3.sh pulls
PROD=$PROD R=$RUNS TAG=ideal ./run_all_beam3.sh plots
```

**RUN THE FITS ON submit50 / 51 / 52.**  The submit8x machines are shared and
sit at load ~900 on 768 cores; `run_tf.sh` asks for `OMP_NUM_THREADS` and
nothing enforces it, so a 20-minute fit becomes an hour.  Check
`/proc/loadavg` first.

**`--maxn` and the card size.**  A `vtx + bsx + bsy` card is 554 MB at 8 000
candidates with `--prune-frac 0.001`, i.e. ~6.9 GB at 1e5; the fit's RSS was
10.7 GB at 8 000 candidates already.  Raising `--prune-frac` folds more
material groups into the fixed baseline: 405 MB at 0.01 and 255 MB at 0.05 on
the same candidates.  Section 14.14 has already established that NO material
group is constrained by any of these channels on a DY sample, so pruning is
close to inert -- but if it is used, build the SAME card at 0.001 and 0.05 on
a common subset first and check that the beam parameters do not move.

## The next questions, in the order they should be answered

1. **The tilt closure at 1e5 candidates.**  The only number the 8 000-candidate
   sample cannot deliver.
2. **`beamcorr_xy`.**  It is 3.9 sigma away from a value the simulation fixes
   exactly at zero, freezing it moves nothing else, and the gen vertices say
   the luminous region has no correlation -- so it is measuring the transverse
   anisotropy of the fit's own vertex-covariance deficit (STATE 14.20).  The
   way to settle it is the joint treatment of the two beam terms (STATE 14.19
   item 2): they are multiplied as if independent, and a real `rho` would
   correlate them.
3. **`beamcentre_x` at +2.6 sigma.**  Check whether it survives the larger
   sample; if it does, it is either a real sub-micron centroid offset the gen
   vertices do not see or a leak from the same anisotropy.
4. **Data.**  The record's per-IOV values are already exported per candidate,
   so the only new thing a data fit needs is one parameter set per IOV rather
   than one for the sample.
