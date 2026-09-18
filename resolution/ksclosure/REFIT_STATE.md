# K_S closure on the FIXED CVH code -- running state

Task: repeat the K_S -> pi pi displaced closure on commit `3d4c926ff461`
(the MS within-step correlation-sign fix (d), the reference EDM fix (a), the
step-record fix (b)).  The 260917 production is the before-fix comparison and
must be kept.

## Where things are

| | |
|---|---|
| fixed production | `/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed/` |
| old production (keep) | `/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal/` |
| CMSSW | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2` @ `3d4c926ff461`, built 11:35 (newer than the sources at 11:32) |
| submit | `prod/submit_ks_half.sh [first] [last] [maxrun]` (1000 half-chunk tasks) |
| resubmit | `prod/resubmit_half.sh [maxrun]` (seals, then resubmits the gaps) |
| seal | `prod/seal_halves.sh` -- `.complete` for chunks with both halves |
| smoke compare | `smoke_compare.py <old task dir> <new task dir>` |

`chunks/` and `allfiles.txt` are copies of the 260917 lists, so the two
productions are candidate-by-candidate comparable.  `truth/` is a SYMLINK to
the 260917 truth dump: the truth is FWLite over the same inputs and does not
touch the CVH code.

## Progress

- [x] new production dir + PROVENANCE + chunk lists + truth symlink
- [x] smoke, 40 files of chunk_0000 run interactively on the fixed build,
      compared candidate-by-candidate with the 260917 `task_0000`
      (`smoke_compare.py`): 108/108 matched, `med|dm/m|` 4.9e-5, p95 6.9e-4,
      38.9 % above 1e-4, `d sigma_m/sigma_m` mean +3.75e-3 -- the
      DEFECTS_STATE (d) table, reproduced.  `niter` mean 10.000 -> 5.528
      (cap no longer always hit).  `ndof` identical.
- [x] CONTROL: the OLD pairs cache refitted with nothing changed reproduces
      `alpha = +0.260199 +- 0.042427` to every printed digit (`ctl_all`),
      so the analysis chain is unchanged.
- [ ] full array, slurm **6474164**, 1000 HALF-chunk tasks

  The `submit` partition was saturated: a 5 min job backfilled at once, a 15 min
  one waited behind the reservations, and the 8 h request of the 260917 array
  never started at all.  Measured window on 2026-09-18 13:04: 12 min backfills,
  15 min does not.  So each 252-file chunk is split into two 126-file halves
  (`chunks_half/half_NNNN_{a,b}.txt`, verified to reproduce the 125 999-file
  list exactly), each task asks for 12 min, and the two halves write into the
  SAME `task_NNNN/` under `outprefix=globalcor_ks_{a,b}`.  `ks_pairs.py` globs
  `globalcor_ks_*.root` in the task directory, so the chunking the truth join
  sees is unchanged and `truth_NNNN.npz` still pairs with `task_NNNN`.
- [ ] pairs cache -> cards -> fits
- [ ] figures + STATE.md

## Next command

```bash
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
PROD=/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed
$KS/prod/resubmit_half.sh 250           # seal, then anything that did not finish
RUNS=$KS/runs/fixed $KS/run_ks_closure.sh $PROD
RUNS=$KS/runs/fixed $KS/run_ks_syst.sh  $PROD
# the a_res forms
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
python3 $KS/ks_cov_extract.py --prod $PROD --out $KS/runs/fixed/kscov_all.npz
python3 $KS/ares_angles.py --cov $KS/runs/fixed/kscov_all.npz \
    --pairs $KS/runs/fixed/kspairs_all.npz --masses pion \
    --write $KS/runs/fixed/kspairs_ares.npz --tag ares_angles_fixed
RUNS=$KS/runs/fixed CACHE=$KS/runs/fixed/kspairs_ares.npz \
    FORMS="mom truth" $KS/run_ares_forms.sh
python3 $KS/ks_compare_closure.py --old $KS/runs --new $KS/runs/fixed
```
