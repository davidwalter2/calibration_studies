# BEAM3 — the luminous region as a floated 3x3 covariance

Working checkpoint for the task "promote the beam-line constraint's fixed
pieces to floated first-principles parameters".  Final results go to
`STATE.md` section 14; this file is the resumable state.

## What the change is

The beam-line constraint is a Gaussian noise block whose covariance is the CMS
beam-spot-fitter form.  Before this change only its two transverse WIDTHS
floated, as two linear variance classes; the x-y correlation was fixed at zero
(the record does not store it), the two tilts and the centre were taken from
the record.  `--beam3` floats the covariance itself plus the centre:

| parameter | role | card unit | prior |
|---|---|---|---|
| `beamwidth_x` / `beamwidth_y` | `k = 1 + eps` on `sigma_x^2` / `sigma_y^2` | eps | record's `BeamWidthError` (`--beamwidth-prior 0` = free) |
| `beamcorr_xy` | `rho = tanh(eta)` | eta | FREE (the record has no rho) |
| `beamtilt_x` / `beamtilt_y` | offset of `dxdz` / `dydz` | 1e-5 | FREE |
| `beamcentre_x` / `beamcentre_y` | offset of `x0` / `y0` | 1e-4 cm = 1 um | FREE |

`sigma_z` is fixed (the z beam row is weightless against a ~100 um vertex
error) and `z0` is inert.

The tilts enter BOTH the covariance and the mean; `MassCFTerm` de-duplicates
the parameter name, so one parameter drives both.

## Where the code is

| what | where |
|---|---|
| the block | `rabbit-vmass/rabbit/unbinned.py`, `MaterialCFTerm` (`beam3_params` / `beam3_units` / `beam3`, `_beam3_share`, `BEAM3_ROLES`) |
| its tests | `rabbit-vmass/tests/test_beam3.py` (9 tests) |
| the extraction | `resolution/vtxres/extract_vtx.py` — now writes `bsmean`, `vbs`, `bsvtx`, `bsspot`, `bswidth`, `bsslope` for EVERY functional |
| the card | `resolution/vtxres/make_vtx_card.py --beam3` (`beam3_block`, `beam3_cov`, `BEAM3_*`) |
| the assembly gate | `resolution/vtxres/gate_beam3.py` |
| the nominal gate | `resolution/vtxres/gate_beam3_card.py` |
| the closure reference | `resolution/vtxres/beam3_gen.py` |
| the report | `resolution/vtxres/beam3_report.py`, `beam3_pulls.py` |
| the production | `resolution/vtxres/run_prod_beam3.sh`, `mark_complete.sh` |

## The productions

| tag | job | geometry | files x events | output |
|---|---|---|---|---|
| `dy_beam3_7b54ce096b27` | 6447928 | aligned (`useIdealGeometry=False`) | 80 x 3500 | `/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/` |
| `dy_beam3_ideal_7b54ce096b27` | 6450187 | IDEAL (`useIdealGeometry=True`) | 80 x 3500 | same root |

Same 80 input files (`production/filelist_dymc_beam3_260917.txt`, lines 7-86 of
the 8.5 M DY list), same build (dev2 @ `7b54ce096b27`), everything else the
`condor_dymc_v2` configuration plus `exportVtxResidual` / `bsConstraint` /
`exportBsResidual`.  So the two legs are the SAME EVENTS and differ only in the
tracker geometry.

The aligned leg was submitted before the instruction to move the MC closure
samples to the ideal geometry arrived and was left to finish (it is the
same-candidate cross-check of what the geometry does to the beam parameters);
the ideal leg is the one to quote.

Resume:
```bash
ssh submit50 'squeue -u david_w'
cd /work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
./mark_complete.sh 6450187 /ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/dy_beam3_ideal_7b54ce096b27
```

## The chain

```bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
BL=/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt

# 1. extract (four functionals, same candidates, same order)
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
for F in bsx bsy vtx mass; do
  python3 -u extract_vtx.py --files "$BL/<prod>/task_*/globalcor_*.root" \
    --functional $F --groups $GRP -j 24 --require-complete --max-chi2-ndof 3 \
    -o $BL/<runs>/dy_$F.npz
done

# 2. the assembly gate (needs nothing but the npz)
python3 gate_beam3.py --dir $BL/<runs>

# 3. the closure reference
python3 beam3_gen.py --npz $BL/<runs>/dy_bsx.npz --nboot 200

# 4. cards: the baseline (widths only) and --beam3, SAME candidates
COMMON="--groups $GRP --maxn 8000 --whiten --prune-frac 0.001 --poi all \
  --hit-prior 1.0 --max-chi2-ndof 3.0 --m-ref 91.1876 --m-window 30 \
  --vtx-npz $BL/<runs>/dy_vtx.npz --bsx-npz $BL/<runs>/dy_bsx.npz \
  --bsy-npz $BL/<runs>/dy_bsy.npz --arm cf --same-candidates --beamwidth-prior 0"
./run_tf.sh python3 -u make_vtx_card.py $COMMON        -o $BL/<runs>/cards/base_vtxbs.hdf5
./run_tf.sh python3 -u make_vtx_card.py $COMMON --beam3 -o $BL/<runs>/cards/b3_vtxbs.hdf5

# 5. the nominal gate
./run_tf.sh python3 gate_beam3_card.py --base .../base_vtxbs.hdf5 --beam3 .../b3_vtxbs.hdf5

# 6. fits, EDM-certified
export R=$BL/<runs>
./run_fit.sh base_vtxbs ; ./run_fit.sh b3_vtxbs

# 7. the report
./run_tf.sh python3 beam3_report.py --fits base=$R/fits/base_vtxbs b3=$R/fits/b3_vtxbs \
   --genref $R/dy_bsx_genref.npz --delta base,b3
./run_tf.sh python3 beam3_pulls.py --npz-dir $R --fit $R/fits/b3_vtxbs --groups $GRP
```

## Gates, measured

| gate | result |
|---|---|
| ASSEMBLY `sum_ab covBS_ab Q_ab / Cov_ii == the exported share` | median **2.2e-8**, max 1.1e-7 on all four functionals (`dy_bs_final`, 10 254 candidates) -- the float32 export precision |
| the same for `d/dk_x`, `d/dk_y` vs `*vbsx` / `*vbsy` | median **2.2e-8** |
| the FORMULA-rebuild derivative vs the maker's `D covBS D` convention | median **2.5e-4** of the derivative, i.e. ~5e-5 of the share; the two differ only in `dC_xz/dk_x` |
| NOMINAL, card level: NLL(0) beam3 vs the two linear width classes | **BIT-IDENTICAL** on all three channels |
| NOMINAL, gradient on the 60 shared parameters | max rel **2e-16** |
| rabbit unit tests (`tests/test_beam3.py`) | 9/9 pass; share(0) - v0 exactly 0; gradients 1e-9..4e-6 vs FD; Hessian 1e-8 vs FD |

## Defects found and fixed

1. **`rho sqrt(C_xx C_yy)` has a NaN SECOND derivative at a zero record row.**
   The correlation term was first written `rho * sqrt(vx) * sqrt(vy)` with
   `vx = k_x sigma_x^2`.  Where the record's width is zero that is
   `0 * inf = NaN` in the Hessian while the VALUE and the GRADIENT stay
   finite -- so the minimiser converges normally and `edmval_cov` dies on
   `array must not contain infs or NaNs`, which is exactly how it presented.
   Measured, on the bare expression at `sigma = 0`, `rho = 0`:

   | form | value | grad | d2/deps2 |
   |---|---|---|---|
   | `rho sqrt(vx) sqrt(vy)` | 0 | 0 | **NaN** |
   | `rho sqrt(k_x) sqrt(k_y) sigma_x sigma_y` | 0 | 0 | 0 |

   The two are identical for any positive record; the second takes the square
   root of the PARAMETER, which is 1 at the nominal point, instead of of the
   variance, which the record can make zero.  Fixed, with
   `tests/test_beam3.py::test_zero_record_row` as the regression.

2. **A zero record row is not hypothetical: the truncation normalisation
   builds them.**  `_vtx_norm_block` gives an EMPTY resolution class the whole
   sample as its members (so its `vg_other` and hit shares are the sample
   means); the first version of the beam3 class-level block used a plain
   `bincount` instead and gave those classes ZERO -- zero `Q`, zero record,
   zero nominal share.  The beam channels hit this every time, because their
   `sigma` is identically 1, the quantile edges collapse and seven of the
   eight classes come out empty.  Fixed to use the same members list.

Either fix alone removes the failure; both are kept, because the first is
about the expression being differentiable and the second about the class rows
being the model.

## The closure reference (`dy_bs_final`, 10 223 gen-matched)

The MC's luminous region is known in CLOSED FORM:
`BetafuncEvtVtxGenerator` + `Realistic25ns13TeV2016CollisionVtxSmearingParameters`
has `Phi = Alpha = 0` and the `+ Z*fdxdz` term commented out, so
**dxdz = dydz = 0 and rho = 0 EXACTLY**, `sigma_x = sigma_y` with marginal rms
`sqrt(emittance (betastar + SigmaZ^2/betastar)/2) = 9.9467 um`, and
`X0, Y0, Z0 = 0.09163, 0.16955, 0.9315 cm`.

| parameter | target (generator) | measured on the gen vertices |
|---|---|---|
| `beamwidth_x` | -0.1570 | -0.1401 +- 0.0123 |
| `beamwidth_y` | -0.0831 | -0.0777 +- 0.0130 |
| `beamcorr_xy` | 0 | -0.0061 +- 0.0098 |
| `beamtilt_x` | +0.5970 | +0.4304 +- 0.2771 |
| `beamtilt_y` | -0.4718 | -0.4037 +- 0.2671 |
| `beamcentre_x` | -0.1590 | -0.1210 +- 0.1029 |
| `beamcentre_y` | +0.1301 | -0.0458 +- 0.1008 |

## NEXT

1. wait for the two productions, `mark_complete.sh`, extract;
2. `beam3_gen.py` on the big sample, then cards + fits at the largest `--maxn`
   the card size allows (744 MB at 8 000 candidates for vtx+bs+mass, so a
   1e5-candidate card needs a harder `--prune-frac`; check first that raising
   it does not move the beam parameters);
3. figures into `~/public_html/ZMass/cvh/260917_beam3/`;
4. STATE.md section 14 to the final state; commit.
