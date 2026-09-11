# vtxres — the VERTEX-CONSTRAINT RESIDUAL of the two-track CVH fit, as a CF term

## RESUME HERE (2026-09-11 10:45) — THE STUDY IS COMPLETE

Everything the coordinator asked for is measured and banked.  NOTES.md entry
appended (`/work/submit/david_w/Documents/Resolution/NOTES.md`, section
"2026-09-11 — THE VERTEX-CONSTRAINT RESIDUAL OF THE TWO-TRACK FIT").
Scripts committed on `resolution-energy-loss-corrections` (c931a47); the maker
on `vtxres-cf-260911` of `CMSSW_15_0_19_patch2_dev2` (4b3984312f6).

**DONE**
| item | where |
|---|---|
| maker export `exportVtxResidual` + `vtxConstraintZeroSeed` | "STEP 1" |
| gates (a)-(e) on a 400-event FREE/FROZEN pair and 20 000 production candidates | "THE GATES", `logs/gates_smoke*.log`, `logs/gates_prod.log` |
| production 160/160, 2000 events each, 328 kB/candidate | ceph `runs_vtxres_260911/prod` |
| distribution + tails + composition, **96 160 candidates** | "STEP 4 RESULT", `logs/plots.log` |
| 13/13 cards, 8/11 fits EDM-certified (the 3 failures are all Gaussian arms) | "STEP 5" |
| the sandwich, 3 arms x 3 channels | `logs/eff_*_p2.log` |
| injections: a material group and an innermost pixel hit class | "THE INJECTIONS" |
| the vertex-mass correlation at three levels | "STEP 6 RESULT", `logs/xcum.log` |
| cost A/B on a pinned CPU + the export bill | "STEP 7", `logs/bill.log` |
| the Z-like check on DY MC | "STEP 8 RESULT" |
| 20 figures + `index.php` | `~/public_html/cvh/260911_vtxres/` |

**OPEN ITEMS (none blocking; each is a new study)**
1. **The mass term's Gaussian arm returns a NaN Hessian.**  `mass_gaussq` and
   `joint_gaussq` have a finite NLL and gradient at theta = 0 but
   `H` is NaN; with `floor="clip"` instead of `"softplus"` the Hessian is
   finite and the NLL is `inf`, so the density is going to <= 0 (or
   underflowing) for some candidates and the softplus floor's SECOND
   derivative is what NaNs there.  Localised to the MASS term with a Gaussian
   arm (the vertex term's three arms and the mass term's CF arm are all fine)
   and time-boxed: no headline needs it.
2. **A non-Gaussian HIT model.**  Both arms treat the hit noise as exactly
   Gaussian.  On the gun the CF is within 1.4-1.7 of the data out to 5 sigma;
   on DY it is a factor 85 short (section "STEP 8"), and a data fit will need
   an outlier component whose size is now measured (~2 %).
3. **The concatenated-tau trick is not ported.**  Every exponent primitive
   depends on `weight * tau` alone, so the mass and the vertex functional
   could share ONE `cvhcf` pass; as it stands the second call costs the same
   as the first (+0.54 s/candidate, +48 %).
4. The `--prior-power` unit trap (section "A THIRD ITEM"): a Fisher matrix
   built through `hitlik_term.build` is in PHYSICAL k units (prior `gprior`,
   `--prior-power 1`), one built through `make_*_card.build_term` is in CARD
   units (prior `gprior**2`, `--prior-power 2`).  Making `build` and the card
   share one convention would remove the trap.
5. `hitlik/recovery.py` prints `nan` in its `/truth` columns on these cards
   (it reads the injected truth under a key `make_vtx_card.py` does not
   write); the recovery numbers in this file were obtained from its own
   `shift` and `f_pri` columns by hand.

**INFRASTRUCTURE NOTES FOR A FRESH AGENT**
* `/work/submit` quota is 500 G and nearly full.  `runs/vtxres/{vtx,mass,
  smoke_*}.npz` and `runs/vtxres/cards` are SYMLINKS into
  `/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/runs/`.
  Anything > 0.5 G goes there.
* ceph is NOT readable from submit82; use submit50/51/52 (helpers
  `/home/submit/david_w/.claude-work/jobs/28e0dfa8/s5{0,1,2}.sh '<cmd>'`).
* **Do not `scram b` in `..._dev2` while a production runs from it**, and do
  not edit a bash script that is executing (both traps were respected here;
  the `.py` modules were edited freely).
* `run_tf.sh` binds ceph when the host can read it.

## THE OBJECT (derivation; a fresh agent does not have to redo it)

Two-track CVH state: 10-dim vertex PCA + 5 per hit.
`ResidualGlobalCorrectionMakerBase::twoTrackCart2pca` (Base.cc:2649) defines

    n_hat = (p_a x p_b).normalized()        a = track 0, b = track 1
    theta_6 = d0 = n_hat . (x_b - x_a)      the SIGNED track-track PCA distance

and `twoTrackPca2cart` places the two reference points symmetrically about the
common vertex `x_v = statepca.tail<3>()`:  `x_a = x_v - d/2 n_hat`,
`x_b = x_v + d/2 n_hat`.  The tree already carries
`Jpsi_d = firstplus ? statepcaupd[6] : -statepcaupd[6]` (maker line 4390),
i.e. d re-signed so that leg "a" is the POSITIVE muon.

`doVtxConstraint_` (maker 2092) removes index 6 from `freestateidxs`.

### THE TWO REGIMES, and the one the production is in
* **index 6 FREE** (`doVtxConstraint=False`, **which is what every production
  uses** -- `production/condor_jpsimc_v2/config_jpsimc_v2.sh:107`,
  `config_jpsimc20M.sh:32`, and the J/psi-gun mass-card production
  `resolution_trackres_jpsigun_ul16_260905d_m0`): the fit REPORTS the DCA.
  Then
      sigma_v^2 = C_66 = covstate(6,6),     C = Cinvd^-1
      w_v       = Vinv F_f C e_6            (= the mass functional's `wmass`
                                             with `afull = e_6`)
      r_v       = statepcaupd[6],  z_v = r_v/sigma_v
* **index 6 FROZEN** (`doVtxConstraint=True`): b is zero on every free index,
  and with `F_6 = Ffull.col(6)`, `h_f6 = F_f^T Vinv F_6`, `Cs = C h_f6`,
      sigma_v^2 = 1/(h_66 - h_f6^T Cs)
      b_6       = -(Vinv F_6) . r                 (minus the half-gradient)
      r_v       = sigma_v^2 b_6                    the unconstrained DCA,
                                                   relative to the FROZEN value
      w_v       = sigma_v^2 (Vinv F_6 - VinvF Cs)
  Both give `sum_b |dV_b^{1/2} w_v,b|^2 = sigma_v^2` EXACTLY (one line:
  w_v^T V w_v).  That is the gate.

### SIGN
`dxfree = -C F^T Vinv r`, so a noise perturbation `n` of the residual rows
moves the free state by `-C F^T Vinv n`: the TRUE influence on theta_6 is
`-w_v`, and the same minus sits in the mass functional's `wmass`.  The mass
code hard-sets `ioniSign = -1` on top of it.  The vertex functional has no
single global ionization sign (an energy loss moves the DCA either way
depending on geometry and charge), so it needs the PER-BLOCK `ressgn`
mechanism `cvhcf::TrackInput::ressgn` that the perhit branch added.
Rule (perhit `ResidualGlobalCorrectionMakerG4e.cc` ~5290): for fam == 11,
`u` = the leading eigenvector of `dV_b` oriented by its qop component, and
the signed weight carries `-sign(w_b . u)`; the leg CHARGE multiplies it.
**Convention is not asserted -- it is GATED**: the same rule applied to
`wmass` must return -1 on every ionization block (the validated mass
convention).  Diagnostic branch `Jpsi_vtxsgnchk` reports the fraction.

## A DEFECT FOUND ON THE WAY (to verify, then fix)
`doVtxConstraint_` freezes index 6 at its SEED value.  Nothing zeroes
`statepca[6]` before the first iteration (refftsarr comes from
`midPropagated`/perigee seeds whose PCA distance is NOT zero, maker
2285-2335), so "the common-vertex constraint" as coded constrains
`d = d_seed`, not `d = 0`.  The path is dormant (every production runs
`doVtxConstraint=False`), which is presumably why it was never caught.

## KEY FACTS
| item | value |
|---|---|
| maker | `.../CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/plugins/ResidualGlobalCorrectionMakerTwoTrackG4e.cc` |
| area HEAD | `perhit-residual-cf-260910` @ `a0f12c7f66d` -- branch `vtxres-cf-260911` from HERE |
| mass-card gun production | `resolution_trackres_jpsigun_ul16_260905d_m0`, 160 tasks, 1875 cand/task = 300k, 121 GiB, **no `cf*` branches** (offline exponents from step records) |
| its input | `/ceph/.../resolution_simprod_jpsigun_ul16/task_*/step2.root` |
| its config | `runCvhJpsiGenMC.py`, `doRes=True fillGrads=True fitFromGenParms=False useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 armijoSlack=1.0 trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False applyHltFilter=False scalarPot3DInitFile=.../polyfit3d_full_coeffs_lmax18_custom50.txt` |
| running jobs at start | NONE (squeue + condor_q + ps on submit50/51/52/82 all empty) -> dev2 is safe to build |
| ceph out dir | `/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/` |
| figures | `~/public_html/cvh/260911_vtxres/` |

## STEP 1 DONE -- THE MAKER EXPORT (branch `vtxres-cf-260911`, area `..._dev2`)

Builds clean (`scram b`, exit 0).  Config switches on `runCvhJpsiGenMC.py`:
`exportVtxResidual` (False), `vtxConstraintZeroSeed` (True).

| branch | what |
|---|---|
| `Jpsi_vtxres` | r_v, cm, the RAW theta_6 (see the sign note) |
| `Jpsi_vtxsig` | sigma_v, cm |
| `Jpsi_vtxz` | r_v/sigma_v |
| `Jpsi_vtxb6`, `Jpsi_vtxbfree` | the half-gradient at index 6, and max_i |g_i| sqrt(C_ii) over the FREE indices (dimensionless, comparable with |z_v|) |
| `Jpsi_vtxdchi2` | z_v^2 |
| `Jpsi_vtxvchk` | \|sum_b v_b/sigma_v^2 - 1\| over fam != 15 |
| `Jpsi_vtxvgf` / `vtxvhit` / `vtxvms` / `vtxvioni` | the variance shares by family |
| `Jpsi_massvms` / `Jpsi_massvioni` | the same split for the MASS functional |
| `Jpsi_vtxsgnchk` | the SIGN GATE (below) |
| `Jpsi_vtxfree`, `Jpsi_vtxfirstplus`, `Jpsi_vtxok` | flags |
| `vtxvarv`, `vtxsgnv` | per-block v_b/sigma_v^2 and ionization sign |
| `resinfvtxv`, `resinfv` | a_b = dV_b^{1/2} w_b for the VERTEX and the MASS functional (5/block, padded) -- the fourth cross cumulant needs these |
| `cfvtx_{ms,del,ioni_re,ioni_im,rad_re,rad_im}` | the exponents at the vertex weights |
| `cfvtx_hitcls`, `cfvtx_hitv` | per-hit-class shares of sigma_v^2 |
| `cfvtx_grp*` (+ `cfmass_grp_vqms/vqio`, new) | the per-material-group split and the FIT'S OWN Q variance per group (the `gaussq` arm) |
| `Jpsi_jacVtx` | d theta_6^unc / d(global params), aligned with `globalidxv` |

### THE SIGN CONVENTION, and a DEFECT IN `Jpsi_d`
`theta_6` is INVARIANT under swapping the two legs (`twoTrackCart2pca` flips
both `n_hat` and `x_b - x_a`), so the raw `theta_6` is already a well-defined
signed DCA.  `Jpsi_d = firstplus ? theta_6 : -theta_6` multiplies an invariant
by the charge of leg 0 -- which does NOT "define the sign wrt charge" (the
code comment says it does) but RANDOMIZES it whenever the leg ordering is not
charge-ordered.  `Jpsi_vtxres` therefore carries the RAW theta_6, and
`Jpsi_vtxfirstplus` is exported so the `Jpsi_d` convention can still be
formed.  This matters: the CF exponents are built from `w_v` in the raw
convention, so re-signing the residual and not the weights would put the
Landau skew on the wrong side.

### THE PER-BLOCK IONIZATION SIGN, GATED not asserted
Rule: for `fam == 11`, `u` = the leading eigenvector of `dV_b` oriented by its
qop component; the block's signed weight carries `q_leg * (-sign(w_b . u))`,
with `ioniSign = 1` and `cvhcf::TrackInput::ressgn` carrying everything.
`Jpsi_vtxsgnchk` applies the IDENTICAL rule to the MASS influence `wmass`,
variance-weighted, where the validated answer is -1 on every block.

## STEP 2 -- THE GATES (in progress)

### A STRUCTURAL FACT that cost an hour, and that a fresh agent must know
In a GBL-type fit the reference trajectory is built BY PROPAGATING, so every
PROCESS-NOISE row of `rfull` is IDENTICALLY ZERO at the linearisation point.
The vertex state enters the residual ONLY through the `ihit == 0` propagation
rows.  Therefore

    F_6^T Vinv rfull == 0   bit for bit, on every candidate,

with `|F_6| = 0.7071` (two entries of +-1/2, one per leg: `x_a = x_v - d/2
n_hat` and `n_hat` is perpendicular to both momenta, so d moves each leg
purely in its own curvilinear transverse plane) and `h_66 = 1e5 - 4e6`.  The
gradient that means anything is the one at the CONSTRAINED OPTIMUM,

    rho = rfull + F_f dxfree,      b_6 = -(Vinv F_6) . rho

-- the same vector the exported `Rr` is built from.  Measured with `rho`:
`max_i |g_i| sqrt(C_ii)` over the FREE indices is **3e-9 .. 7e-9** while
`|b_6| sigma_v = |z_v| = 0.45 .. 1.9`.  That IS gate (a).

### THE GATES (400-event smoke pair, `logs/gates_smoke*.log`)
`--files` the FREE run (`doVtxConstraint=False`, 378-400 candidates),
`--cons-files` the SAME events with `doVtxConstraint=True`.

| gate | number |
|---|---|
| **(a) b zero on every free index, nonzero only on 6** (FROZEN run) | `max_i \|g_i\| sqrt(C_ii)` over the FREE indices / `\|b_6\| sigma_v`: median **1.14e-11**, p90 4.1e-10.  Absolute: free 3.3e-10, `\|b_6\| = 64` (median) |
| **(b) the FROZEN fit reproduces the FREE one** | `\|r_v(frozen) - r_v(free)\|/sigma_v` median **1.5e-3**, p90 1.5e-2, max 0.26; relative `\|ratio-1\|` median **2.9e-3**, p90 2.0e-2 |
| **(b') sigma_v** | `\|sigma_v(frozen)/sigma_v(free) - 1\|` median **8.7e-4**, p90 6.0e-3 |
| **(c) sum_b \|a_b\|^2 = sigma_v^2** | maker `Jpsi_vtxvchk` median **8.4e-11**, p90 2.9e-9; `\|a_b\|^2` vs `vtxvarv` 6.1e-8 (float32); `sum_c hitv` vs `vtxvgf` 1.9e-9 |
| **(d) z_v^2 = Delta chi2** | `<dchi2> = 1.1651` vs `<z_v^2> = 1.1509` (**1.2 % in the mean**); per candidate median 7.2 % (the linearisation); `ndof(frozen) - ndof(free) = 1` on every candidate |
| **(e) the ionization SIGN rule** | applied to the MASS influence it returns -1 on **99.73 %** of candidates at 100 % of their blocks, variance-weighted mean **0.99999** |
| per-group closure | `cfvtx_grp_closure` max **2.1e-15**, `cfmass_grp_closure` max 1.0e-15 |
| `sum_g (vqms+vqio) + vgf` | **1.000000** |

### THE PHYSICS ALREADY IN THE GATES (400 candidates, J/psi gun)
| quantity | value |
|---|---|
| `r_v` | mean **-18 um**, rms 557 um; `sigma_v` median **81 um** |
| `z_v` | mean **-0.0059 +- 0.0552** (consistent with ZERO -- no kernel, no PDG input), **Var 1.148**, skew **+0.075**, kurt **4.61** |
| `vtxvgf` = the HIT (Gaussian) share of `sigma_v^2` | median **0.381** (p10 0.09, p90 0.68) -- so the vertex residual is **62 % MATERIAL** at J/psi momenta |
| the MASS functional on the same candidates | `cfmass_vgf` median **0.098** -- 90 % material |
| `corr(sigma_v, z_v)` | **+0.0003 +- 0.052** -- the self-consistent-sigma coupling is consistent with zero, which is why the vertex term is built with `self_consistent_sigma=False` |
| the per-block ionization signs | -1 on **16 %** of blocks, +1 on 84 % -- i.e. the functional really does change sign block to block, which is why `ressgn` is needed and a single `ioniSign` would be wrong |

### A SECOND DEFECT, CONFIRMED: `Jpsi_d`'s charge re-sign
`leg 0 is the mu+` on **50.7 %** of candidates, and `Jpsi_d = q(leg 0) * theta_6`
on all of them.  Measured on the same 400 candidates:
`skew(z_v)` is **+0.075** in the RAW (swap-invariant) convention and
**-0.378** in the `Jpsi_d` one.  The raw theta_6 IS already the
charge-ordered signed DCA -- `n_hat = p_a x p_b` and `x_b - x_a` both flip
under the swap -- so multiplying by `q(leg 0)` does not define a charge sign,
it destroys one.  `Jpsi_vtxres` carries the raw value; anyone using `Jpsi_d`
for a signed quantity should know this.

## STEP 3 -- THE PRODUCTION (launched 2026-09-11 07:52)

    /ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/prod/task_NNNN/

160 tasks x 2000 events of the SAME J/psi-gun SIM the mass card used
(`resolution_simprod_jpsigun_ul16`), driver `vtxres/run_prod.sh`, 20-way on
each of submit50/51/52 (~31 min/task, 1.61 GB RSS each).  Switches:

    exportVtxResidual=True exportCfGroupExponents=True exportStepRecords=False
    doVtxConstraint=False        (index 6 FREE -- what every production does)
    + the 260905d fit settings verbatim

**328 kB/candidate** measured (the 260905d production was ~400 kB/cand WITH
raw step records; here the in-maker exponents replace them and the per-group
split is carried for BOTH functionals).  Expect ~98 GB / 300 k candidates.

### The offline chain, all TESTED end to end on a 400-event smoke
| script | what |
|---|---|
| `extract_vtx.py --functional {vtx,mass}` | the in-maker `cfvtx_grp_*` / `cfmass_grp_*` into the `matres` CSR npz (27.3 kB/cand, 21.9 groups, 12.2 hit rows) |
| `vtxterm.py` | arms (`cf`/`gauss`/`gaussq`) + the predicted density |
| `make_vtx_card.py` | the `MaterialCFTerm` cards: vtx / mass / joint, 42 material + 18 hit-class parameters |
| `run_ladder.sh`, `run_fit.sh`, `run_tf.sh` | cards and EDM-certified fits |
| `fisher_vtx.py` | H and J in the layout `hitlik/efficiency.py` reads |
| `xcum_vtx.py` | the vertex-mass correlation at three levels |
| `plot_vtx.py` | the figures |
| `run_prod_dy.sh` | the Z-like check on DY MiniAOD |
GATE: card -> fit chain validated on 300 smoke candidates, **EDM 2.4e-17**.

## STEP 4 RESULT -- THE DISTRIBUTION AND THE COMPOSITION (16 040 candidates)
`plot_vtx.py`, `logs/plots.log`, figures in `~/public_html/cvh/260911_vtxres/`.

### the pull
| | untrimmed | trimmed at \|z\| < 5 (drops 0.187 %) | model `cf` | model `gaussq` |
|---|---|---|---|---|
| mean | **-0.0079 +- 0.0189** | -- | 0 | 0 |
| Var | 5.75 | **0.981** | 1.038 | **0.9987** |
| skew | -91.8 | **-0.0097** | **+0.0001** | 0 |
| kurt | 10452 | 3.906 | 25.9 | 3.004 |

Two things.  (i) **The mean is ZERO** -- no kernel, no theory, no PDG input;
that is the whole point of the term.  (ii) The untrimmed moments are set by
0.19 % of candidates, so the moments are not the description: the TAILS TABLE
is.  `gaussq`'s model variance is 0.9987 rather than exactly 1 -- the 0.13 %
is the tau-grid truncation at 7.89, the same number `hitlik` measured.

**The vertex residual has NO LANDAU SKEW**, and that is physics, not an
accident: the ionization share of `sigma_v^2` is **3.8e-6** (below), so the
one-sided channel that skews the MASS residual contributes nothing here.  The
CF's own predicted skew is +0.0001 and the trimmed data skew is -0.0097.

### THE TAILS  P(\|z\| > t)
| t | data | CF | data/CF | Gaussian (fit's Q) | data/chi2 |
|---|---|---|---|---|---|
| 1 | 0.29003 | 0.29567 | 0.98 | 0.31693 | 0.92 |
| 2 | 0.04707 | 0.04433 | 1.06 | 0.04545 | 1.04 |
| 3 | 0.00885 | 0.00644 | **1.37** | 0.00270 | **3.28** |
| 4 | 0.00343 | 0.00201 | **1.71** | 0.000063 | **54.2** |
| 5 | 0.00187 | 0.00097 | **1.94** | 5.7e-7 | **3261** |
The CF is **1.9x (3 sigma) to 1680x (5 sigma)** closer to the data than the
chi2.  (Mass term on the same candidates, for comparison: data/CF is
0.82-0.97 at 3-5 sigma and data/chi2 3.5 / 42 / 2392.)

### THE COMPOSITION -- MS- or hit-dominated?  BOTH, and it crosses over
Median shares of `sigma_v^2`: **hit 0.371, multiple scattering 0.629,
ionization 0.0000** (3.8e-6).  The MASS functional on the SAME candidates:
MS 0.898, ionization 0.0002, hit 0.10.

vs the SOFTER muon's **GEN** pT (8 quantile bins):
| gen pT [GeV] | <1.2 | 1.2-2.0 | 2.0-2.9 | 2.9-3.9 | 3.9-5.1 | 5.1-6.7 | 6.7-8.9 | >8.9 |
|---|---|---|---|---|---|---|---|---|
| hit share | 0.065 | 0.201 | 0.296 | 0.359 | 0.413 | 0.461 | 0.508 | **0.552** |
| MS share | **0.935** | 0.799 | 0.704 | 0.640 | 0.587 | 0.539 | 0.492 | 0.448 |
So the vertex residual is **MS-dominated below ~7 GeV and hit-dominated
above** -- exactly the expected composition (innermost pixel hit resolution
plus multiple scattering in the beam pipe and the first pixel layers), now
measured, with the crossover located.

### WHICH MATERIAL AND WHICH HITS -- and it is COMPLEMENTARY to the mass term
mean share of `sigma_v^2` (MS + ionization), top groups:
| VERTEX | share | MASS | share |
|---|---|---|---|
| `bpix_support6` | **0.324** | `tib_support` | **0.221** |
| `bpix_active_L1` | 0.073 | `tec_structure` | 0.183 |
| `tib_support` | 0.055 | `tob_support` | 0.135 |
| `bpix_active_L2` | 0.041 | `bpix_support6` | 0.100 |
| **`beampipe`** | **0.031** | `tibtid_services` | 0.068 |
| `fpix_support` | 0.030 | `bpix_services` | 0.041 |

hit classes: VERTEX `pix_y_q1` 0.065, `pix_x_q1` 0.041, `pix_y_q3` 0.038,
`pix_y_q2` 0.038 -- **all PIXEL**; MASS `str_N3_lo` 0.024, `str_N2_lo` 0.020,
`str_N1_lo` 0.012 -- **all STRIP**.

**This is the reason to have the term at all**: the vertex residual measures
the INNER tracker (beam pipe, BPix support and active layers, the pixel hit
classes), the mass residual the OUTER one (TIB/TEC/TOB support, the strip hit
classes).  They are two projections of the same fit that see different
material.

### THE SELF-CONSISTENT-SIGMA CHECK
`corr(sigma_v, z_v) = -0.00038 +- 0.00790`, `corr(sigma_v, |z_v|) = +0.0037`
-- ZERO at the 0.8 % level, so the vertex term is built with
`self_consistent_sigma=False` and no `a_res`, and the first-order Jacobian
term it would carry is bounded by that correlation.  (The MASS functional on
the same candidates has `corr(sigma_m, z_m) = +0.033 +- 0.008`, i.e. the
effect the mass term's `a_res` correction exists for is real and 4 sigma --
and the vertex term does not have it.)

## STEP 5 -- THE CARDS (8 000 candidates each, 42 material + 18 hit classes)
`logs/cards_all.log`.  13/13 built.  NLL at MC truth (theta = 0), lower =
describes the data better:

| card | NLL(0) | vs `cf` | per candidate |
|---|---|---|---|
| `vtx_cf` | **-26 049.892** | -- | -- |
| `vtx_gauss` (variance-matched) | -25 881.262 | +168.63 | +0.0211 |
| `vtx_gaussq` (the fit's Q = the chi2) | -25 876.335 | **+173.56** | **+0.0217** |
| `mass_cf` | **-15 893.497** | -- | -- |
| `mass_gaussq` | -15 312.549 | +580.95 | +0.0726 |
| `joint_cf` | -41 943.389 = exactly `vtx_cf` + `mass_cf` | | |

(`hitlik`'s truth-referenced 4-component term was +0.0155/row against its
Gaussian, so the vertex term's non-Gaussianity is of the same size.)

### TWO PIPELINE DEFECTS FOUND AND FIXED HERE
1. **`imap_unordered` in the extraction.**  The vertex and the mass npz were
   assembled in whatever order the 8 workers finished, so row `i` of one was
   NOT the same candidate as row `i` of the other.  The joint card's
   `(run, lumi, event)` gate caught it; the extraction now uses ordered
   `imap`, and `make_vtx_card.common_index` takes the INTERSECTION of the two
   selections up front.
2. **The Gaussian arms underflow float64.**  A Gaussian density at
   `|z| = 40` is `exp(-800) = 1e-348`, below the smallest float64, so
   `gauss`/`gaussq` returned `NLL(0) = inf` and no comparison was possible.
   A `--max-abs-z 40` guard now drops those candidates -- **2 in 16 040
   (0.012 %)** -- applied IDENTICALLY to every arm and channel.  It IS a cut
   on the residual and is labelled as such; the UNCUT tails are reported
   separately, so nothing about the tail is hidden.
   The candidates it removes are NOT fit failures by any convergence measure
   (`Jpsi_vtxbfree`, `Jpsi_vtxvchk`, `edmval`, the fitted vertex position and
   chi2/ndof are all normal), and they are NOT the `n_hat` degeneracy either
   -- their muon opening angles are LARGER than the bulk's (median
   `sin theta` 0.264 vs 0.051), so the collinear-`p_a x p_b` hypothesis is
   refuted.  They are a real tail: a fitted DCA of centimetres on a prompt
   J/psi gun, 0.19 % of candidates beyond 5 sigma.

### The other cut, and what it is
`extract_vtx.py --max-vchk 1e-4` drops **0.245 %** of candidates (49 in
20 000).  Every one of them has `sigma_v` of 46 m to 460 m -- a fit whose
vertex direction is unconstrained -- so this is a cut on the FIT's OWN
COVARIANCE, not on the residual.

## STEP 5 RESULT (part) -- THE SANDWICH ON THE VERTEX CHANNEL
`fisher_vtx.py` (H by 60 HVPs in 83-113 s per arm, J by 200 batch means,
`|sum_m g_m - g| = 0`) + `hitlik/efficiency.py`.  8 000 candidates, at MC
truth, with the parmtype-15 tier priors and 1.0 on every hit class.
Every number is the ACTUAL (sandwich) variance, never the nominal Fisher one.

### sandwich / quoted -- does each arm's own error mean anything?
| arm | marginal | prior-free, material | prior-free, hit classes |
|---|---|---|---|
| **`cf`** | **0.911** | **1.031** | **1.061** |
| `gauss` (variance-matched) | 1.310 | -- | -- |
| **`gaussq`** (the chi2) | **1.502** | **1.902** | **1.177** |
bootstrap/sandwich 0.994-1.004 in every arm (as it must be algebraically).

**The CF's quoted error is right to 3-6 %.  The chi2's is optimistic by 50 %
(marginal) to 90 % (prior-free) IN SIGMA, i.e. 2.3x to 3.6x in variance** --
and that is the direct consequence of the pull's kurtosis (28 trimmed at
40 sigma, 3.9 trimmed at 5): a Gaussian scale estimator's variance is set by
the FOURTH moment of the real residual.

### EFFICIENCY  sigma^2(chi2, ACTUAL) / sigma^2(CF, ACTUAL)
| | marginal | prior-free (standalone) | what the chi2 CLAIMS |
|---|---|---|---|
| MATERIAL groups | median **69.8** (16-84 % 7.6-347) | median **3.30** (16-84 % 1.14-...) | 3.22 |
| HIT CLASSES | median **2.40** (16-84 % 1.77-5.07) | median **1.115** | 0.936 |

The marginal material numbers are enormous because the chi2's ACTUAL error on
several groups (0.03-2.0) exceeds their 0.0025 prior by two orders of
magnitude -- on this residual the Gaussian estimator is not merely
inefficient, it is unusable, and the prior is doing all the work.  The
PRIOR-FREE median 3.30 is the number to quote.

### WHAT THE VERTEX TERM MEASURES (CF, quoted sigma, 8 000 candidates)
material (1 prior sigma = 0.0025 in card units): `material_beampipe`
**0.0061**, `bpix_active_L1` 0.0067, `bpix_support6` 0.0081,
`bpix_active_L2` 0.0081, `tib_support` 0.0139, `fpix_support` 0.0195 --
i.e. 2.4-8 prior sigmas of information on the INNER material at 8 000
candidates.
hit classes (prior 1.0): `pix_y_q1` **0.172**, `pix_y_q3` 0.199,
`pix_y_q2` 0.212, `pix_x_q1` 0.213, `pix_y_q0` 0.252, `pix_x_q0` 0.271 --
the innermost PIXEL classes, measured to 17-27 % from 8 000 candidates.

## A THIRD ITEM -- A UNIT TRAP, *NOT* AN INHERITED DEFECT
### (CORRECTED 2026-09-11 later; the paragraph below is kept for the record
### but its last sentence was WRONG -- see the correction under it)
## A THIRD DEFECT, INHERITED: `hitlik/efficiency.py`'s PRIOR UNITS
A whitened card carries a material parameter in units of its own tier prior,
so **1 prior sigma is `gprior**2` in card units**, not `gprior`
(`make_*_card.py`: `gprior_card = gpriors * gscale`).  `efficiency.py` built
its prior matrix `P` from the RAW `gprior`, i.e. **1/gprior ~ 20x too loose**,
which left the "marginal" numbers effectively prior-free and inflated the
material efficiency ratios (my first run read a median EFF of 69.8 with
individual groups at 623 and `inf`).  Fixed with a `--prior-power` option
(default 1, so every number produced before today is unchanged; pass 2 for a
whitened card).  The corrected numbers are the ones quoted below.

### THE CORRECTION (2026-09-11, later)
"The same correction applies to `hitlik`'s and `perhit`'s material marginal
tables" is **WRONG and withdrawn**.  `efficiency.py` applies the prior in
whatever units the FISHER MATRICES carry, and the two pipelines differ:
* `hitlik/fisher_cmp.py` builds through `hitlik_term.build`, which sets
  **`group_units = ones`** -- the parameter IS the physical `k`, 1 tier prior
  is `gprior`, and `--prior-power 1` (the default) is RIGHT there.
* `make_*_card.py` OVERRIDES `term.group_units` to `1/gprior`, where 1 tier
  prior is `gprior**2`; `vtxres/fisher_vtx.py` builds through
  `make_vtx_card.build_term`, i.e. in CARD units, so **only THIS pipeline
  needs `--prior-power 2`**.
Re-evaluating the STORED matrices (`runs/hitlik/fisherHJ20k.npz`,
`runs/perhit/fisherHJ{,_all}.npz`) at `--prior-power 1` reproduces every
published hitlik/perhit number EXACTLY (1.825 / 2.093, 0.974 / 1.338, 1.204 /
1.221; 1.097 / 1.171, 0.529 / 0.563; 1.486 / 1.393; 1.510 / 1.657).  At
`--prior-power 2` those k-unit matrices get a 20x-too-tight prior, `(H+P)^-1`
collapses onto it and the informativeness test rejects every material group --
the material table comes out EMPTY, which is the diagnostic that found this.
Correction blocks appended to `hitlik/STATE.md`, `hitlik/perhit/STATE.md` and
`Documents/Resolution/NOTES.md`.  **The vtxres numbers below are unaffected**:
they were produced at `--prior-power 2`, which is the correct choice for
card-unit matrices.

## STEP 5 RESULT -- THE SANDWICH, CORRECTED PRIOR (8 000 candidates)
`logs/eff_vtx_p2.log`

### sandwich / quoted
| arm | marginal | prior-free, material | prior-free, hit classes |
|---|---|---|---|
| **`cf`** | **0.885** | **0.961** | **1.061** |
| `gauss` | 1.040 | -- | -- |
| **`gaussq`** (the chi2) | 1.099 | **1.543** | **1.177** |
bootstrap/sandwich 0.995-0.998.

### EFFICIENCY  sigma^2(chi2, ACTUAL) / sigma^2(CF, ACTUAL)
| | marginal | prior-free | what the chi2 CLAIMS |
|---|---|---|---|
| **MATERIAL** | **2.659** (2.02-3.46) | **2.778** (1.85-15.4) | 1.006 |
| **HIT CLASSES** | **1.215** (0.97-1.46) | **1.115** (0.50-1.30) | 0.968 |

**The full PDF constrains the material 2.7x better in variance (1.6x in
sigma) and the hit classes 1.2x, while the chi2 claims parity.**  The CF's own
quoted error is right to 4-6 %; the chi2's is optimistic by 54 % in variance
on the material.  Per group (marginal): `bpix_services` 3.46,
`tib_support` 2.66, `bpix_support6` 2.02.

## STEP 5 RESULT -- VERTEX vs MASS vs JOINT (quoted sigma at MC truth, CF arm)
`logs/eff_{vtx,mass,joint}_p2.log`.  1 tier prior = 0.0025 card units for the
0.05-prior material groups and 1.0 for a hit class.

| parameter | VERTEX | MASS | JOINT | prior |
|---|---|---|---|---|
| `material_bpix_support6` | **0.0020** | 0.0024 | **0.0020** | 0.0025 |
| `material_tib_support` | 0.0024 | 0.0023 | **0.0022** | 0.0025 |
| `material_tec_structure` | (uninformative) | 0.0022 | **0.0021** | 0.0025 |
| `material_tob_support` | -- | 0.0024 | **0.0023** | 0.0025 |
| **`hitres_pix_y_q1`** | **0.167** | 0.958 | **0.166** | 1.0 |
| **`hitres_pix_x_q1`** | **0.206** | 0.800 | **0.204** | 1.0 |
| `hitres_pix_y_q2` | 0.207 | ~0.95 | 0.206 | 1.0 |
| `hitres_str_N1_lo` | 0.377 | 0.440 | **0.300** | 1.0 |
| `hitres_str_N3_lo` | 0.404 | 0.561 | **0.32** | 1.0 |

**On the innermost PIXEL hit classes the vertex term is 4-6x tighter in sigma
(15-33x in variance) than the mass term**, and the joint is no better than the
vertex alone there -- i.e. that information comes from the vertex residual and
from nowhere else.  On the STRIP classes and the OUTER material the mass term
leads and the joint improves on both.

## STEP 5 RESULT -- THE FITS, CERTIFIED (8 000 candidates, 60 parameters)
`logs/certify.log`.  Value + NLL + rabbit EDM, `--max-edm 1e-3`.

| fit | NLL(min) | EDM | `material_bpix_support6` | `hitres_pix_y_q1` |
|---|---|---|---|---|
| `vtx_cf` | -26 059.934 | **1.3e-12** | +0.00366 +- 0.00196 | **-0.400 +- 0.132** |
| `vtx_gaussq` | -25 890.002 | 2.5e-08 | +0.00310 +- 0.00188 | -0.538 +- 0.125 |
| `mass_cf` | -15 896.129 | 6.2e-14 | -0.00044 +- 0.00243 | -0.181 +- 0.958 |
| `joint_cf` | -41 954.970 | 7.7e-12 | +0.00327 +- 0.00192 | -0.408 +- 0.131 |
| `inj_vtx_cf` | -26 061.175 | 1.2e-16 | +0.00270 +- 0.00194 | -0.414 +- 0.131 |
| `inj_vtx_gaussq` | -25 892.592 | 1.4e-08 | +0.00196 +- 0.00187 | -0.544 +- 0.126 |
| `inj_mass_cf` | -15 895.928 | 3.3e-17 | -0.00059 +- 0.00242 | -0.187 +- 0.958 |
| `inj_joint_cf` | -41 956.049 | 2.1e-15 | +0.00226 +- 0.00191 | -0.422 +- 0.130 |
| `injhit_vtx_cf` / `injhit_vtx_gaussq` | | PASS | | |

**8/11 certified.  The three that fail are ALL Gaussian arms**:
`vtx_gauss` (Hessian not positive-definite at its own minimum) and
`mass_gaussq` / `joint_gaussq` (a NaN Hessian -- see the open item).  Same
lesson as `hitlik` ("a misspecified likelihood is also a worse-conditioned
one": CF 7/8 vs Gaussian 5/8 there).

**The vertex term MEASURES the innermost pixel resolutions.**
`hitres_pix_y_q1 = -0.400 +- 0.132` from the vertex residual against
`-0.181 +- 0.958` from the mass term on the SAME candidates -- **7.3x tighter
in sigma, 53x in variance** -- i.e. the fit's assumed variance for that class
is 40 % too large, and only the vertex residual can see it.

## STEP 5 RESULT -- THE INJECTIONS
`logs/recovery.log`, `logs/recovery_hit.log`.  (`recovery.py` prints `nan` in
its `/truth` columns because it reads the truth under a key this card does not
write; the numbers below are its own `shift` and `f_pri` divided by hand.)

### `material_bpix_support6` x1.05 material (truth = +0.00243951 card units)
| channel | baseline | injected | shift | /truth | f_pri | **corrected/truth** | leak rms |
|---|---|---|---|---|---|---|---|
| vertex, CF | +0.00366 | +0.00270 | -0.00096 | -0.394 | 0.400 | **0.984** | 0.034 |
| vertex, fit's Q | +0.00310 | +0.00196 | -0.00114 | -0.467 | 0.443 | **1.055** | 0.118 |
| MASS, CF | -0.00044 | -0.00059 | -0.00015 | -0.062 | 0.063 | **0.976** | 0.013 |
| **JOINT, CF** | +0.00327 | +0.00226 | -0.00101 | -0.414 | 0.419 | **0.988** | 0.034 |
Every channel recovers the 5 % injection to **1.6-5.5 %**.  Largest leakage:
-0.12 sigma onto `hitres_pix_x_q1` (CF) and +0.81 sigma onto
`hitres_pix_y_q3` (the fit's-Q arm -- another sign that arm is worse
conditioned).

### `hitres_pix_x_q2` variance x1.10 (an INNERMOST PIXEL class)
`hit_mode` is LINEAR, so the expected shift is
`-eps_inj/(1+eps_inj) x (1+eps_base)`.
| channel | eps_base | injected | shift | expected | **/truth** | leak rms |
|---|---|---|---|---|---|---|
| vertex, CF | -0.1659 +- 0.291 | -0.2375 | -0.0716 | -0.0758 | **0.944** | **0.001** |
| vertex, fit's Q | -0.2978 +- 0.289 | -0.3593 | -0.0615 | -0.0638 | **0.963** | 0.000 |
Leakage below 0.001 sigma on every other parameter.

## STEP 6 RESULT -- THE CORRELATION WITH THE MASS TERM (20 000 candidates)
`logs/xcum.log`.  Three levels:

1. **THE ALGEBRA.**  `Cov(r_v, dm) = sum_b a_b^v . a_b^m = e_6^T C a_m`
   exactly (every residual row is covered by a registered block -- that is the
   `vtxvchk` gate).  It is NOT zero by construction the way the per-hit
   complement was (`F^T R = 0` there).  What makes it vanish is a **MIRROR
   SYMMETRY**: reflecting the event in the plane spanned by the two momenta
   sends `n_hat = p_a x p_b -> -n_hat`, hence `theta_6 -> -theta_6`, while the
   momenta -- and so the mass -- are unchanged.  MEASURED per candidate:
   median **-0.0049**, **mean +0.00013**, rms **0.192**, p1/p99 -0.42/+0.43.
   So each candidate has a real |rho| ~ 0.19 of random sign and the ENSEMBLE
   average is zero -- exactly the symmetry statement, with the field the only
   thing that could break it.
2. **THE ENSEMBLE.**  `corr(z_v, z_m) = +0.0047 +- 0.0071` with
   `z_m = (m - m_gen)/sigma_m`; `corr(z_v^2, z_m^2) = -0.0014`.  Consistent
   with zero at both first and second order.
3. **THE FOURTH CROSS CUMULANT.**  `kappa(v,v,m,m)/sqrt(kappa4_v kappa4_m)`
   median **0.132** (p16-p84 0.069-0.228) -- the same size as the per-hit
   study's q/p-vs-others (0.09-0.14) and far below its phi-vs-d0 (0.71).

**THE JOINT VERDICT.**  `sandwich/quoted` on the joint is **0.880** against
**0.885** for the vertex alone and 0.432 for the mass alone -- i.e. **NOT
above 1**, so the joint does not over-count relative to what its arms already
do.  The two residuals are uncorrelated in the ensemble and share 13 % of
their fourth cumulant, so the product-of-marginals form is a good
approximation and the sandwich prices what is left.

## STEP 8 RESULT -- THE Z-LIKE CHECK (DY MC, 7 841 candidates)
6 files of `production/condor_dymc_v2`'s own configuration (REAL geometry,
pileup, MiniAOD, `massMin=60 massMax=120`) plus `exportVtxResidual=True`;
`logs/plots_dy.log`.

**The predicted regime change is confirmed.**  Median shares of `sigma_v^2`:
| sample | hit (Gaussian) | multiple scattering | ionization |
|---|---|---|---|
| J/psi gun (muons 1-15 GeV) | 0.371 | **0.629** | 3.8e-6 |
| **DY (muons 20-143 GeV)** | **0.732** | 0.268 | 0.0 |
vs GEN pT on DY: hit share 0.587 (20-26 GeV) -> **0.803** (>43 GeV).
So the vertex residual is **MS-dominated at J/psi momenta and hit-dominated at
Z momenta**, and on DY the leading contribution is a single hit class,
`pix_x_q1` at **0.198** of `sigma_v^2` (0.041 on the gun) -- the innermost
pixel BENDING-plane measurement.  Material drops to `bpix_support6` 0.108
(0.324 on the gun).

**AND A WARNING FOR A DATA FIT.**  On DY the residual has a tail the CF does
NOT model: `P(|z|>5)` is **1.86 %** against a CF 0.022 % (data/CF **85**),
and `Var(z)` is 1123 untrimmed (1.147 trimmed at 5 sigma, skew -0.014, kurt
3.99).  The gun, on ideal geometry, has 0.19 % beyond 5 sigma and data/CF
1.9.  So the extra tail is not the resolution model: it is real geometry
(misalignment), pileup, and genuinely displaced or mispaired muons in a
60-120 GeV MiniAOD dimuon sample.  **A vertex term on data needs an outlier
(background) component**, and the size of it is now measured: ~2 %.

## STEP 7 RESULT -- COST AND THE EXPORT BILL

**Cost, A/B on ONE PINNED CPU** (`taskset -c 40`, 200 events of the J/psi gun,
identical configuration but for the switch; time from the first event to the
last stamped line):
| | wall for 200 events | per candidate |
|---|---|---|
| fit + the MASS export | 226 s | 1.13 s |
| + the VERTEX block | 334 s | 1.68 s |
| **the vertex block** | **+108 s** | **+0.54 s (+48 %)** |
It is the same `cvhcf` call at different weights, so it costs what the mass
functional's own call costs.  The concatenated-tau trick (every primitive
depends on `weight * tau` alone) would make the second functional nearly free
and is NOT ported.

**Export, compressed, measured** (`cost_vtx.py --file`):
| block | kB/cand | TB at 41 M |
|---|---|---|
| vertex: per-group exponents | 26.37 | 1.107 |
| vertex: flat exponents | 1.44 | 0.060 |
| vertex: shares + scalars | 0.61 | 0.026 |
| vertex: the influence `a_b` (`resinfvtxv`) | 1.63 | 0.069 |
| vertex: the D row (`Jpsi_jacVtx`) | 0.91 | 0.038 |
| **the VERTEX block** | **30.96** | **1.300** |
| the MASS block, for comparison | 29.98 | 1.259 |
| (`hesspackedv`, the quadratic term's, for scale) | 105.3 | 4.42 |
**the vertex term costs 1.03x the mass term's export** -- as expected, it is
ONE more component per candidate.  The whole file is 328 kB/candidate.

## Plan / status
- [x] step 0: the object, the state layout, the sign rule, the production to match
- [x] step 1: maker export `exportVtxResidual`
- [x] step 2: gates -- ALL PASS
- [x] step 3: production 160/160
- [x] step 4: distribution + composition (96 160 candidates)
- [x] step 5: fits + sandwich + injections
- [x] step 6: correlation with the mass term
- [x] step 7: cost + export bill
- [x] step 8: Z-like check on DY (beamspot rows: OFF in every config -- see NOTES section 9)
- [x] step 9: figures, NOTES.md, commits
