# perhit — the DATA version of the hit-residual CF likelihood

## RESUME HERE (2026-09-10 17:00)

STATUS: maker DONE and committed; production **160/160 COMPLETE**; offline
chain running.  Everything below "THE OBJECT" is settled and needs no
re-deriving.

**Verified this session (17:00):**
* production `resolution_trackres_mugun_ul16_260910_perhit`: 160 `task_NNNN`
  dirs, **160 `.complete` sentinels, 160 root files, 190 GB** on ceph.  DONE.
  (The task dirs are `task_0000`-style, not `task_0`; a `seq 0 159` check
  reports everything missing.)
* extraction `runs/perhit/perhit.npz` = **8750 tracks / 169 709 rows**, made at
  15:25 from the **70** tasks complete at the time.  A full re-extract at 160
  tasks x `--max-cands 125` gives ~20 k tracks.
* cards: 14 of 15 in `runs/perhit/cards` (on ceph); `ph_gauss.hdf5` still in
  `runs/perhit/cards_local` (open by the running fit); `ph_inj_cf_all` MISSING
  (it died on the /work quota).

**DISK (do not undo this):** `/work/submit` quota is 500 G and was full.
`runs/perhit/cards` and `runs/perhit/perhit.npz` are now SYMLINKS into
`/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_perhit_260910/`, so every path
in the scripts still works and new cards land on ceph.  Anything > 0.5 G goes
there.  **ceph is NOT visible from submit82** (permission denied — cephx
eviction); do ceph I/O from submit50/51/52 through
`/home/submit/david_w/.claude-work/jobs/perhit/s5{0,1,2}.sh '<cmd>'`
(persistent ControlMaster), always with `cd /abs/path || exit 9`.

**Live processes (kill only these PIDs; never `pkill -f`):**
| host | what | state |
|---|---|---|
| submit50 | `run_stage2.sh fits` PID 1757773, log `logs/stage2_fits.log` | RUNNING: ph_cf edm 5.05e-12, ph_gaussq edm 1.64e-9, ph_gauss from 16:37, 12 to go |

**NEXT STEPS, in order**
1. [x] rebuild `ph_inj_cf_all` -- DONE 17:03 on submit52 (`CARDOK`), on ceph
2. [~] let `run_stage2.sh fits` finish; certify with `run_stage2.sh certify`
       (`perhit/certify.py`: value + NLL(reduced) + rabbit EDM, PASS at
       EDM < 1e-3).  Done so far: ph_cf 5.05e-12, ph_gaussq 1.64e-9,
       ph_gauss 7.65e-10; ph_cf_ref running from 16:56.
       ALL 15 CARDS ARE BUILT AND ON CEPH (17:05), `ph_gauss.hdf5` moved out
       of `cards_local`, which is gone.
3. [~] fisher: A = `--comps hit ref --arms cf gauss gaussq` on submit52
       (`logs/stage2_fisher.log` -> `runs/perhit/fisherHJ.npz`);
       B = `--comps all --arms cf gaussq` on submit51
       (`logs/fisher_all.log` -> `fisherHJ_all.npz`).  Both 8000 tracks of
       `perhit.npz`, `--nbatch 200`.
4. [~] `xcum` on the 20 k npz (submit51, `logs/xcum20k.log`)
5. [ ] `run_stage2.sh efficiency effall joint saturation recovery cost plots
       finaltable subfits` -- NEW stages `certify`, `effall` (the SAME-TRACK
       joint, `--cset all`), `joint` (the DISJOINT residual+mass joint via
       `perhit/fisher_joint.py`), `tails`, `subfits` were added to
       `run_stage2.sh` this session.
6. [ ] NOTES.md entry, commit
7. [x] gates re-run on all 160 tasks after fixing two indexing traps in
       `gates.py` -- see "THE GATES, FINAL"
8. [x] the measured export bill at production scale -- see "THE EXPORT BILL,
       MEASURED AT PRODUCTION SCALE"

**SECOND EXTRACTION (this session):** `perhit20k.npz` on ceph,
**20 000 tracks / 388 641 rows** (19.4 comps/track), 227 s, 11.5 GB, from all
160 tasks.  Used for the STATISTICS-hungry steps (xcum, tails, cost, plots);
the FITS and the Fisher/sandwich stay on `perhit.npz` (8750 tracks, 8000 used)
because the whole card ladder was built from it -- quote the N with every
number.

**run_tf.sh CHANGED (this session):** it now binds `/ceph/submit` whenever the
host can read it (it used to need `WANT_CEPH=1`).  Without that the cards
symlink does not resolve inside the container and every fit after the move
would have failed.  `.bak` of the original next to it.

**Paths**: npz `runs/perhit/perhit.npz`; cards/fits `runs/perhit/{cards,fits}`;
figures `~/public_html/cvh/260910_perhit/`; maker branch
`perhit-residual-cf-260910` in `CMSSW_15_0_19_patch2_dev2` (4 commits on top of
ca6058d96fc).

## THE OBJECT (derivation, so a fresh agent does not have to redo it)

CVH single-track fit: `ncons` constraints, `nstatefree` free local params,
`V = sum_b dV_b` block diagonal (the `dVs` the maker already builds),
`F = Ffull(:, freestateidxs)`, `C = (F^T V^-1 F)^-1 = Cinvd^-1`,
`R = V^-1 - V^-1 F C F^T V^-1`.

* post-fit residual in constraint space  `rho = V R r = r + F dxfree`
* `V R V = V - F C F^T`   (one line: `R V R = R`)
* `M = V^{1/2} R V^{1/2}` is an ORTHOGONAL PROJECTOR of rank
  `d = ncons - nstatefree`; `chi2 = r^T R r = |M s|^2`, `s = V^{-1/2} r`.
* row counting: `ncons = 5*nhits + nvalid + nvalidpixel`,
  `nstatefree = 5*(nhits+1)`  =>  **d = n_meas - 5** exactly, `n_meas =
  nvalid + nvalidpixel`.
* the KINK rows of `rho` are a deterministic function of the MEASUREMENT
  rows, so restricting to the measurement rows loses nothing: the Mahalanobis
  form is invariant under a bijective linear map of the support, hence
  `rho_m^T G^+ rho_m = chi2` with `G = (V R V)[m,m] = V_mm - F_m C F_m^T`,
  rank `d`.
* whitening: any `C_w` with `C_w C_w^T = G` gives `z = C_w^+ rho_m` with
  `Cov(z) = I_d` and `sum_k z_k^2 = chi2`.  **Basis chosen: LDL^T of `G` in
  MEASUREMENT-ROW ORDER (= hit order, inner to outer, second pixel coordinate
  after the first)**, null pivots skipped.  Component k is then hit-row k's
  post-fit residual CONDITIONED ON the inner rows' post-fit residuals -- local
  to a hit in the same sense the prototype's q/p component was local to q/p,
  and the natural sequence for a term whose parameters are per-hit-class.
  It is NOT the Kalman filter innovation sequence (Brown-Durbin-Evans
  recursive residuals), which is a different orthonormal basis of the same
  d-space: BDE conditions on the RAW inner measurements and drops the FIRST 5
  components, the LDL of the post-fit covariance drops the LAST 5.  Both give
  `sum z^2 = chi2`; the composite (product-of-marginals) likelihood differs
  between them and the difference is measured, not assumed.
* influence: with `Psi` the whitener (`z = Psi^T rho_m`),
  `W = E_m Psi - V^-1 F C F_m^T Psi`  (ncons x d) is the exact analogue of the
  prototype's `wqop = W5.col(0)`:  `z_k = W[:,k]^T n`.  Then
  `v^(k)_b = W[r0:r0+nb, k]^T dV_b W[r0:r0+nb, k]` and
  `sum_{b not fam15} v^(k)_b = 1` for every k -- the gate.
* the ionization/radiative CF weight is SIGN-SENSITIVE and the sign is
  `charge * sign(W[r0, k])`, per (block, component); the q/p functional's
  single `ioniSign = charge` is the special case `sign(W[r0,0]) = +1`.

## Plan / status
- [x] maker: `cvhcf::TrackInput::ressgn` (per-block sign)
- [x] maker: `exportPerHitResidual_` switch + branches
- [x] maker: fix `genParms[4]` (identically 0) and `genParms[2]` (phi wrap)
- [x] maker: truth-referenced components appended to the same arrays
- [x] GATES 1-5 on 200 tracks (below)
- [ ] production `resolution_trackres_mugun_ul16_260910_perhit` -- RUNNING
- [ ] offline: extract / cards / fits / sandwich / injections / cost
- [ ] the two EXTRA DELIVERABLES at the end of this file

## THE MAKER (branch `perhit-residual-cf-260910`, area `..._dev2`)

Commits `d1d10985cc9` (the export) and `86380e52d35` (the whitener and the
sign).  Config switches on `runCvhResClosure.py`, all off/neutral by default:
`exportPerHitResidual` (False), `perHitCfGroups` (True), `perHitRefComponents`
(True), `perHitShareMin` (0.0).

### Two things that did NOT work, and why (do not re-try them)

1. **Assembling `G = V_mm - F_m C F_m^T` and factorizing it.**  `G` is a Schur
   complement: its five null directions are a DIFFERENCE of nearly equal
   numbers, and the solve behind them (`C = (F^T V^-1 F)^-1`) is badly
   conditioned because a thin layer gives its kink block very small process
   noise.  MEASURED, 200 tracks: the null eigenvalues came out at ~1e-9 of the
   diagonal instead of 1e-16, the LDL kept a spurious SIXTH component on 34/60
   tracks (`d = n_meas - 4`), and on exactly those tracks `sum_b v^(k)_b`
   missed 1 by up to a factor of 19 -- always on the LAST component, the one
   that closes the space.  One step of ITERATIVE REFINEMENT made it worse
   (`vchk` p99 0.69 -> 215), so it was removed again.
2. **`sign(W[r0,k])` as the ionization sign.**  See gate 5.

### What works: an orthogonal projection in the standardized space

With `Fw = V^-1/2 F`,  `R = V^-1/2 (I - Q1 Q1^T) V^-1/2`  is an IDENTITY
(`Q1` from a column-pivoted QR of `Fw`), so the projector never goes through
an inverse.  Then
  * `s = (I - Q1 Q1^T) V^-1/2 r`   the standardized post-fit residual,
    `chi2 = |s|^2 = r^T R r`;
  * `Gs = I - Q1_m Q1_m^T`         its covariance on the measurement rows --
    a submatrix of an orthogonal projector, so its null eigenvalues sit at
    1e-16 of the unit diagonal instead of 1e-9;
  * rank IMPOSED at `dexp = ncons - rank(Fw)`, `S = U sqrt(Lambda)` truncated
    there, modified Gram-Schmidt (one re-orthogonalization pass) on the ROWS
    of `S` in measurement-row order.  The "is this row already spanned" test
    is then on a NORM.  What comes out is the LDL's own triangular whitener,
    `z_k = (s_k - sum_{j<k} c_kj z_j)/|e_k|`, computed stably.
  * influence `Wstd = E_m Psi - Q1 Q1_m^T Psi`, `W = V^-1/2 Wstd`, so
    `sum_b W_b^T dV_b W_b = |Wstd_k|^2 = 1` by construction.
`V` is block diagonal over exactly the blocks `resblockrng` enumerates, so
`V^-1/2` is a per-block symmetric square root of `Vinvfull`.

### THE GATES (200 mu-gun tracks, `runs/perhit/qrref3`, `gates.py`)

| gate | number |
|---|---|
| 1a `d == n_meas - 5` and `rank(Fw) == nstatefree` | **100 %** (200/200) |
| 1b `n_meas == nValidHits + nValidPixelHits` | 100 % |
| 1c `|sum_k z_k^2 - chi2|/chi2` | median **2.6e-8**, p90 6.1e-8, max **1.9e-7** |
| 2 `max_k |sum_b v^(k)_b - 1|` (`phres_vchk`) | median **1.2e-9**, p99 2.4e-7, max 7.7e-7 (per-hit only); **2.3e-4** worst with the truth-referenced components, which is the float32 storage of `refCov` and is the same number hitlik measured for `sum_b B_b B_b^T` |
| 2b rank gap `lambda_d/lambda_(d+1)` | median **1.2e15**, min 1.7e14 (was 4.5e7 with the assembled `G`) |
| 3 | model Var(z_k) under the fit's own Q | **1.00000 EXACTLY** (`valdens.py`, below).  DATA variance pooled **0.926**, mean +0.006, skew +0.04, kurt **2.98** |
| 4 `corr(z_k, truth-referenced pull_j)` | all `|corr| < 0.21` at N=200, i.e. `< 3 sigma` of the 1/sqrt(200)=0.071 statistical error; to be re-measured on the production |
| 5 reference component 0 vs the validated `cfqop_*` | **1.2e-7** worst-track relative on ALL SIX families (`ms`, `del`, `ioni_re`, `ioni_im`, `rad_re`, `rad_im`) -- the float32 storage of the reference |
| 5b `phcf_grp_closure` (`sum_g S_g` vs `S`) | max **2.5e-15** |

### WHICH GATE ACTUALLY BINDS (read this before quoting one)
* **Gate 1c, `sum_k z_k^2 = chi2` (2.6e-8 median)** is the binding one: the
  chi2 side comes from `|sfull|^2 - |Q1^T sfull|^2`, which never touches the
  whitener, so the identity tests the projector, the eigen-truncation, the
  Gram-Schmidt and the residual together.
* **Gate 5, reference component 0 vs `cfqop_*` (1.2e-7)** is the other binding
  one: it tests the influence, the per-block weights, the pooling and the sign
  against an independently validated model.
* **Gate 2, `sum_b v^(k)_b = 1` (1.2e-9)**, is largely BY CONSTRUCTION --
  `|Wstd_k|^2 = t_k^T Gs t_k`, which the Gram-Schmidt sets to 1 exactly.  It
  is still worth exporting (it catches a wrong `dV_b`, a mis-indexed block, a
  missing family) but it is not evidence about the truncation.
* **`phres_gchk`** (median 0.02, max 1.8e4) is NOT a defect: it is
  `|T (Gs - Gs_trunc) T^T|`, i.e. the discarded eigenvalues (~1e-16) times
  `|T_k|^2 ~ 1/pivot_k`, so it simply re-measures `phresinflat`.  A component
  whose conditional variance is 1e-11 of its marginal has an accurate `z`
  (~1e-6 relative, and gate 1c holds) but its own row of the whitener is huge.

### THE GATES, FINAL (all 160 tasks, 20 000 tracks, 2026-09-10 17:05)
`run_all.sh gates --max-tracks 20000`, log `logs/gates_final.log`.  **Two
indexing traps in `gates.py` were fixed first** (same family as the gate-4 one
below): gates 1c and 3 summed the WHOLE `phresz`, which carries the five
truth-referenced components AFTER the `d` per-hit ones.  That added ~5 to a
chi2 of ~13 and made gate 1c read a spurious **0.36** median; it also polluted
the per-k variance table above k = d.  Both now slice `z[:d]`.

| gate | number |
|---|---|
| 1a `d == n_meas - 5` | **20000/20000, 0 violations** |
| 1b `n_meas == nValidHits + nValidPixelHits` | 20000/20000 |
| **1c `\|sum_k z_k^2 - chi2\|/chi2` (THE BINDING GATE)** | median **2.43e-8**, p90 5.76e-8, max 2.08e-5 |
| 1c' in-maker `phres_chi2` vs recomputed | max 4.66e-6 |
| 2 `phres_vchk` | median 3.7e-7, p90 4.7e-6, p99 5.6e-5, max 4.5e-2 |
| 2b rank gap `lambda_d/lambda_(d+1)` | median **1.14e15**, p10 5.3e14, min 7.9e13 |
| 3 DATA Var(z), per-hit pooled (N = 288 991) | **0.9584**, mean +0.0108, skew +0.009, **kurt 3.43** |
| 3' DATA Var(z), truth-referenced pooled (N = 100 000) | 1.2559, skew +0.43, kurt 22.2 |
| 4 `corr(z_k, pull_j)` per component index | all `\|corr\| < 0.02` for k <= 17 (N > 4000); the k >= 20 rows have N < 1600 and scatter by their own 1/sqrt(N) |
| 4b algebraic `\|sum_b A_b[k].A_b[qp]\|` (20 k tracks, `xcum`) | median **2.14e-9**, p99 1.5e-8, max 8.3e-6 |
| 5 reference component 0 vs `cfqop_*` | 1.2e-7, all six families |
| 5b `phcf_grp_closure` | max **9.24e-15** |
| export health | `phres_ok` **100.00 %**, `phcf_nok == d + nref` **100.00 %** |

Per-component DATA variance falls from 1.012 (k=0) to ~0.82-0.90 by k ~ 17:
the OUTERMOST kept components sit nearest the rank boundary, where the
conditional variance is a small difference and the fit's assumed hit error is
the least well matched.  `phcf_msec` median 2092 ms/track, 163 ms/component
under 60-way contention; hit (Gaussian) share median 0.968, p10 0.382.

### THE GATES AT PRODUCTION SCALE (20 000 tracks, 2026-09-10 15:25) -- SUPERSEDED by the table above for gates 1c and 3
`run_all.sh gates --max-tracks 20000` on the completed tasks, and the npz
checks on 8750 extracted tracks:

| gate | number |
|---|---|
| 1a `phres_ok` (d == n_meas - 5 == ncons - rank(Fw), and every component's cvhcf call ok) | **100.00 %** |
| 2 `phres_vchk` | median 3.6e-7, **max 1.6e-2** on one track in 8750 |
| 2b `|A_b[k]|^2` vs `|phresvarv|` | 6.5e-8 median, 1.3e-7 max |
| 3 `gaussq` model variance | 1.00000 exactly (both component sets) |
| 4 pooled `corr(z_k, pull_j)` over ALL per-hit rows | **+0.000 / -0.003 / +0.002 / -0.002 / +0.002** against a 0.003 statistical error, for (q/p, lam, phi, d0, z0); INDEPENDENT of the conditioning cut (1e2 -> 1e12) |
| 4b per-track algebraic `|sum_b A_b[k].A_b[qp]|` | median 2.2e-9, max 3.5e-7 |
| 5 reference component 0 vs `cfqop_*` | 1.2e-7, all six families |
| 5b `phcf_grp_closure` | max 9.2e-15 |

**A trap that cost an hour**: the first gate-4 table read up to **+0.38** at
component 24.  It indexed `phresz[k]` without checking `k < phres_d`, so for a
short track slot 24 is one of the TRUTH-REFERENCED components -- correlated
with the pull it was built from by construction.  Fixed in `gates.py`; the
algebraic per-track check is what caught it.

### The physics the production-scale gates show
| quantity | value |
|---|---|
| DATA Var(z), per-hit components | **0.955** (0.961 with `inflat < 1e4`) -- the fit's assumed hit variances are ~4.5 % too large |
| DATA Var(pull_j), the 5 truth-referenced components | 1.015 / 1.033 / 1.070 / 1.115 / **1.818** |
| `phcf_vgf` (hit share of a per-hit component) | median 0.986, p10 **0.860**, p5 **0.764** |
| `phresinflat` | median 3.9, p90 456, p95 8.3e6 -- `max_inflat = 1e4` keeps **92.7 %** of rows |
| `phcf_msec` per track under 60-way contention | median 2092 ms (993 ms uncontended), 163 ms/component |

Two things to carry forward:
* **`z0` is over-dispersed by 1.35x in sigma** (Var 1.818 against 1.015-1.115
  for the other four).  This observable did not EXIST before the `genParms[4]`
  fix -- the prototype had `Var = 4.0e6` there -- so it is new, and it says
  the fit's `z0` error is underestimated.  Worth its own look; it is the most
  weakly conditioned Cholesky component, so a small mis-modelling is amplified.
* **The material-rich tail.**  `vgf` median 0.986 makes a typical per-hit
  innovation 1.4 % material, but p5 is 0.764, i.e. one row in twenty is ~24 %
  material.  So the per-hit term is not as material-blind as the 120-track
  smoke suggested; the fits decide.

### GATE 3 IN FULL (`valdens.py`, 120 tracks, all three arms)
Row-averaged predicted density by the same inverse Fourier transform the term
uses, integrated on |z| < 40:

| component set | N | arm | norm | mean | var(density) | var(model) | var(DATA) |
|---|---|---|---|---|---|---|---|
| per-hit | 1608 | cf | 1.000000 | +0.000000 | 1.00320 | 1.00318 | **0.89864** |
| per-hit | 1608 | gauss | 1.000000 | +0.000000 | 1.00318 | 1.00318 | 0.89864 |
| per-hit | 1608 | gaussq | 1.000000 | +0.000000 | **1.00000** | **1.00000** | 0.89864 |
| reference | 600 | cf | 0.999996 | +0.000010 | 1.03281 | 1.02836 | **1.00720** |
| reference | 600 | gauss | 1.000000 | +0.000000 | 1.02836 | 1.02836 | 1.00720 |
| reference | 600 | gaussq | 1.000000 | +0.000000 | **1.00000** | **1.00000** | 1.00720 |

Three things at once:
1. `gaussq`'s variance is **1.00000 exactly** for BOTH component sets, so it
   really is the fit's own Q-matrix decomposition of a unit-variance
   component -- i.e. it IS the Gaussian hit-chi2 the study is measured
   against, not an approximation to it.
2. the CF's model variance is **1.0032** for the per-hit components against
   **1.0328** for the reference ones.  Same Rossi-vs-Moliere gap, diluted by
   the hit share: a per-hit innovation is 98.7 % Gaussian hit noise, so 1.3 %
   of a ~25 % gap is 0.3 %.  **That is quantitatively why the non-Gaussianity
   cannot buy anything on the per-hit components and does on the reference
   ones.**
3. `cf`'s norm 0.999996 / var(density) 1.0328 against var(model) 1.0284 on the
   reference set is the tau-grid truncation at 7.89 -- 0.4 % on the second
   moment, the same truncation the prototype ran with.

### GATE 5 IS WHAT FIXED THE SIGN CONVENTION
Reference component 0 IS the q/p functional -- column 0 of `L^-T` is
`e_0/sigma_qp`, so its per-block weight `sqrt(v^(0)_b/sq2)` equals the
`cfqop_*` weight `sqrt(v_b/sq2)/sigma` term by term.  That makes the
comparison a real test, and it distinguished three candidate conventions for
the ionization/radiative sign by their worst-track relative difference on the
IMAGINARY families (the real ones are even and agree at 1.2e-7 under all
three):

| candidate | worst rel |
|---|---|
| `sign(W[r0,k])` | **2.0** -- i.e. the opposite sign, on every track |
| `-sign(W[r0,k])` | 5.3e-2 |
| `-sign(W_b . u)`, `u` = the rank-1 direction of `dQI` oriented by its qop component | **1.2e-7** |

Two separate effects: (i) the process-noise constraint row is built as
(propagated - state), which is one global minus; (ii) `dQI` is
`e_0 e_0^T sigma^2` in the CURVILINEAR frame and `Hm` spreads it over all five
local rows, so the signed coefficient is `W_b . u` and not `W[r0,k]`.

### PHYSICS ALREADY VISIBLE IN THE GATES
* **DATA Var(z) = 0.926 against a model 1.0000**: the hit variances the fit
  assumes are ~7.4 % too large (3.8 % in sigma).  This is the quantity the 18
  hit-class parameters exist to absorb, and it is the opposite sign to the
  truth-referenced pulls (Var 0.98/1.01/0.94/1.21/1.26 on the same tracks),
  which are dominated by material, not by hit resolution.
* **kurtosis 2.98, i.e. Gaussian**, against 4.9-11.9 for the truth-referenced
  pulls.  The reason is in the same export: `phcf_vgf`, the hit (Gaussian)
  share of a component's unit variance, has median **0.987** (p10 0.868).
  Only ~1.3 % of a per-hit innovation is material.  So the per-hit term should
  be strong on hit classes and weak on material -- exactly complementary to
  the reference-parameter term, and a prediction to check in the fits.
* `phresinflat` (`Gs_kk/pivot_k`, the conditioning of the sequential basis)
  median 4.07, p90 506, p99 4.6e10.  The last one or two KEPT components of a
  track are near-degenerate -- intrinsic to a hit-ordered basis, since by the
  rank boundary the residual is nearly determined.  They are still exact
  (gates 1c and 2 hold at float precision); a cut on `phresinflat` is a cut on
  the FIT'S COVARIANCE, not on the residual, and is available offline.

### COST AND SIZE (100 events / 200 tracks, one thread)
| | with per-group split | without |
|---|---|---|
| `phcf_msec` per track (19 components) | **978 ms** | ~340 ms (14 comps) |
| per component | 52 ms | 24 ms |
| `ph*` branches, compressed | **544 kB/track** | 25 kB/track |
| whole file | 630 kB/track | 111 kB/track |
Fit itself ~1 s/track (315 s wall for 200 tracks incl. ~110 s startup), so the
naive `d` separate `cvhcf` passes cost about as much as the fit.  The
concatenated-tau trick (`extract_res5.py`'s, every exponent primitive depends
on `(w, tau)` through the product alone) is therefore worth porting for a
40M-track production but was not needed here.

## EXTRA DELIVERABLE 1 -- DONE (20 000 tracks, `logs/xcum20k.log`, 17:00)

`xcum_perhit.py` on `perhit20k.npz`.  Table (3) reproduces the PROTOTYPE's
reference-parameter cross-cumulants to three digits from a completely
independent route (the maker's `phresbv` instead of the offline step records):
0.138 / 0.088 / 0.095 / 0.329 / 0.279 / 0.712 -- identical.

**Gaussian level (the answer to "are they correlated"): NO, exactly.**
per track algebraic `|sum_b A_b[k].A_b[qp]|` median **2.14e-9**, p99 1.5e-8,
max 8.3e-6 (this IS `F^T R = 0`).  Ensemble `corr(z_k, z_qp)` by hit position:
-0.0005 / +0.0015 / -0.0076 / -0.0086 / -0.0063 / +0.0030 / +0.0132 against a
1/sqrt(N) of 0.0043-0.0094 -- consistent with zero in every bin.

**Fourth order: they DO share non-Gaussianity, and it depends on where the hit
is.** `kappa(z_qp,z_qp,z_k,z_k)/sqrt(kappa4 kappa4)`, per track:

| relpos | 0.00-0.12 | 0.12-0.25 | 0.25-0.38 | 0.38-0.50 | 0.50-0.62 | 0.62-0.75 | 0.75-0.88 | ALL |
|---|---|---|---|---|---|---|---|---|
| median | 0.395 | 0.594 | **0.654** | 0.590 | 0.331 | 0.118 | **0.052** | **0.464** |
| p90 | 0.690 | 0.802 | 0.817 | 0.790 | 0.683 | 0.337 | 0.160 | 0.765 |

i.e. the INNER-to-middle hits share most of their fourth cumulant with the q/p
pull and the outermost almost none -- which is the material distribution seen
from the q/p functional's own influence.  By hit class it splits by
STRIP MULTIPLICITY/charge rather than by layer: the `_hi` classes 0.56-0.61,
the `_lo` ones 0.25-0.42, pixels 0.34 (y) to 0.56 (x).

**Adjacent per-hit innovations**: median **0.505**, p90 0.902 (247 732 pairs),
rising from 0.34 at the innermost pair to 0.63 in the middle of the track.
That is the composite-likelihood sizing between adjacent innovations: the
product form drops a fourth-cumulant correlation of ~0.5 between NEIGHBOURS.

READ IT RIGHT: a per-hit innovation is only ~3 % material (`phcf_vgf` median
0.968), so a 0.46 correlation of the fourth cumulants is 0.46 of a small
number.  The consequence is measured, not assumed, by the same-track joint
sandwich (`--comps all`).

Figures: `xcum_corr_zqp_vs_relpos.pdf`, `xcum_ref0_vs_relpos.pdf`,
`xcum_adjacent_vs_relpos.pdf`, `xcum_ref0_by_class.pdf` in
`~/public_html/cvh/260910_perhit/` (`plot_xcum.py`).

## THE PRODUCTION
`resolution_trackres_mugun_ul16_260910_perhit`, launched 2026-09-10 13:44 from
`perhit/run_prod.sh`, output
`/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_mugun_ul16_260910_perhit/task_*/globalcor_resclosure_*.root`.
SAME input sample and selection as `resolution_trackres_mugun_ul16_260903x_m0`
(`simprod/filelist_mugun_ul16.txt`, generalTracks, ideal geometry, default
field, CgfQoPMode=0, fitFromGenParms=False, tight G4e stepper), so results are
directly comparable.  160 tasks, ~2000 tracks each => ~320 k tracks, ~212 GB
(0.66 MB/track).

SPLIT OVER THREE NODES because one is ~2.4 h per 60-way block and 160 tasks
would be three blocks:
| node | tasks | parallel | log |
|---|---|---|---|
| submit50 | 0-59 | 60 | `runs/perhit/logs/prod_260910.log` |
| submit51 | 60-109 | 32 | `..._s51.log` |
| submit52 | 110-159 | 32 | `..._s52.log` |
Resume is on the per-task `.complete` sentinel, so re-running any range is
safe; do NOT run two ranges that overlap (both would write the same file).
**`run_prod.sh` executes from `CMSSW_15_0_19_patch2_dev2`.  Do NOT `scram b`
in that area until every node reports ALL DONE.**

## THE OFFLINE CHAIN (generalises hitlik, does not fork it)
* `perhit/extract_perhit.py` -- tree -> npz.  Much smaller than
  `../extract_res5.py` because the maker now evaluates the exponents; what is
  left is a re-layout plus the two things that need the per-block influence
  VECTORS: the cross cumulants and the algebraic `F^T R = 0` check.
* `../hitlik_term.py` -- `load()` now DISPATCHES on `row_ptr` in the file and
  calls a new `load_perhit()` for the variable-ncomp layout; `arm_families`
  and `build` are untouched (they are row-based and do not care).  `comps` is
  now a SPEC resolved by the loader: a digit string still indexes the fixed 5
  truth-referenced components, and `"hit"` / `"ref"` / `"ref0123"` / `"all"` /
  `"hit:N"` select from the per-hit file.
* `../make_hitlik_card.py`, `../fisher_cmp.py` -- two lines each: pass `comps`
  through instead of parsing it, and iterate over the components PRESENT.
* `perhit/run_card.sh`, `perhit/run_fit.sh` -- the per-hit wrappers
  (`runs/perhit/{cards,fits}`).
* `perhit/gates.py` -- gates 1-5 straight off the tree.

Validated end to end on a 120-track smoke: extract -> card (60 params,
NLL(0) = 2200.42) -> `rabbit_fit.py` (runs; edmval 0.50, i.e. 120 tracks do
not constrain 60 parameters -- expected).

## THE OFFLINE CHAIN, VALIDATED END TO END (2026-09-10 14:17, 120-track smoke)
`perhit/run_all.sh <step>` drives it; every hitlik script now takes `R=`,
`NPZ=`, `QNPZ=`, `COMPS=` from the environment, so nothing is forked.
All 15 cards of the ladder build:
`ph_{cf,gauss,gaussq}` (per-hit), `ph_cf_ref` (the 5 truth-referenced),
`ph_cf_all` (BOTH on the same tracks), `ph_inj_*`, `ph_{mass,resmass}` and
their injections (the J/psi-gun mass term, 24 000 candidates, 42 parameters,
from `runs/matres/gun_groups_probe.npz`).  `rabbit_fit.py` runs.

### `genParms[4]` IS FIXED, so the reference term now has 5 usable components
The prototype had to drop it (`Var(z_4) = 4.0e6`, median pull +375).  On the
new export the five truth-referenced pull variances are
**0.98 / 1.01 / 0.94 / 1.21 / 1.26** -- z0 is usable, and `--comps ref` means
all five.

## EARLY RESULTS FROM THE SMOKE (120 tracks) -- to be redone at full statistics

### GATE 4, sharpened: the two terms are EXACTLY uncorrelated, per track
`F^T R = 0` is an identity, so `sum_b A_b[k] . A_b[qp] = 0` for every per-hit
component against the truth-referenced q/p one -- not a statistical statement.
MEASURED: median **2.2e-9**, p99 1.7e-8, max 3.5e-7 (float32).  The ensemble
`corr(z_k, z_qp)` is consistent with 0 within `1/sqrt(N)` in every hit-position
bin.  So **on MC their Fisher information ADDS at the Gaussian level.**

### The composite-likelihood approximation is the thing to watch
`xcum_perhit.py`, the same fourth-cross-cumulant correlation the prototype
measured.  It reproduces the prototype's reference-parameter table to the
third digit from a completely independent route (the maker's `phresbv` instead
of the offline step records):

| pair | q/p-lam | q/p-phi | q/p-d0 | lam-phi | lam-d0 | phi-d0 |
|---|---|---|---|---|---|---|
| here | 0.146 | 0.093 | 0.108 | 0.330 | 0.282 | 0.708 |
| prototype | 0.138 | 0.088 | 0.095 | 0.329 | 0.279 | 0.712 |

and gives the two NEW numbers:

| | median | p90 |
|---|---|---|
| per-hit innovation vs the truth-referenced q/p | **0.487** | 0.763 |
| ADJACENT per-hit innovations (k, k-1) | **0.516** | 0.896 |

by hit position (relpos = row/(n_meas-1)): 0.49 / 0.64 / 0.43 / 0.10 for the
four quintiles -- the innermost-to-middle hits share most of their
non-Gaussianity with q/p, the outermost almost none.

READ IT RIGHT: the components are uncorrelated (above); what is 0.49 is the
correlation of the FOURTH cumulants, i.e. of the non-Gaussian part, and a
per-hit innovation is only ~1.3 % material (`phcf_vgf` median 0.987) so its
fourth cumulant is small in absolute terms.  The consequence is that a PRODUCT
of the two terms over-counts the non-Gaussian information, which is exactly
what the joint sandwich (`ph_cf_all`, both component sets on the SAME tracks)
measures.  Note that hitlik's `joint` -- residual + the J/psi-gun MASS term --
is on DISJOINT samples, so there is nothing to over-count there; the
same-track joint is the informative one.

## THE CHAIN IS VALIDATED, AND THE SMOKE ALREADY PREDICTS THE HEADLINE
`fisher_cmp.py` (H and J, 60 params) and `efficiency.py` (the sandwich) both
run on the per-hit npz.  On 120 tracks -- far too few to quote, but the sign is
structural, not statistical -- the CF-over-Gaussian EFFICIENCY on the hit
classes is **1.003** (16-84 % 0.999-1.029), i.e. the full PDF buys nothing
over the chi2 for the per-hit components.

That is what `phcf_vgf` median **0.987** says it must be: a per-hit innovation
is 98.7 % Gaussian hit noise and only ~1.3 % material, so `cf` and `gaussq`
are nearly the same density and their estimators have nearly the same
variance.  The prototype's 1.8-2.1x came from the truth-referenced components,
which are MATERIAL dominated.  So the expected division of labour is:

* the per-hit (complement) term -> the HIT CLASSES, through the extra
  components, with the non-Gaussianity irrelevant;
* the truth-referenced term -> the MATERIAL amounts, where the non-Gaussianity
  is the whole point.

Both to be confirmed at production statistics.  `saturation.py` turns the
prior-free sandwich into `N_sat = n0 (sigma_free/sigma_prior)^2`, the track
count at which each parameter stops needing its prior.

## THE EXPORT BILL, MEASURED AT PRODUCTION SCALE (2026-09-10 17:06)
Read straight off `task_0000` of the production, **2000 tracks**, compressed
branch bytes / entries (the 200-track table further down agrees to 2 %):

| branch | kB/track compressed | raw |
|---|---|---|
| `phcf_grp_rad_im` | 87.25 | 94.23 |
| `phcf_grp_ioni_im` | 87.24 | 94.23 |
| `phcf_grp_del` | 85.57 | 94.23 |
| `phcf_grp_ioni_re` | 84.75 | 94.23 |
| `phcf_grp_ms` | 84.65 | 94.23 |
| `phcf_grp_rad_re` | 84.63 | 94.23 |
| **the six per-group exponent tables** | **514.1** | 565.4 |
| `phresbv` (the influence vectors) | 16.33 | 21.70 |
| `phcf_{ms,del,ioni_re,ioni_im,rad_re,rad_im}` flat | 28.2 | 29.9 |
| **all `ph*`** | **567.0** | |
| (`hesspackedv`, the quadratic term, for scale) | 57.88 | 99.24 |
| whole file | **630.2** | 736.1 |

Production total: 160 tasks x 2000 tracks = **320 000 tracks, 190 GiB**,
634 kB/track including the ROOT overhead.

**At 41 M tracks (7 M Z legs + 34 M J/psi legs):**
| variant | kB/track | TB at 41 M |
|---|---|---|
| everything as produced here | 567 | **23.2** |
| per-group split OFF (one exponent set, all groups summed) | 53 | **2.2** |
| ... + the card's own `--prune-frac 0.001` (discards 83 % of group rows) | ~140 | 5.7 |
| ... + rank-16 tau PCA on top of the pruning | ~35 | 1.4 |
| (the truth-referenced 5-component prototype, for scale) | 33.4 | 1.37 |
The per-group tables are 91 % of the bill and compress by only 9-10 % (they are
float32 noise to zlib), so the compression has to be ALGORITHMIC.

## THE EXPORT BILL, ITEMISED (measured per track, compressed, in the tree)
| branch | kB/track | raw kB/track |
|---|---|---|
| `phcf_grp_{ms,del,ioni_re,ioni_im,rad_re,rad_im}` | **507** (6 x ~84) | 6 x 93.3 |
| `phresbv` (the influence vectors) | 16.4 | 21.8 |
| `phcf_{ms,del,ioni_re,ioni_im,rad_re,rad_im}` (flat, per component) | 28.2 | 30.0 |
| `phresvarv` | 4.2 | 4.4 |
| `phcf_grp_vq{ms,io}` | 2.9 | 3.0 |
| `phcf_hit*`, `phresz/raw/row/hit/dim/cls/piv/inflat` | < 1 | |
| **all `ph*`** | **560** | 620 |
| (`hesspackedv`, the quadratic term, for scale) | 57.9 | 99.4 |
whole file 662 kB/track; 160 tasks x 2000 tracks = ~212 GB.

The per-group exponents are 90 % of it and compress by only 11 % -- they are
float32 noise to zlib -- so the compression path has to be ALGORITHMIC:
turning the per-group split off is 53 kB/track (2.2 TB at 41 M), the card's own
`--prune-frac 0.001` already discards 83 % of the group rows, and a rank-16
tau PCA is another factor 4.  `cost.py` prints the bill at the measured
multiplicities.

## TWO APPROXIMATIONS TO STATE WITH EVERY NUMBER
1. **The composite likelihood.**  The product of whitened marginals drops the
   joint cumulants.  The POINT ESTIMATE stays consistent; the quoted error does
   not, and the sandwich is the accounting -- `fisher_cmp.py` estimates `J` by
   BATCH MEANS over batches of whole tracks, so every within-track correlation,
   including the fourth cross-cumulants above, is inside `J`.  Two caveats:
   the batching is by ROW COUNT and the rows per track VARY, so a fraction
   `nbatch/ntrk` of tracks straddles a boundary and loses its cross terms
   (2 % at `nbatch = 200`, `ntrk = 10 000`); and the efficiency loss relative
   to the true joint density is not measured by any of this -- only the
   disjoint-subsample spread (`subfits`) is assumption-free.
   Note that at SECOND order the components are exactly uncorrelated, and the
   hit-class information is a second-order (variance) quantity, so the
   composite form mis-counts only the SHAPE information -- which is why the
   approximation is expected to be mild for the hit classes and worse for the
   material.
2. **The conditioning cut.**  `max_inflat` cuts ROWS on `Gs_kk/pivot_k`, the
   conditioning of the sequential whitening.  It is a cut on the FIT'S
   COVARIANCE and cannot bias the residual distribution, but it does select
   hit positions: at the default 1e4 it removes ~7 % of rows, concentrated in
   the LAST one or two components of a track (the ones nearest the rank
   boundary), i.e. the outermost hits.

## WHAT A DATA FIT STILL NEEDS: the residual MEAN
On MC with ideal geometry and the simulation's own field the mean of `z` is
zero and the term is a pure WIDTH/SHAPE term.  On data, alignment and field
enter through the MEAN: a shift `da` in a global parameter moves the
constraint residual by `J da` (`Jfull`, already built in the maker), so

    dz_k = W[:,k]^T J da ,

i.e. the per-track mean Jacobian is `D = W^T J`, the same `W` this export
already forms.  Two ways to carry it, and the numbers to choose between them:

* **export `D` per track.**  Its columns are NOT `npars` wide in practice: the
  field modes and the material groups are already collapsed to global blocks
  (50 + 42 = 92 columns) and the alignment block only touches the track's own
  ~17 modules (5 columns each, ~85).  So ~19 x 177 float32 = **13 kB/track**
  plus ~85 global indices, i.e. **0.6 TB at 41 M tracks** -- comparable with
  the exponents themselves, not with the raw step records.
* **or keep the means where they already work**: run the CF term (width and
  shape, means fixed at zero) on one track partition and the ordinary
  quadratic term -- which handles the means exactly, through `gradv` /
  `hesspackedv` -- on the DISJOINT remainder.  Costs no new export at all and
  no double counting, at the price of splitting the statistics.
The saturation number below decides between them.

## EXTRA DELIVERABLES (coordinator, 2026-09-10) — WG question on mass/hit-residual correlation
1. **Cross-dependence of the mass functional with the innovations.** On the same
   tracks, the fourth cross-cumulant correlation
   `kappa(z_qp,z_qp,z_k,z_k)/sqrt(kappa4(qp) kappa4(k))` between the whitened
   TRUTH-REFERENCED q/p pull (what the mass term uses; available on this MC)
   and every per-hit innovation `z_k`; per track median and p90, tabulated BY
   HIT POSITION along the track — the same `xcum` machinery hitlik used for
   the 4 reference components (q/p vs the others was 0.09-0.14 there).
   Plus the Gaussian-level check `Cov(z_qp, z_k) = 0` numerically (it is
   `F^T R = 0`; this is also gate 4).
2. **Sandwich on the JOINT card** (per-hit residual + J/psi-gun mass term, as
   hitlik's `joint`): sandwich/quoted for material groups and hit classes, and
   the joint's efficiency against each arm alone. A sandwich/quoted above 1 on
   the joint that neither arm shows alone is the signature of over-counting.
   This is the answer to "are there correlations between the mass term and the
   hit residuals and how are they accounted for".
