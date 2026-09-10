# perhit — the DATA version of the hit-residual CF likelihood

## RESUME HERE (2026-09-10, session start)

Generalising `../STATE.md`'s TRUTH-REFERENCED prototype (residual = refParms -
genParms, 4 usable components) to the **complement residual** that exists on
DATA: the part of the constraint residual the fit has NOT absorbed.

STATUS: design fixed, maker implementation starting.
Branch `perhit-residual-cf-260910` off `cvh-exports-260906` (ca6058d96fc) in
`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2`.
No production is running from that area (checked `squeue`, `condor_q`, `ps` on
submit50/51/52/82 on 2026-09-10) so it is safe to `scram b` there.

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
