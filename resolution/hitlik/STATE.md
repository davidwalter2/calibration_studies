# hitlik — a per-track CF likelihood of the TRACK-PARAMETER residual vector

Goal (David): build the per-track PDF from the CF method for the *residual
vector* (not just the scalar q/p), use it as a likelihood term, and MEASURE
what the full non-Gaussian densities buy over the Gaussian chi2 for the
material amounts and the hit-class resolutions.

## Step 0 — what the trees contain (decided 2026-09-09)

`resolution_trackres_mugun_ul16_260903x_m0` (320 k tracks, 20-60 GeV mu gun,
both charges, |eta| < 2.4, 17 valid hits median) exports, per resolution
block `b` (aligned with `reseigidx`):

    resinfbv[b]  =  B_b  =  W5[b]^T dV_b^{1/2}     (5 x 5, row-major, padded)
    W5           =  Vinv F C E5                   (influence of the noise on
                                                   ALL FIVE reference params)

so the reference-parameter residual is EXACTLY

    r = refParms - genParms = sum_b B_b u_b ,   u_b standardized, independent

MEASURED (200 tracks, task_0000): `sum_b B_b B_b^T` reproduces `refCov`
element by element to a median 1.1e-7 / p90 3.4e-6 relative (float32 storage),
and `(B_b B_b^T)_00 = resinfvarv[b]` to 1.4e-7. `refCov` is stored
UPPER-TRIANGULAR (the lower triangle is literally 0).

=> the residual vector this prototype uses is the **5 reference-state
parameters** (q/p, lambda, phi, d0, z0), truth-referenced (MC gen), whitened
by V = refCov.  Per-HIT residuals are NOT in this tree (`hitres2_*` has them
— `dxrecgen`/`dxsimgen`/`dxrecsim` — but carries no step records, so it has no
CF exponents; the two productions DO share (run,lumi,event) and could be
paired, but the per-hit influence weights a_{b,h} are not exported by either).
That is the export spec of step 5.

## Status
- [x] step 0: residual vector + influence identified and verified
- [ ] step 1: extraction
- [ ] step 2: parameters
- [ ] step 3: information comparison
- [ ] step 4: injection
- [ ] step 5: cost + export spec
- [ ] step 6: report

## Step 1 — extraction (2026-09-09)

`extract_res5.py` (+ `run_extract.sh`).  Per track: whiten the 5-vector
`r = refParms - genParms` by the LOWER Cholesky factor of `refCov` (so
component 0 IS the established q/p pull), form `a_{b,k} = (L B_b)[k]`,
`v_b^(k) = |a_{b,k}|^2` (sums to 1 over blocks, exactly), and evaluate the
per-material-group log-CF exponents at the per-component weight
`wstd_k = sqrt(v_pool^(k) / sq2)`.

* **the extended-grid trick**: all four exponent primitives depend on
  `(wstd, tau)` through the PRODUCT only, so ONE call per (block, group) on
  `tau_ext = concat_k(TG * wstd_k / wstd_ref)` gives all five components.
  MEASURED 0.63 s/track/worker for 5 components against `matres`'s ~1.9 s for
  ONE (the ditrack functional); 20 000 tracks in 705 s on 26 workers.
* **GATE (`--validate`)**: component 0 rebuilt the single-functional way
  (`sqrt(v/sq2)/sigma`) differs from the 5-component route by
  **max 8.3e-7** against `|S| ~ 10`.  The route reproduces the established
  q/p model exactly; the residual is the float32 storage of `B_b`.
* `sum_b B_b B_b^T` vs `refCov`: median 1.1e-7, p90 3.4e-6, max 2.3e-4
  relative.

### THE ONE THING THAT DOES NOT WORK: component 4
`genParms[4]` (the gen `dsz`) is computed in `ResidualGlobalCorrectionMakerG4e.cc:1147`
as `(vtx.z() - myBeamSpot.z()) * pt/p - (transverse)` with
`myBeamSpot = bsH->position(vtx.z())`, and `BeamSpot::position(z)` returns
`Point(x(z), y(z), z)` -- so `vtx.z() - myBeamSpot.z()` is **identically 0**
and `genParms[4]` is the transverse term alone (~1e-4 cm), while
`refParms[4]` is the fitted z0 (~ genZ, up to several cm).  Measured on 2000
tracks: `Var(z_4) = 4.0e6`, median `z_4 = +375`, against `Var(z_0..3) =
1.070 / 1.113 / 0.985 / 1.056`.
=> **this prototype uses 4 components (q/p, lambda, phi, d0)**, and the
maker fix is one line (`vtx.z() - myBeamSpot.z()` -> `vtx.z() - bs.z0()`,
or simply store the gen z0 alongside).

Cholesky variance inflation `V_kk/d_k` (median over 2000 tracks):
1.00 / 1.002 / 1.39 / 3.67 / 5.41 -- so the nesting is well conditioned;
`cond(refCov)` median 7.5e3.

`globalfit/extract.py`: the unconditional `Jpsi_jacMass` guard rejected a
SINGLE-track production even under `--no-mass`.  Moved inside the mass branch
(one-line fix, committed): the quadratic term over the same 318 390 mu-gun
tracks now builds in 45 s.

## Step 2 — parameters (2026-09-09)

Exactly the parameters `matres` uses, so a joint fit floats ONE set:

* `material_<group>` (42), `k_g` = ln of the group's material amount,
  entering the exponent as `S_f = S^fix + sum_g e^{k_g} S_{f,g}`;
* `hitres_<class>` (18), `eps_c` the LINEAR scale of that class's Gaussian
  variance share, `v_i = v_other + sum_c (1 + eps_c) v_{c,i}`.

Field and alignment are NOT parameters of this term.  On a truth-referenced
residual they move only the MEAN, and the mean of a whitened residual carries
no material or hit-resolution information -- so the residual term is a pure
width/shape term and field/alignment stay in the quadratic one.

## Step 3 — the three arms (`hitlik_term.py`)

| arm | what replaces the per-(row, group) exponent | model Var(z) |
|---|---|---|
| `cf` | nothing -- the extracted log-CF exponents | 1.013-1.077 |
| `gauss` | `-1/2 kappa2_g tau^2`, `kappa2 = -(16 S(t1) - S(2t1))/(6 t1^2)` off the SAME arrays (tau^4 term eliminated); imaginary parts dropped | 1.013-1.068 |
| `gaussq` | the variance the FIT used: `thp2` (Rossi) for MS, `ioni_sq2` for ionization, ZERO radiative and delta | **1.00000 exactly** |

VALIDATED numerically (`/tmp/valdens.py`, 60 tracks, 4 components): all three
densities integrate to 1.000000 and have mean 0.00000; `gaussq`'s variance is
1.00000 to 5 decimals for every component -- i.e. the fit's own Q-matrix
decomposition of `refCov` sums to unity, so `gaussq` IS the pull model the
quadratic hit-chi2 term assumes.  `cf` is 1.3-7.7 % wider: the Rossi-vs-Moliere
gap of NOTES 2026-08-16, seen here directly as a model variance.

## Step 1b — TWO DEFECTS FOUND IN THE FIRST 20 k EXTRACTION (2026-09-09)

1. **`phi` is an angle and the residual was not wrapped.** `genParms[2] =
   g->phi()` is in `(-pi, pi]`; the fitted `refParms[2]` is not, so a track
   near the branch cut has `r_2 = +-2 pi = 3.7e4 sigma(phi)`.  MEASURED: 8
   tracks in 20 000 (0.04 %), `max |z_2| = 24 011`, which alone made
   `Var(z_2) = 2.9e4` against a robust sigma of 0.93 -- and, through the
   Cholesky nesting, `Var(z_3) = 9.7e4` with `max |z_3| = 44 134`.
   `covdev` of those tracks is 3e-7, i.e. the block decomposition is perfect
   and the defect is purely the branch cut.  FIXED in `extract_res5.py`
   (`r[2] = (r[2] + pi) % 2pi - pi`).

2. **The Cholesky conditioning needs a guard.** `V_kk / d_k` (marginal over
   conditional variance) has median 1.00 / 1.00 / 1.39 / 3.67 / 5.41 but a
   tail; it is now exported (`inflat`) and `hitlik_term.load(max_inflat=...)`
   cuts on it.  This is a cut on the FIT'S COVARIANCE, not on the residual
   being measured, so it cannot bias the residual distribution.

The extraction also now stores the raw residual `rres` (5,) per track.

## The composite-likelihood approximation, SIZED (`xcum`)

The whitened components are uncorrelated by construction but NOT independent:
they are different linear functionals of the same non-Gaussian block noises.
The product-of-marginals likelihood therefore drops the joint cumulants.  The
leading one is the fourth, and for the MS blocks (which carry the
non-Gaussianity) its correlation
`kappa(z_j,z_j,z_k,z_k) / sqrt(kappa_4(j) kappa_4(k))` is, per track (median):

| pair | q/p-lam | q/p-phi | q/p-d0 | lam-phi | lam-d0 | phi-d0 |
|---|---|---|---|---|---|---|
| median | 0.138 | 0.088 | 0.095 | 0.329 | 0.279 | **0.712** |

So `q/p` is nearly independent of the other three in its fourth cumulant
(0.09-0.14), while `phi` and `d0` share 71 % of theirs -- unsurprising, they
are the same bending-plane measurement at two lever arms.  The composite
likelihood is therefore a good approximation for what `q/p` adds and a
NOTICEABLE one for the `phi`/`d0` pair: their joint information is
over-counted, and the honest reading is that `phi + d0` together contribute
less than the product form says.

## Step 5 (part) — THE EXPORT SPEC, and what a DATA version needs

### What this prototype consumed
Everything came from `resolution_trackres_mugun_ul16_260903x_m0`, which was
produced with `exportStepRecords_` ON, i.e. the 430 kB/candidate raw mode:

| branch | what it is | bytes/track |
|---|---|---|
| `resinfbv` | `B_b` = `W5^T dV_b^{1/2}`, 5x5 per block | 68 x 25 x 4 = **6.8 kB** |
| `msmoliv` / `ioniurbanv` / `radstepv` / `radstepspecv` | the raw step records the exponents are built from | ~430 kB |
| `refParms`, `refCov`, `genParms` | the residual and its covariance | 140 B |
| `reshitidx` + `hitDetId`/`hitUProj`/`clusterSizeX`/`clusterChargeBin` | the hit classes | ~0.5 kB |

The raw step records are what makes this offline-only.  The maker already
solved that problem once for the q/p functional (`exportCfExponents_`,
`cvhcf::trackExponents`, 1.4 kB/candidate); the residual-vector version is the
SAME evaluator run at `n_res` weights instead of one.

### The change in the maker
`cvhcf::trackExponents` takes the per-block scalar weight
`sqrt(v_b/sq2)/sigma`.  For the residual vector it must take `n_res` of them,

    w^(k)_b = sqrt( v^(k)_b / sq2 ),   v^(k)_b = | (L B_b)[k, :] |^2

with `L = inv(chol_lower(refCov))` -- both `B_b` and `refCov` are already in
scope at that point in `ResidualGlobalCorrectionMakerG4e.cc` (the `W5` block
around line 4650).  And because every exponent primitive depends on
`(w, tau)` through the PRODUCT alone, the `n_res` weights are ONE evaluator
pass on a concatenated `tau` grid -- measured offline at 0.63 s/track/worker
for five components against ~0.5 s for one, i.e. **the cost of the vector is
the cost of the scalar**, not `n_res` times it.

### Two maker defects this study found
* `genParms[4]` (`dsz`) uses `vtx.z() - bsH->position(vtx.z()).z()`, which is
  identically 0 -- the gen `z0` is not recoverable from the tree.
* `genParms[2] = g->phi()` is wrapped and `refParms[2]` is not, so the `phi`
  residual needs an explicit branch-cut wrap (offline here; better in the
  maker, where a `genParms` in the fit's own convention costs nothing).

### What a DATA version needs
This prototype is TRUTH-REFERENCED: `r = refParms - genParms`.  On data there
is no `genParms`, and the residual has to be the hits about the FITTED track --
whose vector lives in the `n_meas - n_free` space the fit has not already
absorbed.  That is exactly the object the quadratic hit-chi2 term is the
Gaussian approximation of, so **the data version of this term is the CF
generalisation of the hit chi2**, and it needs per track:

    r_h            the n_res residual components
    B_b            n_res x 5 per resolution block  (the influence)
    cfres_*        n_res x nfam x NTAU per-group exponents

`B_b` for `n_res = 5` is already written (6.8 kB); for `n_res = n_hits ~ 18`
it is 24.5 kB.  The exponents dominate -- see `cost.py`'s export bill.

## STEP 3 RESULT — the information comparison (2026-09-09, 20 000 tracks)

`fisher_cmp.py --no-hessian` (score covariance, 500 batches, PSD, rank 57/60)
+ `report.py`.  Priors: the parmtype-15 tier priors both terms carry in every
fit; the quadratic is marginalised over its MATERIAL block alone (like-for-
like: the residual term floats no field modes).

**(i) The full CF densities carry LESS information than the Gaussian.**
`I_CF / I_chi2`, marginal, 4 components:

| parameter | ratio |  | parameter | ratio |
|---|---|---|---|---|
| `material_bpix_support6` | 0.647 | | `hitres_pix_x_q0` | 0.358 |
| `material_bpix_services` | 0.724 | | `hitres_pix_x_q2` | 0.689 |
| `material_tib_support` | 0.751 | | `hitres_pix_y_q2` | 0.697 |
| `material_fpix_support` | 0.851 | | `hitres_str_N2_lo` | 0.732 |
| `material_tec_structure` | 0.902 | | `hitres_str_N3_hi` | 0.742 |
| `material_tob_support` | 0.922 | | `hitres_str_N3_lo` | 0.761 |
| median over the 10 informative groups | **0.85** | | median over 18 classes | **0.737** |

**(ii) WHY, from first principles** (`scaleinfo.py`).  The Fisher information
about a pure width, `I(ln s) = E[(1 + z dlnp/dz)^2]`, is 2 for a Gaussian and
`2 nu/(nu+3) < 2` for a Student-t: a heavier tail carries LESS information
about its own scale.  GATES: exact `N(0,1)` -> **2.0000**; `t_5` -> **1.2500**
against the analytic 1.25.  On the actual row-averaged residual densities:

| component | I(ln s) CF | Gaussian | ratio |
|---|---|---|---|
| q/p | 1.6445 | 2.0000 | **0.822** |
| lambda | 1.8658 | 2.0000 | 0.933 |
| phi | 1.6012 | 2.0000 | **0.801** |
| d0 | 1.8764 | 2.0000 | 0.938 |

i.e. the whole of the measured ratio is this one effect.  **The Gaussian
hit-chi2 is not conservative: it reports material errors that are optimistic
by 1/sqrt(0.85) = 1.08x to 1/sqrt(0.65) = 1.24x**, because it treats the
Moliere tail as if it were Gaussian.

**(iii) What the residual VECTOR buys over the q/p scalar** (`I_4comp/I_qp`):

| | material | hit classes |
|---|---|---|
| median | 1.00 (bpix_support6 **4.3**, bpix_services 1.7, tib_support 1.5) | **18.9** |
| range | 1.0 - 4.3 | 3.0 - 538 |

The hit-class resolutions are essentially UNMEASURABLE from `q/p` alone
(sigma(eps) 0.17-0.95) and well measured from the four components
(0.024-0.10).  That is the dominant gain of this whole exercise, and it comes
from the extra COMPONENTS, not from the non-Gaussianity.

**(iv) Where the CF does win: the tails.**  `tails.py`, 20 000 rows/component:

| P(\|z\| > 4) | data | CF | data/CF | Gaussian chi2 | data/chi2 |
|---|---|---|---|---|---|
| q/p | 0.00235 | 0.00248 | **0.95** | 0.00006 | **37** |
| lambda | 0.00215 | 0.00069 | 3.10 | 0.00006 | 34 |
| phi | 0.00365 | 0.00287 | **1.27** | 0.00006 | **58** |
| d0 | 0.00205 | 0.00068 | 3.03 | 0.00006 | 32 |

and at 5 sigma the chi2 is wrong by **1650-3400x** while the CF is within
0.8-4.  In the core (\|z\| > 1, > 2) all three arms agree with the data to
3-20 %.

**(v) NLL at MC truth** (lower = describes the data better):

| rows | CF | Gaussian (variance-matched) | Gaussian (fit's Q = the chi2) | dNLL/row |
|---|---|---|---|---|
| 20 000 (q/p only) | **28 376.25** | 28 660.35 | 28 643.17 | 0.0133 |
| 80 000 (4 components) | **114 064.76** | 115 332.10 | 115 308.25 | 0.0155 |

## STEP 4 RESULT — injection (2026-09-09)

`material_tib_support` scaled by `exp(ln 1.05)` = **5 % more material**,
applied to that group's exponents on the data side of whichever term is being
built.  SIGN: scaling the card's exponents by `exp(+k)` declares the model at
k = 0 to already have that much material, so the MLE moves by `-k`; the test
is `|shift|` against `|truth|`.  PRIOR SHRINKAGE: the group carries its 0.05
tier prior and the injection is about one prior sigma, so the posterior only
moves by `f = 1 - sigma_post^2/sigma_pri^2` of it.

| channel | shift | /truth | f_prior | prior-corrected /truth | pull | leak rms |
|---|---|---|---|---|---|---|
| residual vector, CF | -0.00083 | -0.341 | 0.348 | **0.979** | -0.01 | 0.026 |
| residual vector, fit's Q | -0.00083 | -0.341 | 0.362 | **0.943** | -0.04 | 0.027 |
| J/psi-gun MASS term | -0.00106 | -0.436 | 0.446 | **0.978** | -0.02 | 0.052 |

All three recover the 5 % injection to 2-6 % with a sub-0.05-sigma pull.

## STEP 5 RESULT — cost (4 000 tracks x 4 components, 1 CPU, nt = 64)

| | | per row | per track |
|---|---|---|---|
| NLL | 120 ms | 7.5 us | 29.9 us |
| NLL + gradient | 458 ms | 28.6 us | 114 us |
| one HVP | 2.27 s | 142 us | -- |
| full Hessian (60 HVPs) | 136 s | | |
| card in memory | 0.367 GB | | 91.6 kB |

One NLL+gradient over this production (320 k tracks) is **37 s**; over
7 M Z legs + 34 M J/psi legs = 41 M tracks, **1.3 h CPU** (one HVP 6.5 h CPU),
and the term is a dense `(chunk, nt)` matmul per family, so a GPU is 20-60x
that.  Same order as the mass term.

Export bill per track (float32, 13.7 groups and 9.0 hit classes per (track,
component) after pruning at 1e-3), rank-16 tau PCA:

| variant | total | at 41 M tracks |
|---|---|---|
| `q/p` only (status quo) | 6.7 kB | 0.27 TB |
| 4 reference-state components | **26.7 kB** | **1.10 TB** |
| 5 components | 33.4 kB | 1.37 TB |
| 18 per-HIT residuals | 120.3 kB | 4.93 TB |
