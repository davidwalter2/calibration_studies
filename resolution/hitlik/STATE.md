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
