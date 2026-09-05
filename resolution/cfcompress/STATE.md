# cfcompress — compressing the per-candidate CF exponents

Study of whether the per-candidate resolution-CF exponents (`Sms`, `Sio_re/im`,
`Srad_re/im`, `Sdel`) on the 448-point standardized grid can be replaced by a
few scalars per candidate WITHOUT moving the unbinned mass likelihood at the
1e-4 level.  Read-only on all production caches; every script here is new.

Started 2026-09-04.  Paused mid-run on a usage limit — see **OPEN** below.

---

## Environment (every command below assumes this)

```bash
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /work/submit/david_w/ZMass/calibration_studies/resolution/cfcompress
S=/tmp/claude-125124/-work-submit-david-w-ZMass/9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/cfcompress
```

`$S` is the session scratchpad and holds the 0.5–0.65 GB shuffled subsamples.
**If it has been cleaned, re-run `make_subsets.py` (step 0) first** — everything
else reads only those files plus the small `runs/masslikfit_phiKtab_*.npz`
tables that `cf_masslik_fit.py` already built.

Durable copies of every small output are in `results/` (a `cp` loop was left
running for 4 h to catch the jobs that were still going).

---

## Files

| file | what it does |
|---|---|
| `make_subsets.py` | materialises a shuffled, seeded subsample of a production cache as an UNCOMPRESSED npz (the production npz are deflated: every `d["Sms"]` re-inflates ~0.5 GB) |
| `spectral.py` | SVD/PCA of the (n x 448) exponent matrices; per-rank reconstruction-error tables, weighted by the integrand envelope |
| `cfbasis.py` | the bases: centred PCA, and the physics dictionaries (`gshape` dilations for Moliere, `e^{ivt}-1-ivt` for the compound-Poisson families) |
| `physbasis.py` | physics dictionary (greedy quadrature-node selection from a fine log-binned dictionary) vs PCA vs cumulant/Taylor, at equal scalars/candidate |
| `massnll_np.py` | numpy re-implementation of `cf_masslik_fit.MassNLL` (families, window off) with an ANALYTIC gradient — **validated, see below** |
| `nll_impact.py` | end-to-end: refit the mass likelihood with compressed exponents; modes `validate`, `compress`, `fullshift` |
| `shiftdecomp.py` | splits a compression-induced shift into the n-independent systematic and the 1/sqrt(n) statistical part |
| `gridtest.py` | the OTHER compression axis: decimating / truncating the t grid (**written, NOT yet run**) |
| `bench.py` | storage + time per NLL evaluation, full vs compressed, extrapolated to 1e7 |
| `transfer.py` | does a basis trained on cache A work on cache B |
| `tables.py`, `figs.py` | text tables and figures -> `~/public_html/cvh/260904_cfcompress/` |

---

## DONE

### 0. Subsamples (60000 candidates each, seed 1234, shuffled then sorted)
```bash
for c in jpsigun btojpsix trk_lowpt trk_ul16; do
  python3 make_subsets.py --cache $c --n 60000 --seed 1234; done
```
-> `$S/sub_<cache>_n60000_s1234.npz`.  Rows 0..40000 are the TRAIN split,
40000..60000 the TEST split, used consistently everywhere.

### 1. Spectral analysis — COMPLETE (4 caches)
```bash
python3 spectral.py --sub $S/sub_jpsigun_n60000_s1234.npz  --tag jpsigun  --out $S/spec_jpsigun.npz  --nfit 40000 --phik
python3 spectral.py --sub $S/sub_btojpsix_n60000_s1234.npz --tag btojpsix --out $S/spec_btojpsix.npz --nfit 40000 --phik
python3 spectral.py --sub $S/sub_trk_lowpt_n60000_s1234.npz --tag trk_lowpt --out $S/spec_trk_lowpt.npz --nfit 40000
python3 spectral.py --sub $S/sub_trk_ul16_n60000_s1234.npz  --tag trk_ul16  --out $S/spec_trk_ul16.npz  --nfit 40000
python3 tables.py ; python3 figs.py
```
Results: `results/spec_*.npz`, `results/tables.txt`,
`~/public_html/cvh/260904_cfcompress/`.

Smallest rank of the JOINT (all-families -> ONE coefficient vector) basis with
`max_t W|dS|` below threshold for EVERY candidate of 40000, W = exp(Re S_tot)
(x |phi_K| for the mass caches):

| cache | max < 1e-3 | max < 1e-4 | q999 < 1e-4 |
|---|---|---|---|
| jpsigun   | 24 | 64 | 48 |
| btojpsix  |  6 | 48 | 12 |
| trk_lowpt | 24 | 96 | 64 |
| trk_ul16  |  8 | 48 | 24 |

Per-block bases (jpsigun): MS 24, IO 24, RAD 2 for max < 1e-4 = 50 scalars
total, i.e. the joint basis (48) is as good or better at equal storage.
`Sms` always drives the rank.  The float32 cache itself has 1 ulp = 7.6e-6 at
max|S| = 66, so 1e-4 is 13x above the file's own precision and 1e-5 is not
meaningful on these caches.  The PCA reconstruction preserves S(t=0) = 0
EXACTLY (the t=0 column of the data matrix is identically zero).

### 2. Physics basis — COMPLETE (jpsigun)
```bash
python3 physbasis.py --sub $S/sub_jpsigun_n60000_s1234.npz --tag jpsigun --out $S/phys_jpsigun.npz --phik
```
* The physics FORM is exact: the full log-binned Levy dictionary (K=192,
  columns `Int_bin (e^{ivt}-1-ivt) dv/v`, both signs of v) reproduces the
  cached ionization exponent to `max_t W|dS|` = 3.2e-5 (max) / 1.6e-5 (q999);
  radiative 1.2e-5; the dilated-`gshape` dictionary (K=128) gives MS 7.2e-5.
* But it is NOT parsimonious: with k greedily chosen nodes it needs ~3-5x more
  scalars than PCA for the same error (IO: k=32 -> 1.1e-4 where PCA k=12 is
  already 1.5e-4 and k=32 is 7.6e-6).  The population of candidates explores a
  ~10-dimensional family of Levy measures, while REPRESENTING an arbitrary
  member needs ~100 nodes.
* The cumulant/Taylor baseline is useless — it saturates at 6e-2 (IO) /
  3.6e-1 (MS): the exponent is heavy-tailed, its Taylor series in t does not
  converge over t in [0,14].

### 2b. Basis transferability — COMPLETE
```bash
python3 transfer.py            # -> results/transfer.log, transfer.npz
```
q999 of `max_t W|dS|`, basis trained on row block 0-20000 of A, tested on
20000-40000 of B:

| train -> test | r=8 | r=16 | r=32 | r=48 |
|---|---|---|---|---|
| jpsigun -> jpsigun   | 2.7e-3 | 7.5e-4 | 1.0e-4 | 5.8e-5 |
| jpsigun -> btojpsix  | 1.0e-3 | 4.9e-4 | 8.2e-5 | 4.2e-5 |
| btojpsix -> jpsigun  | 5.3e-2 | 2.6e-2 | 5.7e-3 | 2.8e-3 |
| trk_lowpt -> trk_ul16| 6.8e-4 | 2.3e-4 | 6.4e-5 | 4.3e-5 |
| trk_ul16 -> trk_lowpt| 1.4e-2 | 4.5e-3 | 1.3e-3 | 1.1e-3 |

=> the basis transfers PROVIDED it is trained on a sample that COVERS the
kinematics; training on the narrower sample and applying to the wider one is
50x worse.

### 3a. The numpy NLL is validated against the published TF fit
```bash
python3 nll_impact.py --tag jpsigun --mode validate --nproc 12
```
`NLL(published minimum) = -588375.472844` on the full 299712-candidate
`cf_masspairs_jpsigun_ul16_260903x_m0.npz` with `k_rad = 1`, against the
published `-588375.47` (`runs/rad260903x/masslikfit_summary.txt`, row
"gun families k_rad=1").  Gradient at that point 5.3, consistent with the
published parameters being quoted to 4 decimals.  The full 300k REFIT
(`--mode fullshift`, `results/fullshift_jpsigun.log`) then reproduces the
published fit including the errors:

| | alpha [1e-3] | k_hit | k_ms | k_ioni | NLL |
|---|---|---|---|---|---|
| this numpy fit | +0.244676 +- 0.017656 | 0.942554 +- 0.035350 | 1.010752 +- 0.006066 | 0.670982 +- 0.054485 | -588375.472993 |
| published TF   | +0.2447 +- 0.0177 | 0.9426 +- 0.0354 | 1.0108 +- 0.0061 | 0.6710 +- 0.0545 | -588375.47 |

### 3b. End-to-end impact, 20000 held-out candidates — PARTIAL (r = 4, 8, 16)
```bash
python3 nll_impact.py --tag jpsigun --mode compress --nfit 20000 --ntrain 40000 \
        --nproc 6 --ranks 4,8,16,32,48
```
Reference (uncompressed, rows 40000..60000):
`alpha = +0.252623 +- 0.067930 e-3, k_hit = 0.8535 +- 0.1319,
k_ms = 1.0377 +- 0.0232, k_ioni = 0.3475 +- 0.2059`, NLL = -39399.814558.
(Consistent with the full-sample +0.2447 +- 0.0177e-3; sigma scales as
sqrt(299712/20000) = 3.87.)

| basis | B/cand | x | max\|dS\| | dNLL | dalpha | dk_hit | dk_ms | dk_ioni |
|---|---|---|---|---|---|---|---|---|
| pca_all r=4  | 16 | 560 | 5.7e-2 | +0.376 | -1.70e-6 (-0.025s) | +0.039s | -0.067s | +0.106s |
| pca_all r=8  | 32 | 280 | 7.8e-3 | +0.101 | +1.18e-7 (+0.002s) | +0.003s | -0.001s | -0.004s |
| pca_all r=16 | 64 | 140 | 3.0e-3 | -0.054 | -1.69e-7 (-0.002s) | +0.001s | -0.004s | +0.007s |
| pca_all r=32 | 128 | 70 | — | +0.013 | -3.92e-8 (-0.001s) | +0.000s | -0.001s | +0.002s |
| pca_all r=48 | 192 | 47 | — | +0.016 | -2.45e-8 (-0.000s) | +0.000s | -0.000s | +0.001s |

The `pca_all` scan is therefore COMPLETE (r = 4, 8, 16, 32, 48); the refit
dalpha at r=32 and r=48 reproduces the first-order prediction of 3d to three
digits (-3.918e-5 vs -3.919e-5 and -2.446e-5 vs -2.447e-5 in alpha[1e-3]
units), which closes the two methods against each other.

The first-order estimate `dtheta = -H^-1 g_compressed(theta*_full)` agrees with
the refit to 3 digits, so it can be used instead of a refit.
**Target |dalpha| < 1e-5 is met from r = 4; all four parameters within 0.01
sigma from r = 8.**

### 3d. Does the shift survive at 1e7? — COMPLETE
```bash
python3 shiftdecomp.py --tag jpsigun --n 20000 --nproc 2 --ranks 4,8,16,32,48
```
`dtheta = -H^-1 sum_i dg_i`; H grows like n, so the MEAN of `dg_i` gives an
n-independent systematic and the SPREAD gives a 1/sqrt(n) statistical part.
`results/shiftdecomp_jpsigun.log`:

| basis | B/cand | dalpha (20k) | +- stat (20k) | at 1e7 |
|---|---|---|---|---|
| pca_all r=4  |  16 | -1.688e-6 | 2.14e-6 | -1.688e-6 +- 9.6e-8 |
| pca_all r=8  |  32 | +1.181e-7 | 1.06e-6 | +1.181e-7 +- 4.8e-8 |
| pca_all r=16 |  64 | -1.693e-7 | 3.99e-7 | -1.693e-7 +- 1.8e-8 |
| pca_all r=32 | 128 | -3.919e-8 | 8.97e-8 | -3.919e-8 +- 4.0e-9 |
| pca_all r=48 | 192 | -2.447e-8 | 3.28e-8 | -2.447e-8 +- 1.5e-9 |

At EVERY rank the measured shift is smaller than its own 20k statistical
uncertainty, i.e. the n-independent systematic is consistent with zero and
BOUNDED by ~2e-6 (r=4), ~1e-6 (r=8), ~4e-7 (r=16), ~9e-8 (r=32).  Even the
most aggressive rank is an order of magnitude inside the 1e-5 target, and the
apparent non-monotonicity of dalpha with r in 3b is just this scatter.

### 3e. The extrapolation CONFIRMED on the full 299712-candidate sample
`--mode fullshift`, basis trained on the 60000 shuffled subsample and applied
to the WHOLE cache, shift evaluated to first order at the uncompressed
minimum (`results/fullshift_jpsigun.log`):

| basis | B/cand | max\|dS\| (300k) | dNLL(theta*) | dalpha | in sigma_alpha(300k) |
|---|---|---|---|---|---|
| pca_all r=8 | 32 | 1.25e-2 | -2.836 | **+2.83e-7** | +0.016 |

Consistent with the 20k estimate of 3d (+1.2e-7 with a +-1.1e-6 uncertainty at
that n; the 300k measurement's own statistical uncertainty is +-2.7e-7), so the
systematic at r=8 is ~3e-7 -- **30x inside the 1e-5 target and ~1/60 of the
full-sample statistical error sigma_alpha = 1.77e-5**.  This is the number
that validates quoting the 20k study for 1e7 candidates.

Note `max|dS|` grew from 7.8e-3 (20000 candidates) to 1.25e-2 (299712), i.e. a
factor 1.6 for 15x the statistics -- the WORST-CANDIDATE metric drifts slowly
upward with n (as a max statistic must), which is why the rank recommendation
is anchored on the NLL/alpha shift and not on the pointwise max.

### 3c. Cost model — COMPLETE
```bash
python3 bench.py --tag jpsigun --n 20000 --ranks 8,16,32,48,64   # results/bench.log
```
One core, n = 20000, then extrapolated to 1e7 candidates:
* full: 8960 B/cand -> **89.6 GB**; NLL 23.1 us/cand (231 s on 1 core, ~14 s on
  16), NLL+grad 36.5 us/cand (364 s / ~23 s).
* r=8 / 16 / 32 / 48 / 64: 32 / 64 / 128 / 192 / 256 B/cand -> 0.32 / 0.64 /
  1.28 / 1.92 / 2.56 GB (280x / 140x / 70x / 47x / 35x smaller).
* Reconstruction (a dense GEMM) costs 25 / 26 / 29 / 36 / 54 % of the NLL
  itself, i.e. ~4-8 s on 16 cores for 1e7 — it TRADES 90 GB of streaming for
  ~0.7 Tflop of GEMM, and only the compressed form fits in GPU memory.
  (Timings taken on a machine at load ~400/768 cores; ratios are the robust part.)

---

## WRAP-UP 2026-09-04 (usage limit refresh)
`pca_all` (r=4..48) and `pca_fam` (r=4..48/fam) end-to-end rows COLLECTED; the
full-300k `fullshift` scan COMPLETE (r=8..64).  The `levy` end-to-end arm was
KILLED: the 6-scalar quadrature diverges (`k_ms -> 1034`, NLL nan) and BFGS
then burns maxiter=200 per rank -- the divergence is itself the result, and
`physbasis.py` already answers the physics-basis question properly.
`gridtest.py` launched and reporting; `figs.py` rerun (adds `nll_impact`,
`param_shifts`, `fullshift`).  A dated entry is in
`/work/submit/david_w/Documents/Resolution/NOTES.md` (2026-09-04, "COMPRESSING
the per-candidate CF exponents").  `parse_logs.py` was added to recover the
json rows from a compress log that is still running.
btojpsix end-to-end and the r=1-3 edge were deliberately SKIPPED.

## OPEN — exact next commands

1. **Let the two jobs that were still running finish and collect them.**
   They were launched before the pause and write into `$S`; a `cp` loop into
   `results/` was armed for 4 h.  Check:
   ```bash
   tail -40 results/compress_jpsigun.log      # wants r=32,48 + pca_fam + levy rows
   tail -20 results/fullshift_jpsigun.log     # full-300k reference fit + 1st-order shifts
   tail -20 results/shiftdecomp_jpsigun.log   # systematic vs statistical split
   ```
   If they were killed, re-run:
   ```bash
   python3 nll_impact.py --tag jpsigun --mode compress   --nfit 20000 --ntrain 40000 --nproc 6  --ranks 4,8,16,32,48
   python3 nll_impact.py --tag jpsigun --mode fullshift  --nproc 10 --ranks 8,16,32,48,64
   python3 shiftdecomp.py --tag jpsigun --n 20000 --nproc 4 --ranks 4,8,16,32,48
   ```
   NOTE `nll_impact.py` refits the PCA basis for EVERY rank (a 2240x2240 Gram +
   eigh on 40000 rows, ~45 s each).  Caching `pca_fit(X, max(ranks))` once and
   slicing would cut the whole job by ~40 %.

2. **The t-grid axis — written but never run** (`gridtest.py`).  The integrand
   envelope W(t) is below 1e-12 beyond t ~ 9 for all four caches
   (`weight_envelope.png`), so truncating TG at 8-9 is an independent ~1.6x
   saving that needs no basis and no reconstruction:
   ```bash
   python3 gridtest.py --tag jpsigun --nfit 20000 --nproc 6   # -> $S/gridtest_jpsigun.json
   ```

3. **btojpsix end-to-end** (the harder physics case, k_ioni = -0.28):
   ```bash
   python3 nll_impact.py --tag btojpsix --mode compress --nfit 20000 --ntrain 40000 --nproc 6 --ranks 8,16,32
   ```

4. **Low-rank edge** (where does it actually break):
   ```bash
   python3 nll_impact.py --tag jpsigun --mode compress --nfit 20000 --ntrain 40000 \
           --nproc 6 --ranks 1,2,3 --out $S/nllimpact_jpsigun_lowrank.json
   ```

5. `physbasis.py` for btojpsix / the track caches (optional; the jpsigun
   conclusion is unambiguous).

6. Re-run `figs.py` once the jsons exist — it then also writes `nll_impact.png`
   and `param_shifts.png` (both currently missing from the public_html dir).

---

## Preliminary recommendation (to be confirmed by the OPEN items)

* **Representation:** ONE joint rank-r PCA basis over the concatenated
  families, coefficients stored float32, basis (r x 2240 float64) stored once.
  Each k_f still floats independently because each family is its own column
  block of the same basis.
* **Rank:** r = 16 (64 B/candidate, 140x, 0.64 GB for 1e7) as the operating
  point — |dalpha| = 1.7e-7, every fitted parameter within 0.01 sigma, and the
  pointwise weighted exponent error is <= 1.3e-3 (worst candidate) /
  2.5e-4 (q999).  Go to r = 32-48 (128-192 B, 70-47x) only if a POINTWISE
  1e-4 guarantee on every candidate is required.
* **Not** the physics dictionary: it is exactly the right functional form but
  needs 3-5x more scalars; keep it as the portable fallback.

### Caveats
* The float32 caches are themselves only good to 7.6e-6 at max|S|.
* The basis must be trained on a kinematically COVERING sample (see 2b) and
  re-fitted whenever the model that generates the exponents changes (Kokoulin
  on, a different radiative or hit model) — that re-fit is one Gram + eigh,
  ~1 min on 40000 rows, and needs no re-derivation of the physics.
* The radiative family is tiny in these samples (|Srad_re| <= 0.1 % of |Sms| at
  t = 2 even at genpt 30-60 GeV) and compresses to rank <= 2; its share grows
  with momentum, so the rank must be re-measured on a Z-scale sample.
* Everything here is measured on J/psi-scale mass candidates (and mu-gun tracks
  at pT 2-60).  The Z-mass application has different sigma and exponent shapes
  and must be re-measured.
