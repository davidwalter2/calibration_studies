# cfcompress — compressing the per-candidate resolution-CF exponents

## Purpose

The unbinned mass likelihood stores, per candidate, five log-CF exponents
(`Sms`, `Sio_re/im`, `Srad_re/im`, and `Sdel` for the track functional) on a
448-point standardized `t` grid: **8960 B/candidate = 89.6 GB at 1e7
candidates**. This study asks whether those 2240 floats can be replaced by a
few scalars per candidate **without moving the fitted mass scale `alpha` at the
1e-5 level**, and answers it end to end: representation, rank, the surviving
systematic at production statistics, and the cost.

The answer: **one joint rank-16 PCA basis over the concatenated families,
64 B/candidate (140x), `|dalpha| = 1.7e-7`, every fitted parameter within
0.01 sigma.** The independent `t`-grid axis gives a further free 1.8x by
truncation and 4x by decimation; the 64-point subset grid that came out of it
is the grid the in-maker `cvhcf` export now writes.

Read-only on all production caches; every script here is new.

## The object

For one candidate the five exponents are a single row of a
`(n x 2240)` matrix (5 families x 448 points). Compression is a rank-`r`
truncated PCA of that matrix — ONE joint basis over the concatenated families,
so each family stays its own column block of the same basis and each family
scale `k_f` still floats independently. Coefficients are stored float32 per
candidate; the `(r x 2240)` float64 basis is stored once.

The error metric is **`max_t W |dS|` per candidate**, `W = exp(Re S_tot)`
(times `|phi_K|` for the mass caches) — the integrand envelope, i.e. the error
weighted by where the quadrature actually has support. The PCA reconstruction
preserves `S(t = 0) = 0` EXACTLY (the `t = 0` column of the data matrix is
identically zero).

Two independent axes, measured separately:
* the **coefficient axis** — rank `r` of the PCA (`spectral.py`, `cfbasis.py`,
  `physbasis.py`, `nll_impact.py`, `shiftdecomp.py`);
* the **grid axis** — decimating/truncating the `t` grid (`gridtest.py`), which
  needs no basis and no reconstruction.

## How to run

```bash
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /work/submit/david_w/ZMass/calibration_studies/resolution/cfcompress
S=<a scratch directory with ~3 GB free>     # holds the 0.5-0.65 GB subsamples
```

Inputs (the four production caches, read-only):

```
/work/submit/david_w/ZMass/calibration_studies/resolution/runs/
  cf_masspairs_jpsigun_ul16_260903x_m0.npz   + cf_masskernel_jpsigun_ul16_260903x_m0.npz
  cf_masspairs_btojpsix_v3_260903x_m0.npz    + cf_masskernel_btojpsix_v3_260903x_m0.npz
  cf_trackres_mugun_lowpt_260903x_m0_k0.npz
  cf_trackres_mugun_ul16_260903x_m0_k0.npz
```

```bash
# 0. shuffled, seeded subsamples (rows 0..40000 = TRAIN, 40000..60000 = TEST,
#    used consistently everywhere).  UNCOMPRESSED npz: the production npz are
#    deflated and every d["Sms"] re-inflates ~0.5 GB.
for c in jpsigun btojpsix trk_lowpt trk_ul16; do
  python3 make_subsets.py --cache $c --n 60000 --seed 1234 --out $S/sub_${c}_n60000_s1234.npz
done

# 1. spectra / per-rank reconstruction error
python3 spectral.py --sub $S/sub_jpsigun_n60000_s1234.npz   --tag jpsigun   --out $S/spec_jpsigun.npz   --nfit 40000 --phik
python3 spectral.py --sub $S/sub_btojpsix_n60000_s1234.npz  --tag btojpsix  --out $S/spec_btojpsix.npz  --nfit 40000 --phik
python3 spectral.py --sub $S/sub_trk_lowpt_n60000_s1234.npz --tag trk_lowpt --out $S/spec_trk_lowpt.npz --nfit 40000
python3 spectral.py --sub $S/sub_trk_ul16_n60000_s1234.npz  --tag trk_ul16  --out $S/spec_trk_ul16.npz  --nfit 40000

# 2. physics dictionary vs PCA vs cumulant/Taylor, at equal scalars/candidate
python3 physbasis.py --sub $S/sub_jpsigun_n60000_s1234.npz --tag jpsigun --out $S/phys_jpsigun.npz --phik

# 2b. does a basis trained on cache A work on cache B
python3 transfer.py

# 3. end to end: refit the mass likelihood with compressed exponents
python3 nll_impact.py --tag jpsigun --mode validate  --nproc 12
python3 nll_impact.py --tag jpsigun --mode compress  --nfit 20000 --ntrain 40000 --nproc 6  --ranks 4,8,16,32,48
python3 nll_impact.py --tag jpsigun --mode fullshift --nproc 10 --ranks 8,16,32,48,64
python3 shiftdecomp.py --tag jpsigun --n 20000 --nproc 2 --ranks 4,8,16,32,48

# 4. the grid axis, and the cost model
python3 gridtest.py --tag jpsigun --nfit 20000 --nproc 6
python3 bench.py    --tag jpsigun --n 20000 --ranks 8,16,32,48,64

# 5. tables and figures
python3 tables.py ; python3 figs.py
```

Durable outputs: `results/` (all logs, the small npz, `tables.txt`, the json
rows). `parse_logs.py` recovers the json rows from a `compress` log that is
still running. Figures:
**`~/public_html/ZMass/cvh/260904_cfcompress/`** — `svd_spectra`,
`rank_error_eabs`, `rank_error_ewgt`, `weight_envelope`, `joint_vs_perfamily`,
`physbasis_vs_pca`, `nll_impact`, `param_shifts`, `fullshift`.

## Results

### Spectral rank (4 caches, 40 000 training candidates)

Smallest rank of the JOINT (all-families -> ONE coefficient vector) basis with
`max_t W|dS|` below threshold for EVERY candidate:

| cache | max < 1e-3 | max < 1e-4 | q999 < 1e-4 |
|---|---|---|---|
| jpsigun | 24 | 64 | 48 |
| btojpsix | 6 | 48 | 12 |
| trk_lowpt | 24 | 96 | 64 |
| trk_ul16 | 8 | 48 | 24 |

`Sms` always drives the rank. Per-block bases (jpsigun) need MS 24 + IO 24 +
RAD 2 = 50 scalars for `max < 1e-4`, i.e. the joint basis (48) is as good or
better at equal storage — confirmed end to end below. The float32 cache itself
has 1 ulp = **7.6e-6** at `max|S| = 66`, so 1e-4 is 13x above the file's own
precision and 1e-5 is not meaningful on these caches.

### The physics dictionary is exactly right and not parsimonious

* The FORM is exact: the full log-binned Levy dictionary (K = 192, columns
  `Int_bin (e^{ivt} - 1 - ivt) dv/v`, both signs of `v`) reproduces the cached
  ionization exponent to `max_t W|dS|` = **3.2e-5** (max) / 1.6e-5 (q999);
  radiative **1.2e-5**; the dilated-`gshape` dictionary (K = 128) gives MS
  **7.2e-5**.
* But with `k` greedily chosen nodes it needs **3-5x more scalars than PCA**
  for the same error (IO: `k = 32` -> 1.1e-4 where PCA `k = 12` is already
  1.5e-4 and `k = 32` is 7.6e-6). The population explores a ~10-dimensional
  family of Levy measures, while REPRESENTING an arbitrary member needs ~100
  nodes.
* The cumulant/Taylor baseline is useless — it saturates at 6e-2 (IO) /
  3.6e-1 (MS): the exponent is heavy-tailed and its Taylor series in `t` does
  not converge over `t` in [0, 14].

### The basis transfers, provided it is trained on a covering sample

q999 of `max_t W|dS|`, basis trained on rows 0-20000 of A, tested on
20000-40000 of B:

| train -> test | r=8 | r=16 | r=32 | r=48 |
|---|---|---|---|---|
| jpsigun -> jpsigun | 2.7e-3 | 7.5e-4 | 1.0e-4 | 5.8e-5 |
| jpsigun -> btojpsix | 1.0e-3 | 4.9e-4 | 8.2e-5 | 4.2e-5 |
| btojpsix -> jpsigun | 5.3e-2 | 2.6e-2 | 5.7e-3 | 2.8e-3 |
| trk_lowpt -> trk_ul16 | 6.8e-4 | 2.3e-4 | 6.4e-5 | 4.3e-5 |
| trk_ul16 -> trk_lowpt | 1.4e-2 | 4.5e-3 | 1.3e-3 | 1.1e-3 |

Training on the narrower sample and applying to the wider one is **50x worse**.

### The numpy NLL reproduces the published TF fit

`massnll_np.py` (families, window off, analytic gradient) at the published
minimum on the full 299 712-candidate `cf_masspairs_jpsigun_ul16_260903x_m0`
with `k_rad = 1`: `NLL = -588375.472844` against the published `-588375.47`
(gradient 5.3 there, consistent with the published parameters being quoted to
4 decimals). The full 300 k REFIT reproduces the published fit including the
errors:

| | alpha [1e-3] | k_hit | k_ms | k_ioni | NLL |
|---|---|---|---|---|---|
| this numpy fit | +0.244676 +- 0.017656 | 0.942554 +- 0.035350 | 1.010752 +- 0.006066 | 0.670982 +- 0.054485 | **-588375.472993** |
| published TF | +0.2447 +- 0.0177 | 0.9426 +- 0.0354 | 1.0108 +- 0.0061 | 0.6710 +- 0.0545 | -588375.47 |

### End-to-end impact, 20 000 held-out candidates

Reference (uncompressed, rows 40000..60000): `alpha = +0.252623 +- 0.067930
e-3`, `k_hit = 0.8535 +- 0.1319`, `k_ms = 1.0377 +- 0.0232`,
`k_ioni = 0.3475 +- 0.2059`, `NLL = -39399.814558`. (Consistent with the
full-sample +0.2447 +- 0.0177e-3; sigma scales as `sqrt(299712/20000) = 3.87`.)
`dalpha` is absolute; `s` = the 20 k statistical sigma.

| basis | B/cand | x | max\|dS\| | dNLL | dalpha | dk_hit | dk_ms | dk_ioni |
|---|---|---|---|---|---|---|---|---|
| pca_all r=4 | 16 | 560 | 5.7e-2 | +0.376 | -1.70e-6 (-0.025s) | +0.039s | -0.067s | +0.106s |
| pca_all r=8 | 32 | 280 | 7.8e-3 | +0.101 | +1.18e-7 (+0.002s) | +0.003s | -0.001s | -0.004s |
| pca_all r=16 | 64 | 140 | 3.0e-3 | -0.054 | -1.69e-7 (-0.002s) | +0.001s | -0.004s | +0.007s |
| pca_all r=32 | 128 | 70 | 3.0e-4 | +0.013 | -3.92e-8 (-0.001s) | +0.000s | -0.001s | +0.002s |
| pca_all r=48 | 192 | 47 | 2.0e-4 | +0.016 | -2.45e-8 (-0.000s) | +0.000s | -0.000s | +0.001s |
| pca_fam r=4/fam | 80 | 112 | 4.8e-3 | +0.203 | +2.00e-6 (+0.029s) | | | |
| pca_fam r=8/fam | 160 | 56 | 1.0e-3 | -0.085 | +3.67e-7 (+0.005s) | | | |
| pca_fam r=16/fam | 320 | 28 | 1.9e-4 | +0.027 | -3.84e-8 (-0.001s) | | | |
| pca_fam r=32/fam | 640 | 14 | 5.9e-5 | +0.004 | -1.13e-8 (-0.000s) | | | |
| pca_fam r=48/fam | 960 | 9 | 2.7e-5 | -0.000 | -1.62e-9 (-0.000s) | | | |
| levy k=2+2+2 | 24 | 373 | 2.5e+1 | -2.3e+5 | +4.7e-2 (DIVERGED) | | | |

**Target `|dalpha| < 1e-5` is met from r = 4; all four parameters are within
0.01 sigma from r = 8.** The per-family basis is WORSE at equal storage
(`pca_fam r=4/fam` costs 80 B and gives +2.0e-6, where the joint `pca_all
r=16` costs 64 B and gives -1.7e-7), confirming the spectral result.

The first-order estimate `dtheta = -H^-1 g_compressed(theta*_full)` agrees with
the refit to 3 digits: at r = 32 and r = 48 the refit `dalpha` reproduces the
prediction to -3.918e-5 vs -3.919e-5 and -2.446e-5 vs -2.447e-5 (in alpha[1e-3]
units), which closes the two methods against each other. The prediction can be
used instead of a refit.

### Does the shift survive at 1e7 candidates?

`shiftdecomp.py`: `dtheta = -H^-1 sum_i dg_i`; `H` grows like `n`, so the MEAN
of `dg_i` gives an n-independent systematic and the SPREAD gives a `1/sqrt(n)`
statistical part.

| basis | B/cand | dalpha (20k) | +- stat (20k) | at 1e7 |
|---|---|---|---|---|
| pca_all r=4 | 16 | -1.688e-6 | 2.14e-6 | -1.688e-6 +- 9.6e-8 |
| pca_all r=8 | 32 | +1.181e-7 | 1.06e-6 | +1.181e-7 +- 4.8e-8 |
| pca_all r=16 | 64 | -1.693e-7 | 3.99e-7 | -1.693e-7 +- 1.8e-8 |
| pca_all r=32 | 128 | -3.919e-8 | 8.97e-8 | -3.919e-8 +- 4.0e-9 |
| pca_all r=48 | 192 | -2.447e-8 | 3.28e-8 | -2.447e-8 +- 1.5e-9 |

At EVERY rank the measured shift is smaller than its own 20 k statistical
uncertainty: the n-independent systematic is consistent with zero and BOUNDED
by ~2e-6 (r=4), ~1e-6 (r=8), ~4e-7 (r=16), ~9e-8 (r=32). Even the most
aggressive rank is an order of magnitude inside the 1e-5 target, and the
apparent non-monotonicity of `dalpha` with `r` above is just this scatter.

### The extrapolation, confirmed on the full 299 712-candidate sample

Basis trained on the 60 000 shuffled subsample and applied to the WHOLE cache,
shift evaluated to first order at the uncompressed minimum
(`sigma_alpha(300k) = 1.766e-5`):

| basis | B/cand | max\|dS\| (300k) | dNLL(theta*) | dalpha | in sigma_alpha(300k) |
|---|---|---|---|---|---|
| pca_all r=8 | 32 | 1.25e-2 | -2.836 | **+2.83e-7** | +0.016 |
| pca_all r=16 | 64 | 4.03e-3 | -1.587 | -9.39e-8 | -0.005 |
| pca_all r=32 | 128 | 6.09e-4 | -0.052 | -8.65e-8 | -0.005 |
| pca_all r=48 | 192 | 2.42e-4 | +0.075 | -1.16e-8 | -0.001 |
| pca_all r=64 | 256 | 1.53e-4 | -0.006 | -8.80e-10 | -0.000 |

Consistent with the 20 k estimate (r=8: +1.2e-7 +- 1.1e-6 there; the 300 k
measurement's own statistical uncertainty is +- 2.7e-7), so the systematic at
r = 8 is ~3e-7 — **30x inside the 1e-5 target and ~1/60 of the full-sample
statistical error**. This is the number that licenses quoting the 20 k study
for 1e7 candidates.

Note `max|dS|` grew from 7.8e-3 (20 000 candidates) to 1.25e-2 (299 712), a
factor 1.6 for 15x the statistics: the WORST-CANDIDATE metric drifts slowly
upward with `n`, as a max statistic must, which is why the rank recommendation
is anchored on the NLL/alpha shift and not on the pointwise max.

### The grid axis

The integrand envelope `W(t)` is below 1e-12 beyond `t ~ 9` for all four
caches (`weight_envelope.pdf`), so truncating `TG` is free. `dalpha` absolute;
`s` = the 20 k sigma:

| grid | points | B/cand | x | dNLL(theta*) | dalpha |
|---|---|---|---|---|---|
| full (stride 1, t<=14) | 448 | 8960 | 1.0 | — | — |
| stride 1, t <= 8 | 256 | 5120 | 1.8 | -3.3e-7 | **+1.4e-13** |
| stride 2, t <= 14 | 224 | 4480 | 2.0 | -4.1e-4 | +9.5e-10 (+0.000s) |
| stride 2, t <= 8 | 128 | 2560 | 3.5 | -4.1e-4 | +9.5e-10 (+0.000s) |
| **stride 4, t <= 8** | **64** | **1280** | **7.0** | -1.05e-2 | **-1.61e-8** (-0.000s) |
| stride 4, t <= 6 | 48 | 960 | 9.3 | -1.00e-2 | -1.86e-8 |
| stride 8, t <= 14 | 56 | 1120 | 8.0 | -0.734 | -2.26e-6 (-0.033s) |
| stride 1, t <= 5 | 160 | 3200 | 2.8 | +0.111 | +7.2e-8 (+0.001s), `dk_hit` +0.0010 |
| stride 16, t <= 14 | 28 | 560 | 16.0 | -63.2 | -1.35e-5 (-0.199s) |

Truncation at `t = 8` is exactly free (1.4e-13); decimation by 4 costs
**-1.6e-8 = 1/1000 of the full-sample statistical error**; stride 8 and
truncation below `t = 6` start to cost (`k_hit` moves first). **The
`stride 4, t <= 8` row is the 64-point grid the in-maker `cvhcf` export
writes** (`tau_i = 4 i (14/447)`, `tau_63 = 7.8926`, a strict SUBSET of the
448-point offline grid so the two can be compared with no interpolation).

### Cost model

One core, `n = 20000`, extrapolated to 1e7 candidates (timings taken on a
machine at load ~400/768 cores; the ratios are the robust part):

* full: **8960 B/cand -> 89.6 GB**; NLL 23.1 us/cand (231 s on 1 core, ~14 s on
  16), NLL+grad 36.5 us/cand (364 s / ~23 s).
* r = 8 / 16 / 32 / 48 / 64: 32 / 64 / 128 / 192 / 256 B/cand -> 0.32 / 0.64 /
  1.28 / 1.92 / 2.56 GB (**280x / 140x / 70x / 47x / 35x** smaller).
* Reconstruction (a dense GEMM) costs 25 / 26 / 29 / 36 / 54 % of the NLL
  itself, i.e. ~4-8 s on 16 cores for 1e7. It TRADES 90 GB of streaming for
  ~0.7 Tflop of GEMM, and only the compressed form fits in GPU memory.

## Recommendation

* **Representation:** ONE joint rank-`r` PCA basis over the concatenated
  families, coefficients float32 per candidate, basis (`r x 2240` float64)
  stored once. Each `k_f` still floats independently because each family is its
  own column block of the same basis.
* **Rank `r = 16`** (64 B/candidate, 140x, 0.64 GB for 1e7) as the operating
  point: `|dalpha| = 1.7e-7`, every fitted parameter within 0.01 sigma, and the
  pointwise weighted exponent error <= 1.3e-3 (worst candidate) / 2.5e-4
  (q999). Go to r = 32-48 (128-192 B, 70-47x) only if a POINTWISE 1e-4
  guarantee on every candidate is required.
* **Grid:** truncate at `t = 8` always (free) and decimate by 4 unless a reason
  appears not to.
* **Not** the physics dictionary: exactly the right functional form, but 3-5x
  more scalars; keep it as the portable fallback.

## Defects found and fixed

* **The Levy quadrature arm diverges as a fitting basis.** The 6-scalar
  greedy-node representation drives `k_ms -> 1035` and the NLL to NaN, and BFGS
  then burns `maxiter = 200` per rank. The divergence is itself the result;
  `physbasis.py` answers the physics-basis question properly (by reconstruction
  error at equal scalars) and the end-to-end `levy` arm was dropped.
* **The float32 caches are only good to 7.6e-6** at `max|S| = 66`, so no
  compression claim below ~1e-5 is meaningful on them; rank recommendations are
  anchored on the NLL/alpha shift, not on the pointwise max.
* `nll_impact.py` refits the PCA basis for EVERY rank (a 2240x2240 Gram + eigh
  on 40 000 rows, ~45 s each). Caching `pca_fit(X, max(ranks))` once and slicing
  would cut the whole job by ~40 %. Not done.
* The scripts' default output directory is a hard-coded session scratchpad path
  (`SCRATCH` in `make_subsets.py` / `nll_impact.py`) that no longer exists;
  pass `--out` / `--sub` explicitly, as the commands above do.

## Open items

1. **btojpsix end to end** (the harder physics case, `k_ioni = -0.28`):
   `python3 nll_impact.py --tag btojpsix --mode compress --nfit 20000 --ntrain 40000 --nproc 6 --ranks 8,16,32`.
2. **The low-rank edge** (where it actually breaks): the same with
   `--ranks 1,2,3`.
3. `physbasis.py` on btojpsix and the track caches (optional; the jpsigun
   conclusion is unambiguous).
4. **The basis must be re-fitted whenever the model that generates the
   exponents changes** (Kokoulin on, a different radiative or hit model). That
   re-fit is one Gram + eigh, ~1 min on 40 000 rows, and needs no
   re-derivation of the physics — but it must not be forgotten.
5. **Everything here is measured at J/psi mass scale** (and mu-gun tracks at
   pT 2-60). The radiative family is tiny in these samples (`|Srad_re|` <= 0.1 %
   of `|Sms|` at `t = 2` even at genpt 30-60 GeV) and compresses to rank <= 2;
   its share grows with momentum, so **the rank must be re-measured on a
   Z-scale sample**, which has different sigma and exponent shapes.
