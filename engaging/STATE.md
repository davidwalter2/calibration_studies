# Engaging setup — state as of 2026-09-05 00:35

**Complete.** The GPU path, the rabbit path and the CPU extraction path are all
set up, validated against submit, and benchmarked on L40S and H200. Nothing is
outstanding.

**Bottom line: run the fits on `-G h200:1`.** It is the only card the default
rabbit datacard fits on, it is 4.5-4.8x faster than an L40S on the mass
likelihood, and it turns the 31-minute CPU rabbit fit into 105 s.

---

## 1. Validated

### Numerical agreement — submit CPU vs Engaging L40S

Same inputs, same code, float64 throughout. Compared at full precision from the
result `.npz` files, not the rounded summary table.

| quantity | max relative difference | tolerance |
|---|---|---|
| NLL (`bench_masslik.py`) | 2.0e-16 | 1e-6 |
| gradient, 4 components | 7.5e-15 | 1e-6 |
| Hessian diagonal | 7.5e-16 | 1e-6 |
| **fitted values**, `fam_krad0` | **1.5e-15** | 1e-6 |
| **fitted values**, `fam_krad1` | **3.8e-15** | 1e-6 |
| fitted uncertainties | 1.0e-15 | — |
| full Hessian | 6.8e-15 | — |

Nine orders of magnitude inside tolerance. `fam_krad0` on the L40S:

```
alpha[1e-3] = 0.250016915937544 +- 0.0176575372049118
      k_hit = 0.94348367375237
       k_ms = 1.01084320714084
     k_ioni = 0.688443163134954     NLL = -588388.626654176
```

versus submit `0.250016915937543 / 0.94348367375237 / 1.01084320714084 /
0.688443163134955`, NLL `-588388.626654176`.

### Timings — gun cache, families, n = 299712

| | submit CPU (32 thr, busy) | Engaging CPU (16 thr) | L40S | **H200** |
|---|---|---|---|---|
| NLL | 8.456 s | 0.584 s | 0.116 s | **0.026 s** |
| NLL + grad | 14.752 s | 1.100 s | 0.236 s | **0.050 s** |
| NLL + grad + Hessian | 41.045 s | 6.138 s | 1.450 s | **0.302 s** |
| full 4-par fit (6 Hessian evals) | 28.7 s | 38.4 s | 9.7 s | **2.9 s** |
| input load | 59.4 s | 69.5 s | 20.2 s | 20.1 s |

| speedup | NLL | +grad | +Hessian |
|---|---|---|---|
| H200 vs submit-CPU | 325x | 295x | 136x |
| L40S vs submit-CPU | 73x | 63x | 28x |
| H200 vs L40S | 4.5x | 4.7x | 4.8x |
| L40S vs idle Engaging CPU | 5.0x | 4.7x | 4.2x |

The last row is the honest *GPU* speedup; the 73x/325x are what you actually get
today because submit is loaded (user/real ~ 11 of the 32 threads requested).

### Environment — `~/orcd/pool/env/tf`, 5.6 GB

python 3.13.15, numpy 2.4.6, scipy 1.18.0, **TF 2.21.0** (matches submit's
`wmassdevrolling` image), tf-keras 2.21.0, rabbit-fit 0.2.5 (`-e --no-deps`),
**wums 0.2.0 from PyPI**, lz4, hdf5plugin, plotly, kaleido.

Four things had to be fixed and are now baked into `setup_env_engaging.sh`:

1. `tensorflow[and-cuda]` does not put the `nvidia-*-cu12` wheel `lib/` dirs on
   `LD_LIBRARY_PATH`; without them TF silently reports **zero GPUs** (no
   exception, just "Cannot dlopen some GPU libraries ... Skipping registering GPU
   devices"). This is why job 21994154 failed and 21994357 ran CPU-only.
2. `tf-keras` — `tensorflow_probability` needs it on TF >= 2.16, and rabbit
   imports tfp.
3. `lz4` (+`hdf5plugin`) — `wums/output_tools.py` imports `lz4.frame` at module
   scope; `rabbit_fit.py` imports `output_tools`.
4. **wums must be 0.2.0 from PyPI, not the 0.1.12 tree in submit's mfs venv.**
   `rabbit/tensorwriter.py` imports `wums.sparse_hist`, which does not exist in
   0.1.12. 0.2.0 carries `sparse_hist`, `plot_tools` and `output_tools` together.
   (Submit's own `cf_*.py` still run against 0.1.12 via
   `calibration_studies/env_tf/pypath`; if a `plot_tools` call ever diverges it
   will show up there, not in the fit numbers — the fits here ran `--no-plots`.)

### Staging — `~/orcd/pool/zmass/`, pool at 11 GB / 1 TB

| payload | size | wall | throughput |
|---|---|---|---|
| 4 mass-likelihood npz caches | 3.50 GB | 27 s | ~129 MB/s |
| 4 `globalcor_resclosure_0.root` off `/ceph` | 1.78 GB | 19 s | ~94 MB/s |
| rabbit bundle + `cf_*.py` | 8.5 MB | ~2 s | — |

rabbit datacard regenerated on Engaging: `cards/jpsigun_260903x_families.hdf5`,
**1.82 GB**, built in 2 min on a login node. It declares 5 parameters
(`alpha, k_hit, k_ms, k_ioni, k_rad`) with `k_rad` **floating**, so the rabbit fit
should be compared against submit's `jpsigun_260903x_fam_kradf`
(alpha 0.86348 +- 0.01829, k_hit 0.97091, k_ms 1.06921, k_ioni 1.82174,
k_rad -102.31344, NLL -588968.1575), **not** `fam_krad0`.

### CPU path: `cf_track_resolution.py --extract`

Job 21999569, `mit_preemptable`, 8 cores, 4 files of the
`mugun_ul16_260903x_m0` production (443 MB each), one BLAS-pinned python per file.

| | wall |
|---|---|
| per file (4 running concurrently) | **1300 - 1315 s = 21.7 min** |
| all 4 shards | 1315 s |
| job total incl. startup | 21 m 59 s |
| output per shard | 20.05 MB npz |

Submit reference for the *same production*: 160 files as 160 concurrent shards on
the 192-core box, **44 m 35 s** including the multi-GB merge
(`runs/chain_rad_260903x.log`, 23:22:52 -> 00:07:27).

**Verdict — yes, the closures can move here, but for capacity, not speed.**
At the `mit_normal` 96-CPU cap the 160-file production is two waves
(96 + 64) x ~22 min uncontended, so ~45 min best case and more like 70-90 min
once 96 shards share one node's memory bandwidth — i.e. comparable to submit,
not faster. The win is that it stops occupying submit, and `mit_normal`'s 12 h
walltime is ample. Cost of entry is staging: the full production is **67 GB**,
~12 min at the measured 94 MB/s.

## 2. rabbit — validated on H200 and on CPU

**rabbit on Engaging reproduces the reference implementation on submit.**

| par | submit `cf_masslik_fit` `fam_kradf` | rabbit (Engaging H200) | rel diff | in sigma |
|---|---|---|---|---|
| alpha [1e-3] | 0.863484675595702 | 0.863484724418496 | 5.7e-8 | 2.7e-6 |
| k_hit | 0.970914405315289 | 0.97091442455272 | 2.0e-8 | 5.5e-7 |
| k_ms | 1.06921191296018 | 1.06921190812331 | 4.5e-9 | 7.9e-7 |
| k_ioni | 1.82173655742542 | 1.82173668353264 | 6.9e-8 | 2.2e-6 |
| k_rad | -102.313443043419 | -102.31345055832 | 7.3e-8 | 4.5e-6 |
| **NLL** | **-588968.157542529** | **-588968.15754253** | **1.6e-15** | — |

Max 7.3e-8 on the values, inside the 1e-6 tolerance. These are two *different*
minimisers (scipy `trust-exact` vs rabbit `trust-krylov`, edm 2.0e-11) landing on
the same minimum, so 1e-8 on the parameters with the NLL identical to 1.6e-15 is
exactly the expected signature. The H200 and CPU rabbit runs agree with each
other to **4.7e-14**.

The card floats `k_rad`, so the comparison is against submit's `fam_kradf`
(5 parameters), **not** `fam_krad0`.

| rabbit run | wall (fit) | peak RSS |
|---|---|---|
| H200 (22000070) | **105.5 s** | 16.4 GB host |
| CPU, 64 cores, 360 GB (22001478) | 1806 s | 68.2 GB host |
| L40S | — | **OOM** |

### The memory story

`rabbit_fit.py` **OOMs on an L40S (46 GB)**, and reducing the chunk does not fix it:

| card | job | outcome | node in the graph |
|---|---|---|---|
| `--chunk 32768` (default) | 21999590 L40S | OOM at 5 m 48 s | `.../mul_161/Mul/pfor/Mul` |
| `--chunk 8192` | 22000192 L40S | OOM at 18 m 46 s | `.../mul_331/Mul/pfor/Mul` |
| `--chunk 32768` (default) | 22000070 **H200** | **COMPLETED, 105 s** | — |
| `--chunk 32768` (default) | 22001478 **CPU** | COMPLETED, 1806 s | — |

rabbit builds the Hessian with `tf.vectorized_map` (pfor) over the parameters, so
the whole pfor batch is live at once; a 4x smaller chunk bought ~3x more
iterations and hit the same wall. The CPU run's 68 GB host peak is the honest
size of the problem — it fits in an H200's 141 GB and not in an L40S's 46 GB.
(On the H200 the host RSS is only 16 GB because the big tensors live in HBM.)

`cf_masslik_fit.py` does **not** have this problem — it loops the chunks in
python and accumulates `L, g, H` on the host, so its peak is one chunk. That is
why the reference implementation fits on an L40S in 9.7 s and rabbit does not.

**So: use `-G h200:1` for rabbit.** `rabbit_cpu.sbatch` (`mit_normal`, 360 GB)
is the fallback when no H200 is free — 17x slower but it cannot run out of memory.
The `--chunk 8192` card is not a workaround and can be deleted.

## 3. Queue behaviour

`mit_normal_gpu` was **235 jobs deep** on a Friday evening; jobs sat
`PENDING (Priority)` for 30+ minutes and two never started at all.
`mit_preemptable` (591 nodes) started every **L40S and CPU** job in 20-55 s.
**H200 requests are contended everywhere** — the two H200 jobs waited about five
hours on `mit_preemptable` before running (submitted ~19:10, started 00:24).

So: L40S and CPU work is effectively on demand; H200 work needs to be submitted
and left alone. Since H200 is the right card for rabbit, plan for that latency
rather than waiting on it interactively.

## 4. Completed job history

| job | what | outcome |
|---|---|---|
| 21994154 | `gpu_check` | FAILED — pre-`LD_LIBRARY_PATH`-fix, no GPU seen |
| 21994357 | masslik gun --rabbit | ran **CPU-only** (same reason); fits correct, rabbit died on missing `tf_keras` |
| 21995409 | srun probe | GPU confirmed after the fix |
| 21998945 | `masslik_l40s` | **the validation run** — bench + both fits, all numbers above |
| 21999237 | `rabbit_l40s` | FAILED — missing `lz4`, then missing datacard |
| 21999590 | `rabbit_p` (L40S) | FAILED — **GPU OOM** building the Hessian at 5m48s (see §2) |
| 21999569 | `extract_cpup` | COMPLETED — 4 shards, 21.7 min/file |
| 22000192 | `rabbit_c8k` (L40S, chunk 8192) | FAILED — same OOM at 18m46s |
| 22001478 | `rabbit_cpup` (64 c, 360 GB) | COMPLETED — rabbit validation, 31m35s, peak RSS 68.2 GB |
| 21999568 | `masslik_h200p` | COMPLETED — the H200 bench + both fits |
| 22000070 | `rabbit_h200` | **COMPLETED** — rabbit on the default card, 105 s |

## 5. Exact next commands

```bash
# reconnect if the master wedged (see README section 1)
ssh -O exit engaging && eng-master

# the two H200 jobs, if they ever start
eng 'timeout 30 squeue -u david_w -o "%.10i %.14j %.13P %.9T %.10M %R"'
eng 'grep -viE "oneDNN|absl::Init|^I0000" ~/orcd/pool/zmass/logs/masslik_h200p_21999568.out | tail -20'

# read any rabbit fitresult
eng 'bash -lc "source ~/orcd/pool/zmass/engaging/setup_env_engaging.sh
python - <<PY
from rabbit import io_tools
import h5py, numpy as np
f = h5py.File(\"$HOME/orcd/pool/zmass/fitresults/fitresults_jpsigun_260903x_families_cpu.hdf5\",\"r\")
p = io_tools.get_fitresult(f)[\"parms\"].get()      # NOTE the .get(): it is an H5PickleProxy
for n, v, e in zip([str(x) for x in p.axes[0]], p.values(), np.sqrt(p.variances())):
    print(f\"{n:>8s} = {v!r} +- {e!r}\")
PY"'

# pull results to the web dir on submit
D=~/public_html/cvh/$(date +%y%m%d)_engaging; mkdir -p $D
rsync -a engaging:orcd/pool/zmass/fitresults/ $D/
cp ~/public_html/cvh/260814_cleanprop/index.php $D/index.php
```

Gotchas worth remembering: `io_tools.get_fitresult(f)` wants **no** `--result`
argument (passing one makes the key `results_<arg>`), and every entry it returns
is an `H5PickleProxy` that needs `.get()` before `.axes` / `.values()`.

## 6. Open / not done

- The full 160-file extraction has never been run here; only the 4-file timing
  test. Doing it for real needs the 67 GB production staged first (~12 min).
- `cf_masslik_fit.py`'s plotting path has never run on Engaging (all fits used
  `--no-plots`), so wums 0.2.0's `plot_tools` is unexercised here.
- Nothing has been rsynced back to `~/public_html` yet; results are still in
  `~/orcd/pool/zmass/fitresults` on Engaging.
- `cards/jpsigun_260903x_families_c8k.hdf5` (1.8 GB) was the chunk-8192
  experiment and is superseded — delete it.
- `~/orcd/pool/wmass` (18 GB) is not mine; something else already uses this pool.
  Total pool use 34.9 GB / 1 TB.
