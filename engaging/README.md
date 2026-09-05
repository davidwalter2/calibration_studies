# Running the CVH offline fits on MIT Engaging (ORCD)

Engaging is the GPU execution site for the **offline TensorFlow / rabbit** half of
the calibration: the unbinned mass fits (`cf_masslik_fit.py`, `rabbit_fit.py`), and
later the joint global fits. It is not a second submit — see §7.

Verified 2026-09-05 on L40S and H200: both reproduce the submit numbers to
**7.5e-15** relative, and an H200 runs the NLL **325x** faster than a (loaded)
submit node. **Use `-G h200:1`** — it is the only card the rabbit datacard fits
on, and it is ~4.7x faster than an L40S here.

---

## 1. Login

```bash
eng-master              # 8 h multiplexed master; normally silent (SSH key + Duo grace window)
eng 'timeout 30 sinfo -s'
```

If `eng-master` says authentication needs a human, **David has to run it himself**
in an interactive terminal (in Claude Code prefix the line with `!`) so the Duo
prompt is visible. Never add `BatchMode=yes` — it disables the very stage that
succeeds silently.

Wrap every remote command in `timeout`. `orcd-login.mit.edu` round-robins over
several login nodes, and one with a stuck NFS home hangs *every* command over the
master while TCP and the SSH banner still look perfectly healthy (seen 2026-09-04
on login009). Cure — it re-lands on a different node:

```bash
ssh -O exit engaging && eng-master
```

Batch jobs are unaffected by this; only the control channel is.

## 2. What lives where

| Path | What | Size |
|---|---|---|
| `~/orcd/pool/env/tf` | conda env: python 3.13, TF 2.21 + CUDA wheels | 5.6 GB |
| `~/orcd/pool/zmass/rabbit` | rabbit checkout, branch `unbinned-mass-term` | 4.2 MB |
| `~/orcd/pool/zmass/resolution` | `cf_*.py` + `runs/` npz caches | 3.3 GB |
| `~/orcd/pool/zmass/trackres` | staged `globalcor_resclosure_0.root` inputs | 1.8 GB |
| `~/orcd/pool/zmass/wums` | wums 0.1.12 master copy | 468 KB |
| `~/orcd/pool/zmass/{engaging,cards,fitresults,logs}` | scripts, datacards, output | — |
| `~/orcd/pool/tmp` | conda pkg + pip caches (deletable) | 3.8 GB |

Quotas — check with `cat ~/orcd/.quota`:

| Space | Limit | Backup |
|---|---|---|
| `$HOME` | 200 GB | yes, snapshots |
| `~/orcd/pool` | 1 TB | no |
| `~/orcd/scratch` | 1 TB | no, purged after 6 months idle |

**Everything heavy goes in pool**, never `$HOME` — the CUDA wheels alone are 5 GB.

## 3. Staging

Always push from submit; Engaging can pull nothing itself.

```bash
cd /work/submit/david_w/ZMass/calibration_studies/engaging
./stage_engaging.sh code      # rabbit bundle + cf_*.py + wums + these scripts
./stage_engaging.sh caches    # the ~3.5 GB npz caches, 4 parallel rsync streams
./stage_engaging.sh           # both
```

Measured 2026-09-04, submit -> Engaging, 4 parallel rsync streams over the `eng`
master:

| payload | size | wall | throughput |
|---|---|---|---|
| 4 mass-likelihood npz caches | 3.50 GB | 27 s | ~129 MB/s |
| 4 `globalcor_resclosure_0.root` off `/ceph` | 1.78 GB | 19 s | ~94 MB/s |
| `cf_*.py` + wums + rabbit bundle | 8.5 MB | ~2 s | — |

A single stream is per-stream limited near 50-90 MB/s, so parallelising is worth
it. rsync skips identical files, so re-running after a code edit costs seconds.

rabbit travels as a **`git bundle`** rather than a working-tree copy: one 5.6 MB
file, full history, and it does not depend on Engaging reaching github.com. On the
far side it is `git clone`d and `origin` repointed at `https://github.com/WMass/rabbit`.

For anything much larger, use **Globus** — both ends have managed collections, so
the transfer is server-to-server, asynchronous, resumable and checksummed, and it
neither holds an SSH session open nor routes through a login node:

- subMIT: search `SubMIT` in the Globus File Manager
- Engaging: *MIT ORCD Engaging Collection*, `ec54b570-cac5-47f7-b2a1-100c2078686f`
- Drive it from the web UI at globus.org — neither machine has the `globus` CLI.

**Coming back** — results to the usual web dir on submit:

```bash
D=~/public_html/cvh/$(date +%y%m%d)_engaging; mkdir -p $D
eng-master && rsync -a engaging:orcd/pool/zmass/fitresults/ $D/
cp ~/public_html/cvh/260814_cleanprop/index.php $D/index.php    # required in every figure dir
```

`cf_masslik_fit.py` defaults `--outpath` to `~/public_html/cvh/<YYMMDD>_masslikfit/`,
which on Engaging is an ordinary `$HOME` directory and is **not** web-served.
`masslik_fit_gpu.sbatch` therefore passes `--outpath $ZMASS/fitresults` explicitly;
do the same in any new job.

## 4. The environment

```bash
source ~/orcd/pool/zmass/engaging/setup_env_engaging.sh          # activate
~/orcd/pool/zmass/engaging/setup_env_engaging.sh --install       # rebuild, ~10 min
```

Versions are pinned to the `wmassdevrolling` CVMFS image used on submit so the two
sites compute the same numbers: **python 3.13, numpy 2.4.6, scipy 1.18.0,
TF 2.21.0**. `rabbit` is `-e --no-deps` from the pool checkout; `wums` is the exact
**0.1.12** tree copied from submit's mfs venv (0.2.0 is on PyPI but `plot_tools`
changed between them, so PyPI is the wrong source here).

The install runs fine on a login node — it is conda + pip, a few minutes of
download. Launch it detached so it is not tied to the SSH session:

```bash
eng 'bash -lc "cd ~/orcd/pool/zmass/engaging && setsid nohup ./setup_env_engaging.sh --install \
     > ~/orcd/pool/zmass/logs/env_install.log 2>&1 < /dev/null &"'
```

Three things bit us; all three are handled inside `setup_env_engaging.sh`, but
they will bite again in any env built by hand:

1. **`tensorflow[and-cuda]` does not put the `nvidia-*-cu12` wheel `lib/` dirs on
   `LD_LIBRARY_PATH`.** On a GPU node TF then finds the driver, fails
   `dlopen libcudart.so.12 / libcudnn.so.9`, logs *"Cannot dlopen some GPU
   libraries ... Skipping registering GPU devices"* and returns
   `list_physical_devices('GPU') == []` **with no exception**. Always *source* the
   setup script; putting the env's `bin` on `PATH` is not enough.
2. **`tf-keras`** — `tensorflow_probability` imports it on TF >= 2.16, and rabbit
   imports tfp, so `rabbit_fit.py` cannot start without it.
3. **`lz4`** (and `hdf5plugin`) — `wums/output_tools.py` imports `lz4.frame` at
   module scope and `rabbit_fit.py` imports `output_tools`.

Sanity check on a real GPU node:

```bash
eng 'bash -lc "srun -A mit_general -p mit_normal_gpu -G l40s:1 -c 4 --mem 16G \
     -t 00:15:00 ~/orcd/pool/zmass/engaging/probe.sh"'
```

## 5. Submitting

```bash
eng 'bash -lc "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general masslik_fit_gpu.sbatch gun --rabbit"'
eng 'bash -lc "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -G h200:1 masslik_fit_gpu.sbatch gun"'
eng 'bash -lc "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general --export=ALL,STAGES=2 masslik_fit_gpu.sbatch gun --rabbit"'
eng 'bash -lc "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general extract_cpu.sbatch 4"'
eng 'bash -lc "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general rabbit_cpu.sbatch"'
eng 'timeout 30 squeue -u david_w -o "%.10i %.14j %.9T %.10M %R"'
```

Always pass **`-A mit_general`** from scripts: Engaging's submit-time default-account
lookup is broken, and a non-login remote command never sources the `.bashrc` that
exports `SBATCH_ACCOUNT`.

Partitions that matter:

| Partition | Walltime | Per-user cap | Use for |
|---|---|---|---|
| `mit_normal_gpu` | 6 h (default 2 h) | 2 GPUs, 32 CPU, 515 GB | the mass / global fits |
| `mit_preemptable` | 2 days | 4 GPUs, 1024 CPU, 4 TB | long fits — requeued on preemption, so checkpoint |
| `mit_normal` | 12 h | 96 CPU, 386 GB (CPU only) | extractions, closures |
| `mit_quicktest` | 15 min | — | syntax / smoke |

GPUs are requested with `-G l40s:1` / `-G h200:1`, **not** `--gres`. Ignore
`sbatch --test-only` start-time predictions — they are worst case by hours; L40S
jobs have started in under a minute, but the queue does back up badly:
`mit_normal_gpu` was **235 jobs deep** on a Friday evening and two jobs never
started at all. **`mit_preemptable` (591 nodes) started every L40S and CPU job
within a minute** — for anything short or checkpointable it is the fast lane, not
the fallback. **H200 is contended everywhere**, though: the two H200 jobs waited
about five hours on `mit_preemptable`. Since H200 is the right card for rabbit,
submit and walk away rather than waiting interactively.

`masslik_fit_gpu.sbatch` has three stages, selectable with `--export=ALL,STAGES=...`:

| stage | what |
|---|---|
| 0 | `bench_masslik.py` — NLL / grad / Hessian timings **and values** (the cross-site check) |
| 1 | `cf_masslik_fit.py --model families`, `--krad 0` and `--krad 1` |
| 2 | build the rabbit datacard if absent, then `rabbit_fit.py -t 0 --unblind` |

`CARD=<path>` (via `--export=ALL,CARD=...`) overrides the datacard in stage 2.

**rabbit needs far more memory than `cf_masslik_fit.py`.** It builds the Hessian
with `tf.vectorized_map` (pfor) over the parameters, so the whole pfor batch is
live at once. On the 300k-candidate mass card that **OOMs on an L40S (46 GB)**,
and rebuilding the card with `--chunk 8192` only postpones it (5m48s -> 18m46s
before the same failure) — chunking is not the knob. The CPU run's 68 GB host
peak is the honest size of the problem: it fits in an H200's 141 GB (105 s) and
not in an L40S's 46 GB. **So run rabbit on `-G h200:1`**, with
`rabbit_cpu.sbatch` (`mit_normal`, 360 GB) as the 17x-slower fallback when no
H200 is free. `cf_masslik_fit.py` is unaffected — it loops the chunks in python
and accumulates on the host, so its peak is one chunk.

## 6. Measured performance

Gun cache, `--model families --krad 0`, n = 299712 candidates, float64 throughout.

| | submit CPU (32 thr, busy) | Engaging CPU (16 thr) | L40S | **H200** |
|---|---|---|---|---|
| NLL | 8.456 s | 0.584 s | 0.116 s | **0.026 s** |
| NLL + grad | 14.752 s | 1.100 s | 0.236 s | **0.050 s** |
| NLL + grad + Hessian | 41.045 s | 6.138 s | 1.450 s | **0.302 s** |
| full 4-par fit (6 Hessian evals) | 28.7 s | 38.4 s | 9.7 s | **2.9 s** |
| input load | 59.4 s | 69.5 s | 20.2 s | 20.1 s |
| `rabbit_fit.py` on the 5-par card | — | 1806 s (64 c) | OOM | **105.5 s** |

| speedup | NLL | +grad | +Hessian |
|---|---|---|---|
| H200 vs submit-CPU | 325x | 295x | 136x |
| L40S vs submit-CPU | 73x | 63x | 28x |
| H200 vs L40S | 4.5x | 4.7x | 4.8x |
| L40S vs *idle* Engaging CPU | 5.0x | 4.7x | 4.2x |

That last row is the honest *GPU* speedup; the 73x/325x are what you actually get
today, because submit is loaded (user/real ~ 11 of the 32 threads requested).
Both numbers matter — the big one is the reason to move here at all.

Numerical agreement, submit-CPU vs Engaging-L40S, same inputs:

| quantity | max relative difference |
|---|---|
| NLL | 2.0e-16 |
| gradient (4 components) | 7.5e-15 |
| Hessian diagonal | 7.5e-16 |

Nine orders of magnitude below the 1e-6 tolerance. Fitted values are identical to
every printed digit (`alpha = 0.250017 +- 0.017658` [1e-3], `k_hit = 0.943484`,
`k_ms = 1.010843`, `k_ioni = 0.688443`, NLL `-588388.626654`).

**rabbit** (5-par card, `k_rad` floating; compare against submit's `fam_kradf`)
agrees to **7.3e-8** on the parameters and **1.6e-15** on the NLL — the residual
is the difference between two minimisers (scipy `trust-exact` vs rabbit
`trust-krylov`, edm 2e-11), not between two sites. The H200 and CPU rabbit runs
agree with each other to 4.7e-14.

### Extraction (CPU path)

`cf_track_resolution.py --extract` runs here too — it is numpy + uproot over
already-produced `globalcor_resclosure_0.root` files and needs no CMSSW.
Measured on `mit_preemptable`, 8 cores, 4 files of `mugun_ul16_260903x_m0`:
**21.7 min per file** with 4 running concurrently (1315 s for all four), 20 MB of
npz per file. Submit does the whole 160-file production in 44 m 35 s at 160-way
concurrency, so Engaging is not faster per production — it is *extra* capacity
that leaves submit free. Budget ~12 min to stage the 67 GB first.

`extract_cpu.sbatch` takes the shard count as its only argument; raise it to 96
on `mit_normal` (the per-user CPU cap) for a real run, and expect per-file wall
to grow once that many shards share one node's memory bandwidth.

## 7. The CVMFS rule: CMSSW stays on submit

Engaging has **no `/cvmfs`, no `xrdcp`, no `gfal-copy`**, and no CMS data. So:

- **CMSSW / CVH / the Geant4e refit never run here.** They stay on submit, which
  has CVMFS and the ALCARECO on `/ceph`.
- Only the *offline* stage runs on Engaging: the `.npz` caches that
  `cf_masspairs` / `cf_masskernel` already extracted, the TF likelihood, rabbit —
  and the `--extract` step, which is plain numpy + uproot over already-produced
  `globalcor_resclosure_0.root` files and so needs no CMSSW at all.
- Anything CMS-data-shaped that genuinely needs a GPU belongs on **ml.cern.ch**,
  which mounts `/eos` and `/cvmfs` natively (1 GPU cap) — see the `ml-cern` skill.

Practical consequence: a change to the CVH fit means re-producing ROOT files on
submit and re-staging. A change only to the likelihood or the fit model is a
`stage_engaging.sh code`, which is seconds.

## 8. Files here

| file | what |
|---|---|
| `README.md` | this |
| `STATE.md` | current state, job ids, what is and is not done |
| `setup_env_engaging.sh` | build (`--install`) / activate the env |
| `stage_engaging.sh` | push code and/or caches from submit |
| `masslik_fit_gpu.sbatch` | GPU job: bench + the two fits + rabbit (`STAGES=`) |
| `extract_cpu.sbatch` | CPU job: `cf_track_resolution.py --extract` on N shards |
| `rabbit_cpu.sbatch` | CPU job: `rabbit_fit.py` with 360 GB host RAM (the no-OOM path) |
| `gpu_check.sbatch` | minimal "does TF see the GPU" job |
| `bench_masslik.py` | site-independent NLL/grad/Hessian timing + value dump |
| `probe_cuda.py`, `probe.sh` | CUDA dlopen diagnostic for an srun shell |
| `ref/bench_submit_gun.json` | the submit-CPU baseline |
