# Slurm submission for per-file CVH refits

Generic per-file slurm array submission for any `cmsRun` config that takes
`input=<path>` via `VarParsing.analysis`. One array task = one input file.
Defaults: 1 CPU, 4 GB, 12 h, partition `submit`.

This is the *generic* machinery, used for the single-track / cosmics / V0 /
pixel-hit productions. The two large dimuon productions have their own
chunk-list machinery (event ranges rather than whole files) and their own
reference — see `../production/PRODUCTIONS.md`.

## Files

| file | role |
|---|---|
| `submit.sh` | builds the array spec and `sbatch`es |
| `array.sbatch` | per-task script: picks a file from `$FILELIST` by `$SLURM_ARRAY_TASK_ID`, then drives `run_one.sh` |
| `run_one.sh` | inner script: sources `cmsenv` and `exec cmsRun "$CFG" input="$INPUT"` |
| `resubmit_failed.sh` | re-drive the FAILED / TIMEOUT / OUT_OF_MEMORY tasks of an array |
| `submit_cosmics_grads.sh` | cosmics CVH grads production (Run2016G+H NoBPTX `TkAlCosmicsInCollisions`), one task per repacked file |
| `submit_pixelhits_genclosure.sh` | pixel edge / single-column hit A/B on B→J/ψ+X MC (baseline vs `keepPixelEdgeHits`) |
| `submit_resolution_closure.sh` | single-track resolution gen-closure, one array per ionization-truncation α |
| `filelists/`, `filelist_*.txt` | input filelists |

`array.sbatch` dispatches on the area: an **el9-native** area (15_0+, detected by
`lib/el9_*`) runs directly on the host; a legacy **el7** area (10_6) goes through
the `cmssw-el7` container with the standard bind list. `CMSSW_AREA` is forwarded
into the container, so the dev area can move while a production array is running
without affecting queued tasks.

## Quick start — local files

```bash
# 1. build a filelist (one absolute path per line)
ls /ceph/submit/data/.../ALCARECO/*/*.root > slurm/filelists/jpsi_2016.txt

# 2. submit (test the wiring with --dry-run first)
slurm/submit.sh \
  --config   /work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/runCvhJpsi.py \
  --filelist slurm/filelists/jpsi_2016.txt \
  --outdir   /ceph/submit/data/user/d/david_w/ZMass/cvh/jpsi_stage2_2016 \
  --max-running 50 \
  --dry-run
```

Per-task output lands in `$OUTDIR/task_<NNNN>/globalcor_*.root`; slurm
stdout/stderr in `$OUTDIR/logs/`. Retry the failures of an array with

```bash
./resubmit_failed.sh <ARRAY_JOB_ID> <ORIG_FILELIST> <ORIG_OUTDIR> [extra submit.sh args…]
```

## Quick start — xrootd (`/store/...` from CMS)

For inputs that are not pre-staged, build a filelist of `root://` URLs and let
cmsRun stream them.

```bash
> filelists/jpsi_2016_xrootd.txt
for era in F G H; do
  dasgoclient -query="file dataset=/Charmonium/Run2016${era}-TkAlJpsiMuMu-21Feb2020_UL2016-v1/ALCARECO" \
    | sed 's|^|root://cms-xrd-global.cern.ch/|' >> filelists/jpsi_2016_xrootd.txt
done

voms-proxy-init -voms cms -valid 192:00       # refresh BEFORE submitting
slurm/submit.sh --config … --filelist filelists/jpsi_2016_xrootd.txt --outdir … --max-running 50
```

`submit.sh` stages the proxy from `$X509_USER_PROXY` (default
`/tmp/x509up_u$UID`) to `/work/submit/$USER/.x509up_slurm.proxy` so the worker
nodes reach it through the bind mount, and warns below 24 h. If it exits with no
output and an unset proxy variable, run `voms-proxy-init` and try again.

> **Never use a redirector for repacked inputs.** For the J/ψ MC ALCARECO the
> same LFN exists centrally with *different* content (split-99 vs our split-1),
> and reading the central one destroys the refit silently. See
> `../production/PRODUCTIONS.md` §8.1.

## Tunables

| flag | default | notes |
|---|---|---|
| `--time` | `12:00:00` | bump if Geant4e on full files runs over |
| `--mem` | `4G` | fine for single-track work; two-track dimuon tasks have a tail to **3.56 GB** at 1 thread — size those at 4.5–5 G |
| `--cpus` | `1` | CVH in 15_0 *is* multithreading-capable and scales ~3.9× at 4 threads; see `../production/PRODUCTIONS.md` §6 |
| `--partition` | `submit` | the default 247 GB nodes are plenty |
| `--max-running` | (unbounded) | cap concurrency, e.g. 50, when reading from `/ceph` |
| `--name` | `cvh` | sets `--job-name` and the log filename pattern |
| `--cmsrun-args` | (none) | extra VarParsing args as one space-separated string |
| `--cmssw-area` | `…/CMSSW_10_6_26_prod` | see below |
| `--dry-run` | off | prints the sbatch command, does not submit |

Slurm limits here: `MaxArraySize = 1001`, `MaxJobCount = 50000`, QOS `normal`
`MaxJobsPU = 5000`. The binding constraint in practice is fair share, not a
limit — two productions in the queue interleave rather than adding throughput.

## Storage choice

I/O benchmark on a single 2.2 GB file
(`Analysis/HitAnalyzer/test/benchmark_io/`):

| storage | cold sequential read | warm cache |
|---|---|---|
| `/scratch` (NFS-NVMe) | ~4 GB/s | ~17 GB/s |
| `/ceph` (CephFS-HDD) | ~10 MB/s | ~12 GB/s |

The cold-cache penalty on `/ceph` (~3–4 min for 2.2 GB) is amortised away by
hours of Geant4e per file, so reading directly from `/ceph` is fine for
production. Pre-stage to `/scratch` only if it fits *and* the job is short
enough that 3 minutes matters.

Node caveats: **submit82's ceph client is evicted** (EPERM on every
`/ceph/submit` path) and **submit60 has ceph but no AVX2**, so CMSSW 15_0 dies
there with an illegal instruction.

## Config requirements

The config must accept `input=<absolute path or root:// URL>` via
`VarParsing.analysis` and write its output ROOT file relative to the current
directory. The 10_6 production driver is
`Analysis/HitAnalyzer/test/runCvhJpsi.py` (reads raw `ALCARECOTkAlJpsiMuMu`
directly and falls back to the legacy in-module pair loop when `srcCandidates`
is unset); the 15_0 drivers are `runCvhJpsiGenMC.py`, `runCvhDimuonMiniAOD.py`
and `runCvhSingleTrack.py`.

## Dev / prod separation

`submit.sh` defaults `--cmssw-area` to a dedicated tree
`/work/submit/david_w/ZMass/CMSSW_10_6_26_prod/`, so the dev tree at
`…/CMSSW_10_6_26/` is free for active work without affecting queued or running
tasks. **A `scram b` in an area a production is running from relinks the `.so`
under the running jobs and they segfault in the same second.** Advance prod
when ready:

```bash
cd /work/submit/david_w/ZMass/CMSSW_10_6_26_prod/src
git fetch && git checkout <branch-or-hash>
eval $(scramv1 runtime -sh)   # SCRAM_ARCH=slc7_amd64_gcc700
scram b -j 8                  # incremental
```

The submission auto-appends the prod HEAD short hash to `--outdir`, so which
build produced any given output is recorded by the path alone.
