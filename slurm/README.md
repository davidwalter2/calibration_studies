# Slurm submission for the CVH stage-2 refit

Generic per-file Slurm array submission for any cmsRun config in CMSSW
10_6_26 that takes `input=<path>` via VarParsing. One array task = one
ALCARECO file. Single-threaded (Geant4e is not thread-safe in the CVH
chain), 1 CPU + 4 GB by default, 12 h walltime, on the `submit` partition.

## Files

- `submit.sh`     — wrapper that builds the array spec and `sbatch`es
- `array.sbatch`  — per-task SBATCH script; picks a file from `$FILELIST`
                    using `$SLURM_ARRAY_TASK_ID`, then drives `cmssw-el7`
- `run_one.sh`    — inner script (runs inside the el7 container): sources
                    `cmsenv` and `exec cmsRun "$CFG" input="$INPUT"`
- `filelists/`    — convention: keep input filelists here

## Quick start — local files

```bash
# 1. Build a filelist (one absolute path per line)
ls /scratch/submit/cms/dwalter/data/TkAlJpsiMuMu-21Feb2020_UL2016-v1/60000/*.root \
   > slurm/filelists/jpsi_2016_scratch.txt

# 2. Submit (test the wiring with --dry-run first)
slurm/submit.sh \
  --config   /work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/benchmark_io/bench_cmsrun_cfg.py \
  --filelist slurm/filelists/jpsi_2016_scratch.txt \
  --outdir   /ceph/submit/data/user/d/david_w/ZMass/cvh/jpsi_stage2_2016 \
  --max-running 50 \
  --dry-run
```

Drop `--dry-run` to actually submit. Per-task output lands in
`$OUTDIR/task_<NNNN>/globalcor_*.root`; Slurm stdout/stderr in
`$OUTDIR/logs/`.

## Quick start — xrootd (`/store/data/...` from CMS)

For inputs that aren't pre-staged locally, build a filelist of `root://`
URLs and let cmsRun stream them. The CVH config
(`bench_cmsrun_cfg.py`) auto-detects the `root://` prefix.

```bash
# 1. DAS -> filelist with the global redirector
> filelists/jpsi_2016_xrootd.txt
for era in F G H; do
  dasgoclient -query="file dataset=/Charmonium/Run2016${era}-TkAlJpsiMuMu-21Feb2020_UL2016-v1/ALCARECO" \
    | sed 's|^|root://cms-xrd-global.cern.ch/|' \
    >> filelists/jpsi_2016_xrootd.txt
done
wc -l filelists/jpsi_2016_xrootd.txt   # 250 across F/G/H

# 2. Refresh proxy with a long lifetime (8 days), THEN submit
voms-proxy-init -voms cms -valid 192:00

slurm/submit.sh \
  --config   /work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/benchmark_io/bench_cmsrun_cfg.py \
  --filelist slurm/filelists/jpsi_2016_xrootd.txt \
  --outdir   /ceph/submit/data/user/d/david_w/ZMass/cvh/jpsi_stage2_2016_xrootd \
  --max-running 50
```

`submit.sh` stages the proxy from `$X509_USER_PROXY` (defaulting to
`/tmp/x509up_u$UID`) to `/work/submit/$USER/.x509up_slurm.proxy` so the
worker nodes can reach it through the bind-mount, and warns if it has
< 24 h left. If `submit.sh` exits with no output and unset proxy var,
run `voms-proxy-init` and try again.

## Tunables

| Flag            | Default        | Notes                                   |
|-----------------|----------------|-----------------------------------------|
| `--time`        | `12:00:00`     | bump if Geant4e on full files runs over |
| `--mem`         | `4G`           | CVH peak RSS is ~1 GB; 4 G is headroom  |
| `--partition`   | `submit`       | the default 247 GB nodes are plenty     |
| `--max-running` | (unbounded)    | cap concurrency, e.g. 50, when reading from `/ceph` to keep the cluster polite |
| `--name`        | `cvh`          | sets `--job-name` and the log filename pattern |
| `--dry-run`     | off            | prints the sbatch command, doesn't submit |

## Storage choice

I/O benchmark on a single 2.2 GB file (see
`Analysis/HitAnalyzer/test/benchmark_io/`):

| Storage   | Cold sequential read    | Warm cache |
|-----------|-------------------------|------------|
| `/scratch` (NFS-NVMe) | ~4 GB/s     | ~17 GB/s   |
| `/ceph`    (CephFS-HDD) | ~10 MB/s  | ~12 GB/s   |

The cold-cache penalty on `/ceph` (~3-4 min for 2.2 GB) is amortized away
by hours of Geant4e per file, so reading directly from `/ceph` is fine for
production. Pre-stage to `/scratch` only if you can fit (it's ~2 TB free
shared) AND the job is short enough that 3 minutes matters.

## Cmsrun config requirements

The config must accept `input=<absolute path>` via `VarParsing.analysis`
and write its output ROOT file relative to the current directory.
`Analysis/HitAnalyzer/test/benchmark_io/bench_cmsrun_cfg.py` is a working
template (raw JPsi ALCARECO -> CVH refit, with optional `srcCandidates`
fallback to in-module pair selection — works on raw ALCARECO without a
prior candidate-producer step).
