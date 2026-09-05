# Grid → ceph dataset transfers

Machinery for pulling a CMS dataset off the grid onto the submit ceph group
store, mirroring the LFN layout, with per-file verification and atomic publish.

First use: the Z→μμ arm of the CVH full-scale feasibility test.

```
dataset  /DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos
         /RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM
size     4.386 TB, 96,424,820 events, 1731 files, 201 blocks
source   T1_US_FNAL_Disk  (also T1_FR_CCIN2P3_Disk, T1_IT_CNAF_Tape; no T2_US_MIT copy)
dest     /ceph/submit/data/group/cms + <LFN>
```

## Design: why the final name is trustworthy

`xrdcp`'s exit code is not evidence of a good copy — with `--retry` it can leave
a truncated file behind and still exit 0. So every file goes to `<dest>.part.<pid>`,
is checked independently against the DAS **size and adler32** (`verify_file.py`,
streaming `zlib.adler32`), and only then `mv`'d to its final name. Since ceph is
POSIX and source and target are the same filesystem, that `mv` is a rename, i.e.
atomic. **A file present at its final name has been verified**, which is what
makes the job resumable and lets `make_filelists.py` run while the copy is still
going. `xrdcp --cksum adler32:<expected>` is also passed, as a cheap on-the-fly
first line of defence, but it is never the only check.

## Files

| file | role |
|---|---|
| `make_manifest.py` | DAS → manifest TSV (`lfn size nevents adler32 source`); picks the endpoint US-disk-first and skips tape replicas |
| `dy_miniaod_full.tsv` | complete dataset, 1731 files / 4.386 TB |
| `dy_miniaod_subset.tsv` | 8.5M-event priority head, 104 files / 0.387 TB, 2 blocks |
| `transfer_one.sh` | one file: fetch → verify → atomic publish → record |
| `transfer_dy.sh` | parallel driver over a manifest (`NSTREAM` concurrent) |
| `run_dy_transfer.sh` | two-phase launcher: priority subset first, then the full set |
| `verify_file.py` | independent size + adler32 check |
| `make_filelists.py` | published files → driver-ready filelists |
| `report.sh` | progress / completion accounting |
| `transfer_status.tsv` | append-only log: lfn, bytes, ok/FAIL, wall_s, MB/s, source, utc |

## Running / resuming

```bash
cd /work/submit/david_w/ZMass/calibration_studies/transfer
export X509_USER_PROXY=/tmp/x509up_u$(id -u)          # do NOT voms-proxy-init
nohup setsid ./run_dy_transfer.sh > logs/transfer_$(date +%y%m%d).log 2>&1 &
```

Re-running the same command after any interruption is the resume: published
files are skipped (fast path if already in `transfer_status.tsv`, otherwise
re-validated from disk), and anything missing or corrupt is refetched.

Progress at any time:

```bash
./report.sh                                    # full dataset
./report.sh dy_miniaod_subset.tsv              # priority subset only
tail -f logs/transfer_260905.log
```

## Extending to more files / another dataset

The manifest is the only thing that is dataset-specific:

```bash
# whole dataset
python3 make_manifest.py --dataset <DS> --out my.tsv
# smallest set of whole files reaching N events, from the fewest blocks
python3 make_manifest.py --dataset <DS> --out my.tsv --min-events 20000000
# force a particular door
python3 make_manifest.py --dataset <DS> --out my.tsv --source root://cms-xrd-global.cern.ch/
NSTREAM=12 ./transfer_dy.sh my.tsv
```

`--min-events` groups by block, takes the largest blocks and the largest files
first, so a truncated subset spans as few blocks as possible (the 8.5M subset
here lands in 2 of the 201 blocks).

## Throughput notes

Measured 2026-09-05, submit → T1_US_FNAL_Disk:

| host | concurrency | aggregate |
|---|---|---|
| submit82 (load avg ~375) | 1 × 4 substreams | 62 MB/s |
| submit82 (load avg ~375) | 12 × 4 substreams | ~160 MB/s |
| **submit50 (idle)** | 12 × 4 substreams | **~434 MB/s** |

**Pick an unloaded interactive node.** The NIC is 200 Gbps and was carrying
1.5 Gbps on submit82, so the wall was never the network — it was the host. The
same manifest, same source, same concurrency ran 2.7× faster on an idle node.
Check `uptime` before launching. If a single node still saturates, split the
manifest and give one half `--source root://ccxrootdcms.in2p3.fr/` (CCIN2P3 also
holds a complete disk replica) rather than raising `NSTREAM` past ~12.
Verification read-back costs ~18 s per 4 GB file (ceph read ~220 MB/s) and runs
inside the per-file wall time.

## Incident 2026-09-05: ceph client eviction on submit82

Mid-run, every path under `/ceph/submit` started returning `Permission denied`
on submit82 — including files the job itself had just written. `dmesg` showed

```
libceph: auth protocol 'cephx' msgr authentication failed: -13
```

i.e. the node's ceph client lost its cephx authentication and the whole mount
went away. It was node-local: submit50/51/80/81 all had healthy `/ceph/submit`
throughout, and every byte already published was intact and readable from them.

Recovery was simply to relaunch on another node (`scp` the proxy to that node's
local `/tmp` first — `/tmp` is not shared) and let the resume logic skip the
1000+ files already done.

Two guards were added afterwards so this fails loudly instead of silently:

* `transfer_dy.sh` refuses to start if `$DESTBASE/store` is not writable (exit 4).
* `transfer_one.sh` exits 3 with `[HALT]` if the destination directory is
  unreachable, rather than recording a per-file `FAIL`.

Without them a storage outage walks the whole manifest in minutes and marks
every file failed, which is noisy but — by design — not destructive: `FAIL`
rows never block a retry, since the resume decision is made from what is
actually on disk.

## Filelists for the CVH drivers

```bash
python3 make_filelists.py --manifest dy_miniaod_subset.tsv
```

writes into `../resolution/simprod/`:

* `filelist_dy_miniaod_260905.txt` — one `file:<path>` per line, **one file per task**
* `filelist_dy_miniaod_260905_paths.txt` — same, bare paths, for the existing
  `resolution/run_local_*.sh` drivers which pass the raw line to `cmsRun`
* `filelist_dy_miniaod_260905_evt3000.txt` — `file:<path>,<skipEvents>,<maxEvents>`

The third form exists because these MiniAOD files hold **~53k events each**
(median; max 88,479) — far more than the ~2–4k events per task used by the gun
productions. Chunking *whole files* cannot reach that granularity: one file is
already >10× a gun task. Only slicing within a file can, hence the explicit
`(skipEvents, maxEvents)` triplets. The driver needs to translate those into
`process.source.skipEvents` / `process.maxEvents.input`; if a coarser task is
acceptable, use the plain one-file-per-line list instead.

Only files already published are listed, so this can be re-run at any time to
pick up whatever has landed since.
