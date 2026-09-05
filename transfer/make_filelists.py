#!/usr/bin/env python3
"""Emit driver-ready filelists from a transfer manifest, for files already published.

Only files that exist on ceph at their final name are listed -- the copy driver
publishes a final name only after size+adler32 verification, so "present" means
"good".  That lets the filelist be regenerated while the bulk copy is still
running, and the Z ditrack test can start on whatever has landed.

Two products:

  <prefix>.txt          one `file:<path>` per line -- one file per task.
  <prefix>_evt<N>.txt   `file:<path>,<skipEvents>,<maxEvents>` per line, split so
                        each task gets ~N events.

The event-range form exists because MiniAOD files here hold ~53k events each,
far more than the ~2-4k events per task used by the gun productions; chunking
whole files cannot reach that granularity, only slicing within a file can.
A bare-path variant (no `file:` prefix) is written alongside for the existing
resolution/run_local_*.sh drivers, which pass the raw line through to cmsRun.
"""
import argparse
import os

DEFAULT_DEST = "/ceph/submit/data/group/cms"
SIMPROD = "/work/submit/david_w/ZMass/calibration_studies/resolution/simprod"


def read_manifest(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            lfn, size, nev, adler, src = line.rstrip("\n").split("\t")
            rows.append((lfn, int(size), int(nev)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--destbase", default=DEFAULT_DEST)
    ap.add_argument("--outdir", default=SIMPROD)
    ap.add_argument("--out-prefix", default="filelist_dy_miniaod_260905")
    ap.add_argument("--events-per-task", type=int, default=3000)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    present, missing, ev_tot, by_tot = [], 0, 0, 0
    for lfn, size, nev in rows:
        p = args.destbase + lfn
        if os.path.isfile(p) and os.path.getsize(p) == size:
            present.append((p, nev))
            ev_tot += nev
            by_tot += size
        else:
            missing += 1
    present.sort()

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, args.out_prefix)

    with open(base + ".txt", "w") as fh:
        for p, _ in present:
            fh.write("file:%s\n" % p)
    with open(base + "_paths.txt", "w") as fh:      # bare paths for run_local_*.sh
        for p, _ in present:
            fh.write("%s\n" % p)

    ntask = 0
    n = args.events_per_task
    with open("%s_evt%d.txt" % (base, n), "w") as fh:
        for p, nev in present:
            for skip in range(0, nev, n):
                fh.write("file:%s,%d,%d\n" % (p, skip, min(n, nev - skip)))
                ntask += 1

    print("%s: %d/%d files present (%d missing), %d events, %.3f TB"
          % (os.path.basename(args.manifest), len(present), len(rows), missing,
             ev_tot, by_tot / 1e12))
    print("  %s.txt            %d lines (1 file/task)" % (base, len(present)))
    print("  %s_paths.txt      %d lines (bare paths)" % (base, len(present)))
    print("  %s_evt%d.txt      %d lines (~%d events/task)" % (base, n, ntask, n))


if __name__ == "__main__":
    main()
