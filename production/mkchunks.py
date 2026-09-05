#!/usr/bin/env python3
"""Build the input filelist and the per-task chunk list for the 20M-event
UL16 J/psi MC CVH production.

Input is a scan file with "<path> <nevents>" per line, where nevents is -1 for
a file ROOT could not open.  THE SCAN IS NOT OPTIONAL: the repacked ALCARECO
carries a ~5 % tail of TRUNCATED files (zombie, "no keys recovered", typically
a few hundred MB instead of ~2 GB).  cmsRun runs with skipBadFiles=True, so a
truncated input is SILENTLY SKIPPED and the task writes a valid but empty
output -- the same silent-empty-output failure mode as a ceph EPERM.  Files
that fail to open are therefore dropped here, and the first NFILES *good*
files in sorted order are taken.

Each file is then split into NCHUNK equal event ranges (skipEvents/nEvents),
because one 53k-event file is 10+ h of Geant4e in a single task.

usage: mkchunks.py <scan.txt> [<scan2.txt> ...] --nfiles 410 --nchunk 4 \
                   --filelist OUT.txt --chunks OUT_chunks.txt
"""
import argparse
import math
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="+")
    ap.add_argument("--nfiles", type=int, default=410)
    ap.add_argument("--nchunk", type=int, default=4,
                    help="nominal chunks per file; the ACTUAL count is "
                         "ceil(nevents/target) clamped to [1, 2*nchunk]")
    ap.add_argument("--target-events", type=int, default=13500,
                    help="target events per task. Splitting every file into a "
                         "fixed 4 would make 240-event tasks out of the short "
                         "files in this sample (they range 959..62k events), "
                         "each still paying the ~3 min Geant4/geometry "
                         "startup. Sizing by events instead keeps every task "
                         "at roughly the same wall time. Rounded, not ceiled: "
                         "ceil would turn a 54001-event file against a 13500 "
                         "target into 5 chunks rather than 4. Rounding bounds "
                         "the largest chunk at 1.5*target.")
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--counts", default=None)
    a = ap.parse_args()

    seen, rows, nbad = set(), [], 0
    for s in a.scans:
        with open(s) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                path, n = line.rsplit(None, 1)
                n = int(n)
                if path in seen:
                    continue
                seen.add(path)
                if n <= 0:
                    nbad += 1
                    continue
                rows.append((path, n))

    # The scans are supplied in sorted-order segments and each is written in
    # order, so `rows` is already the sorted sequence with the bad files
    # removed. Sort again anyway: correctness must not depend on that.
    rows.sort(key=lambda r: r[0])
    if len(rows) < a.nfiles:
        sys.exit("only %d good files available, need %d (scan more)"
                 % (len(rows), a.nfiles))
    rows = rows[:a.nfiles]

    with open(a.filelist, "w") as fl, open(a.chunks, "w") as ch:
        nev_tot = nch = 0
        for path, n in rows:
            fl.write(path + "\n")
            k_n = int(round(n / float(a.target_events)))
            k_n = max(1, min(k_n, 2 * a.nchunk))
            c = int(math.ceil(n / float(k_n)))
            for k in range(k_n):
                skip = k * c
                nev = min(c, n - skip)
                if nev <= 0:
                    break
                ch.write("%s %d %d\n" % (path, skip, nev))
                nch += 1
                nev_tot += nev
    if a.counts:
        with open(a.counts, "w") as fc:
            for path, n in rows:
                fc.write("%s %d\n" % (path, n))

    print("scanned      : %d files (%d unreadable, dropped)" % (len(seen), nbad))
    print("selected     : %d good files" % len(rows))
    print("events       : %d  (mean %.0f/file)" % (nev_tot, nev_tot / len(rows)))
    print("chunks/tasks : %d  (mean %.0f events/task)" % (nch, nev_tot / nch))
    print("filelist     : %s" % a.filelist)
    print("chunks       : %s" % a.chunks)


if __name__ == "__main__":
    main()
