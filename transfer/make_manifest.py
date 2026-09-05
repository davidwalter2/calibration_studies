#!/usr/bin/env python3
"""Build a transfer manifest (TSV) for a CMS dataset from DAS.

Queries `dasgoclient -query="file dataset=..."` and emits one row per file:

    lfn  size  nevents  adler32  source

`source` is the xrootd endpoint prefix the copy driver should try first; it is
chosen from the dataset's site list with a US-disk-first preference.  DAS
sometimes splits a file's attributes across several JSON records (one per DBS
instance), so records are merged by LFN before writing.

Usage:
    python3 make_manifest.py --dataset <DS> --out dy_miniaod_subset.tsv
    python3 make_manifest.py --dataset <DS> --out sub.tsv --min-events 8500000
"""
import argparse
import json
import os
import subprocess
import sys

DASGOCLIENT = "/cvmfs/cms.cern.ch/common/dasgoclient"

# xrootd door preference: MIT first (free/local), then US T1 disk, then global.
SITE_ENDPOINT = {
    "T2_US_MIT": "root://xrootd.cmsaf.mit.edu/",
    "T1_US_FNAL_Disk": "root://cmsxrootd.fnal.gov/",
    "T2_US_Nebraska": "root://xrootd-local.unl.edu/",
    "T1_FR_CCIN2P3_Disk": "root://ccxrootdcms.in2p3.fr/",
}
GLOBAL_ENDPOINT = "root://cms-xrd-global.cern.ch/"
SITE_PRIORITY = ["T2_US_MIT", "T1_US_FNAL_Disk", "T2_US_Nebraska", "T1_FR_CCIN2P3_Disk"]


def das(query):
    env = dict(os.environ)
    env.setdefault("X509_USER_PROXY", "/tmp/x509up_u%d" % os.getuid())
    out = subprocess.run([DASGOCLIENT, "-query=" + query, "-json"],
                         capture_output=True, text=True, env=env, timeout=900)
    if out.returncode != 0:
        sys.exit("dasgoclient failed for %r:\n%s" % (query, out.stderr[:2000]))
    return json.loads(out.stdout)


def merged_files(dataset):
    """Return {lfn: {size, nevents, adler32, block_name, is_file_valid}}."""
    recs = {}
    for entry in das("file dataset=%s" % dataset):
        for f in entry.get("file", []):
            lfn = f.get("name")
            if not lfn or not lfn.startswith("/store/"):
                continue
            r = recs.setdefault(lfn, {})
            for k in ("size", "nevents", "adler32", "block_name", "is_file_valid"):
                if f.get(k) is not None:
                    r[k] = f[k]
    return recs


def pick_source(dataset):
    """Pick the preferred xrootd endpoint from the dataset's site replicas."""
    sites = set()
    for entry in das("site dataset=%s" % dataset):
        for s in entry.get("site", []):
            name = s.get("name")
            # skip tape: staging is not instantaneous and we want streaming reads
            if not name or s.get("kind") == "TAPE" or name.endswith("_Tape"):
                continue
            sites.add(name)
    for name in SITE_PRIORITY:
        if name in sites:
            return name, SITE_ENDPOINT[name]
    return ("global", GLOBAL_ENDPOINT) if not sites else \
        (sorted(sites)[0], GLOBAL_ENDPOINT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-events", type=int, default=0,
                    help="stop after the smallest set of whole files reaching "
                         "this many events (0 = the complete dataset)")
    ap.add_argument("--source", default=None,
                    help="override the endpoint prefix, e.g. root://cmsxrootd.fnal.gov/")
    args = ap.parse_args()

    site, endpoint = pick_source(args.dataset)
    if args.source:
        endpoint, site = args.source, "override"
    recs = merged_files(args.dataset)
    bad = [l for l, r in recs.items()
           if not all(k in r for k in ("size", "nevents", "adler32"))]
    if bad:
        sys.exit("incomplete DAS metadata for %d file(s), e.g. %s" % (len(bad), bad[0]))
    invalid = [l for l, r in recs.items() if r.get("is_file_valid") == 0]
    if invalid:
        print("[warn] %d file(s) flagged invalid in DBS; dropping them" % len(invalid))
        for l in invalid:
            recs.pop(l)

    # Group by block so a truncated subset spans as few blocks as possible, then
    # order blocks by descending event count: fewest whole files for a target.
    by_block = {}
    for lfn, r in recs.items():
        by_block.setdefault(r.get("block_name", ""), []).append(lfn)
    blocks = sorted(by_block, key=lambda b: -sum(recs[l]["nevents"] for l in by_block[b]))

    rows, tot_ev, tot_sz = [], 0, 0
    for b in blocks:
        for lfn in sorted(by_block[b], key=lambda l: -recs[l]["nevents"]):
            if args.min_events and tot_ev >= args.min_events:
                break
            r = recs[lfn]
            rows.append((lfn, r["size"], r["nevents"], r["adler32"], endpoint))
            tot_ev += r["nevents"]
            tot_sz += r["size"]
        if args.min_events and tot_ev >= args.min_events:
            break

    rows.sort()  # deterministic order for reproducible resumes
    with open(args.out, "w") as fh:
        fh.write("# dataset: %s\n" % args.dataset)
        fh.write("# preferred site: %s (%s)\n" % (site, endpoint))
        fh.write("# files: %d  events: %d  bytes: %d (%.3f TB)\n"
                 % (len(rows), tot_ev, tot_sz, tot_sz / 1e12))
        fh.write("#lfn\tsize\tnevents\tadler32\tsource\n")
        for r in rows:
            fh.write("%s\t%d\t%d\t%s\t%s\n" % r)
    print("wrote %s: %d files, %d events, %d bytes (%.3f TB), source %s"
          % (args.out, len(rows), tot_ev, tot_sz, tot_sz / 1e12, site))


if __name__ == "__main__":
    main()
