#!/usr/bin/env python3
"""Same-candidate comparison of two CVH task directories.

Two productions that differ only in `useIdealGeometry` read the same events
through the same pair-finding, so the CANDIDATE SET must agree up to fit
failures while every FITTED quantity differs. This reports both halves.

Candidates are keyed on ``(run, lumi, event, Muplustrk_pt, Muminustrk_pt)``:
the ``*trk_*`` branches are the PRE-REFIT reco track quantities, which are a
property of the input file and so are identical in both productions. Keying on
position within the event would not work -- one dropped candidate shifts every
later rank.

usage:
    python spotcheck_geometry.py <taskdir A> <taskdir B> [--label-a X --label-b Y]
                                 [--branches Jpsi_mass ...]
"""
import argparse
import os
import sys

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "resolution"))
import prodfiles  # noqa: E402

KEYCOLS = ["run", "lumi", "event", "Muplustrk_pt", "Muminustrk_pt"]


def load(taskdir, branches):
    """Every stream of one task, concatenated, as a dict of arrays."""
    files = prodfiles.stream_files(taskdir)
    if not files:
        raise SystemExit("no stream files under " + taskdir)
    want = list(dict.fromkeys(KEYCOLS + list(branches)))
    parts = []
    for f in files:
        t = uproot.open(f)["tree"]
        parts.append(t.arrays(want, library="np"))
    return {k: np.concatenate([p[k] for p in parts]) for k in want}, len(files)


def keys_of(d):
    # The pre-refit track pt comes straight off the input file and is written
    # from the same float in both productions, so equality is exact.
    return list(zip(d["run"].astype(np.int64), d["lumi"].astype(np.int64),
                    d["event"].astype(np.int64),
                    d["Muplustrk_pt"].astype(np.float32),
                    d["Muminustrk_pt"].astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--branches", nargs="*", default=["Jpsi_mass"])
    args = ap.parse_args()

    da, na = load(args.a, args.branches)
    db, nb = load(args.b, args.branches)
    ka, kb = keys_of(da), keys_of(db)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    if len(ia) != len(ka) or len(ib) != len(kb):
        print("[warn] duplicate keys: %d/%d (A), %d/%d (B)"
              % (len(ia), len(ka), len(ib), len(kb)))
    common = [k for k in ka if k in ib]

    print("%-10s %s  (%d streams, %d candidates)" % (args.label_a, args.a, na, len(ka)))
    print("%-10s %s  (%d streams, %d candidates)" % (args.label_b, args.b, nb, len(kb)))
    print("common candidates : %d   only in %s: %d   only in %s: %d"
          % (len(common), args.label_a, len(ka) - len(common),
             args.label_b, len(kb) - len(common)))

    ja = np.array([ia[k] for k in common])
    jb = np.array([ib[k] for k in common])
    for br in args.branches:
        va, vb = da[br][ja].astype(np.float64), db[br][jb].astype(np.float64)
        d = vb - va
        finite = np.isfinite(d)
        d = d[finite]
        if d.size == 0:
            print("%-18s no finite entries" % br)
            continue
        print("%-18s n=%d  median(%s-%s)=%+.6g  mean=%+.6g  rms=%.6g  "
              "p16=%+.6g p84=%+.6g  identical=%d"
              % (br, d.size, args.label_b, args.label_a, np.median(d), d.mean(),
                 d.std(), np.percentile(d, 16), np.percentile(d, 84),
                 int((d == 0).sum())))
        if br.endswith("mass"):
            rel = d / va[finite]
            print("%-18s   relative: median=%+.4g  rms=%.4g   "
                  "(x %.1f MeV at the Z: %+.2f MeV median, %.2f MeV rms)"
                  % ("", np.median(rel), rel.std(), 91187.6,
                     91187.6 * np.median(rel), 91187.6 * rel.std()))


if __name__ == "__main__":
    main()
