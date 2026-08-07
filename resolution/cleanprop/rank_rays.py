"""Rank the rays produced by scan_ray.sh by how cleanly they cross the tracker.

The clean propagation test compares the simulated state to a deterministic
reference ON a given plane, so only events sharing the modal (module, entry
face) pattern can be used. Everything else is dropped, and that drop IS a
selection effect -- the one the test exists to avoid. So the ray is chosen
empirically: highest modal fraction, with enough crossed planes to be worth
running.

usage:
    python rank_rays.py '/ceph/.../rayscan/scan_pt40_*_pdg13.root'
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
import uproot

TAGRE = re.compile(r"scan_pt([\d.]+)_eta([\d.-]+)_phi([\d.-]+)_pdg(-?\d+)\.root$")


def modal_fraction(path):
    """(nplanes, modal fraction, nevents) for one scan file."""
    t = uproot.open(path)["simstates/simstates"]
    a = t.arrays(["detid", "locz"], library="np")
    seqs = [tuple(zip(d.tolist(), np.round(z, 4).tolist()))
            for d, z in zip(a["detid"], a["locz"])]
    if not seqs:
        return 0, 0.0, 0
    modal, nmodal = Counter(seqs).most_common(1)[0]
    return len(modal), nmodal / len(seqs), len(seqs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern", help="glob of scan_*.root (quote it)")
    p.add_argument("--min-planes", type=int, default=0,
                   help="ignore rays crossing fewer planes than this")
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"no files match {args.pattern}")

    rows = []
    for f in files:
        m = TAGRE.search(os.path.basename(f))
        if not m:
            continue
        pt, eta, phi, pdg = m.groups()
        try:
            npl, frac, nev = modal_fraction(f)
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {type(e).__name__}")
            continue
        rows.append((float(pt), float(eta), float(phi), int(pdg), npl, frac, nev))

    rows = [r for r in rows if r[4] >= args.min_planes]
    # best = cleanest; break ties toward more crossed planes
    rows.sort(key=lambda r: (-r[5], -r[4]))

    print(f"\n{'pt':>6} {'eta':>6} {'phi':>6} {'pdg':>5} {'planes':>7} "
          f"{'clean':>8} {'nev':>6}")
    for pt, eta, phi, pdg, npl, frac, nev in rows:
        print(f"{pt:6.1f} {eta:6.2f} {phi:6.2f} {pdg:5d} {npl:7d} "
              f"{100*frac:7.2f}% {nev:6d}")
    if rows:
        pt, eta, phi, pdg, npl, frac, _ = rows[0]
        print(f"\nbest: pt={pt:g} eta={eta:g} phi={phi:g} pdg={pdg} "
              f"-> {npl} planes, {100*frac:.2f}% clean")


if __name__ == "__main__":
    main()
