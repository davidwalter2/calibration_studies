#!/usr/bin/env python3
"""How much of the clean-propagation non-closure is ACCEPTANCE, not physics?

cf_propagation_test.load_sim keeps only rays whose (module, entry-face)
sequence equals the modal one, and drops the rest. The docstring calls this
"the one selection effect this test cannot avoid, so it must be small and it
must be quoted". Measured:

    pT = 40 GeV   drop = 0.086 %
    pT =  3 GeV   drop = 5.587 %          <-- 65x larger

The dropped rays are exactly those that scattered enough to miss or clip a
module, so the cut removes the TAIL of the very distribution being tested.
Truncating the tail makes the data narrower than the model, i.e. E[exp(-u z^2)]
too LARGE, i.e. data - model > 0 -- the sign and the momentum dependence of
the observed residual (+0.0146 at pT=3, -0.0029 at pT=40).

This script measures the bias instead of arguing about it. For every plane of
the modal pattern it recovers the local position of the DROPPED rays too --
they are only dropped from the closure, the hit is still in the tree whenever
they crossed that same (detid, entry-face) -- and compares

    E[exp(-u z^2)]  over kept rays        (what the closure uses)
    E[exp(-u z^2)]  over kept + recovered (what it should use)

The difference is the acceptance bias, in the same units as the closure number.

usage:
  python cleanprop/acceptance_bias.py --sim <dir-or-glob> [--sim <...>] [--label ...]
"""
import argparse
import glob
import logging
from collections import Counter

import numpy as np
import uproot

logger = logging.getLogger("acceptance_bias")

PROBES = (0.01, 0.05, 0.1, 0.2, 1.0, 2.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="append", required=True,
                   help="runCleanPropSim.py output dir or glob (repeatable)")
    p.add_argument("--label", action="append", default=None,
                   help="label per --sim (default: basename)")
    p.add_argument("--var", default="locx", choices=("locx", "locy"),
                   help="local coordinate to test (default locx, the "
                        "Moliere-dominated bending-plane one)")
    return p.parse_args()


def load(path):
    if any(c in path for c in "*?["):
        files = sorted(glob.glob(path))
    else:
        files = sorted(glob.glob(f"{path}/**/*.root", recursive=True)) or [path]
    keys = ["detid", "locx", "locy", "locz"]
    parts = {k: [] for k in keys}
    for fn in files:
        try:
            t = uproot.open(fn)["simstates/simstates"]
        except Exception:
            continue
        a = t.arrays(keys, library="np")
        for k in keys:
            parts[k].append(a[k])
    if not parts["detid"]:
        raise SystemExit(f"no readable files under {path}")
    return {k: np.concatenate(v) for k, v in parts.items()}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    labels = args.label or [p.rstrip("/").split("/")[-1] for p in args.sim]

    for path, label in zip(args.sim, labels):
        arr = load(path)
        det, loc, lz = arr["detid"], arr[args.var], arr["locz"]
        seqs = [tuple(zip(d.tolist(), np.round(z, 4).tolist()))
                for d, z in zip(det, lz)]
        counts = Counter(seqs)
        modal, nmodal = counts.most_common(1)[0]
        ntot = len(seqs)
        kept = np.array([s == modal for s in seqs])
        print(f"\n=== {label} ===")
        print(f"  {ntot} rays, modal pattern covers {nmodal} "
              f"({100. * nmodal / ntot:.3f}%), DROPPED {ntot - nmodal} "
              f"({100. * (ntot - nmodal) / ntot:.3f}%)")

        # Recover the dropped rays plane by plane. A dropped ray is dropped for
        # its WHOLE sequence, but on the planes it did cross with the modal
        # (detid, entry-face) its local position is perfectly good -- and it is
        # precisely the position the closure never sees.
        hdr = (f"  {'plane':>5} {'detid':>10} {'nkept':>7} {'nrec':>6} "
               f"{'rms_kept':>9} {'rms_all':>9} {'ratio':>6}   "
               + " ".join(f"du={u:<6g}" for u in PROBES))
        print(hdr)

        # Flatten the jagged per-ray arrays once: (ray index, detid, face, value).
        # Looping rays x planes in Python is 200k x 19 x 19 and far too slow.
        nper = np.array([len(d) for d in det])
        ray = np.repeat(np.arange(ntot), nper)
        fdet = np.concatenate([np.asarray(d) for d in det])
        # Round in the NATIVE dtype, exactly as load_sim does. locz is float32
        # and these values sit on a rounding boundary: np.round(x, 4) in float32
        # gives 0.0142 where promoting to float64 first gives 0.0143, so a
        # float64 promotion matches ZERO planes.
        fface = np.concatenate([np.round(np.asarray(z), 4).astype(np.float64) for z in lz])
        fval = np.concatenate([np.asarray(v, dtype=np.float64) for v in loc])
        fkept = kept[ray]

        tot = np.zeros(len(PROBES))
        nplane = 0
        for j, (mdet, mlz) in enumerate(modal):
            sel = (fdet == mdet) & (fface == mlz)
            vk = fval[sel & fkept]
            vr = fval[sel & ~fkept]
            if vk.size < 100:
                continue
            # residual about the KEPT mean: the reference is a single
            # deterministic propagation, so a constant offset is common to both
            # samples and cancels in the comparison.
            mu = vk.mean()
            zk = vk - mu
            za = np.concatenate([zk, vr - mu]) if vr.size else zk
            row = []
            for iu, u in enumerate(PROBES):
                ek = np.exp(-u * zk ** 2).mean()
                ea = np.exp(-u * za ** 2).mean()
                row.append(ek - ea)          # closure bias: kept is too LARGE
                tot[iu] += ek - ea
            nplane += 1
            print(f"  {j:5d} {mdet:10d} {vk.size:7d} {vr.size:6d} "
                  f"{zk.std():9.4f} {za.std():9.4f} {za.std() / zk.std():6.4f}   "
                  + " ".join(f"{v:+8.5f}" for v in row))
        if nplane:
            print(f"  {'MEAN':>5} {'':>10} {'':>7} {'':>6} {'':>9} {'':>9} {'':>6}   "
                  + " ".join(f"{v / nplane:+8.5f}" for v in tot))
            print("  (positive = the closure's kept-only E[exp(-u z^2)] is too "
                  "large,\n   i.e. it fakes 'model over-predicts' by this much)")


if __name__ == "__main__":
    main()
