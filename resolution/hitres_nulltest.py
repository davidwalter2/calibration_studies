#!/usr/bin/env python3
"""Sim-position null test, per species: do the HIT POSITIONS carry the bias?

`fitSimHitPositions=True` substitutes the PSimHit truth positions and leaves
the assigned covariances untouched -- the same estimator with one input
changed. So:
    the gen bias survives   -> it is transport / field / material, not hits
    the gen bias vanishes   -> it is hit reconstruction
and, because sigma is unchanged, the width difference reads off the HIT SHARE
of the q/p variance directly:  var(sim)/var(reco) = 1 - f_hit.

For MUONS this repeats an existing measurement (Documents/Resolution/HIT_RESOLUTION.md:
|shift| < 0.45e-4 at 3 sigma on mugun_lowpt and mugun_ul16) and is run here as
a control on this production. For HADRONS it depends on the sim-hit match being
species-aware: `usesimpos = fitSimHitPositions_ && simhit != nullptr`, so a
null `simhit` on a hadron gun falls back to reco positions silently and the
"sim-position" arm is then a copy of the nominal one.

PAIR THE COMPARISON. The two arms select the same tracks but EMIT THEM IN A
DIFFERENT ORDER, so index pairing matches ~15 % and is meaningless, and
treating the arms as independent overestimates the error by ~4x. Tracks are
keyed on (file, run, lumi, event, genPt, charge sign).

usage:
  python hitres_nulltest.py --tags mugun_lowpt kaongun_ul16
"""
import argparse
import glob
import os

import numpy as np
import prodfiles

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
BR = ["refParms", "refCov", "genParms", "chisqval", "ndof", "nValidHits",
      "genPt", "genEta", "run", "lumi", "event"]


def load_keyed(tag, nfiles=0, subdir="hitres"):
    import uproot
    fs = prodfiles.resolve(
        f"{CEPH}/{subdir}_{tag}/task_*/globalcor_resclosure_*.root", nfiles)
    rec = {}
    for fn in fs:
        # a task without .complete is a truncated file, not a short one
        if not os.path.exists(os.path.join(os.path.dirname(fn), ".complete")):
            continue
        # Key on the TASK NUMBER, not on the position in this arm's glob:
        # two arms with different numbers of completed tasks would otherwise
        # be paired against different input files and match on nothing.
        fi = int(os.path.basename(os.path.dirname(fn)).split("_")[-1])
        try:
            a = uproot.open(fn)["tree"].arrays(BR, library="np")
        except Exception:                                        # noqa: BLE001
            continue
        for i in range(len(a["refCov"])):
            qg = a["genParms"][i][0]
            c0 = a["refCov"][i][0]
            if qg == 0.0 or c0 <= 0.0:
                continue
            k = (fi, int(a["run"][i]), int(a["lumi"][i]), int(a["event"][i]),
                 round(float(a["genPt"][i]), 5), int(np.sign(qg)))
            rec[k] = ((a["refParms"][i][0] - qg) / qg,             # Dk/k
                      (a["refParms"][i][0] - qg) / np.sqrt(c0),    # pull z
                      float(a["chisqval"][i]) / max(float(a["ndof"][i]), 1.0),
                      float(a["genPt"][i]))
    return rec, len(fs)


def med_err(x, nboot, rng):
    m = float(np.median(x))
    bs = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)])
    return m, float(bs.std())


def trim_err(x, nboot, rng, frac=0.1):
    def tm(v):
        v = np.sort(v)
        lo, hi = int(frac * len(v)), max(int((1 - frac) * len(v)), int(frac * len(v)) + 1)
        return float(v[lo:hi].mean())
    m = tm(x)
    bs = np.array([tm(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)])
    return m, float(bs.std())


def one(tag, args, rng):
    ra, nfa = load_keyed(tag, args.nfiles, args.subdir)
    rb, nfb = load_keyed(f"{tag}{args.simsuffix}", args.nfiles, args.subdir)
    if not ra or not rb:
        print(f"{tag}: missing arm (reco {len(ra)} tracks, simpos {len(rb)})")
        return
    common = sorted(set(ra) & set(rb))
    da = np.array([ra[k][0] for k in common])
    db = np.array([rb[k][0] for k in common])
    za = np.array([ra[k][1] for k in common])
    zb = np.array([rb[k][1] for k in common])
    c2a = np.array([ra[k][2] for k in common])
    c2b = np.array([rb[k][2] for k in common])
    # 5-sigma trim on the RECO arm, applied to both, so the pairing survives
    s = 0.5 * np.diff(np.percentile(da, [15.865, 84.135]))[0]
    keep = np.abs(da - np.median(da)) < 5 * s
    da, db, za, zb, c2a, c2b = (v[keep] for v in (da, db, za, zb, c2a, c2b))

    ra68 = 0.5 * np.diff(np.percentile(da, [15.865, 84.135]))[0]
    rb68 = 0.5 * np.diff(np.percentile(db, [15.865, 84.135]))[0]
    mda, eda = med_err(da, args.nboot, rng)
    mdb, edb = med_err(db, args.nboot, rng)
    tma, eta_ = trim_err(da, args.nboot, rng)
    tmb, etb = trim_err(db, args.nboot, rng)
    md, ed = med_err(db - da, args.nboot, rng)
    td, etd = trim_err(db - da, args.nboot, rng)
    fhit = 1.0 - (rb68 / ra68) ** 2

    print(f"\n=== {tag} ===")
    print(f"  files reco/simpos {nfa}/{nfb}   key-matched {len(common)}   "
          f"after 5-sigma trim {len(da)}")
    print(f"  {'arm':<12} {'median Dk/k':>18} {'trim-mean Dk/k':>20} "
          f"{'rob68':>10} {'chi2/ndof':>10}")
    print(f"  {'reco hits':<12} {mda*1e4:+9.3f} +-{eda*1e4:5.3f} "
          f"{tma*1e4:+11.3f} +-{eta_*1e4:5.3f} {ra68*1e4:10.2f} "
          f"{np.median(c2a):10.4f}   (x1e-4)")
    print(f"  {'sim hits':<12} {mdb*1e4:+9.3f} +-{edb*1e4:5.3f} "
          f"{tmb*1e4:+11.3f} +-{etb*1e4:5.3f} {rb68*1e4:10.2f} "
          f"{np.median(c2b):10.4f}")
    print(f"  PAIRED (sim - reco):  median {md*1e4:+.3f} +- {ed*1e4:.3f} "
          f"({abs(md)/max(ed,1e-12):.1f} sigma)   "
          f"trim-mean {td*1e4:+.3f} +- {etd*1e4:.3f} "
          f"({abs(td)/max(etd,1e-12):.1f} sigma)   x1e-4")
    if abs(md) > 3 * ed:
        print("  => the bias MOVES with the hits: hit reconstruction.")
    else:
        print(f"  => the bias does NOT move (|shift| < {3*ed*1e4:.2f}e-4 at 3 sigma):"
              f" the hit POSITIONS are exonerated; the covariance is untouched"
              f" by this switch and remains the open hit-side suspect.")
    print(f"  var(sim)/var(reco) = {(rb68/ra68)**2:.4f}  ->  hit share of the "
          f"q/p variance f_hit = {fhit:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", default=["mugun_lowpt", "mugun_ul16"])
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--subdir", default="resolution_trackres",
                   help="production family; the sim-position ablation needs the "
                        "NOMINAL-fit arms (fitFromGenParms=False), because with "
                        "the reference frozen to gen there is no q/p residual "
                        "to compare -- refParms[0] IS genParms[0].")
    p.add_argument("--simsuffix", default="_simhit")
    p.add_argument("--nboot", type=int, default=300)
    args = p.parse_args()
    rng = np.random.default_rng(20260829)
    for t in args.tags:
        one(t, args, rng)


if __name__ == "__main__":
    main()
