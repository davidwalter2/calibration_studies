#!/usr/bin/env python3
"""Per-momentum-bin dE/dx response and the scale each bin actually requires.

The four existing J/psi-gun productions (dE/dx x 1.000 / 0.960 / 0.930 / 0.887)
form a scan. For each momentum bin this measures

    R    = d(bias)/ds                  the response to the scale
    ds_req = bias(s=1) / R             the scale change that would zero the bin

and compares ds_req with the CGF-predicted mean-vs-mode shift expressed as a
fraction of the mean loss (0.210 at p = 3.1, 0.270 at p = 41.8; see
cgf_saddlepoint / NOTES 2026-08-12). If the mechanism explained the bias, the
two would agree bin by bin.

Candidates are KEY-matched on (run, lumi, event, gen pt), never index-matched:
the configurations emit tracks in a different order, and index pairing matches
~15 % and returns nonsense (NOTES 2026-08-09).

Two observables, because they answer different questions:
  * single-track dp/p vs gen -- directly comparable to the muon-gun a1
  * dimuon dm/m             -- what the earlier scan quoted

usage: python cgf_dedx_scan.py [--nfiles 0] [--obs trk|mass]
"""
import argparse
import glob
import os

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
CONFIGS = [("jpsigun_ul16", 1.000),
           ("jpsigun_dedx096", 0.960),
           ("jpsigun_dedx093", 0.930),
           ("jpsigun_dedx887", 0.887)]

# CGF-predicted (mean - mode)/mean at the two momenta where cleanprop rays exist
CGF_P = np.array([3.14, 41.81])
CGF_F = np.array([0.210, 0.270])
# The model's mode over-predicts Geant4's; the prediction must carry that.
# Use the WANDER-FREE Geant4 mode (~3.7 at |dy| -> 0, cleanprop/wander_test.py),
# NOT the all-sample 3.298: the latter is contaminated by material sampling,
# which is a limitation of the test rather than a defect of the model, and
# which accounts for ~40% of the raw gap.
G4CORR = 3.70 / 4.295

BRANCHES = ["run", "lumi", "event",
            "Muplus_pt", "Muplus_eta", "Muminus_pt", "Muminus_eta",
            "Muplusgen_pt", "Muplusgen_eta", "Muminusgen_pt", "Muminusgen_eta",
            "Jpsi_mass", "Jpsigen_mass"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nfiles", type=int, default=0, help="0 = all")
    p.add_argument("--obs", default="trk", choices=("trk", "mass"))
    p.add_argument("--nboot", type=int, default=200)
    return p.parse_args()


def load(tag, nfiles):
    fs = sorted(glob.glob(f"{CEPH}/resolution_trackres_{tag}/task_*/globalcor_0.root"))
    if nfiles:
        fs = fs[:nfiles]
    cols = {k: [] for k in BRANCHES}
    for fn in fs:
        try:
            a = uproot.open(fn)["tree"].arrays(BRANCHES, library="np")
        except Exception:
            continue
        for k in BRANCHES:
            cols[k].append(np.asarray(a[k], dtype=np.float64))
    if not cols["run"]:
        return None
    d = {k: np.concatenate(v) for k, v in cols.items()}
    # composite key: run/lumi/event alone can repeat if an event yields more
    # than one candidate, so pin the gen kinematics too
    d["key"] = np.array([f"{int(r)}:{int(l)}:{int(e)}:{gp:.5f}"
                         for r, l, e, gp in zip(d["run"], d["lumi"], d["event"],
                                                d["Muplusgen_pt"])])
    return d


def observable(d, obs):
    """Returns (value, momentum) per candidate."""
    pp = d["Muplus_pt"] * np.cosh(d["Muplus_eta"])
    pm = d["Muminus_pt"] * np.cosh(d["Muminus_eta"])
    gp = d["Muplusgen_pt"] * np.cosh(d["Muplusgen_eta"])
    gm = d["Muminusgen_pt"] * np.cosh(d["Muminusgen_eta"])
    if obs == "trk":
        # average of the two legs' dp/p, and the harmonic momentum: this is
        # exactly what enters dm/m = 1/2 (dp1/p1 + dp2/p2)
        v = 0.5 * ((pp - gp) / gp + (pm - gm) / gm)
        p = 2.0 / (1.0 / gp + 1.0 / gm)
    else:
        v = (d["Jpsi_mass"] - d["Jpsigen_mass"]) / d["Jpsigen_mass"]
        p = 2.0 / (1.0 / gp + 1.0 / gm)
    ok = np.isfinite(v) & np.isfinite(p) & (p > 0) & (np.abs(v) < 0.2)
    return v, p, ok


def med_err(x, nboot, rng):
    m = float(np.median(x))
    if nboot <= 0 or len(x) < 20:
        return m, 0.0
    bs = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)])
    return m, float(bs.std())


def main():
    args = parse_args()
    rng = np.random.default_rng(7)

    data = {}
    for tag, s in CONFIGS:
        d = load(tag, args.nfiles)
        if d is None:
            print(f"[skip] {tag}: no files")
            continue
        data[s] = d
        print(f"loaded {tag:22s} s={s:.3f}  n={len(d['key'])}")

    scales = sorted(data, reverse=True)
    # key-match across ALL configurations
    common = set(data[scales[0]]["key"])
    for s in scales[1:]:
        common &= set(data[s]["key"])
    common = np.array(sorted(common))
    print(f"\nkey-matched candidates common to all {len(scales)} configs: "
          f"{len(common)}\n")
    if len(common) < 500:
        print("too few matched candidates -- aborting")
        return

    idx = {}
    for s in scales:
        pos = {k: i for i, k in enumerate(data[s]["key"])}
        idx[s] = np.array([pos[k] for k in common])

    vals, mom = {}, None
    for s in scales:
        d = {k: v[idx[s]] for k, v in data[s].items() if k != "key"}
        v, p, ok = observable(d, args.obs)
        vals[s] = v
        if mom is None:
            mom, okall = p, ok
        else:
            okall &= ok
    for s in scales:
        okall &= np.isfinite(vals[s])

    edges = np.quantile(mom[okall], np.linspace(0, 1, 6))
    print(f"observable = {args.obs};  bias in units of 1e-4\n")
    hdr = "  ".join(f"s={s:.3f}" for s in scales)
    print(f"{'<p>':>7} {'n':>7}  {hdr}  | {'bias(s=1)':>13} {'R':>7} "
          f"{'ds_req':>13} {'CGF*G4':>8} {'pull':>6}")
    print("-" * 118)

    for i in range(len(edges) - 1):
        sel = okall & (mom >= edges[i]) & (mom < edges[i + 1])
        if sel.sum() < 200:
            continue
        pmed = float(np.median(mom[sel]))
        ms, es = [], []
        for s in scales:
            m, e = med_err(vals[s][sel], args.nboot, rng)
            ms.append(m); es.append(e)
        ms, es = np.array(ms), np.array(es)
        sarr = np.array(scales)
        # response: linear fit of bias vs scale
        R = float(np.polyfit(sarr, ms, 1)[0])
        b1 = float(ms[sarr == 1.000][0]) if np.any(sarr == 1.000) else np.nan
        ds_req = b1 / R if R != 0 else np.nan
        e1 = float(es[sarr == 1.000][0])
        ds_err = abs(e1 / R) if R != 0 else np.nan
        # CGF interpolated, then corrected by the measured model-vs-G4 mode
        # over-prediction (SPA 4.295 vs G4 3.298 at the outermost plane)
        cgf = float(np.interp(np.log(pmed), np.log(CGF_P), CGF_F)) * G4CORR
        pull = (ds_req - cgf) / ds_err if ds_err > 0 else np.nan
        row = "  ".join(f"{m*1e4:+7.2f}" for m in ms)
        print(f"{pmed:7.2f} {int(sel.sum()):7d}  {row}  | {b1*1e4:+6.2f}+-{e1*1e4:.2f} "
              f"{R*1e4:7.1f} {ds_req:6.3f}+-{ds_err:.3f} {cgf:8.3f} {pull:+6.1f}")

    sel = okall
    ms = np.array([float(np.median(vals[s][sel])) for s in scales])
    sarr = np.array(scales)
    R = float(np.polyfit(sarr, ms, 1)[0])
    b1 = float(ms[sarr == 1.000][0])
    print("-" * 104)
    row = "  ".join(f"{m*1e4:+7.2f}" for m in ms)
    print(f"{'ALL':>7} {int(sel.sum()):7d}  {row}  | {b1*1e4:+12.2f} "
          f"{R*1e4:7.1f} {b1/R:13.3f}")
    print("\nds_req = the dE/dx scale CHANGE that would zero that bin's bias.")
    print("CGF d/dE = independently predicted (mean-mode)/mean, interpolated in log p.")
    print("ratio ~ 1 everywhere would mean the mean-vs-mode mechanism explains "
          "the bias bin by bin.")


if __name__ == "__main__":
    main()
