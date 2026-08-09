#!/usr/bin/env python3
"""Where does the +1 MeV CVH J/psi mass bias come from: momenta or angles?

The invariant mass of a dimuon depends on BOTH leg momenta and the opening
angle,

    m^2 = 2(E1 E2 - p1.p2) + 2 m_mu^2,

so a mass bias can come from a momentum scale OR from an angular bias, and the
angular term is dangerous at high boost: d ln m / d theta = (1/2) cot(theta/2),
which diverges as the muons become collinear. That is the natural candidate for
the observed RISE of the bias at high momentum, which a pure momentum-scale
model cannot produce.

The decomposition here is exact rather than linearised. Four masses are built
per candidate from the same 4-vector machinery:

    m(p_reco, u_reco) = m_reco        (cross-checked against Jpsi_mass)
    m(p_gen , u_gen ) = m_gen
    m(p_reco, u_gen )                 -> MOMENTUM-only effect
    m(p_gen , u_reco)                 -> ANGLE-only effect

with u the unit direction from (eta, phi) and p the magnitude pt*cosh(eta).
The two partial shifts should approximately sum to the total; any remainder is
the cross term.

Run for each mass flavour available (CVH refit, raw tracks, kinematic fit) so
the comparison isolates what the CVH refit itself does.

usage: python jpsi_bias_decompose.py [--nfiles 60] [--maxdr 0.05]
"""
import argparse
import glob

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
MMU = 0.1056583745

FLAVOURS = {
    "CVH refit": ("Muplus_pt", "Muplus_eta", "Muplus_phi",
                  "Muminus_pt", "Muminus_eta", "Muminus_phi"),
    "raw tracks": ("Muplustrk_pt", "Muplustrk_eta", "Muplustrk_phi",
                   "Muminustrk_pt", "Muminustrk_eta", "Muminustrk_phi"),
    "kin fit": ("Mupluskin_pt", "Mupluskin_eta", "Mupluskin_phi",
                "Muminuskin_pt", "Muminuskin_eta", "Muminuskin_phi"),
}
GEN = ("Muplusgen_pt", "Muplusgen_eta", "Muplusgen_phi",
       "Muminusgen_pt", "Muminusgen_eta", "Muminusgen_phi")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="jpsigun_ul16")
    p.add_argument("--nfiles", type=int, default=60)
    p.add_argument("--maxdr", type=float, default=0.0,
                   help="cut on Mu*gen_dr; 0 = none. Soft muons have larger dR "
                        "so a mis-association would fake a low-p bias -- this "
                        "is the control for that.")
    p.add_argument("--nboot", type=int, default=100)
    return p.parse_args()


def unit(eta, phi):
    th = 2. * np.arctan(np.exp(-eta))
    return np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], axis=-1)


def mass(p1, u1, p2, u2):
    v1, v2 = p1[:, None] * u1, p2[:, None] * u2
    e1 = np.sqrt(p1 ** 2 + MMU ** 2)
    e2 = np.sqrt(p2 ** 2 + MMU ** 2)
    m2 = (e1 + e2) ** 2 - np.sum((v1 + v2) ** 2, axis=1)
    return np.sqrt(np.clip(m2, 0., None))


def med_err(x, nboot, rng):
    m = float(np.median(x))
    bs = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)])
    return m, float(bs.std())


def main():
    args = parse_args()
    rng = np.random.default_rng(4)
    fs = sorted(glob.glob(f"{CEPH}/resolution_trackres_{args.tag}/task_*/globalcor_0.root"))[:args.nfiles]
    need = list(GEN) + ["Jpsi_mass", "Jpsigen_mass", "Muplusgen_dr", "Muminusgen_dr"]
    for v in FLAVOURS.values():
        need += list(v)
    cols = {k: [] for k in dict.fromkeys(need)}
    for fn in fs:
        try:
            a = uproot.open(fn)["tree"].arrays(list(cols), library="np")
        except Exception:
            continue
        for k in cols:
            cols[k].append(np.asarray(a[k], dtype=np.float64))
    d = {k: np.concatenate(v) for k, v in cols.items()}

    gp1 = d["Muplusgen_pt"] * np.cosh(d["Muplusgen_eta"])
    gp2 = d["Muminusgen_pt"] * np.cosh(d["Muminusgen_eta"])
    gu1 = unit(d["Muplusgen_eta"], d["Muplusgen_phi"])
    gu2 = unit(d["Muminusgen_eta"], d["Muminusgen_phi"])
    ok = np.isfinite(gp1) & np.isfinite(gp2) & (gp1 > 0) & (gp2 > 0) & (d["Muplusgen_pt"] > 0)
    if args.maxdr > 0:
        ok &= (d["Muplusgen_dr"] < args.maxdr) & (d["Muminusgen_dr"] < args.maxdr)
    mg = mass(gp1, gu1, gp2, gu2)
    ok &= np.isfinite(mg) & (mg > 2.5)
    phar = 2. / (1. / gp1 + 1. / gp2)
    cos12 = np.sum(gu1 * gu2, axis=1)
    theta = np.arccos(np.clip(cos12, -1., 1.))

    print(f"files={len(fs)}  candidates={int(ok.sum())}"
          + (f"   [gen dR < {args.maxdr}]" if args.maxdr > 0 else ""))
    print(f"gen opening angle: median {np.median(theta[ok]):.4f} rad, "
          f"cot(theta/2) median {np.median(1./np.tan(theta[ok]/2)):.2f}\n")

    edges = np.quantile(phar[ok], np.linspace(0, 1, 7))
    for name, (a1, e1, f1, a2, e2, f2) in FLAVOURS.items():
        rp1 = d[a1] * np.cosh(d[e1]); rp2 = d[a2] * np.cosh(d[e2])
        ru1 = unit(d[e1], d[f1]);     ru2 = unit(d[e2], d[f2])
        g = ok & np.isfinite(rp1) & np.isfinite(rp2) & (rp1 > 0) & (rp2 > 0)
        m_rr = mass(rp1, ru1, rp2, ru2)      # reco p, reco angle
        m_rg = mass(rp1, gu1, rp2, gu2)      # reco p, GEN angle -> momentum only
        m_gr = mass(gp1, ru1, gp2, ru2)      # gen  p, reco angle -> angle only
        g &= np.isfinite(m_rr) & (m_rr > 2.) & (m_rr < 4.5)
        tot = (m_rr[g] - mg[g]) / mg[g]
        mom = (m_rg[g] - mg[g]) / mg[g]
        ang = (m_gr[g] - mg[g]) / mg[g]
        t, et = med_err(tot, args.nboot, rng)
        mo, em = med_err(mom, args.nboot, rng)
        an, ea = med_err(ang, args.nboot, rng)
        print(f"=== {name}   n={int(g.sum())}")
        print(f"    TOTAL dm/m = {t*1e4:+8.3f} +- {et*1e4:.3f}   "
              f"MOMENTUM {mo*1e4:+8.3f} +- {em*1e4:.3f}   "
              f"ANGLE {an*1e4:+8.3f} +- {ea*1e4:.3f}   "
              f"[sum {(mo+an)*1e4:+8.3f}]")
        print(f"    {'<p_harm>':>9} {'n':>7}  {'TOTAL':>9} {'MOMENTUM':>10} {'ANGLE':>9}")
        for i in range(len(edges) - 1):
            s = g & (phar >= edges[i]) & (phar < edges[i + 1])
            if s.sum() < 500:
                continue
            sub = s[g]
            print(f"    {np.median(phar[s]):9.2f} {int(s.sum()):7d}  "
                  f"{np.median(tot[sub])*1e4:+9.3f} {np.median(mom[sub])*1e4:+10.3f} "
                  f"{np.median(ang[sub])*1e4:+9.3f}")
        print()


if __name__ == "__main__":
    main()
