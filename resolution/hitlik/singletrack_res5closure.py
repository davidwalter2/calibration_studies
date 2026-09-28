#!/usr/bin/env python3
"""CF closure of all five track parameters (q/p, lambda, phi, d0, z0) on an
`extract_res5.py` cache -- the clean-propagation closure statistic applied to
the single-track refit against gen.

Arms (model densities of the pull z_k):
  cf      the CF product: MS/ionisation/radiative blocks + Gaussian hits
  cfcls   the same with the MEASURED per-class hit densities (hitres3 bank)
  gaussq  the fit's own Gaussian (Rossi MS + the fit's hit variances), i.e. a
          unit Gaussian in the fit's normalisation -- what the fit assumes

Per (sample, charge, parameter): closure(u) = <e^{-u z^2}>_data - <.>_model at
u = 0.1, 1 (data error std/sqrt(n)), sigma68 data/model, P(|z|>3) data/model.
Figures: pull histogram vs the three model densities, one file per parameter.

usage: singletrack_res5closure.py --npz A.npz [--npz B.npz ...] --tag lowpt ul16
"""
import argparse
import datetime
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hitlik_term as HT  # noqa: E402

CN = ("q/p", "lambda", "phi", "d0", "z0")
CNTEX = ("q/p", r"\lambda", r"\phi", r"d_0", r"z_0")
ARMS = ("cf", "cfcls", "gaussq")
ZG = np.linspace(-12, 12, 2401)
UPROBES = (0.1, 1.0)


def row_exponents(sel, arm, bank=None):
    """Per-row complex log-CF on sel['tgrid']."""
    base = "gaussq" if arm == "gaussq" else "cf"
    fam, ptr, gid, _, _, _, _ = HT.arm_families(sel, base)
    n = len(sel["z"])
    tg = sel["tgrid"]
    seg = np.repeat(np.arange(n), np.diff(ptr))
    S = np.zeros((n, len(tg)), np.complex128)
    for m in fam:
        if "re" in m:
            np.add.at(S.real, seg, m["re"].astype(np.float64))
        if "im" in m:
            np.add.at(S.imag, seg, m["im"].astype(np.float64))
        if "fix_re" in m:
            S.real += m["fix_re"]
        if "fix_im" in m:
            S.imag += m["fix_im"]
    vg = sel["vgf"].astype(np.float64)
    if arm == "cfcls":
        import hitres_classes
        names = sel["hit_classes"]
        hp, hc, hv = sel["hit_ptr"], sel["hit_cls"], sel["hit_v"]
        hseg = np.repeat(np.arange(n), np.diff(hp))
        have = np.array([nm in bank for nm in names])
        arg = np.sqrt(np.maximum(hv, 0.0))[:, None] * tg[None, :]
        val = np.zeros(arg.shape, np.complex128)
        gshare = np.zeros(n)
        for ci in np.unique(hc):
            mk = hc == ci
            if have[ci]:
                val[mk] = hitres_classes.logphi(bank[names[ci]], arg[mk])
                np.add.at(gshare, hseg[mk], hv[mk])
            else:
                val[mk] = -0.5 * arg[mk] ** 2
                np.add.at(gshare, hseg[mk], hv[mk])
        np.add.at(S, hseg, val)
        vg = np.maximum(vg - gshare, 0.0)       # the non-hit Gaussian remainder
    elif arm == "gaussq":
        pass                                    # hits are the fit's own variances
    S += -0.5 * np.outer(vg, tg ** 2)
    return S


def density(S, tg, up=8):
    from scipy.interpolate import CubicSpline
    tf = np.linspace(tg[0], tg[-1], (len(tg) - 1) * up + 1)
    phib = np.exp(S).mean(axis=0)
    ph = CubicSpline(tg, phib.real)(tf) + 1j * CubicSpline(tg, phib.imag)(tf)
    w = np.gradient(tf); w[0] *= 0.5; w[-1] *= 0.5
    return ((np.cos(np.outer(ZG, tf)) * ph.real + np.sin(np.outer(ZG, tf)) * ph.imag) @ w) / np.pi


def weier(S, tg, u):
    w = np.exp(-tg ** 2 / (4. * u)) / np.sqrt(np.pi * u)
    return np.trapezoid(np.exp(S).real.mean(axis=0) * w, tg)


def s68_from_density(p):
    c = np.cumsum(p) * (ZG[1] - ZG[0]); c /= c[-1]
    q = np.interp([0.15865, 0.84135], c, ZG)
    return 0.5 * (q[1] - q[0])


def ptail(p, t):
    m = np.abs(ZG) > t
    return np.sum(p[m]) * (ZG[1] - ZG[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--outpath", default=os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today().strftime('%y%m%d')}_singletrack_res5"))
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outpath, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from wums import output_tools, plot_tools
    import ratiopanel
    import hitres_classes
    hep.style.use(hep.style.ROOT)
    bank, _ = hitres_classes.build_cf_bank("hitres3", "mugun_lowpt", 25)
    lines = []
    P = lambda s: (print(s, flush=True), lines.append(s))

    stats = {}
    for npz, tag in zip(args.npz, args.tag):
        sel = HT.load(npz, max_tracks=args.max_tracks)
        # (the extraction already drops non-positive-definite covariances and
        # tracks whose q/p variance share does not close; covdev is the max over
        # ALL elements and is dominated by near-zero off-diagonals -- reported only)
        keep = np.ones(len(sel["z"]), bool)
        P(f"{tag}: {sel['ntrk']} tracks; covdev > 1e-3 on {int((sel['covdev'] > 1e-3).sum())} (kept)")
        tg = sel["tgrid"]
        comp = sel["comp"]
        ch = np.repeat(sel["charge"], sel["ncomp_used"])
        S = {a: row_exponents(sel, a, bank) for a in ARMS}
        arms = list(ARMS)
        comp = np.where(keep, comp, -1)
        for k in range(5):
            for qn, qm in (("mu-", ch < 0), ("mu+", ch > 0), ("all", np.ones_like(ch, bool))):
                m = (comp == k) & qm
                z = sel["z"][m]
                rec = {"n": m.sum()}
                for u in UPROBES:
                    e = np.exp(-u * z ** 2)
                    rec[f"d{u}"] = e.mean()
                    rec[f"e{u}"] = e.std() / np.sqrt(len(z))
                    for a in arms:
                        rec[f"m{u}_{a}"] = weier(S[a][m], tg, u)
                q = np.percentile(z, [15.865, 84.135])
                rec["s68"] = 0.5 * (q[1] - q[0])
                if k == 0 and qn != "all":
                    # the ODD statistic (q/p only: only there are the block
                    # weights signed): data median vs each arm's model median,
                    # and each arm's model MEAN from the slope of Im S at tau -> 0
                    rec["med"] = np.median(z)
                    rec["med_e"] = 1.2533 * rec["s68"] / np.sqrt(len(z))
                    for a in arms:
                        dd = density(S[a][m], tg)
                        c_ = np.cumsum(dd); c_ /= c_[-1]
                        rec[f"med_{a}"] = np.interp(0.5, c_, ZG) + 0.5 * (ZG[1] - ZG[0])
                        rec[f"mean_{a}"] = float(np.mean(S[a][m][:, 1].imag) / tg[1])
                rec["p3"] = np.mean(np.abs(z) > 3)
                if qn == "all":
                    rec["dens"] = {a: density(S[a][m], tg) for a in arms}
                    for a in arms:
                        rec[f"s68_{a}"] = s68_from_density(rec["dens"][a])
                        rec[f"p3_{a}"] = ptail(rec["dens"][a], 3.0)
                    rec["z"] = z
                stats[(tag, qn, k)] = rec
        stats[(tag, "arms")] = arms

    allarms = [a for a in ARMS if all(a in stats[(t, "arms")] for t in args.tag)]
    for u in UPROBES:
        for a in allarms:
            P(f"\nclosure(u={u}) = <e^(-u z^2)>_data - <.>_model[{a}]")
            P(f"{'sample':7s} {'q':4s} " + " ".join(f"{c:>17s}" for c in CN))
            for tag in args.tag:
                for qn in ("mu-", "mu+"):
                    cells = []
                    for k in range(5):
                        r = stats[(tag, qn, k)]
                        cells.append(f"{r[f'd{u}'] - r[f'm{u}_{a}']:+.4f} ± {r[f'e{u}']:.4f}")
                    P(f"{tag:7s} {qn:4s} " + " ".join(f"{c:>17s}" for c in cells))
    P("\nq/p LOCATION: data median pull vs model median [sigma]; model mean in brackets")
    for tag in args.tag:
        for qn in ("mu-", "mu+"):
            r = stats[(tag, qn, 0)]
            P(f"{tag:7s} {qn:4s} data {r['med']:+.4f} ± {r['med_e']:.4f}   " + "   ".join(
                f"{a}: {r['med_' + a]:+.4f} ({r['mean_' + a]:+.4f}) d-m {r['med'] - r['med_' + a]:+.4f}"
                for a in allarms))
    P("\nsigma68 data / model  and  P(|z|>3) data / model   (both charges)")
    P(f"{'sample':7s} {'param':7s} {'s68 data':>9s} " + "".join(f"{'s68 ' + a:>14s}" for a in allarms)
      + f" {'P3 data':>9s}" + "".join(f"{'P3 d/m ' + a:>17s}" for a in allarms))
    for tag in args.tag:
        for k in range(5):
            r = stats[(tag, "all", k)]
            P(f"{tag:7s} {CN[k]:7s} {r['s68']:9.4f} " + "".join(f"{r['s68_' + a]:14.4f}" for a in allarms)
              + f" {r['p3']:9.5f}" + "".join(f"{r['p3'] / max(r['p3_' + a], 1e-12):17.3f}" for a in allarms))
    with open(os.path.join(args.outpath, "res5_closure.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    sty = {"cf": ("crimson", "-"), "cfcls": ("darkorange", "-"),
           "gaussq": ("C0", "-")}
    lab = {"cf": "CF, Gaussian hits", "cfcls": "CF, measured hit shapes",
           "gaussq": "fit's Gaussian"}
    zb = np.linspace(-8, 8, 161)
    for tag in args.tag:
        for k in range(5):
            r = stats[(tag, "all", k)]
            fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(10., 7.6))
            zc = np.clip(r["z"], zb[0], zb[-1])
            cnt, _ = np.histogram(zc, bins=zb)
            ax.hist(zc, bins=zb, density=True, histtype="step", color="k", label="pulls (gen-matched MC)")
            cen = 0.5 * (zb[1:] + zb[:-1])
            for a in allarms:
                c_, ls_ = sty[a]
                pz = np.interp(zb, ZG, r["dens"][a])
                ax.plot(zb, pz, color=c_, ls=ls_, label=lab[a])
                pb = np.interp(cen, ZG, r["dens"][a])
                rax.plot(cen, cnt / (cnt.sum() * (zb[1] - zb[0])) / np.maximum(pb, 1e-300),
                         ".", color=c_, ms=4)
            rax.axhline(1, color="k", lw=0.8)
            rax.set_ylim(0.6, 1.4)
            rax.set_ylabel("data / model")
            rax.set_xlabel(rf"$z = \Delta {CNTEX[k]}/\sigma_{{\mathrm{{fit}}}}$")
            ax.set_yscale("log"); ax.set_ylim(1e-6, 1e3)
            ax.set_ylabel("density")
            ax.legend(fontsize="x-small", title_fontsize="x-small", loc="upper left", ncol=2,
                      title=f"{tag}, {r['n']} tracks")
            name = f"res5_pulls_{tag}_{CN[k].replace('/', '')}"
            plot_tools.save_pdf_and_png(args.outpath, name, fig)
            output_tools.write_logfile(args.outpath, name, args=args, wd=_HERE)
            plt.close(fig)
    print("wrote", args.outpath)


if __name__ == "__main__":
    main()
