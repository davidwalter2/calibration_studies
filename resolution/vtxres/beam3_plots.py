#!/usr/bin/env python3
"""Figures for the floated luminous region (`--beam3`).

One file per panel, into ``~/public_html/ZMass/cvh/<YYMMDD>_beam3/``.

--closure    the seven beam parameters, fitted against the SIMULATED luminous
             region (the generator's closed form and the same quantity measured
             on this sample's gen vertices), with a pull panel underneath
--tilt       the gen production vertex x and y against z, with the record's
             tilt, the fitted tilt and the generator's (zero) drawn on
--pulls      the two whitened beam pulls before and after the fitted beam
             parameters are applied, with a data/model ratio panel
"""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_vtx_card as MC  # noqa: E402
import pubhtml  # noqa: E402
from beam3_gen import GEN, gen_sigma_t, irls_slope  # noqa: E402
from beam3_report import BEAM, UNIT, get, read_fit  # noqa: E402

hep.style.use(hep.style.ROOT)

LAB = {"beamwidth_x": r"$\epsilon_x$  ($\sigma_x^2$ scale)",
       "beamwidth_y": r"$\epsilon_y$  ($\sigma_y^2$ scale)",
       "beamcorr_xy": r"$\mathrm{atanh}\,\rho_{xy}$",
       "beamtilt_x": r"$\delta(dx/dz)\ [10^{-5}]$",
       "beamtilt_y": r"$\delta(dy/dz)\ [10^{-5}]$",
       "beamcentre_x": r"$\delta x_0\ [\mu\mathrm{m}]$",
       "beamcentre_y": r"$\delta y_0\ [\mu\mathrm{m}]$"}


def closure(fits, ref, outdir, tag):
    names = [n for n in BEAM if any(n in f["names"] for f in fits.values())]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(names) + 2.2))
    tgt = np.array([float(ref[f"target_{n}"]) for n in names])
    gm = np.array([float(ref[f"genmeas_{n}"]) for n in names])
    ge = np.array([float(ref[f"generr_{n}"]) for n in names])
    for i, yy in enumerate(y):
        ax.plot([tgt[i]], [yy + 0.22], marker="*", ms=16, color="k",
                zorder=5, ls="none",
                label="generator (exact)" if i == 0 else None)
        ax.errorbar([gm[i]], [yy + 0.22], xerr=[ge[i]], fmt="s", ms=6,
                    color="#555555", capsize=3,
                    label="gen vertices" if i == 0 else None)
    cols = ["#d62728", "#1f77b4", "#2ca02c"]
    for k, (lab, f) in enumerate(fits.items()):
        v = np.array([get(f, n)[0] for n in names])
        e = np.array([get(f, n)[1] for n in names])
        ax.errorbar(v, y - 0.18 * (k + 1), xerr=e, fmt="o", ms=6,
                    color=cols[k % len(cols)], capsize=3, label=f"fit: {lab}")
    ax.axvline(0.0, color="0.7", lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([LAB[n] for n in names])
    ax.set_xlabel("value in card units (see the label)")
    ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
    ax.legend(loc="best", fontsize=11)
    ax.set_title("Luminous region: fitted vs simulated", fontsize=13)
    pubhtml.savefig(fig, os.path.join(outdir, f"closure_{tag}.pdf"))
    plt.close(fig)

    # the PULL panel, one point per parameter per fit
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(names) + 2.0))
    for k, (lab, f) in enumerate(fits.items()):
        p = []
        for n in names:
            v, e = get(f, n)
            i = names.index(n)
            p.append((v - tgt[i]) / e if e > 0 else np.nan)
        ax.plot(p, y - 0.15 * k, "o", ms=7, color=cols[k % len(cols)],
                label=f"fit: {lab}")
    for s, c in ((1, "#bbbbbb"), (2, "#dddddd")):
        ax.axvspan(-s, s, color=c, zorder=0)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LAB[n] for n in names])
    ax.set_xlabel(r"(fit $-$ generator) / $\sigma_\mathrm{fit}$")
    ax.set_xlim(-5, 5)
    ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
    ax.legend(loc="best", fontsize=11)
    ax.set_title("Closure against the generator's luminous region", fontsize=13)
    pubhtml.savefig(fig, os.path.join(outdir, f"closure_pull_{tag}.pdf"))
    plt.close(fig)


def tilt(npz, fit, outdir, tag, nbin=24):
    d = np.load(npz, allow_pickle=False)
    gx, gy, gz = (np.asarray(d[f"genvtx_{k}"], np.float64) for k in "xyz")
    ok = (np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gz)
          & (np.abs(gx) < 5) & (np.abs(gy) < 5) & (np.abs(gz) < 60))
    gx, gy, gz = gx[ok], gy[ok], gz[ok]
    rec = np.median(np.asarray(d["bsslope"], np.float64)[ok], axis=0)
    spot = np.median(np.asarray(d["bsspot"], np.float64)[ok], axis=0)
    for j, (g, nm, rs, s0) in enumerate(
            ((gx, "x", rec[0], spot[0]), (gy, "y", rec[1], spot[1]))):
        lo, hi = np.quantile(gz, [0.005, 0.995])
        edges = np.linspace(lo, hi, nbin + 1)
        ib = np.clip(np.digitize(gz, edges[1:-1]), 0, nbin - 1)
        ctr, med, err = [], [], []
        for b in range(nbin):
            m = ib == b
            if m.sum() < 20:
                continue
            v = g[m]
            q = np.quantile(v, [0.005, 0.995])
            v = v[(v >= q[0]) & (v <= q[1])]
            ctr.append(gz[m].mean())
            med.append(v.mean())
            err.append(v.std() / np.sqrt(len(v)))
        ctr = np.asarray(ctr)
        med = np.asarray(med) * 1e4
        err = np.asarray(err) * 1e4
        fig, ax = plt.subplots(figsize=(8, 5.4))
        ax.errorbar(ctr, med, yerr=err, fmt="o", ms=5, color="k",
                    capsize=2, label="gen vertices (trimmed mean)")
        zz = np.linspace(lo, hi, 100)
        sfit, bfit, _ = irls_slope(gz, g)
        ax.plot(zz, (s0 + rs * (zz - spot[2])) * 1e4, "-", color="#1f77b4",
                lw=2, label=f"beam-spot record ({rs:+.2e})")
        ax.plot(zz, (sfit * zz + bfit) * 1e4, "--", color="#d62728", lw=2,
                label=f"gen-vertex fit ({sfit:+.2e})")
        g0 = GEN["X0"] if nm == "x" else GEN["Y0"]
        ax.axhline(g0 * 1e4, color="#2ca02c", lw=2, ls=":",
                   label="generator (no tilt)")
        if fit is not None:
            p = fit[0].get(f"beamtilt_{nm}", 0.0) * MC.BEAM3_UNITS[f"beamtilt_{nm}"]
            c = fit[0].get(f"beamcentre_{nm}", 0.0) * MC.BEAM3_UNITS[f"beamcentre_{nm}"]
            ax.plot(zz, (s0 + c + (rs + p) * (zz - spot[2])) * 1e4, "-",
                    color="#ff7f0e", lw=2,
                    label=f"record + fit ({rs+p:+.2e})")
        ax.set_xlabel(r"gen vertex $z$ [cm]")
        ax.set_ylabel(rf"gen vertex $\langle {nm}\rangle$ [$\mu$m]")
        ax.legend(loc="best", fontsize=10)
        ax.set_title(f"The luminous region's {nm}-z tilt", fontsize=13)
        pubhtml.savefig(fig, os.path.join(outdir, f"tilt_{nm}_{tag}.pdf"))
        plt.close(fig)


def pulls(npzdir, prefix, fit, outdir, tag, nbin=60, zr=4.0):
    import ratiopanel
    pars = fit[0]
    for ch in ("bsx", "bsy"):
        d = np.load(os.path.join(npzdir, f"{prefix}_{ch}.npz"),
                    allow_pickle=False)
        z = np.asarray(d["m0"], np.float64)
        bm = np.asarray(d["bsmean"], np.float64)
        vtx = np.asarray(d["bsvtx"], np.float64)
        spot = np.asarray(d["bsspot"], np.float64)
        lev = vtx[:, 2] - spot[:, 2]
        dz = (bm[:, 0] * MC.BEAM3_UNITS["beamcentre_x"] * pars.get("beamcentre_x", 0.)
              + bm[:, 1] * MC.BEAM3_UNITS["beamcentre_y"] * pars.get("beamcentre_y", 0.)
              + bm[:, 0] * lev * MC.BEAM3_UNITS["beamtilt_x"] * pars.get("beamtilt_x", 0.)
              + bm[:, 1] * lev * MC.BEAM3_UNITS["beamtilt_y"] * pars.get("beamtilt_y", 0.))
        edges = np.linspace(-zr, zr, nbin + 1)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        h0, _ = np.histogram(z, edges)
        h1, _ = np.histogram(z + dz, edges)
        fig, ax, rx = ratiopanel.make_ratio_fig(figsize=(8, 6.4))
        ax.step(ctr, h0, where="mid", color="#1f77b4", lw=2,
                label=f"record beam spot (Var {np.var(z):.3f})")
        ax.step(ctr, h1, where="mid", color="#d62728", lw=2,
                label=f"+ fitted beam parameters (Var {np.var(z+dz):.3f})")
        ax.set_ylabel("candidates")
        ax.legend(loc="best", fontsize=10)
        ax.set_title(f"whitened beam pull {ch}", fontsize=13)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(h0 > 0, h1 / np.maximum(h0, 1), np.nan)
        rx.step(ctr, r, where="mid", color="k", lw=1.6)
        rx.axhline(1.0, color="0.6", lw=1, ls=":")
        rx.set_ylim(0.8, 1.2)
        rx.set_ylabel("corr. / record")
        rx.set_xlabel(f"$z_{{{ch[-1]}}}$")
        pubhtml.savefig(fig, os.path.join(outdir, f"pull_{ch}_{tag}.pdf"))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--prefix", default="dy")
    ap.add_argument("--fits", nargs="+", required=True, help="LABEL=dir")
    ap.add_argument("--genref", required=True)
    ap.add_argument("--main", default=None,
                    help="label of the fit used for the tilt / pull overlays")
    ap.add_argument("--outpath", default=None)
    ap.add_argument("--tag", default="dy")
    ap.add_argument("--closure", action="store_true")
    ap.add_argument("--tilt", action="store_true")
    ap.add_argument("--pulls", action="store_true")
    a = ap.parse_args()

    outdir = a.outpath or pubhtml.figdir("beam3")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir)
    print(f"figures -> {outdir}")

    fits, raw = {}, {}
    for spec in a.fits:
        lab, _, dd = spec.partition("=")
        fp = dd if dd.endswith(".hdf5") else os.path.join(dd, "fitresults.hdf5")
        if not os.path.exists(fp):
            print(f"[missing] {lab}")
            continue
        f = read_fit(fp)
        fits[lab] = f
        raw[lab] = (dict(zip(f["names"], f["val"])),
                    dict(zip(f["names"], f["err"])))
    ref = np.load(a.genref, allow_pickle=False)
    main_lab = a.main or list(fits)[-1]

    if a.closure or not (a.tilt or a.pulls):
        closure(fits, ref, outdir, a.tag)
    if a.tilt:
        tilt(os.path.join(a.npz_dir, f"{a.prefix}_bsx.npz"),
             raw.get(main_lab), outdir, a.tag)
    if a.pulls:
        pulls(a.npz_dir, a.prefix, raw[main_lab], outdir, a.tag)
    print(f"done {datetime.date.today()}")


if __name__ == "__main__":
    main()
