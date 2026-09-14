#!/usr/bin/env python3
"""Standalone Photos++ runs against the generator sample and the analytic kernel.

The standalone runs (`photos_standalone/`) are generated flat in ``m_pre`` band
so that every band has the same precision; a band is recombined into a wider one
with the *sample's* own band weights, which restores the sample's conditional
``m_pre`` distribution exactly (checked by ``<m_pre>`` per band).

Errors: the sample's are ``sqrt(p(1-p)/N_eff)`` with ``N_eff = (sum w)^2/sum w^2``
from the clipped MiNNLO weights; the standalone's are Poisson.

One file per panel, ratio panel under every density-vs-model plot, mplhep ROOT
style, output under ``~/public_html/ZMass/cvh/<date>_fsr_photos/``.
"""
import argparse
import datetime
import math
import os
import struct

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import logging

import pubhtml
import ratiopanel

import fsr_analytic as FA

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

D = "data/photos"
#: analysis bands of ``00_moments.txt``; every edge is a generated band edge
BANDS = [(50, 60), (60, 70), (70, 80), (80, 86), (86, 96), (96, 110),
         (110, 130), (130, 150), (150, 200)]
TAILS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.2, 0.3,
         0.5, 1.0]
U_FINE, U_TAIL0, U_TAILW = 2e-5, 2.0, 0.05


# --------------------------------------------------------------------------
def sample_frac(path=f"{D}/mpre_bands.bin"):
    """The sample's weight fraction in each generated band."""
    blob = open(path, "rb").read()
    nb = struct.unpack_from("<i", blob, 0)[0]
    o = np.array([struct.unpack_from("<d", blob, 4 + i * 40 + 16)[0]
                  for i in range(nb)])
    return o / o.sum()


class Run:
    """One accumulation (the sample, or a standalone configuration)."""

    def __init__(self, path, label, is_mc=False):
        self.d = np.load(path, allow_pickle=True)
        self.label = label
        self.is_mc = is_mc
        self.label_tag = None
        self.frac = sample_frac()

    def idx(self, lo, hi):
        b0, b1 = self.d["bands_lo"], self.d["bands_hi"]
        return np.where((b0 >= lo - 1e-9) & (b1 <= hi + 1e-9))[0]

    def band(self, lo, hi):
        """Sample-weighted aggregation of the generated bands inside [lo, hi)."""
        i = self.idx(lo, hi)
        d = self.d
        if self.is_mc:
            r = np.ones(len(i))
        else:
            # the standalone is flat in band index; restore the sample's
            # population so <m_pre> and the kernel are the sample's
            r = self.frac[i] / np.maximum(d["n"][i], 1.0)
        out = {}
        for k in ("n", "n_nophot", "n_noemit", "n_norad", "mpre_s1", "over",
                  "nphot_s1", "npair"):
            if k in d.files:
                out[k] = float(np.sum(r * d[k][i]))
        # the sample's `npre == 0` is "Photos touched nothing", i.e. no photon
        # AND no pair; the standalone's photon-only counter is the same thing
        # only when pair emission is off
        out["p0"] = out.get("n_noemit", out["n_nophot"]) / out["n"]
        out["mom"] = np.einsum("i,ij->j", r, d["mom"][i])
        for k in ("fine_s0", "fine_s1", "fine_s2",
                  "tail_s0", "tail_s1", "tail_s2", "logu_h"):
            out[k] = np.einsum("i,ij->j", r, d[k][i])
        # raw generated (or weighted) count, for the Poisson error; a mixed
        # run carries its own generated count in `ngen`
        out["N"] = float(np.sum(d["ngen" if "ngen" in d.files else "n"][i]))
        out["neff"] = (float(np.sum(d["n"][i])) ** 2 /
                       float(np.sum(d["w2"][i]))) if "w2" in d.files else out["N"]
        return out

    def halves(self, lo, hi):
        """The two half-sample aggregations, for half-split errors."""
        i = self.idx(lo, hi)
        d = self.d
        out = []
        for t in ("A", "B"):
            if f"half{t}_n" not in d.files:
                return None
            r = (np.ones(len(i)) if self.is_mc
                 else self.frac[i] / np.maximum(d[f"half{t}_n"][i], 1.0))
            o = {"n": float(np.sum(r * d[f"half{t}_n"][i])),
                 "mom": np.einsum("i,ij->j", r, d[f"half{t}_mom"][i]),
                 "n_nophot": float(np.sum(r * d[f"half{t}_n_nophot"][i])),
                 "fine_s0": np.einsum("i,ij->j", r, d[f"half{t}_fine_s0"][i]),
                 "tail_s0": np.einsum("i,ij->j", r, d[f"half{t}_tail_s0"][i]),
                 "over": float(np.sum(r * d[f"half{t}_over"][i]))}
            out.append(o)
        return out


def u_edges(nfine, ntail):
    return np.concatenate([np.arange(nfine) * U_FINE,
                           U_TAIL0 + np.arange(ntail) * U_TAILW])


def tail_curve(b):
    """P(u > u_edge) on the histogram edges."""
    h = np.concatenate([b["fine_s0"], b["tail_s0"]])
    ue = u_edges(len(b["fine_s0"]), len(b["tail_s0"]))
    above = np.concatenate([np.cumsum(h[::-1])[::-1], [0.0]]) + b["over"]
    return ue, above / b["n"]


def tail_at(b, u0):
    ue, p = tail_curve(b)
    return float(p[np.searchsorted(ue, u0)])


def mean_u(b):
    return b["mom"][1] / b["n"]


class PairOnly:
    """The exact O(alpha^2) real-pair kernel with the photon switched off.

    ``K_u(u) = 2 z R_pair(z)`` from `fsr_analytic.pair_radiator`, the object a
    pair-only Photos run measures.  Photos 3.61 emits e+e- and mu+mu- pairs
    only (photosC.cxx: ``PHOPAR(..., 11, 0.000511, ...)`` and
    ``PHOPAR(..., 13, 0.1057, ...)``), so the matching species set is
    ``("e", "mu")``; ``table`` selects the exact or the eikonal (= what Photos
    generates) matrix element.
    """

    def __init__(self, m, species=("e", "mu"), table=None):
        self.m, self.species, self.table = float(m), tuple(species), table
        k = FA.FSRKernel(m, variant="born", pair=species, pair_table=table)
        self._c = k.pair_cells(n=2000)
        self._e = np.geomspace(k._pair._u[0],
                               min(k._pair._u[-1], k._u_max()), 2001)
        self.rate = float(self._c[0].sum())
        self.mean_u = float(self._c[1].sum())

    def pdf_u(self, u):
        return FA.pair_pdf_u(self.m, u, self.species, path=self.table)

    def tail(self, u0):
        """``P(u > u0)``: the pair kernel puts no weight at ``u = 0``."""
        u0 = np.atleast_1d(np.asarray(u0, float))
        cum = np.concatenate([np.cumsum(self._c[0][::-1])[::-1], [0.0]])
        j = np.clip(np.searchsorted(self._e, u0) - 1, 0, len(self._c[0]) - 1)
        frac = np.clip((np.log(self._e[j + 1]) - np.log(np.maximum(u0, 1e-300)))
                       / (np.log(self._e[j + 1]) - np.log(self._e[j])), 0, 1)
        return cum[j + 1] + frac * self._c[0][j]

    def p_norad(self, u0=1e-5):
        return 1.0 - float(self.tail(u0)[0])


def pair_table(runs, fh, tag_paironly="paironly",
               eik_table="data/fsr/pairkern_eik.npz"):
    def w(s=""):
        print(s)
        fh.write(s + "\n")
    r = [x for x in runs if x.label_tag == tag_paironly]
    if not r:
        return
    r = r[0]
    w()
    w("Pair emission alone (Photos with photon emission switched off) against "
      "the exact\nO(alpha^2) pair radiator and against its EIKONAL limit, "
      "which is the matrix\nelement Photos actually uses (pairs.cxx YOT1).")
    w("%-9s %10s %12s %9s %12s %9s %12s %9s" % (
        "band", "<m_pre>", "<u> Photos", "+-", "<u> eikonal", "Ph/eik",
        "<u> exact", "Ph/exact"))
    for lo, hi in BANDS:
        b = r.band(lo, hi)
        m = b["mpre_s1"] / b["n"]
        ke, kx = PairOnly(m, table=eik_table), PairOnly(m)
        sd = math.sqrt(max(b["mom"][2] / b["n"] - mean_u(b) ** 2, 0))
        w("%-9s %10.4f %12.5e %9.1e %12.5e %9.4f %12.5e %9.4f" % (
            f"{lo}-{hi}", m, mean_u(b), sd / math.sqrt(b["N"]),
            ke.mean_u, mean_u(b) / ke.mean_u,
            kx.mean_u, mean_u(b) / kx.mean_u))
    w()
    b = r.band(86, 96)
    m = b["mpre_s1"] / b["n"]
    w(f"Pair rate and mass loss by species, m_pre in [86,96), <m> = {m:.4f}:")
    w("%-26s %12s %12s %12s" % ("species", "N_pair", "<u>", "<u> eikonal"))
    for sp in (("e",), ("e", "mu"), ("e", "mu", "tau", "had")):
        kx, ke = PairOnly(m, sp), PairOnly(m, sp, table=eik_table)
        w("%-26s %12.5e %12.5e %12.5e"
          % ("/".join(sp), kx.rate, kx.mean_u, ke.mean_u))
    w()
    w(f"Tail of the pair-only kernel, m_pre in [86,96), <m> = {m:.4f}:")
    w("%9s %12s %12s %9s %12s %9s" % ("u0", "Photos", "eikonal", "Ph/eik",
                                      "exact", "Ph/exact"))
    ke, kx = PairOnly(m, table=eik_table), PairOnly(m)
    for u0 in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 0.5]:
        p = tail_at(b, u0)
        a, x = float(ke.tail(u0)[0]), float(kx.tail(u0)[0])
        w("%9.1e %12.6e %12.6e %9.4f %12.6e %9.4f"
          % (u0, p, a, p / a if a > 0 else 0, x, p / x if x > 0 else 0))
    w()
    w("Mean number of pair particles per event (2 per pair): %.5f"
      % (b.get("npair", 0.0) / b["n"]))


def syst_table(groups, fh):
    """Production-model dependence: variations against their own nominal.

    The O(alpha) FSR spectrum integrated over production angles is
    production-independent, so a kernel generated with a different Born
    angular distribution, a different incoming quark flavour or a boosted Z
    must come out the same.  Photos's *per-event* weight need not be, and with
    the exact Z matrix-element correction on it is not: that weight is a ratio
    of matrix elements at the generated production angle.
    """
    def w(s=""):
        print(s)
        fh.write(s + "\n")
    w()
    w("Production-model dependence.  Each variation against its own nominal, in\n"
      "the peak band and inclusively over the bands; the quoted error is the\n"
      "Poisson error on the ratio.")
    for nom, vars_ in groups:
        w()
        w(f"nominal: {nom.label}")
        w("%-26s %12s %10s %10s %10s %10s %10s" % (
            "variation", "<u> ratio", "+-", "P(u>1e-3)", "P(u>1e-2)",
            "P(u>0.1)", "P(u>0.5)"))
        bn = nom.band(86, 96)
        un = mean_u(bn)
        sd = math.sqrt(max(bn["mom"][2] / bn["n"] - un ** 2, 0))
        for v in vars_:
            bv = v.band(86, 96)
            uv = mean_u(bv)
            sv = math.sqrt(max(bv["mom"][2] / bv["n"] - uv ** 2, 0))
            e = (uv / un) * math.sqrt((sv / uv) ** 2 / bv["N"] +
                                      (sd / un) ** 2 / bn["N"])
            line = "%-26s %12.5f %10.5f" % (v.label, uv / un, e)
            for u0 in (1e-3, 1e-2, 0.1, 0.5):
                line += " %10.5f" % (tail_at(bv, u0) / tail_at(bn, u0))
            w(line)


# --------------------------------------------------------------------------
def table(runs, out, fh):
    def w(s=""):
        print(s)
        fh.write(s + "\n")

    w("Bands, sample-weighted.  MC errors are half-sample splits; the "
      "standalone's are Poisson\non its own N (given per row).")
    w()
    hdr = "%-9s %10s %10s %9s %9s" % ("band", "<m_pre>", "<u>", "+-",
                                       "P(none)")
    for r in runs[1:]:
        hdr += " %12s" % r.label[:12]
    w(hdr)
    for lo, hi in BANDS:
        b = [r.band(lo, hi) for r in runs]
        sd0 = math.sqrt(max(b[0]["mom"][2] / b[0]["n"] - mean_u(b[0]) ** 2, 0))
        line = "%-9s %10.4f %10.5e %9.2e %9.6f" % (
            f"{lo}-{hi}", b[0]["mpre_s1"] / b[0]["n"], mean_u(b[0]),
            sd0 / math.sqrt(b[0]["neff"]), b[0]["p0"])
        for k in range(1, len(runs)):
            line += " %12.5f" % (mean_u(b[k]) / mean_u(b[0]))
        w(line + "   <- <u> ratio to MC")
    w()
    # analytic reference
    w("<u> against the analytic kernels at the band's <m_pre>:")
    hdr = "%-9s %11s %11s %11s %9s %9s" % ("band", "<u> MC", "exp1", "exp2",
                                           "MC/exp1", "MC/exp2")
    for r in runs[1:]:
        hdr += " %11s" % (r.label[:11] + "/exp1")
    w(hdr)
    for lo, hi in BANDS:
        b = [r.band(lo, hi) for r in runs]
        m = b[0]["mpre_s1"] / b[0]["n"]
        e1 = FA.FSRKernel(m, variant="exp1").moments()["u"]
        e2 = FA.FSRKernel(m, variant="exp2").moments()["u"]
        line = "%-9s %11.5e %11.5e %11.5e %9.5f %9.5f" % (
            f"{lo}-{hi}", mean_u(b[0]), e1, e2, mean_u(b[0]) / e1,
            mean_u(b[0]) / e2)
        for k in range(1, len(runs)):
            line += " %11.5f" % (mean_u(b[k]) / e1)
        w(line)
    w()
    # tails in the peak band
    lo, hi = 86, 96
    b = [r.band(lo, hi) for r in runs]
    m = b[0]["mpre_s1"] / b[0]["n"]
    k1 = FA.FSRKernel(m, variant="exp1")
    k2 = FA.FSRKernel(m, variant="exp2")
    w(f"Tail P(u > u0), m_pre in [{lo},{hi}), <m_pre> = {m:.4f} GeV.")
    hdr = "%9s %11s %11s %11s" % ("u0", "P MC", "exp1", "exp2")
    for r in runs[1:]:
        hdr += " %11s" % r.label[:11]
    w(hdr + "     " + " ".join("%9s" % (r.label[:7] + "/MC") for r in runs[1:]))
    for u0 in TAILS:
        p = [tail_at(x, u0) for x in b]
        line = "%9.1e %11.6f %11.6f %11.6f" % (
            u0, p[0], float(k1.tail(u0)[0]), float(k2.tail(u0)[0]))
        for k in range(1, len(runs)):
            line += " %11.6f" % p[k]
        line += "     " + " ".join("%9.5f" % (p[k] / p[0])
                                   for k in range(1, len(runs)))
        w(line)
    w()
    w("Tail ratios to the sample, with the significance of the difference:")
    hdr = "%9s" % "u0"
    for r in runs[1:]:
        hdr += " %13s %7s" % (r.label[:11] + "/MC", "sigma")
    w(hdr)
    for u0 in TAILS:
        p = [tail_at(x, u0) for x in b]
        e0 = math.sqrt(p[0] * (1 - p[0]) / b[0]["neff"])
        line = "%9.1e" % u0
        for k in range(1, len(runs)):
            ek = math.sqrt(e0 ** 2 + p[k] * (1 - p[k]) / b[k]["N"])
            line += " %13.5f %7.1f" % (p[k] / p[0], (p[k] - p[0]) / ek)
        w(line)
    w()
    w("Same tails as ratios to the analytic exp1 / exp2:")
    hdr = "%9s %9s %9s" % ("u0", "MC/exp1", "MC/exp2")
    for r in runs[1:]:
        hdr += " %11s %11s" % (r.label[:9] + "/e1", r.label[:9] + "/e2")
    w(hdr)
    for u0 in TAILS:
        p = [tail_at(x, u0) for x in b]
        a1, a2 = float(k1.tail(u0)[0]), float(k2.tail(u0)[0])
        line = "%9.1e %9.5f %9.5f" % (u0, p[0] / a1, p[0] / a2)
        for k in range(1, len(runs)):
            line += " %11.5f %11.5f" % (p[k] / a1, p[k] / a2)
        w(line)
    w()
    w("Unradiated fraction: P(Photos emitted nothing at all).  The sample's flag\n"
      "is npre == 0 (no status-746 pre-Photos muon copy, written whenever Photos\n"
      "modified the muons - by a photon or by a pair).")
    hdr = "%-9s %11s %9s %11s %11s" % ("band", "MC", "+-", "exp1", "exp2")
    for r in runs[1:]:
        hdr += " %11s %7s" % (r.label[:11], "sigma")
    w(hdr)
    for lo, hi in BANDS:
        b = [r.band(lo, hi) for r in runs]
        m = b[0]["mpre_s1"] / b[0]["n"]
        p = b[0]["p0"]
        e = math.sqrt(p * (1 - p) / b[0]["neff"])
        line = "%-9s %11.6f %9.1e %11.6f %11.6f" % (
            f"{lo}-{hi}", p, e,
            FA.FSRKernel(m, variant="exp1").p_norad(),
            FA.FSRKernel(m, variant="exp2").p_norad())
        for k in range(1, len(runs)):
            q = b[k]["p0"]
            ek = math.sqrt(e ** 2 + q * (1 - q) / b[k]["N"])
            line += " %11.6f %7.1f" % (q, (q - p) / ek)
        w(line)


# --------------------------------------------------------------------------
COLORS = ["C0", "C1", "C2", "C4", "C5", "C6", "C7"]


def fig_tail_ratio(runs, out, band=(86, 96), ref="exp2"):
    runs = [r for r in runs if r.label_tag != "paironly"]
    b = [r.band(*band) for r in runs]
    m = b[0]["mpre_s1"] / b[0]["n"]
    k = FA.FSRKernel(m, variant=ref)
    # P(u > u0) is read off the histogram at the edge at or above u0, so the
    # model has to be evaluated at that same edge or the 2e-5 binning shows up
    # as a sawtooth below u ~ 1e-4
    ue_ = u_edges(len(b[0]["fine_s0"]), len(b[0]["tail_s0"]))
    u0 = np.unique(ue_[np.searchsorted(ue_, np.geomspace(4e-5, 1.2, 60))])
    a = np.array([float(k.tail(x)[0]) for x in u0])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    for i, (r, bb) in enumerate(zip(runs, b)):
        p = np.array([tail_at(bb, x) for x in u0])
        e = np.sqrt(np.maximum(p * (1 - p), 0) / bb["neff"])
        if r.is_mc:
            ax.errorbar(u0, p / a, yerr=e / a, fmt="o", ms=3.4, color="black",
                        lw=0, elinewidth=0.9, label=r.label, zorder=5)
        else:
            ax.plot(u0, p / a, color=COLORS[i % len(COLORS)], lw=1.8,
                    label=r.label)
            ax.fill_between(u0, (p - e) / a, (p + e) / a, alpha=0.18,
                            color=COLORS[i % len(COLORS)], lw=0)
    ax.axhline(1.0, color="grey", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("$u_0$")
    ax.set_ylabel(f"$P(u>u_0)$ / analytic {ref}")
    ax.set_title(f"$m_{{pre}}\\in[{band[0]},{band[1]})$ GeV, "
                 f"$\\langle m\\rangle={m:.3f}$ GeV, $\\beta={FA.beta_fsr(m):.5f}$",
                 fontsize=13, loc="left")
    ax.legend(fontsize=12, frameon=False, loc="lower left")
    ax.set_ylim(0.94, 1.07)
    pubhtml.savefig(fig, os.path.join(out, f"01_tail_ratio_{ref}.pdf"))
    plt.close(fig)


def fig_mean_u(runs, out):
    runs = [r for r in runs if r.label_tag != "paironly"]
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    xs, ys, es, mm = [], [], [], []
    for lo, hi in BANDS:
        b = runs[0].band(lo, hi)
        m = b["mpre_s1"] / b["n"]
        sd = math.sqrt(max(b["mom"][2] / b["n"] - mean_u(b) ** 2, 0))
        xs.append(m)
        ys.append(mean_u(b))
        es.append(sd / math.sqrt(b["neff"]))
        mm.append(m)
    xs, ys, es = map(np.array, (xs, ys, es))
    ax.errorbar(xs, ys * 1e3, yerr=es * 1e3, fmt="o", ms=4.5, color="black",
                lw=0, elinewidth=1.0, label=runs[0].label, zorder=5)
    curves = {}
    for nm, lab, col, ls in (("exp1", "analytic exp1", "C3", "-"),
                             ("exp2", "analytic exp2", "C9", "--")):
        y = np.array([FA.FSRKernel(m, variant=nm).moments()["u"] for m in xs])
        curves[lab] = y
        ax.plot(xs, y * 1e3, color=col, ls=ls, lw=1.7, label=lab)
    for i, r in enumerate(runs[1:], 1):
        y = np.array([mean_u(r.band(lo, hi)) for lo, hi in BANDS])
        curves[r.label] = y
        ax.plot(xs, y * 1e3, color=COLORS[i % len(COLORS)], lw=1.7, marker="s",
                ms=3.4, label=r.label)
    ax.set_ylabel(r"$\langle u\rangle \times 10^{3}$")
    ax.legend(fontsize=12, frameon=False)
    ax.set_title("mass dependence of the radiated fraction", fontsize=13,
                 loc="left")
    for lab, y in curves.items():
        col = "C3" if lab == "analytic exp1" else (
            "C9" if lab == "analytic exp2" else
            COLORS[list(curves).index(lab) % len(COLORS)])
        rax.plot(xs, ys / y, color=col, lw=1.6)
    rax.errorbar(xs, np.ones_like(xs), yerr=es / ys, fmt="o", ms=3.0,
                 color="black", lw=0, elinewidth=0.9)
    rax.axhline(1.0, color="grey", lw=0.9)
    rax.set_xlabel(r"$\langle m_{pre}\rangle$ [GeV]")
    rax.set_ylabel("MC / model")
    rax.set_ylim(0.96, 1.05)
    pubhtml.savefig(fig, os.path.join(out, "02_mean_u_vs_mpre.pdf"))
    plt.close(fig)


def fig_density(runs, out, band=(86, 96)):
    runs = [r for r in runs if r.label_tag != "paironly"]
    b = [r.band(*band) for r in runs]
    m = b[0]["mpre_s1"] / b[0]["n"]
    nlog = len(b[0]["logu_h"])
    e = np.geomspace(1e-6, 2.0, nlog + 1)
    c = np.sqrt(e[1:] * e[:-1])
    bw = np.diff(e)
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    dens = []
    for i, (r, bb) in enumerate(zip(runs, b)):
        d = bb["logu_h"] / (bb["n"] * bw)
        dens.append(d)
    ref = FA.FSRKernel(m, variant="exp2")
    y = ratiopanel.bin_average(ref.pdf_u(e[:-1]), ref.pdf_u(c), ref.pdf_u(e[1:]))
    ax.plot(c, y, color="C3", lw=1.8, label="analytic exp2", zorder=4)
    for i, (r, d) in enumerate(zip(runs, dens)):
        if r.is_mc:
            err = d / np.sqrt(np.maximum(b[i]["logu_h"] * b[i]["neff"] /
                                         max(b[i]["n"], 1), 1))
            ax.errorbar(c, d, yerr=err, fmt="o", ms=3.0, color="black", lw=0,
                        elinewidth=0.8, label=r.label, zorder=5)
            rax.errorbar(c, d / y, yerr=err / y, fmt="o", ms=2.6, color="black",
                         lw=0, elinewidth=0.8, zorder=5)
        else:
            ax.plot(c, d, color=COLORS[i % len(COLORS)], lw=1.5, label=r.label)
            rax.plot(c, d / y, color=COLORS[i % len(COLORS)], lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u$")
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    ax.set_title(f"$m_{{pre}}\\in[{band[0]},{band[1]})$ GeV", fontsize=13,
                 loc="left")
    rax.axhline(1.0, color="grey", lw=0.9)
    rax.set_xscale("log")
    rax.set_ylim(0.90, 1.10)
    rax.set_xlabel("$u$")
    rax.set_ylabel("/ exp2")
    pubhtml.savefig(fig, os.path.join(out, "03_u_density.pdf"))
    plt.close(fig)


def fig_pair(runs, out, band=(86, 96), tag="paironly"):
    r = [x for x in runs if x.label_tag == tag]
    if not r:
        return
    b = r[0].band(*band)
    m = b["mpre_s1"] / b["n"]
    nlog = len(b["logu_h"])
    e = np.geomspace(1e-6, 2.0, nlog + 1)
    c = np.sqrt(e[1:] * e[:-1])
    bw = np.diff(e)
    d = b["logu_h"] / (b["n"] * bw)
    err = d / np.sqrt(np.maximum(b["logu_h"] * b["N"] / max(b["n"], 1e-300), 1))
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.errorbar(c, d, yerr=err, fmt="o", ms=3.2, color="black", lw=0,
                elinewidth=0.9, label="Photos++, pairs only (photons off)",
                zorder=5)
    mods = {}
    for sp, lab, col, ls in ((("e", "mu"), r"analytic, $e$+$\mu$ pairs", "C3", "-"),
                             (("e",), r"analytic, $e^+e^-$ only", "C0", "--"),
                             (("e", "mu", "tau", "had"),
                              r"analytic, $e,\mu,\tau$, hadrons", "C2", ":")):
        k = PairOnly(m, sp)
        y = ratiopanel.bin_average(k.pdf_u(e[:-1]), k.pdf_u(c), k.pdf_u(e[1:]))
        mods[lab] = (y, col, ls)
        ax.plot(c, y, color=col, ls=ls, lw=1.8,
                label=lab + r", $N_p$=" + f"{k.rate:.3e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u$")
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    ax.set_title("real pair emission, " + r"$m_{pre}\in$" + f"[{band[0]},{band[1]}) GeV",
                 fontsize=13, loc="left")
    for lab, (y, col, ls) in mods.items():
        rax.plot(c, d / y, color=col, ls=ls, lw=1.6)
    rax.axhline(1.0, color="grey", lw=0.9)
    rax.set_xscale("log")
    rax.set_yscale("log")
    rax.set_xlabel("$u$")
    rax.set_ylabel("Photos/ana.")
    pubhtml.savefig(fig, os.path.join(out, "04_pair_only.pdf"))
    plt.close(fig)


def fig_syst(groups, out):
    """Tail ratio of each production-model variation to its own nominal."""
    for gi, (nom, vars_) in enumerate(groups):
        bn = nom.band(86, 96)
        u0 = np.geomspace(1e-5, 1.0, 40)
        pn = np.array([tail_at(bn, x) for x in u0])
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        for i, v in enumerate(vars_):
            bv = v.band(86, 96)
            pv = np.array([tail_at(bv, x) for x in u0])
            e = pv / pn * np.sqrt(np.maximum(1 - pv, 0) / (pv * bv["N"]) +
                                  np.maximum(1 - pn, 0) / (pn * bn["N"]))
            ax.plot(u0, pv / pn, color=COLORS[i % len(COLORS)], lw=1.6,
                    label=v.label)
            ax.fill_between(u0, pv / pn - e, pv / pn + e, alpha=0.15,
                            color=COLORS[i % len(COLORS)], lw=0)
        ax.axhline(1.0, color="grey", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("$u_0$")
        ax.set_ylabel(r"$P(u>u_0)$ / " + nom.label)
        ax.set_title(f"production-model dependence, nominal = {nom.label}",
                     fontsize=13, loc="left")
        ax.legend(fontsize=11, frameon=False, ncol=2)
        ax.set_ylim(0.985, 1.015)
        pubhtml.savefig(fig, os.path.join(out, f"05_syst_{nom.label}.pdf"))
        plt.close(fig)


def _syst_groups(spec):
    out = []
    for g in spec:
        nom, _, vv = g.partition(":")
        n = Run(f"{D}/gen_{nom}.npz", nom)
        n.label_tag = nom
        vs = []
        for t in vv.split(","):
            if not t:
                continue
            r = Run(f"{D}/gen_{t}.npz", t)
            r.label_tag = t
            vs.append(r)
        out.append((n, vs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="tag=label pairs, e.g. mcA='ME off'")
    ap.add_argument("--mcref", default=f"{D}/mcref.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="fsr_photos")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--syst", nargs="*", default=[],
                    help="nominal:var1,var2,... groups of run tags")
    a = ap.parse_args()
    out = a.out or pubhtml.figdir(a.tag)
    os.makedirs(out, exist_ok=True)

    runs = [Run(a.mcref, "Photos in the sample", is_mc=True)]
    for s in a.runs:
        t, _, lab = s.partition("=")
        r = Run(f"{D}/gen_{t}.npz", lab or t)
        r.label_tag = t
        runs.append(r)
    with open(os.path.join(out, "00_moments.txt"), "w") as fh:
        table(runs, out, fh)
        pair_table(runs, fh)
        if a.syst:
            syst_table(_syst_groups(a.syst), fh)
    if not a.no_figs:
        fig_tail_ratio(runs, out, ref="exp2")
        fig_tail_ratio(runs, out, ref="exp1")
        fig_mean_u(runs, out)
        fig_density(runs, out)
        fig_pair(runs, out)
        if a.syst:
            fig_syst(_syst_groups(a.syst), out)
        pubhtml.ensure_index(out, logger=logger)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
