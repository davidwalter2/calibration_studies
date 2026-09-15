#!/usr/bin/env python3
"""The machinery-only benchmark in the cell-integrated table representation.

Reads the MC's own selection-conditional kernel table (`fsr_table.py cond`)
and the two-leg model's table (`fsr_table.py corr`) and draws them against
each other band by band, plus the fit shifts of the atom and the table form of
both sides.
"""
import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wums import logging

import pubhtml
import ratiopanel

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

#: the mass bands the kernel is drawn in
SHOW = ((80.0, 85.0), (88.0, 91.0), (91.0, 94.0), (98.0, 105.0))


def load(path):
    with np.load(path, allow_pickle=True) as d:
        out = {k: np.asarray(d[k], float) for k in
               ("m_nodes", "u_edges", "K", "u_mean", "p0")}
        if "provenance" in d.files:
            out["prov"] = json.loads(str(d["provenance"][0]))
    return out


def row_at(t, m):
    """The table's row at mass ``m``, interpolated the way the provider does."""
    mn = t["m_nodes"]
    x = float(np.clip(m, mn[0], mn[-1]))
    i = int(np.clip(np.searchsorted(mn, x) - 1, 0, len(mn) - 2))
    f = (x - mn[i]) / (mn[i + 1] - mn[i])
    K = (1.0 - f) * t["K"][i] + f * t["K"][i + 1]
    M1 = ((1.0 - f) * t["K"][i] * t["u_mean"][i]
          + f * t["K"][i + 1] * t["u_mean"][i + 1])
    p0 = (1.0 - f) * t["p0"][i] + f * t["p0"][i + 1]
    return K, M1, p0


def survival(u_edges, K, p0, grid):
    """``P(u > u_0)`` of a table row on ``grid``.

    The survival, not the density: the standalone kernel's cells are 2e-5 wide
    and a finer display grid turns it into a comb, while the survival is the
    same object however the cells fall and is exactly what the fold integrates.
    """
    tail = np.concatenate([np.cumsum(K[::-1])[::-1], [0.0]])
    return np.interp(grid, u_edges, tail)


def fig_ksel(out, specs, name="50_ksel_band", n_show=80):
    """`P(u > u_0 | m)` of every table, band by band, over the reference."""
    grid = np.geomspace(1e-5, 1.5, n_show)
    fig, axs = plt.subplots(2, len(SHOW), figsize=(4.6 * len(SHOW), 7.4),
                            sharex=True, height_ratios=(3.0, 1.2),
                            gridspec_kw=dict(hspace=0.06))
    for j, band in enumerate(SHOW):
        m = 0.5 * (band[0] + band[1])
        ax, rax = axs[0, j], axs[1, j]
        ref = None
        for i, (lab, t, sty) in enumerate(specs):
            K, M1, p0 = row_at(t, m)
            S = survival(t["u_edges"], K, p0, grid)
            if ref is None:
                ref = S
            ax.plot(grid, S, color=sty[0], ls=sty[1], lw=1.7,
                    label=f"{lab}   $p_0$ = {p0:.4f}")
            with np.errstate(divide="ignore", invalid="ignore"):
                rax.plot(grid, np.where(ref > 0, S / ref, np.nan),
                         color=sty[0], ls=sty[1], lw=1.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{band[0]:.0f} < $m_{{\\rm pre}}$ < {band[1]:.0f} GeV",
                     fontsize=12, loc="left")
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
        rax.axhline(1.0, color="grey", lw=1.0)
        rax.set_ylim(0.9, 1.1)
        rax.set_xlabel(r"$u_0$")
        rax.grid(alpha=0.2)
        if j == 0:
            ax.set_ylabel(r"$P(u > u_0\,|\,m)$, selected")
            rax.set_ylabel("/ MC", fontsize="small")
            ax.legend(fontsize=8.5, loc="lower left")
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def fig_moments(out, specs, name="51_moments"):
    """`p_0(m)` and `<u|m>` of every table against the MC's own."""
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=(3.0, 1.2), hspace=0.06,
                          wspace=0.25)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    r0, r1 = fig.add_subplot(gs[1, 0], sharex=ax0), fig.add_subplot(gs[1, 1],
                                                                    sharex=ax1)
    mm = np.linspace(62.0, 125.0, 300)
    ref = None
    for lab, t, sty in specs:
        p0 = np.empty(len(mm))
        mu = np.empty(len(mm))
        for i, m in enumerate(mm):
            K, M1, p = row_at(t, m)
            p0[i], mu[i] = p, M1.sum()
        if ref is None:
            ref = (p0, mu)
        ax0.plot(mm, p0, color=sty[0], ls=sty[1], lw=1.7, label=lab)
        ax1.plot(mm, mu * 1e3, color=sty[0], ls=sty[1], lw=1.7, label=lab)
        r0.plot(mm, p0 - ref[0], color=sty[0], ls=sty[1], lw=1.7)
        r1.plot(mm, mu / ref[1], color=sty[0], ls=sty[1], lw=1.7)
    ax0.set_ylabel(r"$p_0(m)$, selected")
    ax1.set_ylabel(r"$\langle u | m\rangle \times 10^{3}$")
    r0.set_ylabel(r"$-$ MC", fontsize="small")
    r1.set_ylabel("/ MC", fontsize="small")
    r0.axhline(0.0, color="grey", lw=1.0)
    r1.axhline(1.0, color="grey", lw=1.0)
    r1.set_ylim(0.94, 1.06)
    for a in (ax0, ax1):
        a.tick_params(labelbottom=False)
        a.grid(alpha=0.25)
        a.legend(fontsize=9)
    for a in (r0, r1):
        a.set_xlabel(r"$m_{\rm pre}$ [GeV]")
        a.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


#: the discretisation scans, `(panel, label, suite, [(x, row)])`
CONVERGE = (
    ("cond table: cells (2 GeV bands)", "full sample", "conv",
     ((1000, "cond table, 2 GeV, 1000 cells"), (2000, "cond table, 2 GeV bands"),
      (4000, "cond table, 2 GeV, 4000 cells"), (8000, None))),
    ("cond table: cells (2 GeV bands)", "half sample", "conv2",
     ((1000, "cond table, 2 GeV, half A, 1000 cells"),
      (2000, "cond table, 2 GeV, half A"),
      (4000, "cond table, 2 GeV, half A, 4000 cells"), (8000, None))),
    ("model table: mass nodes [GeV]", "standalone K", "conv2",
     ((2.0, None), (1.0, None), (0.5, "model mc K, 0.5 GeV nodes"),
      (0.25, "model mc K, 0.25 GeV nodes"))),
    ("model table: mass nodes [GeV]", "sample K", "conv2",
     ((1.0, "model sample K, 1 GeV nodes"),
      (0.5, "model sample K, 0.5 GeV nodes"),
      (0.25, "model sample K, 0.25 GeV nodes"))),
)
#: points a panel's own suite does not carry, `(panel, label) -> {x: (suite, row)}`
CONVERGE_EXTRA = {
    ("model table: mass nodes [GeV]", "standalone K"): {
        2.0: ("conv", "model table, 2 GeV nodes"),
        1.0: ("conv", "model table, 1 GeV nodes")},
    ("cond table: cells (2 GeV bands)", "full sample"): {
        8000: ("noise", "cond table, 2 GeV, 8000 cells")},
    ("cond table: cells (2 GeV bands)", "half sample"): {
        8000: ("noise", "cond table, 2 GeV, half A, 8000 cells")},
}


def fig_converge(out, fits, name="53_converge"):
    """The fit shift against each discretisation knob, both parameters."""
    panels = []
    for pan, lab, suite, pts in CONVERGE:
        if pan not in panels:
            panels.append(pan)
    fig, axs = plt.subplots(2, len(panels), figsize=(6.4 * len(panels), 8.0),
                            sharex="col")
    seen = {}
    for pan, lab, suite, pts in CONVERGE:
        j = panels.index(pan)
        x, v, e = [], [[], []], [[], []]
        for xv, row in pts:
            src = (suite, row)
            if row is None:
                src = CONVERGE_EXTRA.get((pan, lab), {}).get(xv)
                if src is None:
                    continue
            d = fits.get(src[0], {}).get(src[1])
            if d is None:
                continue
            x.append(xv)
            for k, p in enumerate(("m_Z", "Gamma_Z")):
                v[k].append(d[p][0])
                e[k].append(d[p][1])
        if not x:
            continue
        c = f"C{seen.setdefault(lab, len(seen))}"
        for k in range(2):
            axs[k, j].errorbar(x, v[k], e[k], fmt="o-", ms=6, lw=1.5,
                               color=c, label=lab, capsize=3)
    for j, pan in enumerate(panels):
        axs[0, j].set_title(pan, fontsize=12, loc="left")
        for k, lab in enumerate((r"$\Delta m_Z$ [MeV]",
                                 r"$\Delta\Gamma_Z$ [MeV]")):
            a = axs[k, j]
            a.set_xscale("log")
            a.grid(alpha=0.25)
            xs = sorted({p[0] for pn, _, _, ps in CONVERGE if pn == pan
                         for p in ps})
            a.set_xticks(xs)
            a.set_xticks([], minor=True)
            a.set_xticklabels([f"{v:g}" for v in xs])
            if j == 0:
                a.set_ylabel(lab)
            if k == 1:
                a.set_xlabel(pan.split(":")[1].strip())
        axs[0, j].legend(fontsize=9)
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def diag_record(gen, out_path, pt_cut=25.0, eta_cut=2.4, wclip=100.0):
    """The record-level effects the machinery test can still carry.

    * the **eta decision**: the model takes it on the PRE-FSR muons and calls
      it FSR-independent; the record says how often FSR moves a muon across
      ``|eta| = 2.4`` and what that is worth in ``A(m)``;
    * the **soft floor**: the mass the record cannot resolve, ``u < 1e-5``,
      which the measured kernel puts in its ``p0`` while the standalone
      histograms resolve it -- the residual offset in ``<u>`` and what it is
      worth on ``m_Z``;
    * the **weight clipping**, which must be the same on both sides.
    """
    import fit_gen as FG

    d = np.load(gen)
    w = FG.clip_weights(np.asarray(d["weight"], np.float64), wclip)
    m = np.asarray(d["m_pre"], np.float64)
    mp = np.asarray(d["m_post"], np.float64)
    pt1, pt2 = np.asarray(d["pt1"], float), np.asarray(d["pt2"], float)
    e1, e2 = np.abs(np.asarray(d["eta1"], float)), np.abs(d["eta2"]).astype(float)
    e1p = np.abs(np.asarray(d["eta1_pre"], float))
    e2p = np.abs(np.asarray(d["eta2_pre"], float))
    L = []
    A = L.append
    A("the machinery test's own record-level floors")
    A("=" * 68)
    sw = w.sum()
    epost = (e1 < eta_cut) & (e2 < eta_cut)
    epre = (e1p < eta_cut) & (e2p < eta_cut)
    flip = epost != epre
    A(f"eta decision: pre != post on {np.sum(w * flip) / sw:.3e} of the sample "
      f"({int(flip.sum())} events)")
    ptok = (pt1 > pt_cut) & (pt2 > pt_cut)
    A(f"  ... and inside the pT cut: "
      f"{np.sum(w * flip * ptok) / sw:.3e}")
    win = (mp > 60.0) & (mp < 120.0)
    a_post = np.sum(w * ptok * epost * win) / max(np.sum(w * win), 1e-30)
    a_pre = np.sum(w * ptok * epre * win) / max(np.sum(w * win), 1e-30)
    A(f"  A(60-120) with the post-FSR eta = {a_post:.6f}, with the pre-FSR "
      f"eta = {a_pre:.6f}, ratio - 1 = {a_pre / a_post - 1:.3e}")
    u = -np.log(np.minimum(mp / m, 1.0))
    sel = ptok & epost
    for lab, s_ in (("inclusive", np.ones(len(u), bool)), ("selected", sel)):
        ws = np.where(s_, w, 0.0)
        t = ws.sum()
        soft = (u > 0.0) & (u < 1e-5)
        A(f"soft floor, {lab}: P(0 < u < 1e-5) = "
          f"{np.sum(ws * soft) / t:.5f}, its <u> contribution = "
          f"{np.sum(ws * soft * u) / t:.3e}  "
          f"(-> {91.1535e3 * np.sum(ws * soft * u) / t:+.3f} MeV of m_Z "
          f"if it moved into p0)")
        A(f"  P(u = 0 exactly) = {np.sum(ws * (u <= 0.0)) / t:.5f}")
    aw = np.abs(np.asarray(d["weight"], np.float64))
    A(f"weights: modal |w| = {np.median(aw):.6g}, clipped at {wclip}x, "
      f"{np.sum(aw > wclip * np.median(aw))} events clipped, "
      f"Neff/N = {sw ** 2 / np.sum(w * w) / len(w):.4f}")
    txt = "\n".join(L)
    print(txt)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(txt + "\n")


#: the rows the summary reads, `(label, suite, row)`
SUM_ROWS = (
    ("cond:true atoms, per-leg record", "table", "cond:true atoms, per-leg record"),
    ("cond:true atoms", "table", "cond:true atoms"),
    ("cond:true table, atom bands", "table", "cond:true table"),
    ("cond:true table, 2 GeV bands", "noise", "cond table, 2 GeV"),
    ("cond:true table, 2 GeV, 8000 cells", "noise", "cond table, 2 GeV, 8000 cells"),
    ("model multi atoms, mc K", "table", "model multi atoms"),
    ("model multi table, mc K, 1 GeV", "table", "model multi table"),
    ("model multi table, mc K, 0.5 GeV", "conv2", "model mc K, 0.5 GeV nodes"),
    ("model multi table, sample K, 1 GeV", "table", "model multi table, sample K"),
    ("model multi table, sample K, 0.5 GeV", "noise", "model sample K, 0.5 GeV"),
)

#: the same-run differences the verdict rests on, `(label, suite, a, b)`
SUM_DIFF = (
    ("machinery, atoms (model - cond, per-leg record)", "table",
     "model multi atoms", "cond:true atoms, per-leg record"),
    ("machinery, atoms (model - cond, same events)", "table",
     "model multi atoms", "cond:true atoms"),
    ("machinery, tables (mc K, 1 GeV nodes)", "table",
     "model multi table", "cond:true table"),
    ("MACHINERY, converged tables, the MC's own K", "noise",
     "model sample K, 0.5 GeV", "cond table, 2 GeV, 8000 cells"),
    ("atom bias, model side (atoms - table)", "table",
     "model multi atoms", "model multi table"),
    ("atom bias, cond side (atoms - table)", "table",
     "cond:true atoms", "cond:true table"),
    ("... of which the acceptance grid (model side)", "table",
     "model multi table, atom A", "model multi table"),
    ("the per-leg record (per-leg - same events)", "table",
     "cond:true atoms, per-leg record", "cond:true atoms"),
    ("the kernel floor (mc K - sample K), 1 GeV nodes", "table",
     "model multi table", "model multi table, sample K"),
    ("the kernel floor, inclusive tables", "incl",
     "inclusive mc standalone table", "inclusive sample table"),
    ("the kernel floor, inclusive atoms", "incl",
     "inclusive mc standalone atoms", "inclusive sample atoms"),
    ("the sharing (multi - single), sample K", "conv2",
     "model sample K, 0.5 GeV nodes", "model sample K, single, 0.5 GeV"),
    ("rho from the sample's own K", "conv2",
     "model sample K + rho, 1 GeV", "model sample K, 1 GeV nodes"),
    ("cond table statistical floor (half A - half B)", "conv", 
     "cond table, 2 GeV, half A", "cond table, 2 GeV, half B"),
    ("model table statistical floor (half A - half B)", "noise",
     "model sample K, 0.5 GeV, half A", "model sample K, 0.5 GeV, half B"),
)


def summary(fits, path):
    """The rows, the same-run differences and the verdict, as text."""
    L = []
    A = L.append
    A("the machinery-only benchmark in the cell-integrated table representation")
    A("=" * 74)
    A("")
    A(f"{'row':52s} {'d m_Z':>10s} {'d Gamma_Z':>11s}")
    for lab, suite, row in SUM_ROWS:
        d = fits.get(suite, {}).get(row)
        if d is None:
            A(f"{lab:52s}   [missing: {suite}/{row}]")
            continue
        A(f"{lab:52s} {d['m_Z'][0]:+10.3f} {d['Gamma_Z'][0]:+11.3f}")
    A("")
    A("same-run differences")
    A("-" * 74)
    for lab, suite, a, b in SUM_DIFF:
        da, db = fits.get(suite, {}).get(a), fits.get(suite, {}).get(b)
        if da is None or db is None:
            A(f"{lab:52s}   [missing]")
            continue
        A(f"{lab:52s} {da['m_Z'][0]-db['m_Z'][0]:+10.3f} "
          f"{da['Gamma_Z'][0]-db['Gamma_Z'][0]:+11.3f}")
    A("")
    a = fits.get("noise", {})
    b = fits.get("conv", {})
    if a and b:
        sc = [abs(b["cond table, 2 GeV, half A"][k][0]
                  - b["cond table, 2 GeV, half B"][k][0]) / 2.0
              for k in ("m_Z", "Gamma_Z")]
        sm = [abs(a["model sample K, 0.5 GeV, half A"][k][0]
                  - a["model sample K, 0.5 GeV, half B"][k][0]) / 2.0
              for k in ("m_Z", "Gamma_Z")]
        A("the test's own statistical floor (half-sample splits)")
        A("-" * 74)
        A(f"{'sigma(cond table)':52s} {sc[0]:10.3f} {sc[1]:11.3f}")
        A(f"{'sigma(model table)':52s} {sm[0]:10.3f} {sm[1]:11.3f}")
        A(f"{'sigma(difference)':52s} "
          f"{(sc[0]**2 + sm[0]**2)**0.5:10.3f} "
          f"{(sc[1]**2 + sm[1]**2)**0.5:11.3f}")
    A("")
    A("convergence")
    A("-" * 74)
    for suite, pat in (("conv", "cond table"), ("conv2", "model"),
                       ("noise", "")):
        for row, d in fits.get(suite, {}).items():
            if pat and not row.startswith(pat):
                continue
            A(f"{suite:6s} {row:45s} {d['m_Z'][0]:+10.3f} "
              f"{d['Gamma_Z'][0]:+11.3f}")
    txt = "\n".join(L)
    print(txt)
    with open(path, "w") as fh:
        fh.write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", action="append", default=[],
                    help="label=path.npz[:color:style]; the FIRST is the "
                         "reference of every ratio")
    ap.add_argument("--fit", action="append", default=[],
                    help="label=fit_*.json (repeatable)")
    ap.add_argument("--summary", action="append", default=[],
                    help="suite=fit_*.json (repeatable): the summary table")
    ap.add_argument("--record", default=None,
                    help="a gen record: print the test's own floors")
    ap.add_argument("--outpath", default=None)
    ap.add_argument("--tag", default="fsr_machinery")
    a = ap.parse_args()

    out = a.outpath or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today():%y%m%d}_{a.tag}")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out, logger=logger)
    logger.info(f"figures -> {out}")

    specs = []
    for i, spec in enumerate(a.table):
        lab, _, rest = spec.partition("=")
        parts = rest.split(":")
        path = parts[0]
        col = parts[1] if len(parts) > 1 else f"C{i}"
        ls = parts[2] if len(parts) > 2 else "-"
        if not os.path.exists(path):
            logger.warning(f"missing {path}")
            continue
        specs.append((lab, load(path), (col, ls)))
    if specs:
        fig_ksel(out, specs)
        fig_moments(out, specs)
    if a.fit:
        import cmp_perleg as CP
        CP.fig_fitshifts(out, a.fit, name="52_fitshifts")
    if a.record:
        diag_record(a.record, os.path.join(out, "01_floors.txt"))
    if a.summary:
        fits = {}
        for spec in a.summary:
            lab, _, path = spec.partition("=")
            if os.path.exists(path):
                with open(path) as fh:
                    fits[lab] = json.load(fh)
            else:
                logger.warning(f"missing {path}")
        summary(fits, os.path.join(out, "00_machinery.txt"))
        fig_converge(out, fits)


if __name__ == "__main__":
    main()
