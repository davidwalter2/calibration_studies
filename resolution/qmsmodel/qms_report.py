#!/usr/bin/env python3
"""Tables and figures for the `material_tec_services` tension between the
quadratic hit-chi2 term and the unbinned mass (Moliere/Urban CF) term.

Inputs (all made by the other scripts in this directory):
  tmp/gun_qms_24f.npz        qms_steps.py       -- per-group Q(Rossi) vs Moliere
  tmp/gun_quadgrad.npz       qms_quadgrad.py    -- per-candidate quadratic gradient
  tmp/gun_quadgrad_nocut.npz qms_quadgrad.py --max-chi2-ndof 0 ...
  tmp/gun_census36.npz       qms_census.py      -- per-candidate step census

Figures go one panel per file, with a lower panel wherever there is a
reference to divide by.
"""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "matres"))
import groups as G                                             # noqa: E402
import pubhtml                                                 # noqa: E402
from wums import logging                                       # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

GROUPFILE = ("/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
             "Analysis/HitAnalyzer/data/materialGroups50.txt")
AUX = ["gradmax", "hessmax", "ptp", "etap", "ptm", "etam",
       "dErefp", "dErefm", "ndof", "chisqval"]
MAIN = ["tec_services", "tob_services", "pixel_patch", "tibtid_services",
        "bpix_services", "tec_structure", "tib_support", "tob_support",
        "tid_support", "fpix_support"]


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def two_panel(figsize=(11.0, 8.0), hr=(3.0, 1.2)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=list(hr), hspace=0.07)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    ax.tick_params(labelbottom=False)
    return fig, ax, rax


def load(tmp):
    names, priors = G.read_groups(GROUPFILE)
    q = np.load(os.path.join(tmp, "gun_quadgrad.npz"))
    qn = np.load(os.path.join(tmp, "gun_quadgrad_nocut.npz"))
    s = np.load(os.path.join(tmp, "gun_qms_24f.npz"))
    c = np.load(os.path.join(tmp, "gun_census36.npz"))
    pt, sub = q["parmtype"], q["subidx"]
    mat = np.where(pt == 15)[0]
    lab = [names.get(int(sub[i]), "?") for i in mat]
    pri = np.array([priors.get(int(sub[i]), 0.2) for i in mat])
    return dict(names=names, priors=priors, q=q, qn=qn, s=s, c=c,
                mat=mat, lab=lab, pri=pri, gid=[int(sub[i]) for i in mat])


def khat(g, k):
    K = k.sum()
    if K <= 1e-12:
        return np.nan, np.nan, np.nan
    return -g.sum() / K, np.sqrt(2.0 / K), np.sqrt((g ** 2).sum()) / K


# ---------------------------------------------------------------- tables
def table_ratio(D, out, npzdir):
    s = D["s"]
    n, thp2, seff, s2ndk = s["n"], s["thp2"], s["seff"], s["s2ndk"]
    w = s["w"]
    nc = int(s["ncand"])
    live = np.where(n > 0)[0]
    tot = thp2[live].sum()
    rows = []
    for g in live:
        if n[g] < 50:
            continue
        rows.append(dict(gid=int(g), name=D["names"].get(int(g), "?"),
                         nstep=int(n[g]), xg=s["xg"][g] / nc,
                         dox0=s["dox0"][g] / nc,
                         xgstep=s["xg"][g] / n[g],
                         effZ=s["w_effZ"][g] / w[g], x0g=s["w_x0g"][g] / w[g],
                         zza=s["w_zza"][g] / w[g], lm=s["w_lm"][g] / w[g],
                         Rcore=seff[0][g] / thp2[g], Rlo=seff[1][g] / thp2[g],
                         Rhi=seff[2][g] / thp2[g], R2nd=s2ndk[g] / thp2[g],
                         share=thp2[g] / tot))
    rows.sort(key=lambda r: -r["share"])
    print(f"\n=== TABLE 1. Q(Rossi) vs Moliere per material group "
          f"({nc} J/psi-gun candidates, {int(s['nstep'])} steps) ===")
    print(f"{'group':22s} {'MSshare':>8s} {'steps/cand':>10s} {'xg/step':>9s} "
          f"{'sum d/X0':>9s} {'effZ':>6s} {'X0_g':>7s} {'Lm':>6s} "
          f"{'R_core':>7s} {'R_2nd':>7s} {'R(u/3)':>7s} {'R(3u)':>7s}")
    for r in rows:
        print(f"{r['name']:22s} {r['share']*100:7.2f}% {r['nstep']/nc:10.2f} "
              f"{r['xgstep']:9.4f} {r['dox0']:9.5f} {r['effZ']:6.2f} "
              f"{r['x0g']:7.2f} {r['lm']:6.2f} {r['Rcore']:7.4f} "
              f"{r['R2nd']:7.4f} {r['Rlo']:7.4f} {r['Rhi']:7.4f}")
    print(f"{'ALL GROUPS':22s} {100.0:7.2f}% {int(s['nstep'])/nc:10.2f} "
          f"{'':9s} {s['dox0'][live].sum()/nc:9.5f} {'':6s} {'':7s} {'':6s} "
          f"{seff[0][live].sum()/tot:7.4f} {s2ndk[live].sum()/tot:7.4f} "
          f"{seff[1][live].sum()/tot:7.4f} {seff[2][live].sum()/tot:7.4f}")
    np.savez(os.path.join(npzdir, "table_ratio.npz"),
             name=np.array([r["name"] for r in rows]),
             gid=np.array([r["gid"] for r in rows]),
             **{k: np.array([r[k] for r in rows]) for k in
                ("share", "xgstep", "dox0", "effZ", "x0g", "zza", "lm",
                 "Rcore", "R2nd", "Rlo", "Rhi", "nstep")})
    return rows


def table_quad(D, out, npzdir):
    q, lab, mat, pri = D["q"], D["lab"], D["mat"], D["pri"]
    Gv, K = q["G"], q["K"]
    gi, kd = q["gi"], q["kd"]
    n = len(Gv)
    P = np.zeros(n)
    P[mat] = 1.0 / pri ** 2
    A = K + 2.0 * np.diag(P)
    th = np.linalg.solve(A, -Gv)
    cov = 2.0 * np.linalg.inv(A)
    err = np.sqrt(np.diag(cov))
    S = (gi ** 2).sum(0)
    Kd = np.diag(K)[mat]
    cg = S / np.maximum(2.0 * Kd, 1e-300)
    print(f"\n=== TABLE 2. the quadratic (hit-chi2) term's material block, "
          f"{int(q['nsel'])} gun candidates ===")
    print(f"{'group':22s} {'k(marg)':>9s} {'sigma':>8s} {'pull':>7s} "
          f"{'k(stand)':>9s} {'sigF':>8s} {'sigSW':>8s} {'c_g':>6s} "
          f"{'occ%':>6s} {'top30 %G':>9s}")
    rows = []
    for i, nm in enumerate(lab):
        if Kd[i] <= 0:
            continue
        j = mat[i]
        o = np.argsort(-np.abs(gi[:, i]))
        f30 = np.cumsum(gi[o, i])[29] / gi[:, i].sum() if gi[:, i].sum() else np.nan
        rows.append((nm, th[j], err[j], th[j] / err[j],
                     -gi[:, i].sum() / Kd[i], np.sqrt(2.0 / Kd[i]),
                     np.sqrt(S[i]) / Kd[i], cg[i], (kd[:, i] != 0).mean() * 100,
                     f30 * 100))
    for r in sorted(rows, key=lambda r: r[3]):
        print(f"{r[0]:22s} {r[1]:9.4f} {r[2]:8.4f} {r[3]:7.2f} {r[4]:9.4f} "
              f"{r[5]:8.4f} {r[6]:8.4f} {r[7]:6.2f} {r[8]:6.1f} {r[9]:9.1f}")
    np.savez(os.path.join(npzdir, "table_quad.npz"),
             name=np.array([r[0] for r in rows]),
             **{k: np.array([r[i + 1] for r in rows]) for i, k in enumerate(
                 ("kmarg", "sig", "pull", "kstand", "sigF", "sigSW", "cg",
                  "occ", "f30"))})
    return rows


def table_fr(D, out, npzdir):
    qn, lab = D["qn"], D["lab"]
    gi, kd, c2, aux = qn["gi"], qn["kd"], qn["c2"], qn["aux"]
    A = {c: i for i, c in enumerate(AUX)}
    pp = aux[:, A["ptp"]] * np.cosh(aux[:, A["etap"]])
    pm = aux[:, A["ptm"]] * np.cosh(aux[:, A["etam"]])
    fr = np.maximum(np.abs(aux[:, A["dErefp"]]) / np.maximum(pp, 1e-9),
                    np.abs(aux[:, A["dErefm"]]) / np.maximum(pm, 1e-9))
    base = (np.abs(aux[:, A["gradmax"]]) < 1e6) & (np.abs(aux[:, A["hessmax"]]) < 1e8)
    edges = [0.0, 0.003, 0.01, 0.03, 0.1, 1.0]
    print("\n=== TABLE 3. k_hat (standalone, NO chi2 cut) vs the fractional "
          "reference energy loss dE_ref/p of the more affected leg ===")
    print(f"{'group':20s}" + "".join(f"{f'{lo:g}-{hi:g}':>18s}"
                                     for lo, hi in zip(edges[:-1], edges[1:])))
    res = {}
    for nm in MAIN:
        i = lab.index(nm)
        out_s, vals = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = base & (fr >= lo) & (fr < hi)
            v, e, _ = khat(gi[m, i], kd[m, i])
            vals.append((v, e, kd[m, i].sum()))
            out_s.append(f"{v:8.3f}+-{e:6.3f}" if np.isfinite(v) else f"{'--':>16s}")
        print(f"{nm:20s}" + "".join(f"{o:>18s}" for o in out_s))
        res[nm] = vals
    print(f"\n{'group':20s}" + "".join(f"{f'%info {lo:g}-{hi:g}':>18s}"
                                       for lo, hi in zip(edges[:-1], edges[1:])))
    for nm in MAIN:
        i = lab.index(nm)
        Kt = kd[:, i].sum()
        print(f"{nm:20s}" + "".join(
            f"{kd[base & (fr >= lo) & (fr < hi), i].sum()/Kt*100:17.1f}%"
            for lo, hi in zip(edges[:-1], edges[1:])))
    np.savez(os.path.join(npzdir, "table_fr.npz"), edges=np.array(edges),
             name=np.array(MAIN),
             val=np.array([[v for v, e, k in res[n]] for n in MAIN]),
             err=np.array([[e for v, e, k in res[n]] for n in MAIN]),
             info=np.array([[k for v, e, k in res[n]] for n in MAIN]))
    return fr, base, res


def table_cut(D):
    qn, lab = D["qn"], D["lab"]
    gi, kd, c2, aux = qn["gi"], qn["kd"], qn["c2"], qn["aux"]
    A = {c: i for i, c in enumerate(AUX)}
    base = (np.abs(aux[:, A["gradmax"]]) < 1e6) & (np.abs(aux[:, A["hessmax"]]) < 1e8)
    print("\n=== TABLE 4. is it the chi2/ndof selection? k_hat vs the cut ===")
    sel = ["tec_services", "tob_support", "tec_structure", "tib_support"]
    print(f"{'cut':>10s} {'keep%':>7s}" + "".join(f"{n:>19s}" for n in sel))
    for cut in (1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9):
        m = base & (c2 < cut)
        o = []
        for nm in sel:
            i = lab.index(nm)
            v, e, _ = khat(gi[m, i], kd[m, i])
            o.append(f"{v:9.4f}+-{e:6.4f}")
        print(f"{cut:10.1f} {m.mean()*100:6.2f}%" + "".join(f"{x:>19s}" for x in o))


# ---------------------------------------------------------------- figures
def fig_ratio(D, rows, outdir):
    x = np.arange(len(rows))
    Rc = np.array([r["Rcore"] for r in rows])
    R2 = np.array([r["R2nd"] for r in rows])
    Rlo = np.array([r["Rlo"] for r in rows])
    Rhi = np.array([r["Rhi"] for r in rows])
    nm = [r["name"] for r in rows]
    sh = np.array([r["share"] for r in rows])
    tot_c = (Rc * sh).sum() / sh.sum()
    fig, ax, rax = two_panel(figsize=(13.0, 8.5))
    ax.fill_between(x, Rhi, Rlo, color="0.85", step="mid",
                    label=r"core scale $u^\ast/3 \ldots 3u^\ast$")
    ax.step(x, Rc, where="mid", color="k", lw=2,
            label=r"$\sigma^2_{\rm Moliere\ core}/\sigma^2_Q$")
    ax.step(x, R2, where="mid", color="tab:red", lw=2,
            label=r"$\sigma^2_{\rm 2nd\ moment}/\sigma^2_Q$")
    ax.axhline(1.0, color="0.4", ls=":")
    ax.axhline(np.exp(0.465), color="tab:blue", ls="--", lw=2,
               label=r"$e^{0.465}$ = what a variance fit would need")
    j = nm.index("tec_services")
    ax.plot([x[j]], [Rc[j]], "o", ms=12, mfc="none", mec="tab:green", mew=3,
            label="tec_services")
    ax.set_ylabel("Moliere / $Q$\nprojected-angle variance", fontsize=15)
    ax.set_ylim(0.75, 1.75)
    ax.legend(fontsize=12, ncol=2, loc="upper left")
    rax.step(x, Rc / tot_c, where="mid", color="k", lw=2)
    rax.plot([x[j]], [Rc[j] / tot_c], "o", ms=12, mfc="none", mec="tab:green",
             mew=3)
    rax.axhline(1.0, color="0.4", ls=":")
    rax.set_ylabel("/ all-group", fontsize=13)
    rax.set_ylim(0.9, 1.12)
    rax.set_xticks(x)
    rax.set_xticklabels(nm, rotation=90, fontsize=9)
    rax.set_xlabel("material group, ordered by share of the tracker MS variance")
    save(fig, outdir, "ratio_per_group")


def fig_ratio_vs_lm(D, rows, outdir):
    lm = np.array([r["lm"] for r in rows])
    Rc = np.array([r["Rcore"] for r in rows])
    R2 = np.array([r["R2nd"] for r in rows])
    sh = np.array([r["share"] for r in rows])
    nm = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    sc = ax.scatter(lm, R2, s=40 + 3000 * sh, c="tab:red", alpha=0.6,
                    label=r"second moment $\chi_c^2 L_m/2$ (with the G4 electron ceiling)")
    ax.scatter(lm, Rc, s=40 + 3000 * sh, c="k", alpha=0.6,
               label=r"core-matched $-2S(u^\ast)/u^{\ast 2}$")
    for a, b, n in zip(lm, Rc, nm):
        if n in ("tec_services", "tib_support", "tec_structure", "tob_support",
                 "pixel_patch", "beampipe"):
            ax.annotate(n, (a, b), fontsize=10, xytext=(4, -12),
                        textcoords="offset points")
    ax.axhline(1.0, color="0.4", ls=":")
    ax.set_xlabel(r"$L_m = \ln(\theta_{FF}^2/\chi_a^2) - 1$  ($x_g$-weighted group mean)")
    ax.set_ylabel(r"Moliere / $Q$ variance ratio")
    ax.legend(fontsize=13)
    ax.set_title("marker area $\\propto$ the group's share of the MS variance",
                 fontsize=13)
    _ = sc
    save(fig, outdir, "ratio_vs_moliere_log")


def fig_leverage(D, outdir):
    q, lab = D["q"], D["lab"]
    gi, kd = q["gi"], q["kd"]
    fig, ax, rax = two_panel(figsize=(10.0, 8.0))
    N = gi.shape[0]
    rank = np.arange(1, N + 1)
    for nm, col in (("tec_services", "tab:green"), ("tob_support", "k"),
                    ("tec_structure", "tab:red"), ("tib_support", "tab:blue")):
        i = lab.index(nm)
        o = np.argsort(-np.abs(gi[:, i]))
        cg = np.cumsum(gi[o, i]) / gi[:, i].sum()
        ck = np.cumsum(kd[o, i]) / kd[:, i].sum()
        ax.plot(rank, cg, color=col, lw=2, label=nm)
        rax.plot(rank, ck, color=col, lw=2)
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.4", ls=":")
    ax.set_ylabel(r"cumulative fraction of $\sum_i G_i$", fontsize=15)
    ax.set_ylim(-0.2, 1.6)
    ax.legend(fontsize=13, loc="lower right")
    rax.set_xscale("log")
    rax.set_ylabel(r"of $\sum_i K_{ii}$", fontsize=13)
    rax.set_xlabel(f"candidates ranked by $|G_i|$  (of {N})")
    save(fig, outdir, "leverage_concentration")


def fig_fr(D, outdir, npzdir):
    z = np.load(os.path.join(npzdir, "table_fr.npz"))
    ed, nm, val, err, info = (z["edges"], list(z["name"]), z["val"], z["err"],
                              z["info"])
    ctr = np.sqrt(ed[1:] * np.maximum(ed[:-1], 1e-4))
    fig, ax, rax = two_panel(figsize=(10.0, 8.0))
    cols = {"tec_services": "tab:green", "tob_services": "tab:orange",
            "pixel_patch": "tab:purple", "tibtid_services": "tab:brown",
            "tec_structure": "tab:red", "tib_support": "tab:blue",
            "tob_support": "k", "tid_support": "0.5",
            "bpix_services": "tab:cyan", "fpix_support": "0.7"}
    for i, n in enumerate(nm):
        n = str(n)
        if n not in ("tec_services", "tob_services", "pixel_patch",
                     "tec_structure", "tib_support", "tob_support"):
            continue
        m = np.isfinite(val[i]) & (err[i] < 1.0)
        ax.errorbar(ctr[m], val[i][m], yerr=err[i][m], marker="o", ms=6,
                    color=cols.get(n, "k"), lw=2, capsize=3, label=n)
        rax.plot(ctr[m], info[i][m] / info[i][np.isfinite(info[i])].sum(),
                 marker="s", ms=5, color=cols.get(n, "k"), lw=1.5)
    ax.axhline(0.0, color="0.4", ls=":")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\hat{k}_g$  (hit-$\chi^2$, standalone)", fontsize=15)
    ax.set_ylim(-1.6, 1.3)
    ax.legend(fontsize=12, loc="upper left", ncol=2)
    ax.set_title("MC truth is $k_g = 0$; no $\\chi^2/$ndof cut applied",
                 fontsize=13)
    rax.set_xscale("log")
    rax.set_ylabel("share of\n" + r"$\sum K_{gg}$", fontsize=12)
    rax.set_xlabel(r"$\Delta E_{\rm ref}/p$ of the more affected leg")
    save(fig, outdir, "khat_vs_fractional_eloss")


def fig_bias_vs_x(D, outdir, npzdir):
    """The fr>0.1 bias against two candidate explanatory variables, one file
    each: the group's areal density per Geant4 step (the energy-loss
    hypothesis) and the group's Moliere/Q MS variance ratio (the Q hypothesis).
    """
    tr = np.load(os.path.join(npzdir, "table_ratio.npz"))
    tf = np.load(os.path.join(npzdir, "table_fr.npz"))
    rmap = {str(n): i for i, n in enumerate(tr["name"])}
    xs, rc, ys, es, ns = [], [], [], [], []
    for i, n in enumerate(tf["name"]):
        n = str(n)
        if n not in rmap:
            continue
        v, e = tf["val"][i][-1], tf["err"][i][-1]
        if not np.isfinite(v) or e > 0.5:
            continue
        j = rmap[n]
        xs.append(tr["xgstep"][j])
        rc.append(tr["Rcore"][j])
        ys.append(v)
        es.append(e)
        ns.append(n)
    for name, x, xlab, logx in (
            ("bias_vs_stepthickness", np.array(xs),
             r"areal density per Geant4 step $\langle x_g\rangle$ [g/cm$^2$]",
             True),
            ("bias_vs_msratio", np.array(rc),
             r"Moliere-core / $Q$ MS variance ratio $R_{\rm core}$", False)):
        fig, ax = plt.subplots(figsize=(9.5, 7.0))
        ax.errorbar(x, ys, yerr=es, fmt="o", ms=9, color="k", capsize=4)
        for a, b, n in zip(x, ys, ns):
            ax.annotate(n, (a, b), fontsize=10, xytext=(6, 6),
                        textcoords="offset points")
        ax.axhline(0.0, color="0.4", ls=":")
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\hat{k}_g$ at $\Delta E_{\rm ref}/p > 0.1$",
                      fontsize=15)
        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0.88, 0.98)
        r = np.corrcoef(np.log(x) if logx else x, ys)[0, 1]
        ax.set_title(f"Pearson r = {r:.2f}", fontsize=14)
        save(fig, outdir, name)


def fig_sandwich(D, outdir, npzdir):
    z = np.load(os.path.join(npzdir, "table_quad.npz"))
    nm = [str(x) for x in z["name"]]
    cg = z["cg"]
    occ = z["occ"]
    keep = occ > 1.0
    o = np.argsort(-cg[keep])
    x = np.arange(keep.sum())
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.bar(x, cg[keep][o], color=["tab:green" if np.array(nm)[keep][o][i]
                                  == "tec_services" else "0.5"
                                  for i in range(keep.sum())])
    ax.axhline(1.0, color="k", ls="--", lw=2,
               label=r"$\sum_i G_{i,g}^2 = 2K_{gg}$ (the fit's own weighting)")
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(nm)[keep][o], rotation=90, fontsize=9)
    ax.set_ylabel(r"$c_g=\sum_i G_{i,g}^2 / (2K_{gg})$", fontsize=15)
    ax.set_title(r"the quoted error on $k_g$ is too small by $\sqrt{c_g}$",
                 fontsize=14)
    ax.legend(fontsize=13)
    save(fig, outdir, "sandwich_factor")


def fig_frcut(D, outdir, tmp):
    """The exact 92-parameter quadratic fit, restricted to candidates whose
    reference energy loss is a small fraction of the momentum.  This is the
    resolution of the tension: at dE_ref/p < 0.01 the hit-chi2 term and the
    mass CF term give the same number for every material group."""
    f = os.path.join(tmp, "gun_quadgrad_frcut.npz")
    if not os.path.exists(f):
        logger.warning("no gun_quadgrad_frcut.npz -- skipping fig_frcut")
        return
    z = np.load(f)
    names, priors = G.read_groups(GROUPFILE)
    pt, sub = z["parmtype"], z["subidx"]
    mat = np.where(pt == 15)[0]
    lab = [names.get(int(sub[i]), "?") for i in mat]
    pri = np.array([priors.get(int(sub[i]), 0.2) for i in mat])
    P = np.zeros(len(z["G"]))
    P[mat] = 1.0 / pri ** 2

    def solve(Gv, K):
        A = K + 2.0 * np.diag(P)
        return np.linalg.solve(A, -Gv), np.sqrt(np.diag(2.0 * np.linalg.inv(A)))

    sets = [("no cut", z["G"], z["K"])]
    for j, cut in enumerate(z["fr_cuts"]):
        sets.append((rf"$\Delta E_{{\rm ref}}/p<{cut:g}$", z["Gf"][j], z["Kf"][j]))
    res = [solve(g, k) for _, g, k in sets]
    order = [i for i in np.argsort([res[0][0][mat[i]] / res[0][1][mat[i]]
                                    for i in range(len(mat))])
             if any(abs(r[0][mat[i]] / r[1][mat[i]]) > 0.4 for r in res)]
    x = np.arange(len(order))
    fig, ax, rax = two_panel(figsize=(11.0, 8.0), hr=(2.6, 1.2))
    cols = ("k", "tab:orange", "tab:blue")
    for isx, ((nm, _, _), r) in enumerate(zip(sets, res)):
        ax.errorbar(x + 0.22 * (isx - 1), [r[0][mat[i]] for i in order],
                    yerr=[r[1][mat[i]] for i in order], fmt="o", ms=7,
                    color=cols[isx], capsize=3, lw=2, label=nm)
    ax.errorbar([x[[lab[i] for i in order].index("tec_services")] + 0.66],
                [-0.0087], yerr=[0.095], fmt="s", ms=9, color="tab:green",
                capsize=4, lw=2, label="mass CF term (24k cand)")
    ax.axhline(0.0, color="0.4", ls=":")
    ax.set_ylabel(r"$k_g$   (MC truth 0)", fontsize=15)
    ax.legend(fontsize=12, ncol=2)
    for isx, ((nm, _, _), r) in enumerate(zip(sets, res)):
        rax.plot(x + 0.22 * (isx - 1), [r[0][mat[i]] / r[1][mat[i]]
                                        for i in order], "o", ms=7,
                 color=cols[isx])
    rax.axhline(0.0, color="0.4", ls=":")
    for y in (-1, 1):
        rax.axhline(y, color="0.75", ls="--", lw=1)
    rax.set_ylabel("pull", fontsize=13)
    rax.set_xticks(x)
    rax.set_xticklabels([lab[i] for i in order], rotation=90, fontsize=10)
    save(fig, outdir, "frcut_material_fit")


def fig_toy(outdir):
    """The two mechanisms, one file each, in the toy of qms_toy.py."""
    import qms_toy as TOY
    N, m = 12, np.full(11, 0.02)
    v = (1.0 * m) ** 2
    sh2 = 0.05 ** 2
    rs = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    kr, er, rr = [], [], []
    for i, r in enumerate(rs):
        ks, _, _, sig = TOY.run(20000, N, m, v, sh2, rvar=r, seed=101 + i)
        kr.append(ks.mean())
        er.append(ks.std() / np.sqrt(len(ks)))
        rr.append(ks.std() / sig)
    fig, ax, rax = two_panel(figsize=(9.5, 8.0), hr=(2.4, 1.2))
    ax.errorbar(rs, kr, yerr=er, fmt="o", ms=9, color="k", capsize=4,
                label=r"toy $\langle\hat{k}\rangle$, mean-only derivative")
    ax.plot(rs, np.zeros_like(rs), "k:", lw=1)
    ax.plot(rs, -np.log(rs), "--", color="tab:blue", lw=2,
            label=r"$-\ln r$: what a VARIANCE fit would give")
    ax.axhline(-0.465, color="tab:green", lw=2,
               label=r"the fitted $k$(tec_services)")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\langle\hat{k}\rangle$", fontsize=16)
    ax.legend(fontsize=12, loc="lower left")
    ax.set_title(r"A wrong process-noise $Q$ shifts no parameter",
                 fontsize=14)
    rax.plot(rs, rr, "s-", color="tab:red", ms=7, lw=2)
    rax.plot(rs, np.sqrt(rs), ":", color="0.4", lw=2)
    rax.set_xscale("log")
    rax.set_ylabel("rms / Fisher $\\sigma$", fontsize=12)
    rax.set_xlabel(r"$r$ = true straggling variance / model $Q$")
    save(fig, outdir, "toy_wrong_variance")

    rhos = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.5])
    kp, ep = [], []
    for i, rho in enumerate(rhos):
        ks, _, _, _ = TOY.run(20000, N, m, v, sh2, rho=rho, seed=201 + i)
        kp.append(ks.mean())
        ep.append(ks.std() / np.sqrt(len(ks)))
    kp = np.asarray(kp)
    fig, ax, rax = two_panel(figsize=(9.5, 8.0), hr=(2.4, 1.2))
    ax.errorbar(rhos, kp, yerr=ep, fmt="o", ms=9, color="k", capsize=4,
                label=r"toy $\langle\hat{k}\rangle$")
    ax.plot(rhos, rhos - 1.0, "--", color="tab:blue", lw=2,
            label=r"$\rho-1$, the linearisation of $\ln\rho$ the fit uses")
    ax.axhline(-0.465, color="tab:green", lw=2,
               label=r"the fitted $k$(tec_services)")
    ax.axvline(1.0 - 0.465, color="tab:green", ls=":", lw=2)
    ax.set_ylabel(r"$\langle\hat{k}\rangle$", fontsize=16)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_title("A wrong MEAN energy loss is recovered exactly",
                 fontsize=14)
    rax.plot(rhos, kp - (rhos - 1.0), "s-", color="tab:red", ms=7, lw=2)
    rax.axhline(0.0, color="0.4", ls=":")
    rax.set_ylabel(r"$\langle\hat{k}\rangle-(\rho-1)$", fontsize=12)
    rax.set_xlabel(r"$\rho$ = true mean loss / model mean loss")
    save(fig, outdir, "toy_wrong_mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=os.path.join(HERE, "tmp"))
    ap.add_argument("--outpath", default=None)
    args = ap.parse_args()
    stamp = datetime.date.today().strftime("%y%m%d")
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{stamp}_qmsmodel")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    D = load(args.tmp)
    rows = table_ratio(D, outdir, args.tmp)
    table_quad(D, outdir, args.tmp)
    table_fr(D, outdir, args.tmp)
    table_cut(D)
    fig_ratio(D, rows, outdir)
    fig_ratio_vs_lm(D, rows, outdir)
    fig_leverage(D, outdir)
    fig_fr(D, outdir, args.tmp)
    fig_bias_vs_x(D, outdir, args.tmp)
    fig_sandwich(D, outdir, args.tmp)
    fig_toy(outdir)
    fig_frcut(D, outdir, args.tmp)
    logger.info(f"figures in {outdir}")


if __name__ == "__main__":
    main()
