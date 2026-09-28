#!/usr/bin/env python3
"""The error budget: closure residuals -> momentum-scale bias of the Z and
J/psi mass fits.

Inputs
  runs/error_budget_260928/response_{z_gprof,jpsi}.json   d alpha / d eps per
        family on the fit (error_budget.py response)
  runs/error_budget_260928/frame_*.json   the closure's d(even, odd)/d eps per
        family, per plane, in each closure frame (error_budget_frames.py)
  the published closure JSONs (realmat_closure.py, realmat_ditrack.py)

THE CONVERSION.  A family g (a physical change of one channel: its kick size,
rate, tail at fixed variance, odd part) with amplitude eps moves the closure by
eps s_g(u) (frame) and the fitted scale by eps a_g (fit).  An observed residual
Delta(u) is read as eps_g = argmin sum_u (Delta - eps s_g)^2/err^2 and is worth
d alpha = eps_g a_g.  The nine probes share events, so the error of eps_g is
taken as the best SINGLE probe's, err(u)/|s_g(u)| minimised over u (exact for
fully correlated probes, conservative otherwise); the bound is |eps_g| + 2 err.
A family enters an item only where its channel carries >= 5 % of the frame's
core (its width family's response at u = 1; a frame blind to a channel says
nothing about it), and only in the statistic that measures it: the even probes
the width, rate and tail families, the odd probes the odd-part family.  The
item's worth is the largest bound over its families.  Location (odd) probes of the ditracks
are also read directly: the mass location dm(u) is a mass shift, d alpha =
dm/m, on the ladder (the loss up to the lever arm's middle) and on the
outermost plane (the whole track).

The conversion coefficient reported per frame is  C_g = a_g / s_g(u = 1):
the scale bias per unit of even (or odd) closure at u = 1.

usage: python error_budget_apply.py [--figs]
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
sys.path.insert(0, RES)
RUNS = os.path.join(RES, "runs", "error_budget_260928")
MZ, MJPSI = 91.1876, 3.0969
TARGET = 1e-5
U = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
IU1 = U.index(1.0)
FITS = ("z", "zG", "jpsi")          # Z alpha only, Z with Gamma profiled, J/psi
SHARE_MIN = 0.05


def load_fits():
    z = json.load(open(os.path.join(RUNS, "response_z_gprof.json")))
    j = json.load(open(os.path.join(RUNS, "response_jpsi.json")))
    a = {}
    for k, r in z["families"].items():
        a.setdefault(k, {})["z"] = r["dalpha"]
        a[k]["zG"] = r["dalpha_gprof"]
    for k, r in j["families"].items():
        a.setdefault(k, {})["jpsi"] = r["dalpha"]
    return a, z, j


def load_frame(name):
    return json.load(open(os.path.join(RUNS, f"frame_{name}.json")))


def frame_sig(fr, fam, stat, where):
    """The family's closure signature on one plane or the ladder mean."""
    P = fr["planes"]
    ks = sorted(P, key=int)
    if where == "outer":
        return np.asarray(P[ks[-1]]["fam"][fam][stat])
    return np.mean([P[k]["fam"][fam][stat] for k in ks], axis=0)


def frame_share(fr, c, where):
    """The channel's share of the frame, measured by the frame's response to
    the channel's width family at u = 1 (the core): |s_W_c(1)| / sum_c'.
    (The small-t curvature of a Moliere or Landau channel is its full second
    moment, set by the extreme tail, and is no measure of what the core sees.)"""
    w = {cc: abs(frame_sig(fr, f"W_{cc}", "even", where)[IU1])
         for cc in ("ioni", "rad", "ms")}
    return float(w[c] / max(sum(w.values()), 1e-30))


def fit_eps(delta, err, sig):
    delta, err, sig = map(np.asarray, (delta, err, sig))
    m = np.isfinite(delta) & np.isfinite(err) & (err > 0) & (np.abs(sig) > 0)
    w = 1.0 / err[m] ** 2
    eps = float(np.sum(w * delta[m] * sig[m]) / np.sum(w * sig[m] ** 2))
    e1 = float(np.min(err[m] / np.abs(sig[m])))
    return eps, e1


# -------------------------------------------------------------------------
# the observed closures
# -------------------------------------------------------------------------

def single_obs(lab, func, where):
    p = os.path.join(RES, "runs", "realmat_closure_260925",
                     f"closure_{lab}_{func}_off_nucel0_mat_kj1_qx1.json")
    c = json.load(open(p))["cells"][func]
    r, e = np.asarray(c["rows"]), np.asarray(c["errs"])
    if where == "outer":
        return r[-1], e[-1], p
    return r.mean(axis=0), np.asarray(c["err_ladder"]), p


def ditrack_obs(path, stat, where):
    d = json.load(open(os.path.join(RES, "runs", path)))
    if where == "ladder":
        return (np.asarray(d["ladder"][stat]), np.asarray(d["ladder"][stat + "_err"]),
                d)
    r = d["planes"][-1]
    return np.asarray(r[stat]), np.asarray(r[stat + "_err"]), d


# -------------------------------------------------------------------------
# the items
# -------------------------------------------------------------------------

FAMS = ("W", "R", "T", "O")
# which families a statistic measures: the even probes the even part of the
# channel (its width, rate, tail), the odd probes its odd part (and the shift)
STAT_FAMS = {"even": ("W", "R", "T"), "odd": ("O",)}


def evaluate(delta, err, fr, stat, where, A, chans=("ioni", "rad", "ms")):
    """Per family in the frame: eps, its error, and the bias on each fit."""
    rows = []
    for c in chans:
        sh = frame_share(fr, c, where)
        if sh < SHARE_MIN:
            continue
        for f in STAT_FAMS[stat]:
            fam = f"{f}_{c}"
            if f == "O" and c == "ms":
                continue
            s = frame_sig(fr, fam, stat, where)
            if not np.any(np.abs(s) > 0):
                continue
            eps, e1 = fit_eps(delta, err, s)
            rec = dict(fam=fam, share=sh, eps=eps, eps_err=e1, s_u1=float(s[IU1]))
            for fit in FITS:
                a = A[fam][fit]
                rec[fit] = dict(central=eps * a, bound=(abs(eps) + 2 * e1) * abs(a),
                                coef_per_u1=(a / s[IU1] if s[IU1] != 0 else float("nan")))
            rows.append(rec)
    return rows


def worst(rows, fit):
    if not rows:
        return None
    r = max(rows, key=lambda x: x[fit]["bound"])
    return dict(fam=r["fam"], central=r[fit]["central"], bound=r[fit]["bound"],
                eps=r["eps"], eps_err=r["eps_err"])


def items(A):
    """Every closure number that bears on the Z-mass scale, with its frame."""
    fr = {n: load_frame(n) for n in ("mum_qop", "mum_locx", "mum_dxdz",
                                     "mup_qop", "mup_locx", "mup_dxdz",
                                     "ditrack_jpsi", "ditrack_zmm")}
    out = []
    # --- published single-track muon closures, realmat_full, pT 3
    for lab in ("mum", "mup"):
        for func in ("qop", "locx", "dxdz"):
            for where in ("ladder", "outer"):
                d, e, src = single_obs(lab, func, where)
                rows = evaluate(d, e, fr[f"{lab}_{func}"], "even", where, A)
                out.append(dict(item=f"{'mu-' if lab == 'mum' else 'mu+'} {func} "
                                     f"(realmat, pT 3), {where}",
                                kind="measured", delta_u1=float(d[IU1]),
                                err_u1=float(e[IU1]), rows=rows, source=src))
    # --- ditracks: even and odd
    for name, path, fname, m in (
            ("J/psi -> mumu (realmat, 200 k)",
             "realmat_ditrack_260926/ditrack_closure_off_nucel0_mat_kj1_qx1.json",
             "ditrack_jpsi", MJPSI),
            ("Z -> mumu (realmat, pT 48, 1.8 M)",
             "realmat_ditrack_zmm_260927_xstat/ditrack_closure_kj1_qx1.json",
             "ditrack_zmm", MZ)):
        for where in ("ladder", "outer"):
            for stat in ("even", "odd"):
                d, e, raw = ditrack_obs(path, stat, where)
                rows = evaluate(d, e, fr[fname], stat, where, A)
                rec = dict(item=f"{name}, {stat}, {where}", kind="measured",
                           delta_u1=float(d[IU1]), err_u1=float(e[IU1]), rows=rows,
                           source=path)
                if stat == "odd":
                    rec["location"] = location(raw, where, m)
                out.append(rec)
    return out, fr


def location(d, where, m):
    """The mass location dm(u) read directly: u = 0.01 (the mean, tails
    included) and u = 0.1; only probes where D(u) > 0.2 D(0.001) (the odd
    probe's response to a shift has not turned over)."""
    P = d["planes"]
    sel = [P[-1]] if where == "outer" else P
    res = {}
    for u in (0.01, 0.1):
        i = U.index(u)
        dm = np.mean([r["dm_MeV"][i] for r in sel])
        er = np.mean([abs(r["dm_err_MeV"][i]) for r in sel])
        okD = all(r["D"][i] > 0.2 * r["D"][0] for r in sel)
        res[f"u={u}"] = dict(dm_MeV=float(dm), err_MeV=float(er), valid=bool(okD),
                             rel=float(dm / (1e3 * m)), rel_err=float(er / (1e3 * m)))
    return res


def open_items(A, fr):
    """CLOSURE_STATE.md's open items, each converted with the frame that
    carries its channel (a muon frame standing in where the item is measured on
    another species or geometry -- marked 'proxy')."""
    out = []

    def single(item, frame, delta_u1, err_u1, where, chans, note, kind="proxy",
               stat="even"):
        f = fr[frame]
        rows = []
        for c in chans:
            if frame_share(f, c, where) < SHARE_MIN:
                continue
            for fl in STAT_FAMS[stat]:
                fam = f"{fl}_{c}"
                if fam not in f["planes"][sorted(f["planes"], key=int)[-1]]["fam"]:
                    continue
                if fl == "O" and c == "ms":
                    continue
                s = frame_sig(f, fam, stat, where)[IU1]
                if s == 0:
                    continue
                eps = delta_u1 / s
                e1 = err_u1 / abs(s)
                rec = dict(fam=fam, eps=eps, eps_err=e1, s_u1=float(s))
                for fit in FITS:
                    a = A[fam][fit]
                    rec[fit] = dict(central=eps * a, bound=(abs(eps) + 2 * e1) * abs(a),
                                    coef_per_u1=a / s)
                rows.append(rec)
        out.append(dict(item=item, kind=kind, delta_u1=delta_u1, err_u1=err_u1,
                        rows=rows, note=note))

    def direct(item, fam, eps, note, kind="direct"):
        rows = [dict(fam=fam, eps=eps, eps_err=0.0,
                     **{fit: dict(central=eps * A[fam][fit], bound=abs(eps * A[fam][fit]),
                                  coef_per_u1=float("nan")) for fit in FITS})]
        out.append(dict(item=item, kind=kind, rows=rows, note=note))

    def rel(item, r_central, r_bound, applies, note):
        out.append(dict(item=item, kind="location", rows=[], note=note,
                        rel=dict(central=r_central, bound=r_bound, applies=applies)))

    # 1. real-tracker q/p radial growth (pT 3, eta 0.3), outermost plane u = 1
    single("real tracker q/p, outermost plane, +0.0676 (u = 1) -- as a shape",
           "mum_qop", 0.0676, 0.0006, "outer", ("ioni", "rad"),
           "the realmat mu- q/p frame (same ray, same material sequence) as the "
           "stand-in for the real tracker; the item is a LOCATION effect, read "
           "below")
    # the same as a location: model - data median +1.10 sigma of q/p at plane 18,
    # sigma(q/p)/(q/p) = 4.8e-4 (realmat, same ray) -> 5.3e-4 of p at the
    # outermost plane; ladder mean of the published per-plane medians
    med = [0.0064, 0.0023, -0.0540, -0.0639, -0.0458, 0.0484, 0.1514, 0.2626,
           0.3663, 0.6692, 1.0979]
    rel("real tracker q/p location (median, model - data), pT 3",
        float(np.mean(med)) * 3.0e-4, 1.0979 * 4.83e-4, "J/psi muons (pT ~ 3)",
        "outermost plane 1.10 sigma x sigma(q/p)/(q/p) 4.8e-4 = 5.3e-4 of p; "
        "the plane average of the eleven published medians (x the ladder's "
        "typical sigma 3e-4) is the central; a fixed-reference closure "
        "quantity -- a fit's reference follows each track's own path")
    # 2. 16-19 % wider at the outer two planes (q/p width): W on the loss channels
    direct("real tracker q/p 16-19 % wider (outer two planes), as a loss-width "
           "error eps = +0.17", "W_ioni", 0.17,
           "applied to the ionisation width of the fit's candidates; an upper "
           "reading, since the fit's q/p block is the whole-track one")
    # 3. layered-toy locx outermost regression, -0.0022 at u = 1 (mu, seven corr.)
    single("layered toy local x, outermost plane, -0.0022 (u = 1, mu)",
           "mum_locx", -0.0022, 0.0004, "outer", ("ms",),
           "the realmat mu- local-x frame as the stand-in; on realmat the same "
           "direction closes (-0.3e-3)")
    # 4. pT 0.8 pion scattering directions +0.9e-3 (ladder)
    single("pT 0.8 pi scattering directions +0.9e-3 (ladder), if present for mu",
           "mum_dxdz", 0.0009, 0.0002, "ladder", ("ms",),
           "hadron item; converted as if the same offset sat in the muon's "
           "bending-plane angle")
    rel("Lambda mass location +0.02...+0.03 MeV (outer planes)", 2.0e-5, 2.8e-5,
        "Lambda only (51 deg incidence, pT 0.8 pion)",
        "slope second-order term at steep incidence; at pT 3 tan(alpha) <= 0.2 "
        "of it and the J/psi location closes at 1e-5")
    # 5. pi+- q/p -0.5e-3
    single("pi+- q/p -0.5e-3 (ladder), if present for mu", "mum_qop", -0.0005,
           0.0002, "ladder", ("ioni", "rad"), "hadron item; muon q/p frame")
    # 6. K+ elastic dx/dz +0.5e-3 (outer) and plane-0 dx/dz -1.25e-3
    single("K+ dx/dz +0.5e-3 (elastic, outer planes), if present for mu",
           "mum_dxdz", 0.0005, 0.0002, "outer", ("ms",),
           "elastic kicks < 10 mrad; hadron-only channel")
    # 7. delta-ray production-cut floor 0.00027 on toy closures (q/p)
    single("delta-ray production-cut floor 0.00027 (toy q/p)", "mum_qop", 0.00027,
           0.0, "ladder", ("ioni",), "a floor on the toys, <= 2 % on the real "
           "tracker")
    return out


# -------------------------------------------------------------------------

def coefficient_table(A, fr):
    """C_g = a_g / s_g(u = 1) per frame, family and fit, ladder and outer."""
    rows = []
    for name, f in fr.items():
        for where in ("ladder", "outer"):
            for c in ("ioni", "rad", "ms"):
                sh = frame_share(f, c, where)
                for fl in FAMS:
                    fam = f"{fl}_{c}"
                    if fl == "O" and c == "ms":
                        continue
                    for stat in ("even", "odd"):
                        if stat == "odd" and not name.startswith("ditrack"):
                            continue
                        s = frame_sig(f, fam, stat, where)[IU1]
                        rows.append(dict(frame=name, where=where, fam=fam, stat=stat,
                                         share=sh, s_u1=float(s),
                                         **{fit: (A[fam][fit] / s if s else None)
                                            for fit in FITS}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args()
    A, z, j = load_fits()
    meas, fr = items(A)
    opn = open_items(A, fr)
    coef = coefficient_table(A, fr)
    out = dict(fit_response=A, fit_meta=dict(
        z={k: v for k, v in z.items() if k != "families"},
        jpsi={k: v for k, v in j.items() if k != "families"}),
        measured=meas, open_items=opn, coefficients=coef)
    for it in meas + opn:
        it["worst"] = {fit: worst(it["rows"], fit) for fit in FITS}
    p = os.path.join(RUNS, "error_budget.json")
    json.dump(out, open(p, "w"), indent=1)
    # console summary
    print(f"{'item':<66} {'Z bound':>9} {'Z(G)':>9} {'J/psi':>9}")
    for it in meas + opn:
        if it.get("rel"):
            print(f"{it['item'][:66]:<66} location {it['rel']['central']:+.2e} "
                  f"(bound {it['rel']['bound']:.2e}) on {it['rel']['applies']}")
            continue
        w = it["worst"]
        cols = [f"{w[f]['bound']:9.2e}" if w[f] else f"{'-':>9}" for f in FITS]
        print(f"{it['item'][:66]:<66} " + " ".join(cols)
              + (f"   [{w['jpsi']['fam']}]" if w["jpsi"] else ""))
        if "location" in it:
            for u, r in it["location"].items():
                print(f"{'':<8}location {u}: {r['dm_MeV']:+.4f} +- {r['err_MeV']:.4f} MeV"
                      f" = {r['rel']:+.2e} +- {r['rel_err']:.2e}"
                      f"{'' if r['valid'] else '  (D(u) turned over: not a location)'}")
    print(f"-> {p}")
    if a.figs:
        figures(out, fr)


def figures(out, fr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    import pubhtml
    hep.style.use(hep.style.ROOT)
    tag = datetime.date.today().strftime("%y%m%d")
    od = os.path.join(pubhtml.FIGROOT, f"{tag}_error_budget")
    os.makedirs(od, exist_ok=True)
    pubhtml.ensure_index(od)
    A = out["fit_response"]
    # 1. fit response per family
    fams = [k for k in A if not k.startswith("shift")]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (fit, lab) in enumerate((("z", r"$Z$, $\alpha$ only"),
                                    ("zG", r"$Z$, $\Gamma_Z$ floated"),
                                    ("jpsi", r"$J/\psi$"))):
        ax.bar(x + (i - 1) * 0.27, [abs(A[f][fit]) for f in fams], 0.27, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x, fams, rotation=60, fontsize=14)
    ax.set_ylabel(r"$|\delta\alpha / \epsilon|$")
    ax.axhline(TARGET / 0.01, ls=":", c="k", lw=1)
    ax.text(len(fams) - 0.5, TARGET / 0.01 * 1.2, r"$10^{-5}$ at $\epsilon = 1\%$",
            ha="right", fontsize=14)
    ax.legend(fontsize=15)
    pubhtml.savefig(fig, os.path.join(od, "fit_response.pdf"))
    plt.close(fig)
    # 2. signatures of the width / tail families in the key frames
    for name, fams2 in (("ditrack_zmm", ("W_ioni", "T_ioni", "W_rad", "O_ioni")),
                        ("ditrack_jpsi", ("W_ms", "T_ms", "W_ioni", "O_ioni")),
                        ("mum_locx", ("W_ms", "T_ms", "R_ms")),
                        ("mum_qop", ("W_ioni", "T_ioni", "O_ioni", "W_rad"))):
        f = fr[name]
        fig, ax = plt.subplots(figsize=(8, 6))
        for fam in fams2:
            for where, ls in (("outer", "-"), ("ladder", "--")):
                s = frame_sig(f, fam, "even", where)
                ax.plot(U, s, ls, marker="o", ms=4,
                        label=f"{fam}, {where}")
        ax.set_xscale("log")
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xlabel(r"$u$ [units of $s_F^{-2}$]")
        ax.set_ylabel(r"$\partial\,\mathrm{closure}(u)/\partial\epsilon$ (even)")
        ax.set_title(name.replace("_", " "), fontsize=16)
        ax.legend(fontsize=11, ncol=2)
        pubhtml.savefig(fig, os.path.join(od, f"signature_{name}.pdf"))
        plt.close(fig)
    # 3. the budget
    its = [it for it in out["measured"] + out["open_items"]]
    labs, zc, jc = [], [], []
    for it in its:
        if it.get("rel"):
            labs.append(it["item"][:60])
            zc.append(np.nan)
            jc.append(it["rel"]["bound"])
            continue
        w = it["worst"]
        if not w["z"]:
            continue
        labs.append(it["item"][:60])
        zc.append(w["z"]["bound"])
        jc.append(w["jpsi"]["bound"])
    y = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(12, 0.34 * len(labs) + 2))
    ax.scatter(zc, y, marker="o", label=r"$Z$ fit, $\alpha$ only (2$\sigma$ bound)")
    ax.scatter(jc, y, marker="s", label=r"$J/\psi$ fit (2$\sigma$ bound)")
    ax.axvline(TARGET, c="r", ls="--", lw=1.2, label=r"$10^{-5}$")
    ax.set_xscale("log")
    ax.set_yticks(y, labs, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"$|\delta p / p|$")
    ax.legend(fontsize=12, loc="lower right")
    pubhtml.savefig(fig, os.path.join(od, "budget.pdf"))
    plt.close(fig)
    print(f"figures -> {od}")


if __name__ == "__main__":
    main()
