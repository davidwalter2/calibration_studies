#!/usr/bin/env python3
"""Validation of the two-track maker's VARIANCE (log-det) gradient/Hessian.

The C++ claim (`exportVarianceGrads`, 2026-09-06) is that the exported
per-candidate ``gradv`` / ``hesspackedv`` are the derivatives of the MARGINAL
objective

    -2 lnL = r^T R r + ln|V| + ln|C|,   R = V^-1 - V^-1 F C^-1 F^T V^-1

with respect to EVERY global parameter present, including the ones that enter
only (parmtypes 8/9/10/11) or also (parmtype 15) through the covariance.  The
gates here are the ones that can fail:

  B1 FD, FIXED LINEARIZATION  the maker's own `VARFD` lines: V -> V + s dV_i
                            with r, F, J held fixed, the profile re-done from
                            scratch, against the exported column.  This tests
                            the ASSEMBLY -- traces, projector, sign, the ln|C|
                            term -- and must be O(s^2).
  B2 FD, PROPAGATOR LEVEL   central difference of the exported ``objval`` under
                            a real perturbation of the propagator: parmtype 15
                            via a group's ``k_init``, parmtype 10 via
                            ``CVH_MS_SCALE``.  This additionally tests that
                            `dV` IS dV/dtheta.  It cannot be arbitrarily sharp:
                            the reference trajectory moves with the parameter
                            and the Gauss-Newton stopping tolerance leaves a
                            delta-INDEPENDENT absolute error, so the residual
                            grows as 1/delta below some delta.
  C  SUM RULE               sum_g d/dk_g == d/d(global MS+ioni scale), because
                            sum_g dQ_g == dQMS + dQI per propagation.  Checked
                            on the gradient AND on the Hessian block.
  D  TWO-TRACK vs SINGLE    the same parmtype-10 global index, from the same
                            muon fitted alone and fitted as half of a pair.
  E  POSITIVE SEMI-DEFINITE the exported Hessian, per candidate and pooled,
                            on the ``make_global_term.py`` reconstruction path.

    python3 check_variance_grads_260906.py [--only B,C] [--fdroot ...]
"""
import argparse
import datetime
import glob
import os
import sys

import numpy as np
import uproot

SMOKE = "/work/submit/david_w/ZMass/scratch_smoke_260906"
FDROOT = os.path.join(SMOKE, "fdvar260906")
OUTDIR = ("/work/submit/david_w/ZMass/calibration_studies/resolution/runs/"
          "variancegrads260906")
OUTFILE = os.path.join(OUTDIR, "check_variance.txt")

_LINES = []


def log(msg=""):
    print(msg, flush=True)
    _LINES.append(str(msg))


def head(title):
    log()
    log("=" * 100)
    log(title)
    log("=" * 100)


def tri_index(n):
    return np.triu_indices(n)


def unpack_hess(h, n):
    H = np.zeros((n, n))
    H[tri_index(n)] = h
    return H + H.T - np.diag(np.diag(H))


def load(fn, extra=()):
    """Per-candidate arrays plus the runtree parmtype/rawdetid maps."""
    f = uproot.open(fn)
    t = f["tree"]
    want = ["run", "lumi", "event", "nParms", "globalidxv", "gradv"]
    keys = set(t.keys())
    for b in ("gradllv", "gradchisqv", "hesspackedv", "objval", "objchisq",
              "objlogdetv", "objlogdetc", "nRank", "hessfactorv",
              "Muplus_pt", "Muplus_eta", "Muplus_phi",
              "Muminus_pt", "Muminus_eta", "Muminus_phi",
              "Muplus_nvalid", "Muminus_nvalid", "objnullv",
              "trackPt", "trackEta", "trackPhi", "refParms", "edmval",
              "chisqval", "ndof", "niter") + tuple(extra):
        if b in keys:
            want.append(b)
    a = t.arrays(want, library="np")
    rt = f["runtree"]
    a["_parmtype"] = rt["parmtype"].array(library="np")
    a["_rawdetid"] = rt["rawdetid"].array(library="np")
    a["_file"] = fn
    a["_nent"] = t.num_entries
    return a


def key_of(a):
    return np.stack([a["run"], a["lumi"], a["event"]], axis=1)


def match_events(a, b):
    """index arrays into a and b for the events present in both, once each."""
    ka = [tuple(int(x) for x in r) for r in key_of(a)]
    kb = {tuple(int(x) for x in r): i for i, r in enumerate(key_of(b))}
    ia, ib = [], []
    seen = set()
    for i, k in enumerate(ka):
        if k in kb and k not in seen:
            seen.add(k)
            ia.append(i)
            ib.append(kb[k])
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def cols_of_family(a, ic, fams):
    """positions in this candidate's parameter vector whose parmtype is in fams."""
    gi = np.asarray(a["globalidxv"][ic])
    pt = a["_parmtype"][gi]
    return np.where(np.isin(pt, fams))[0], gi


def col_of_global(a, ic, gidx):
    gi = np.asarray(a["globalidxv"][ic])
    w = np.where(gi == gidx)[0]
    return int(w[0]) if len(w) else -1


# --------------------------------------------------------------- gate B1
def parse_varfd(logfn):
    """the maker's VARFD lines -> list of dicts"""
    out = []
    with open(logfn, errors="replace") as fh:
        for line in fh:
            if not line.startswith("VARFD "):
                continue
            d = {}
            for tok in line.split()[1:]:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                d[k] = v if k == "ev" else float(v)
            out.append(d)
    return out


def gate_B1(args):
    head("GATE B1 -- FD AT FIXED LINEARIZATION (the maker's own VARFD)")
    dirs = sorted(glob.glob(os.path.join(args.inmaker, "eps*")),
                  key=lambda d: -float(d.rsplit("eps", 1)[1]))
    if not dirs:
        log(f"  MISSING {args.inmaker}/eps* -- run fdinmaker_260906.sh first")
        return
    # The in-maker FD perturbs ONLY the covariance, so what it reproduces is
    # the VARIANCE part of the column.  For parmtypes 10/11 that is the whole
    # column (their J columns are identically zero); for parmtype 15 the MEAN
    # loss part has to come off first, and it is exactly the baseline
    # (mean-only) run's gradient for the same event and the same global index.
    meanf = os.path.join(args.defroot, "gun_tt", "globalcor_0.root")
    mean = {}
    if os.path.exists(meanf):
        a = load(meanf)
        for ic in range(a["_nent"]):
            k = ":".join(str(int(x)) for x in key_of(a)[ic])
            gi = np.asarray(a["globalidxv"][ic])
            g = np.asarray(a["gradv"][ic], dtype=float)
            for j, gx in enumerate(gi):
                mean[(k, int(gx))] = g[j]
        log(f"  mean-only baseline: {meanf}")
    else:
        log(f"  NO mean-only baseline at {meanf} -- parmtype-15 rows will "
            f"carry their mean-loss part and cannot close")
    log()
    log(f"  {'eps':>8} {'nrow':>7} {'ncand':>6} {'max|d|/scale':>14} "
        f"{'med|d|/scale':>14} {'max|d| abs':>12} {'med|d| abs':>12}")
    prev = None
    for d in dirs:
        eps = d.rsplit("eps", 1)[1]
        rows = parse_varfd(os.path.join(d, "cmsrun.log"))
        if not rows:
            log(f"  {eps:>8}  no VARFD lines")
            continue
        bycand = {}
        for r in rows:
            bycand.setdefault(r["ev"], []).append(r)
        rel, absd = [], []
        for ev, rs in bycand.items():
            scale = max(abs(r["an"] - mean.get((ev, int(r["glob"])), 0.0))
                        for r in rs)
            if scale <= 0:
                continue
            for r in rs:
                anvar = r["an"] - mean.get((ev, int(r["glob"])), 0.0)
                dd = abs(r["fd"] - anvar)
                rel.append(dd / scale)
                absd.append(dd)
        a = np.asarray(rel)
        b = np.asarray(absd)
        log(f"  {eps:>8} {len(rows):>7} {len(bycand):>6} {a.max():>14.3e} "
            f"{np.median(a):>14.3e} {b.max():>12.3e} {np.median(b):>12.3e}")
        prev = (a.max(), float(eps))
    log()
    log("  `an` is the exported column MINUS its mean-loss part; `fd` moves "
        "only V.  The floor is the FLOAT32 storage of gradv/gradchisqv "
        "(~1e-7 of the value), which is why the residual does not shrink "
        "with eps -- it is not FD truncation.")


# ---------------------------------------------------------------- gate B
def _fd_report(label, an, anc, anl, fd, fdc, fdl):
    an, anc, anl = map(np.asarray, (an, anc, anl))
    fd, fdc, fdl = map(np.asarray, (fd, fdc, fdl))
    if not len(an):
        log(f"  {label}  no candidate")
        return
    sc = max(np.abs(an).max(), 1e-300)
    log(f"  {label}  n={len(an)}")
    log(f"      TOTAL   sum an {an.sum():>11.5g}  fd {fd.sum():>11.5g}   "
        f"max|d|/max|an| {np.abs(fd-an).max()/sc:.3e}  "
        f"med|d|/max|an| {np.median(np.abs(fd-an))/sc:.3e}")
    log(f"      CHISQ   sum an {anc.sum():>11.5g}  fd {fdc.sum():>11.5g}   "
        f"rel {abs(fdc.sum()/anc.sum()-1) if anc.sum() else float('nan'):.3e}")
    log(f"      LOGDET  sum an {anl.sum():>11.5g}  fd {fdl.sum():>11.5g}   "
        f"rel {abs(fdl.sum()/anl.sum()-1) if anl.sum() else float('nan'):.3e}")


def gate_B(args):
    head("GATE B2 -- FD AT THE PROPAGATOR LEVEL (a real parameter change)")
    nomf = os.path.join(args.fdroot, "nom", "globalcor_0.root")
    if not os.path.exists(nomf):
        log(f"  MISSING {nomf} -- run fd_variance_260906.sh first")
        return
    nom = load(nomf)
    log(f"  nominal: {nom['_nent']} candidates   {nomf}")
    if "objval" not in nom:
        log("  objval absent -- the nominal run needs exportObjective=True")
        return
    log(f"  quality cut: both legs with >= {args.min_nvalid} valid hits "
        "(a one-hit leg makes the fit degenerate: the Qinv eigenvalue floor "
        "fires and the objective is not smooth in k at k = 0)")

    def good(a, ic):
        if "Muplus_nvalid" not in a:
            return True
        return (int(a["Muplus_nvalid"][ic]) >= args.min_nvalid
                and int(a["Muminus_nvalid"][ic]) >= args.min_nvalid)

    grpid = {}
    for i, (pt, rd) in enumerate(zip(nom["_parmtype"], nom["_rawdetid"])):
        if pt == 15:
            grpid[int(rd)] = i

    def collect(fp, fm, delta, colfun):
        ap, am = load(fp), load(fm)
        kp = {tuple(int(x) for x in r): i for i, r in enumerate(key_of(ap))}
        km = {tuple(int(x) for x in r): i for i, r in enumerate(key_of(am))}
        out = [[] for _ in range(6)]
        nsame = 0
        for ic in range(nom["_nent"]):
            k = tuple(int(x) for x in key_of(nom)[ic])
            if k not in kp or k not in km or not good(nom, ic):
                continue
            ip, im = kp[k], km[k]
            if "objnullv" in nom and not (int(ap["objnullv"][ip])
                                          == int(am["objnullv"][im])
                                          == int(nom["objnullv"][ic])):
                continue
            nsame += 1
            cols = colfun(ic)
            if cols is None or not len(cols):
                continue
            g = np.asarray(nom["gradv"][ic], dtype=float)
            gc = np.asarray(nom["gradchisqv"][ic], dtype=float)
            gl = np.asarray(nom["gradllv"][ic], dtype=float)
            out[0].append(g[cols].sum())
            out[1].append(gc[cols].sum())
            out[2].append(gl[cols].sum())
            out[3].append((ap["objval"][ip] - am["objval"][im]) / (2 * delta))
            out[4].append((ap["objchisq"][ip] - am["objchisq"][im]) / (2 * delta))
            out[5].append(((ap["objlogdetv"][ip] + ap["objlogdetc"][ip])
                           - (am["objlogdetv"][im] + am["objlogdetc"][im]))
                          / (2 * delta))
        return out

    log()
    log("  parmtype 15 -- a material group's k_g (it moves the mean loss AND "
        "the width)")
    for gdir in sorted(glob.glob(os.path.join(args.fdroot, "mat_g*_p*"))):
        base = os.path.basename(gdir)
        g = int(base.split("_")[1][1:])
        dtxt = base.split("_p")[1]
        fp = os.path.join(gdir, "globalcor_0.root")
        fm = os.path.join(args.fdroot, f"mat_g{g}_m{dtxt}", "globalcor_0.root")
        if not (os.path.exists(fp) and os.path.exists(fm)):
            log(f"    group {g} delta {dtxt}: MISSING")
            continue
        gi = grpid.get(g)
        if gi is None:
            log(f"    group {g}: not a parmtype-15 global")
            continue
        res = collect(fp, fm, float(dtxt),
                      lambda ic, gi=gi: ([col_of_global(nom, ic, gi)]
                                         if col_of_global(nom, ic, gi) >= 0
                                         else []))
        _fd_report(f"  group {g:>3}  delta {dtxt}", *res)

    tightf = os.path.join(args.fdroot, "tight_nom", "globalcor_0.root")
    if os.path.exists(tightf):
        log()
        log("  the same, with edmConvergence=1e-10 nIters=40 on BOTH the "
            "nominal and the arms:")
        log("  what shrinks with that is the Gauss-Newton stopping tolerance "
            "(the chi2 is")
        log("  stationary in the state only to O(edm)); what does NOT is the "
            "reference-trajectory")
        log("  movement in ln|V|, which the analytic dV deliberately omits "
            "('the fit never")
        log("  differentiates through the weights').")
        tnom = nom
        nom = load(tightf)
        for gdir in sorted(glob.glob(os.path.join(args.fdroot, "tight_g*_p*"))):
            base = os.path.basename(gdir)
            g = int(base.split("_")[1][1:])
            dtxt = base.split("_p")[1]
            fp = os.path.join(gdir, "globalcor_0.root")
            fm = os.path.join(args.fdroot, f"tight_g{g}_m{dtxt}",
                              "globalcor_0.root")
            if not (os.path.exists(fp) and os.path.exists(fm)):
                continue
            gi = grpid.get(g)
            res = collect(fp, fm, float(dtxt),
                          lambda ic, gi=gi: ([col_of_global(nom, ic, gi)]
                                             if col_of_global(nom, ic, gi) >= 0
                                             else []))
            _fd_report(f"  TIGHT group {g:>3}  delta {dtxt}", *res)
        nom = tnom

    log()
    log("  parmtype 10 -- a COMMON log scale on every step's MS covariance "
        "(CVH_MS_SCALE)")
    for pdir in sorted(glob.glob(os.path.join(args.fdroot, "ms_p*"))):
        dtxt = os.path.basename(pdir)[4:]
        fp = os.path.join(pdir, "globalcor_0.root")
        fm = os.path.join(args.fdroot, "ms_m" + dtxt, "globalcor_0.root")
        if not (os.path.exists(fp) and os.path.exists(fm)):
            log(f"    delta {dtxt}: MISSING")
            continue
        res = collect(fp, fm, float(dtxt),
                      lambda ic: cols_of_family(nom, ic, [10])[0])
        _fd_report(f"  MS scale  delta {dtxt}", *res)


# ---------------------------------------------------------------- gate C
def gate_C(args):
    head("GATE C -- SUM RULE  sum_g d/dk_g == d/d(MS+ioni scale)")
    nomf = os.path.join(args.fdroot, "nom", "globalcor_0.root")
    deff = os.path.join(args.defroot, "gun_tt", "globalcor_0.root")
    if not (os.path.exists(nomf) and os.path.exists(deff)):
        log("  MISSING inputs")
        return
    nom = load(nomf)
    dfl = load(deff)
    ia, ib = match_events(nom, dfl)
    log(f"  {len(ia)} candidates matched between the log-det run and the "
        f"mean-only baseline")
    d15, d1011, dll15, dll1011 = [], [], [], []
    h15, h1011 = [], []
    for ic, jc in zip(ia, ib):
        c15, gi = cols_of_family(nom, ic, [15])
        c1011, _ = cols_of_family(nom, ic, [10, 11])
        if not len(c15) or not len(c1011):
            continue
        gnom = np.asarray(nom["gradv"][ic], dtype=float)
        gdef = np.asarray(dfl["gradv"][jc], dtype=float)
        gdefi = np.asarray(dfl["globalidxv"][jc])
        pos = {int(g): k for k, g in enumerate(gdefi)}
        # the parmtype-15 columns are shared between the two layouts
        s15 = 0.0
        ok = True
        for c in c15:
            g = int(gi[c])
            if g not in pos:
                ok = False
                break
            s15 += gnom[c] - gdef[pos[g]]
        if not ok:
            continue
        d15.append(s15)
        d1011.append(gnom[c1011].sum())
        if "gradllv" in nom and len(nom["gradllv"][ic]):
            gll = np.asarray(nom["gradllv"][ic], dtype=float)
            dll15.append(gll[c15].sum())
            dll1011.append(gll[c1011].sum())
        if "hesspackedv" in nom and "hesspackedv" in dfl:
            n = int(nom["nParms"][ic])
            m = int(dfl["nParms"][jc])
            Hn = unpack_hess(np.asarray(nom["hesspackedv"][ic], dtype=float), n)
            Hd = unpack_hess(np.asarray(dfl["hesspackedv"][jc], dtype=float), m)
            pcols = [pos[int(gi[c])] for c in c15]
            h15.append(Hn[np.ix_(c15, c15)].sum() - Hd[np.ix_(pcols, pcols)].sum())
            h1011.append(Hn[np.ix_(c1011, c1011)].sum())
    def report(name, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if not len(x):
            log(f"  {name}: no candidate")
            return
        dd = x - y
        sc = np.maximum(np.abs(x), np.abs(y))
        rel = np.abs(dd) / np.maximum(sc, 1e-300)
        log(f"  {name:<34} n={len(x):<5} sum {x.sum():>13.6g} vs {y.sum():>13.6g}"
            f"   max rel {rel.max():.2e}   median rel {np.median(rel):.2e}")
    report("gradient  (chisq + log-det)", d15, d1011)
    report("gradient  (log-det only)", dll15, dll1011)
    report("Hessian block sum", h15, h1011)


# ---------------------------------------------------------------- gate D
def gate_D(args):
    head("GATE D -- TWO-TRACK vs SINGLE-TRACK log-det gradient (parmtype 10)")
    ttf = os.path.join(args.fdroot, "nom", "globalcor_0.root")
    stf = os.path.join(args.stroot, "globalcor_resclosure_0.root")
    if not (os.path.exists(ttf) and os.path.exists(stf)):
        log(f"  MISSING: tt={os.path.exists(ttf)} st={os.path.exists(stf)}")
        return
    tt = load(ttf)
    st = load(stf)
    log(f"  two-track {tt['_nent']} candidates, single-track {st['_nent']} tracks")
    log("  the same muons, the same J/psi gun file, the same parmtype-10 "
        "global indices.")
    log("  NOTE both legs of a pair can cross the SAME module, in which case "
        "the two-track")
    log("  column carries the SUM of the two legs, so the single-track side "
        "must be summed")
    log("  per index before comparing (it is here).")
    ev_st = {}
    for i, k in enumerate(key_of(st)):
        ev_st.setdefault(tuple(int(x) for x in k), []).append(i)
    recs = []
    for ic in range(tt["_nent"]):
        k = tuple(int(x) for x in key_of(tt)[ic])
        if k not in ev_st or "gradllv" not in tt or not len(tt["gradllv"][ic]):
            continue
        c10, gi = cols_of_family(tt, ic, [10])
        if not len(c10):
            continue
        gll = np.asarray(tt["gradllv"][ic], dtype=float)
        ttmap = {int(gi[c]): gll[c] for c in c10}
        stmap = {}
        for j in ev_st[k]:
            if "gradllv" not in st or not len(st["gradllv"][j]):
                continue
            gj = np.asarray(st["globalidxv"][j])
            gllj = np.asarray(st["gradllv"][j], dtype=float)
            for c in np.where(st["_parmtype"][gj] == 10)[0]:
                g = int(gj[c])
                stmap[g] = stmap.get(g, 0.0) + gllj[c]
        for g, v in stmap.items():
            if g in ttmap:
                recs.append((k, g, ttmap[g], v))
    if not recs:
        log("  no matched event with shared parmtype-10 globals")
        return
    A = np.asarray([r[2] for r in recs])
    B = np.asarray([r[3] for r in recs])
    d = A - B
    ev = {}
    for k, g, a, b in recs:
        x = ev.setdefault(k, [0.0, 0.0])
        x[0] += a
        x[1] += b
    ea = np.asarray([v[0] for v in ev.values()])
    eb = np.asarray([v[1] for v in ev.values()])
    log(f"  shared indices           : {len(recs)} over {len(ev)} events")
    log(f"  sum tt / sum st          : {A.sum()/B.sum():.6f}")
    log(f"  correlation              : {np.corrcoef(A,B)[0,1]:.6f}")
    log(f"  |tt - st|                : median {np.median(np.abs(d)):.3e}  "
        f"p99 {np.percentile(np.abs(d),99):.3e}  max {np.abs(d).max():.3e}")
    rel = np.abs(d)/np.maximum(np.abs(B), 1e-30)
    log(f"  |tt - st| / |st|         : median {np.median(rel):.3e}  "
        f"p90 {np.percentile(rel,90):.3e}   (the p99 is a division by a "
        f"near-zero st, use the absolute column)")
    log(f"  per-event sums, tt/st    : total {ea.sum()/eb.sum():.6f}  "
        f"median {np.median(ea/eb):.6f}  16-84% "
        f"{np.percentile(ea/eb,16):.5f}..{np.percentile(ea/eb,84):.5f}")
    o = np.argsort(-np.abs(d))[:5]
    log(f"  {'ev':>14} {'glob':>8} {'tt':>11} {'st(sum)':>11} {'diff':>11}")
    for j in o:
        k, g, a, b = recs[j]
        log(f"  {str(k):>14} {g:>8} {a:>11.6f} {b:>11.6f} {a-b:>11.6f}")
    log("  INTERPRETATION: tr(dV_i R) is a LOCAL quantity -- how much of that "
        "block's noise")
    log("  the surrounding measurements determine -- so the pair fit's common "
        "vertex moves it")
    log("  only where the vertex is close.  The residual IS that effect.")


# ---------------------------------------------------------------- gate E
def gate_E(args):
    head("GATE E -- POSITIVE SEMI-DEFINITENESS of the exported Hessian")
    for tag, fn in (("log-det on (families 10,11,15)",
                     os.path.join(args.fdroot, "nom", "globalcor_0.root")),
                    ("baseline (mean only)",
                     os.path.join(args.defroot, "gun_tt", "globalcor_0.root"))):
        if not os.path.exists(fn):
            log(f"  {tag}: MISSING")
            continue
        a = load(fn)
        if "hesspackedv" not in a:
            log(f"  {tag}: no hesspackedv")
            continue
        worst = 0.0
        worst_sub = 0.0
        pooled = {}
        for ic in range(a["_nent"]):
            n = int(a["nParms"][ic])
            H = unpack_hess(np.asarray(a["hesspackedv"][ic], dtype=float), n)
            w = np.linalg.eigvalsh(0.5 * (H + H.T))
            lm = max(w.max(), 1e-300)
            worst = min(worst, w.min() / lm)
            # the make_global_term.py projection: parmtypes 14 and 15 only
            gi = np.asarray(a["globalidxv"][ic])
            sub = np.where(np.isin(a["_parmtype"][gi], [14, 15]))[0]
            if len(sub):
                Hs = H[np.ix_(sub, sub)]
                ws = np.linalg.eigvalsh(0.5 * (Hs + Hs.T))
                worst_sub = min(worst_sub, ws.min() / max(ws.max(), 1e-300))
                for x, gx in enumerate(gi[sub]):
                    for y, gy in enumerate(gi[sub]):
                        pooled[(int(gx), int(gy))] = \
                            pooled.get((int(gx), int(gy)), 0.0) + Hs[x, y]
        keys = sorted({k[0] for k in pooled})
        idx = {g: i for i, g in enumerate(keys)}
        K = np.zeros((len(keys), len(keys)))
        for (gx, gy), v in pooled.items():
            K[idx[gx], idx[gy]] += v
        wk = np.linalg.eigvalsh(0.5 * (K + K.T))
        log(f"  {tag}")
        log(f"     per-candidate  min(lambda)/max(lambda), full  : {worst:.3e}")
        log(f"     per-candidate  min/max on the (14,15) subset  : {worst_sub:.3e}")
        log(f"     POOLED K over {len(keys)} (14,15) params      : "
            f"min {wk.min():.6g}  max {wk.max():.6g}  ratio {wk.min()/max(wk.max(),1e-300):.3e}")


# ---------------------------------------------------------------- gate F
def gate_F(args):
    head("GATE F -- THE FACTORED STORAGE, and what the switch costs in bytes")
    log("  hess = B^T B + scatter(hessvarpackedv on hessvaridxv). The runs "
        "below carry BOTH")
    log("  Hessian exports, so the reconstruction can be compared against the "
        "packed dense one.")
    log()
    log(f"  {'config':<8} {'n':>4} {'nParms':>7} {'ndof':>6} {'nRank':>6} "
        f"{'nHessVar':>9} {'max|BtB+V-H|/max|H|':>21} {'B/entry(prod)':>14} "
        f"{'delta':>9}")
    base = None
    for tag in ("off", "var15", "varall"):
        fn = os.path.join(args.factored, tag, "globalcor_0.root")
        if not os.path.exists(fn):
            log(f"  {tag:<8}  MISSING {fn}")
            continue
        f = uproot.open(fn)
        t = f["tree"]
        keys = set(t.keys())
        want = ["nParms", "nRank", "hesspackedv", "hessfactorv", "ndof",
                "hessdroppedmass", "hessrankgap", "Muplus_nvalid",
                "Muminus_nvalid"]
        if "hessvaridxv" in keys:
            want += ["hessvaridxv", "hessvarpackedv"]
        a = t.arrays([w for w in want if w in keys], library="np")
        n = t.num_entries
        rels, nhv = [], []
        for ic in range(n):
            npar = int(a["nParms"][ic])
            nr = int(a["nRank"][ic])
            H = unpack_hess(np.asarray(a["hesspackedv"][ic], dtype=float), npar)
            B = np.asarray(a["hessfactorv"][ic], dtype=float).reshape(nr, npar)
            Hf = B.T @ B
            m = 0
            if "hessvaridxv" in a:
                vi = np.asarray(a["hessvaridxv"][ic], dtype=np.int64)
                m = len(vi)
                if m:
                    vp = np.asarray(a["hessvarpackedv"][ic], dtype=float)
                    HV = np.zeros((m, m))
                    HV[tri_index(m)] = vp
                    HV = HV + HV.T - np.diag(np.diag(HV))
                    Hf[np.ix_(vi, vi)] += HV
            nhv.append(m)
            rels.append(np.abs(Hf - H).max() / max(np.abs(H).max(), 1e-300))
        rels = np.asarray(rels)
        # production layout = everything except hesspackedv
        prod = sum(t[k].compressed_bytes for k in t.keys()
                   if k != "hesspackedv") / n
        if base is None:
            base = prod
        log(f"  {tag:<8} {n:>4} {np.mean(a['nParms']):>7.1f} "
            f"{np.mean(a['ndof']):>6.1f} {np.mean(a['nRank']):>6.1f} "
            f"{np.mean(nhv):>9.1f} {rels.max():>21.3e} {prod:>14.0f} "
            f"{100*(prod/base-1):>+8.1f}%")
        bad = np.where(rels > 1e-6)[0]
        for ic in bad:
            log(f"      candidate {ic}: |BtB+V-H|/max|H| = {rels[ic]:.3e}, "
                f"droppedmass {float(a['hessdroppedmass'][ic]):.3e}, "
                f"rankgap {float(a['hessrankgap'][ic]):.3f}, "
                f"ndof {int(a['ndof'][ic])}, "
                f"nvalid {int(a['Muplus_nvalid'][ic])}/"
                f"{int(a['Muminus_nvalid'][ic])}")
    log()
    log("  'B/entry(prod)' is the sum of compressed branch bytes EXCLUDING "
        "hesspackedv, i.e.")
    log("  the layout a production runs (fillGradsFactored, "
        "exportStepRecords=False). It is not")
    log("  the production's absolute number -- these files have 24 entries and "
        "ROOT's per-basket")
    log("  overhead does not amortize -- but the DELTA is what the switch "
        "costs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="B1,B,C,D,E,F")
    ap.add_argument("--inmaker", default=os.path.join(SMOKE, "fdinmaker"))
    ap.add_argument("--min-nvalid", type=int, default=5)
    ap.add_argument("--factored", default=os.path.join(SMOKE, "factored"))
    ap.add_argument("--fdroot", default=FDROOT)
    ap.add_argument("--defroot", default=os.path.join(SMOKE, "v4_default"))
    ap.add_argument("--stroot", default=os.path.join(SMOKE, "st_jpsigun"))
    ap.add_argument("-o", "--out", default=OUTFILE)
    args = ap.parse_args()

    log(f"VARIANCE-GRADIENT VALIDATION -- "
        f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"numpy {np.__version__}   uproot {uproot.__version__}")
    log(f"fd root  : {args.fdroot}")
    log(f"baseline : {args.defroot}")
    only = {x.strip().upper() for x in args.only.split(",")}
    for g, fn in (("B1", gate_B1), ("B", gate_B), ("C", gate_C),
                  ("D", gate_D), ("E", gate_E), ("F", gate_F)):
        if g in only:
            try:
                fn(args)
            except Exception as exc:  # a missing input must not kill the rest
                log(f"  GATE {g} FAILED: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    log()
    log(f"written: {args.out}")


if __name__ == "__main__":
    main()
