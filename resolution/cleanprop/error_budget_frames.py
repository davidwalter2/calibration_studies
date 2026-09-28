#!/usr/bin/env python3
"""The closure side of the error budget: what each perturbation family of
`error_budget.py` does to the clean-propagation closure statistic, in the
closure's own frame.

A frame is one closure functional: a local direction of a single track at a
plane (`realmat_closure.py`), or the one-plane vertex mass of a ditrack
(`realmat_ditrack.py`), standardised by the published s_F of that plane.  Its
model CF is the product of the channels

    phi(t) = exp S(t),   S = S_ioni + S_ms + S_rad + S_joint

(S_joint = S - the three single-channel exponents: the knock-on and radiative
joint laws of loss and deflection, kept as a channel of its own)

each evaluated with the closure's own builder (`cgf_channels.block_cf_exponent`
with one channel switched on), so the channel split is the one the closure
tests.  For a family eps g_c(t) acting on channel c (width, rate, tail, odd
part; definitions in error_budget.py) the closure moves by

    even(u) = int_0 Re[phi g] w_u dt,      odd(u) = int_0 Im[phi g] (t/2u) w_u dt,
    w_u = e^{-t^2/4u}/sqrt(pi u),

per unit eps: exactly the statistic's own Weierstrass integrals applied to the
first-order change of the model CF.  The location probe of a ditrack converts
as in the closure, dm(u) = s_F odd(u)/D(u).

usage:
  python error_budget_frames.py single --pdg 13 --funcs qop locx dxdz
  python error_budget_frames.py ditrack --decay zmm --xstat
  python error_budget_frames.py ditrack --decay jpsi
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
sys.path.insert(0, RES)
sys.path.insert(0, HERE)
os.environ["RES_NO_PHI_CACHE"] = "1"

RUNS = os.path.join(RES, "runs", "error_budget_260928")
CH = ("ioni", "ms", "rad")
FAMS = ("W", "R", "T", "O")


def signatures(Sc, S, tau, probes, sF):
    """Per family and channel: even(u), odd(u) and dm(u) [GeV] per unit eps;
    the model's own even/odd/D for reference."""
    phi = np.exp(S)
    dS = {c: np.gradient(Sc[c], tau) for c in Sc}

    def ws(x, u):
        w = np.exp(-tau ** 2 / (4.0 * u)) / np.sqrt(np.pi * u)
        return float(np.trapezoid(x.real * w, tau))

    def wo(x, u):
        w = (tau / (2.0 * u)) * np.exp(-tau ** 2 / (4.0 * u)) / np.sqrt(np.pi * u)
        return float(np.trapezoid(x.imag * w, tau))

    def wd(x, u):
        w = (tau ** 2 / (2.0 * u)) * np.exp(-tau ** 2 / (4.0 * u)) / np.sqrt(np.pi * u)
        return float(np.trapezoid(x.real * w, tau))

    D = np.array([wd(phi, u) for u in probes])
    out = dict(model_even=[ws(phi, u) for u in probes],
               model_odd=[wo(phi, u) for u in probes], D=D.tolist(),
               var_share={c: float(-2.0 * Sc[c][1].real / tau[1] ** 2) for c in Sc},
               fam={})
    for c in Sc:
        for f in FAMS:
            if f == "W":
                g = 0.5 * tau * dS[c]
            elif f == "R":
                g = Sc[c]
            elif f == "T":
                g = Sc[c] - 0.5 * tau * dS[c]
            else:
                g = 1j * Sc[c].imag
            x = phi * g
            ev = np.array([ws(x, u) for u in probes])
            od = np.array([wo(x, u) for u in probes])
            out["fam"][f"{f}_{c}"] = dict(even=ev.tolist(), odd=od.tolist(),
                                          dm=(sF * od / D).tolist())
    # the pure location shift of the frame by one s_F (reference)
    x = phi * 1j * tau
    od = np.array([wo(x, u) for u in probes])
    out["fam"]["shift_all"] = dict(even=[ws(x, u) for u in probes],
                                   odd=od.tolist(), dm=(sF * od / D).tolist())
    return out


_CTX = None


def _one(k):
    """One plane's channel exponents and family signatures (forked worker)."""
    if _CTX[0] == "single":
        _, cc, legs, avs, sF, tau, probes = _CTX
        s = float(sF[k])
        Sc = {c: cc.block_cf_exponent(legs, k, avs[k], s, tau, channels=(c,))
              for c in CH}
        S = cc.block_cf_exponent(legs, k, avs[k], s, tau, channels=CH)
        Sc["joint"] = S - sum(Sc.values())    # the knock-on joint piece, its own
        return signatures(Sc, S, tau, probes, s)
    _, cc, legs, vec, rows, tau, probes, names = _CTX
    s = float(rows[k]["sF"])
    Sc = {c: sum(cc.block_cf_exponent(legs[lg], k, vec[lg][k][0], s, tau,
                                      channels=(c,)) for lg in names) for c in CH}
    S = sum(cc.block_cf_exponent(legs[lg], k, vec[lg][k][0], s, tau, channels=CH)
            for lg in names)
    Sc["joint"] = S - sum(Sc.values())
    sig = signatures(Sc, S, tau, probes, s)
    sig["model_even_published"] = rows[k]["model_even"]
    return sig


# -------------------------------------------------------------------------
# single-track frames
# -------------------------------------------------------------------------

def cmd_single(a):
    import realmat_closure as rc
    import cgf_channels as cc
    import fisher_norm as fn
    rc.MODEL_TAG = a.model_tag
    cpt, ctr, cne, hb, hp = rc.cpt, rc.ctr, rc.cne, rc.hb, rc.hp
    hb.set_use_h(True)
    cne.NUCEL_CHANNEL = False
    cpt.RAD_CHANNEL = True
    probes = np.asarray(fn.UCURVE)
    tau = fn.closure_tau(float(np.max(probes)))
    for pdg in a.pdg:
        ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        path = rc.model_path(pdg)
        legs = cpt.load_model(path)
        hb.bind(legs, path, pdg=pdg)
        for func in a.funcs:
            src = rc._res_path(pdg, func)
            cell = json.load(open(src))["cells"][func]
            sF = np.asarray(cell["sF"])
            avs = hb.avecs(legs, func)
            ks = cell["ks"] if a.planes is None else a.planes

            global _CTX
            _CTX = ("single", cc, legs, avs, sF, tau, probes)

            rows = dict(zip(ks, fn.pmap(_one, ks, nproc=a.nproc)))
            out = dict(kind="single", pdg=pdg, func=func, source=src,
                       model=path, ucurve=probes.tolist(),
                       planes={str(k): dict(sF=float(sF[k]), **rows[k]) for k in ks})
            os.makedirs(RUNS, exist_ok=True)
            p = os.path.join(RUNS, f"frame_{rc.lab(pdg)}_{func}.json")
            json.dump(out, open(p, "w"), indent=1)
            print(f"-> {p}", flush=True)


# -------------------------------------------------------------------------
# ditrack frames
# -------------------------------------------------------------------------

def cmd_ditrack(a):
    import realmat_ditrack as rd
    rd.configure(a.decay)
    rd.rc.MODEL_TAG = a.model_tag
    rd.XSTAT = a.xstat
    if a.xstat:
        rd.RES = rd.RES + "_xstat"
    rd._setup_physics()
    cc, fn, cpt, hb = rd.cc, rd.fn, rd.cpt, rd.hb
    src = os.path.join(rd.RES, a.source)
    J = json.load(open(src))
    probes = np.asarray(fn.UCURVE)
    tau = fn.closure_tau(float(np.max(probes)))
    m0, g, worst = rd._check_gradient()
    legs = {}
    for leg in rd.LEGS:
        path = rd.model_path(leg)
        legs[leg] = cpt.load_model(path)
        hb.bind(legs[leg], path, pdg=rd.LEGS[leg]["pdg"])
    vec = {leg: rd.leg_vectors(legs[leg], g[leg]) for leg in rd.LEGS}
    rows = {r["k"]: r for r in J["planes"]}
    ks = sorted(rows) if a.planes is None else a.planes

    global _CTX
    _CTX = ("ditrack", cc, legs, vec, rows, tau, probes, list(rd.LEGS))

    res = dict(zip(ks, fn.pmap(_one, ks, nproc=a.nproc)))
    out = dict(kind="ditrack", decay=a.decay, source=src, ucurve=probes.tolist(),
               m_parent=rd.M_PARENT,
               planes={str(k): dict(sF=float(rows[k]["sF"]), **res[k]) for k in ks})
    os.makedirs(RUNS, exist_ok=True)
    p = os.path.join(RUNS, f"frame_ditrack_{a.decay}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}", flush=True)
    for k in ks:
        me = np.asarray(res[k]["model_even"])
        mp = np.asarray(rows[k]["model_even"])
        print(f"  k={k:>2} model even max|recomputed - published| "
              f"{np.abs(me - mp).max():.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("single")
    s.add_argument("--pdg", type=int, nargs="+", default=[13])
    s.add_argument("--funcs", nargs="+", default=["qop", "locx", "dxdz"])
    s.add_argument("--model-tag", default="_mat")
    s.add_argument("--planes", type=int, nargs="+", default=None)
    s.add_argument("--nproc", type=int, default=19)
    d = sub.add_parser("ditrack")
    d.add_argument("--decay", required=True)
    d.add_argument("--xstat", action="store_true")
    d.add_argument("--source", default="ditrack_closure_kj1_qx1.json")
    d.add_argument("--planes", type=int, nargs="+", default=None)
    d.add_argument("--model-tag", default="",
                   help="realmat_closure's MODEL_TAG of the source closure "
                        "(`_mat` for the J/psi file tagged `_off_nucel0_mat`)")
    d.add_argument("--nproc", type=int, default=19)
    a = ap.parse_args()
    {"single": cmd_single, "ditrack": cmd_ditrack}[a.cmd](a)


if __name__ == "__main__":
    main()
