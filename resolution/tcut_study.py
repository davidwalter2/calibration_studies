#!/usr/bin/env python3
"""Fixed-ENERGY truncation (T_cut) of the ionization straggling: additivity,
cost, and the de-confounded three-way closure comparison.

WHY. The CVH fit needs a finite Gaussian process-noise variance for each
ionization block. Geant4's `Dispersion()` obtains one by truncating the
single-collision delta-ray spectrum at a fixed QUANTILE alpha = 0.999 (plus a
handful of other per-step regularizations: the alfa/namean Gaussian split, the
`+1/3` excitation floor, the [0, 2 eav] truncated-Gaussian correction, and the
Gaussian/Glandz regime switch). Every one of those is a per-STEP construction,
so the resulting variance is NOT proportional to path length and NOT invariant
under subdividing a step. That non-additivity is what makes the fit's answer
depend on `StepLengthLimit` (NOTES 2026-08-13 X / XI).

A fixed-ENERGY restriction of the same spectrum does not have that problem. The
untruncated Urban model of one step is a compound Poisson with rates
(a1, a2, a3) all proportional to the step's mean loss (hence to path length) and
jump energies (e1, e2, [e0, tmax]) that depend only on the material and on
beta*gamma. Restricting the jump spectrum to E <= T_cut leaves that structure
intact, so EVERY cumulant stays proportional to path length and is exactly
additive under any subdivision.

  kappa_n(T) = a1 e1^n + a2 e2^n + a3 C Int_{e0}^{min(T,tmax)} E^{n-2} dE,
  C = e0 tmax / (tmax - e0)      [normalization of the 1/E^2 spectrum]

with all energies multiplied by the record's `scaling`.

WHAT THIS DOES NOT FIX. T_cut is still a convention (a different one), and the
block density stays skewed: the mean-vs-mode displacement, which is the actual
reason for the CGF/IRLS programme, is untouched. The two motivations are
separated explicitly in the `motivations` subcommand.

Subcommands
  validate    reproduce the recorded gsig2 from the untruncated record, for
              three alpha conventions -- proves the offline recipe is the
              in-fit recipe before anything is concluded from it
  additivity  Task 1: subdivision invariance, model-level and against a real
              10x re-stepping of the same trajectory
  cost        Task 2: retained variance / mode shift / tail removed vs T_cut
  closure     Task 3: the NOTES (XI) three-way comparison at FIXED
              standardization
  motivations Task 4: how much of the closure error is step-dependence and how
              much is the estimator

Record layout (ioniurbanv, 11 columns), mirrors cf_track_resolution:
    0 regime   1 gsig2   2 a1   3 e1   4 a2   5 e2
    6 a3       7 e0      8 tmax   9 scaling   10 qop-per-MeV (x1e-3)
`gsig2` for regime 1 is the RETURNED alpha-truncated variance in MeV^2, i.e.
exactly what enters the fit's Q matrix; e1/e2/e0/tmax are the raw (pre-scaling)
energies in MeV.
"""
import argparse
import os
import sys

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad")

# G4UniversalFluctuationForExtrapolator.hh constants
NMAXCONT = 8.0
INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------
# 1. the in-fit recipe, reimplemented offline
# --------------------------------------------------------------------------

def _sample_gauss_var(eav, esig2):
    """Variance added by SampleGauss(): the Urban model replaces a Poisson sum
    by a Gaussian of mean eav and variance esig2 RESTRICTED to [0, 2 eav].
    Vectorized transcription of the inline in the header."""
    eav = np.asarray(eav, dtype=np.float64)
    esig2 = np.asarray(esig2, dtype=np.float64)
    out = np.zeros_like(eav)
    sig = np.sqrt(np.maximum(esig2, 0.0))
    small = eav < 0.25 * sig
    out[small] = eav[small] ** 2 / 3.0
    big = ~small & (sig > 0)
    if big.any():
        s = sig[big]
        b = eav[big] / s
        a = -b
        z = 0.5 * (erf(b / np.sqrt(2.0)) - erf(a / np.sqrt(2.0)))
        pa = INV_SQRT2PI * np.exp(-0.5 * a * a)
        pb = INV_SQRT2PI * np.exp(-0.5 * b * b)
        out[big] = s * s * (1.0 + (a * pa - b * pb) / z - ((pa - pb) / z) ** 2)
    return out


def g4_variance(steps, alpha):
    """Per-step ionization variance [MeV^2] exactly as
    G4UniversalFluctuationForExtrapolator::Dispersion() returns it, for an
    arbitrary quantile alpha. Regime-0 (thick-target Gaussian) steps have no
    alpha dependence and are passed through.

    steps: (n, 11) raw ioniurbanv records (column 10 NOT weighted).
    """
    steps = np.atleast_2d(np.asarray(steps, dtype=np.float64))
    n = len(steps)
    out = np.zeros(n)
    reg = steps[:, 0]
    g0 = reg == 0
    out[g0] = steps[g0, 1]
    m = ~g0
    if not m.any():
        return out

    a1, e1 = steps[m, 2].copy(), steps[m, 3].copy()
    a2, e2 = steps[m, 4].copy(), steps[m, 5].copy()
    a3, e0, tmax = steps[m, 6].copy(), steps[m, 7].copy(), steps[m, 8].copy()
    scal = steps[m, 9]

    esig2tot = np.zeros(m.sum())
    emean = np.zeros_like(esig2tot)
    sig2e = np.zeros_like(esig2tot)

    # --- excitations, via AddExcitation()
    for a, e in ((a1, e1), (a2, e2)):
        cont = a > NMAXCONT
        emean[cont] += a[cont] * e[cont]
        sig2e[cont] += a[cont] * e[cont] ** 2
        disc = (~cont) & (a > 0)
        esig2tot[disc] += (a[disc] + 1.0 / 3.0) * e[disc] ** 2
    hs = sig2e > 0
    if hs.any():
        esig2tot[hs] += _sample_gauss_var(emean[hs], sig2e[hs])

    # --- delta rays
    emean[:] = 0.0
    sig2e[:] = 0.0
    p3 = a3.copy()
    alfa = np.ones_like(a3)
    act = a3 > 0
    w1 = np.where(e0 > 0, tmax / np.maximum(e0, 1e-300), 0.0)
    big = act & (a3 > NMAXCONT) & (w1 > 1.0)
    if big.any():
        al = w1[big] * (NMAXCONT + a3[big]) / (w1[big] * NMAXCONT + a3[big])
        al1 = al * np.log(al) / (al - 1.0)
        nam = a3[big] * w1[big] * (al - 1.0) / ((w1[big] - 1.0) * al)
        alfa[big] = al
        emean[big] += nam * e0[big] * al1
        sig2e[big] += e0[big] ** 2 * nam * (al - al1 ** 2)
        p3[big] = a3[big] - nam
    w2 = alfa * e0
    tail = act & (tmax > w2)
    if tail.any():
        w = (tmax[tail] - w2[tail]) / tmax[tail]
        # alpha = 1 is the untruncated limit: f2 -> w2*tmax, finite
        f2 = alpha * w2[tail] ** 2 / (1.0 - alpha * w)
        esig2tot[tail] += p3[tail] * f2          # = p3 (f^2 + sigf2)
    hs = sig2e > 0
    if hs.any():
        esig2tot[hs] += _sample_gauss_var(emean[hs], sig2e[hs])

    out[m] = esig2tot * scal ** 2
    return out


def alpha_effective_cut(steps, alpha=0.999):
    """The ENERGY at which the alpha-quantile convention actually truncates
    each step, in MeV (post-scaling).

    The quantile is taken on the single-collision spectrum of the delta-ray
    TAIL branch, which lives on [w2, tmax] with w2 = alfa*e0, so
        F(E) = (1 - w2/E)/w,  w = (tmax - w2)/tmax
        F(E_a) = alpha  =>  E_a = w2 / (1 - alpha w).
    alfa is a function of a3, and a3 is proportional to the step's mean loss,
    hence to path length -- which is why E_a moves with the step length and the
    convention is not additive.
    """
    steps = np.atleast_2d(np.asarray(steps, dtype=np.float64))
    out = np.full(len(steps), np.nan)
    m = steps[:, 0] != 0
    if not m.any():
        return out
    a3, e0, tmax, s = steps[m, 6], steps[m, 7], steps[m, 8], steps[m, 9]
    w1 = tmax / e0
    alfa = np.ones_like(a3)
    big = (a3 > NMAXCONT) & (w1 > 1.0)
    alfa[big] = w1[big] * (NMAXCONT + a3[big]) / (w1[big] * NMAXCONT + a3[big])
    w2 = alfa * e0
    w = (tmax - w2) / tmax
    Ea = np.where(w > 0, w2 / np.maximum(1.0 - alpha * w, 1e-300), tmax)
    out[m] = np.minimum(Ea, tmax) * s
    return out


# --------------------------------------------------------------------------
# 2. the fixed-T_cut (additive) prescription
# --------------------------------------------------------------------------

def tcut_cumulants(steps, tcut, order=4):
    """Cumulants 1..order [MeV^n] of the CENTERED untruncated Urban compound
    Poisson of each step, with the single-collision spectrum restricted to
    E <= tcut (tcut = np.inf for no restriction).

    Exactly linear in (a1, a2, a3) by construction, and (a1, a2, a3) are each
    proportional to the step's mean loss, hence to path length. That is the
    whole additivity argument.
    """
    steps = np.atleast_2d(np.asarray(steps, dtype=np.float64))
    n = len(steps)
    K = np.zeros((order, n))
    reg = steps[:, 0]
    g0 = reg == 0
    if g0.any():                      # already-Gaussian steps: kappa2 only
        K[1, g0] = steps[g0, 1]
    m = ~g0
    if not m.any():
        return K
    s = steps[m, 9]
    a1, e1 = steps[m, 2], steps[m, 3] * s
    a2, e2 = steps[m, 4], steps[m, 5] * s
    a3, e0, tmax = steps[m, 6], steps[m, 7] * s, steps[m, 8] * s
    sub = np.zeros((order, m.sum()))
    for a, e in ((a1, e1), (a2, e2)):
        keep = (a > 0) & (e > 0) & (e <= tcut)
        for p in range(order):
            sub[p, keep] += a[keep] * e[keep] ** (p + 1)
    ok = (a3 > 0) & (tmax > e0) & (e0 > 0)
    if ok.any():
        E0, TM, A3 = e0[ok], tmax[ok], a3[ok]
        T = np.minimum(TM, tcut)
        good = T > E0
        C = E0 * TM / (TM - E0)
        idx = np.where(ok)[0][good]
        E0, T, A3, C = E0[good], T[good], A3[good], C[good]
        sub[0, idx] += A3 * C * np.log(T / E0)
        if order > 1:
            sub[1, idx] += A3 * C * (T - E0)
        if order > 2:
            sub[2, idx] += A3 * C * (T ** 2 - E0 ** 2) / 2.0
        if order > 3:
            sub[3, idx] += A3 * C * (T ** 3 - E0 ** 3) / 3.0
    K[:, m] = sub
    return K


def apply_tcut(steps, tcut):
    """Records rewritten so that the EXISTING untruncated CGF/CF machinery
    (cgf_saddlepoint.ioni_cgf_derivs, cf_track_resolution.ioni_step_exponent)
    evaluates the T_cut-restricted model.

    The restriction of a 1/E^2 spectrum on [e0, tmax] to [e0, T] is again a
    1/E^2 spectrum on [e0, T] with rate a3 * P(E <= T), so only (a3, tmax) move.
    tmax/e0 are stored PRE-scaling, so the cut is converted with `scaling`.
    """
    out = np.array(steps, dtype=np.float64, copy=True)
    if not np.isfinite(tcut):
        return out
    m = out[:, 0] != 0
    if not m.any():
        return out
    s = out[m, 9]
    e0, tmax, a3 = out[m, 7] * s, out[m, 8] * s, out[m, 6]
    T = np.minimum(tmax, tcut)
    frac = np.where(T > e0,
                    (1.0 / e0 - 1.0 / np.maximum(T, 1e-300)) /
                    (1.0 / e0 - 1.0 / tmax), 0.0)
    newa3, newtmax = a3 * frac, T
    dead = ~(T > e0)
    newa3[dead] = 0.0
    newtmax[dead] = tmax[dead]
    o = out[m]
    o[:, 6] = newa3
    o[:, 8] = newtmax / s
    # excitations above the cut are dropped too (never happens for keV cuts)
    for ja, je in ((2, 3), (4, 5)):
        o[o[:, je] * s > tcut, ja] = 0.0
    out[m] = o
    return out


# --------------------------------------------------------------------------
# 3. i/o helpers
# --------------------------------------------------------------------------

def _load(path):
    from cf_propagation_test import load_model
    return load_model(path if os.path.isabs(path) else os.path.join(SCRATCH, path))


def raw_steps(legs, k=None):
    """All ionization records of legs 0..k (default: all legs), unweighted."""
    sel = legs if k is None else legs[:k + 1]
    a = [l["ioni"] for l in sel if len(l["ioni"])]
    return np.concatenate(a) if a else np.zeros((0, 11))


def weighted_steps(legs, k, avec, sigma=1.0):
    """All ionization steps up to plane k with the transport weight folded into
    column 10 -- the same convention model_phi / ioni_cgf_derivs expect."""
    from cf_propagation_test import step_transports
    _, A_ioni, _ = step_transports(legs, k)
    out = []
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        st = leg["ioni"].copy().astype(np.float64)
        st[:, 10] *= w
        out.append(st)
    return np.concatenate(out) if out else np.zeros((0, 11))


def qop_weights(legs, k, avec):
    """Per-step d(qop-at-plane-k) / d(energy loss at that step) [1/MeV]."""
    from cf_propagation_test import step_transports
    _, A_ioni, _ = step_transports(legs, k)
    out = []
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0]
        out.append(w * leg["ioni"][:, 10] * 1e-3)
    return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------------------
# subcommand: validate
# --------------------------------------------------------------------------

def cmd_validate(args):
    from cf_propagation_test import FUNCTIONALS
    print(__doc__.split("Subcommands")[0].strip()[:0] or "", end="")
    print("VALIDATION -- reproduce the recorded gsig2 offline\n"
          "The offline recipe must BE the in-fit recipe before any conclusion "
          "is drawn from it.\n")
    trios = [("mod_sl10.0.root", 0.999, "per-step 0.999, L=10mm"),
             ("perleg_sl10.0.root", None, "per-leg equiv, L=10mm"),
             ("notrunc_sl10.0.root", 1.0, "no truncation, L=10mm"),
             ("mod_sl1.0.root", 0.999, "per-step 0.999, L= 1mm"),
             ("perleg_sl1.0.root", None, "per-leg equiv, L= 1mm"),
             ("notrunc_sl1.0.root", 1.0, "no truncation, L= 1mm"),
             ("realgeo_sl10.0.root", 0.999, "realgeo pT=3, L=10mm"),
             ("realgeo_sl1.0.root", 0.999, "realgeo pT=3, L= 1mm"),
             ("rg_pt40.root", 0.999, "realgeo pT=40"),
             ("rg_pt100.root", 0.999, "realgeo pT=100")]
    print(f"{'file':24s} {'alpha':>12s} {'nsteps':>7s} {'max |rel err|':>14s}")
    print("-" * 62)
    for fn, alpha, lab in trios:
        legs = _load(fn)
        st = raw_steps(legs)
        if alpha is None:                 # per-leg: solve for the alpha used
            alpha = _fit_alpha(st)
        v = g4_variance(st, alpha)
        rel = np.abs(v - st[:, 1]) / np.maximum(np.abs(st[:, 1]), 1e-300)
        print(f"{fn:24s} {alpha:12.9f} {len(st):7d} {rel.max():14.3e}")
    print("\n(alpha for the per-leg files is solved for from the records, "
          "not assumed)")


def _fit_alpha(steps):
    """Recover the alpha that produced the recorded gsig2, by bisection on the
    total variance (monotone in alpha)."""
    tgt = steps[:, 1].sum()
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g4_variance(steps, mid).sum() < tgt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# subcommand: additivity
# --------------------------------------------------------------------------

def _subdivide(steps, nsub):
    """Model-level subdivision of every step into nsub equal pieces: the Urban
    rates a1, a2, a3 are each proportional to the step's mean loss, hence to
    path length, and nothing else in the record is."""
    out = np.repeat(steps, nsub, axis=0)
    for c in (2, 4, 6):
        out[:, c] /= nsub
    out[:, 1] /= nsub          # regime-0 Gaussian variance is additive by def.
    return out


def cmd_additivity(args):
    from cf_propagation_test import FUNCTIONALS
    avec = FUNCTIONALS["qop"]
    print("TASK 1 -- ADDITIVITY UNDER SUBDIVISION\n")

    # ---------- (A) model-level subdivision, real-geometry records ----------
    print("(A) MODEL-LEVEL SUBDIVISION")
    print("    Each recorded step is split into n pieces by a_i -> a_i/n "
          "(the only extensive\n    quantities in the record). Everything else "
          "is a material/kinematic constant.\n    Quoted: sum over pieces / "
          "single step, for the total ionization variance of\n    the track.\n")
    for fn, lab in (("realgeo_sl10.0.root", "realgeo pT=3  L=10mm"),
                    ("rg_pt40.root", "realgeo pT=40 L=10mm"),
                    ("rg_pt100.root", "realgeo pT=100 L=10mm")):
        legs = _load(fn)
        st = raw_steps(legs)
        base_a = g4_variance(st, 0.999).sum()
        base_a1 = g4_variance(st, 1.0).sum()
        rows = []
        for n in (2, 4, 10, 100):
            sub = _subdivide(st, n)
            rows.append((n,
                         g4_variance(sub, 0.999).sum() / base_a,
                         g4_variance(sub, 1.0).sum() / base_a1))
        print(f"  {lab}   ({len(st)} steps)")
        print(f"    {'n':>5s} {'alpha=0.999':>14s} {'alpha=1 (G4)':>14s}"
              f" {'T_cut=1MeV k2':>15s} {'T_cut=inf k2':>14s} "
              f"{'T_cut=1MeV k3':>15s} {'T_cut=1MeV k4':>15s}")
        b2 = tcut_cumulants(st, 1.0)[1].sum()
        b2i = tcut_cumulants(st, np.inf)[1].sum()
        b3 = tcut_cumulants(st, 1.0)[2].sum()
        b4 = tcut_cumulants(st, 1.0)[3].sum()
        for (n, ra, ra1) in rows:
            sub = _subdivide(st, n)
            Kc = tcut_cumulants(sub, 1.0)
            Ki = tcut_cumulants(sub, np.inf)
            print(f"    {n:5d} {ra:14.6f} {ra1:14.6f} {Kc[1].sum()/b2:15.12f} "
                  f"{Ki[1].sum()/b2i:14.12f} {Kc[2].sum()/b3:15.12f} "
                  f"{Kc[3].sum()/b4:15.12f}")
        print()

    # ---------- (B) real 10x re-stepping ----------
    print("(B) REAL RE-STEPPING -- the same trajectory propagated with")
    print("    StepLengthLimit = 10 mm and 1 mm. Not a model subdivision: "
          "Geant4e re-walks\n    the geometry, re-evaluates dE/dx and re-fills "
          "every Urban record.\n")
    pairs = [("homogeneous toy", "mod_sl10.0.root", "mod_sl1.0.root"),
             ("real geometry pT=3", "realgeo_sl10.0.root", "realgeo_sl1.0.root")]
    for lab, fa, fb in pairs:
        la, lb = _load(fa), _load(fb)
        sa, sb = raw_steps(la), raw_steps(lb)
        print(f"  {lab}: {len(sa)} steps at 10 mm -> {len(sb)} at 1 mm "
              f"({len(sb)/len(sa):.2f}x)")
        # the reference trajectory must be the same, else this is not a
        # subdivision test
        dq = max(abs(la[k]["refqop"] - lb[k]["refqop"]) / abs(la[k]["refqop"])
                 for k in range(len(la)))
        print(f"    max relative refqop difference: {dq:.2e}  "
              f"(same trajectory: {'yes' if dq < 1e-6 else 'NO'})")
        print(f"    {'quantity':<34s} {'L=10mm':>14s} {'L=1mm':>14s} "
              f"{'ratio':>10s}")
        items = [("G4 variance, alpha=0.999 [MeV2]",
                  g4_variance(sa, 0.999).sum(), g4_variance(sb, 0.999).sum()),
                 ("G4 variance, alpha=1     [MeV2]",
                  g4_variance(sa, 1.0).sum(), g4_variance(sb, 1.0).sum()),
                 ("kappa2, T_cut = 100 keV  [MeV2]",
                  tcut_cumulants(sa, 0.1)[1].sum(),
                  tcut_cumulants(sb, 0.1)[1].sum()),
                 ("kappa2, T_cut = 1 MeV    [MeV2]",
                  tcut_cumulants(sa, 1.0)[1].sum(),
                  tcut_cumulants(sb, 1.0)[1].sum()),
                 ("kappa2, T_cut = inf      [MeV2]",
                  tcut_cumulants(sa, np.inf)[1].sum(),
                  tcut_cumulants(sb, np.inf)[1].sum()),
                 ("kappa1 (mean loss), T=1MeV [MeV]",
                  tcut_cumulants(sa, 1.0)[0].sum(),
                  tcut_cumulants(sb, 1.0)[0].sum()),
                 ("kappa3, T_cut = 1 MeV  [MeV3]",
                  tcut_cumulants(sa, 1.0)[2].sum(),
                  tcut_cumulants(sb, 1.0)[2].sum())]
        for nm, x, y in items:
            print(f"    {nm:<34s} {x:14.6e} {y:14.6e} {y/x:10.5f}")
        # the quantity the fit actually uses: block variance transported to the
        # outermost plane, in q/p units
        k = len(la) - 1
        wa, wb = qop_weights(la, k, avec), qop_weights(lb, k, avec)
        va = float(np.sum(g4_variance(sa, 0.999) * wa ** 2))
        vb = float(np.sum(g4_variance(sb, 0.999) * wb ** 2))
        ta = float(np.sum(tcut_cumulants(sa, 1.0)[1] * wa ** 2))
        tb = float(np.sum(tcut_cumulants(sb, 1.0)[1] * wb ** 2))
        print(f"    {'--- transported to plane ' + str(k) + ', qop units':<34s}")
        print(f"    {'sigma_qop, alpha=0.999':<34s} {np.sqrt(va):14.6e} "
              f"{np.sqrt(vb):14.6e} {np.sqrt(vb/va):10.5f}")
        print(f"    {'sigma_qop, T_cut=1MeV':<34s} {np.sqrt(ta):14.6e} "
              f"{np.sqrt(tb):14.6e} {np.sqrt(tb/ta):10.5f}")
        print()

    # ---------- (C) the mechanism ----------
    print("(C) MECHANISM -- the ENERGY at which alpha = 0.999 actually cuts")
    print("    E_a = w2/(1 - alpha w) with w2 = alfa(a3) e0: alfa ~ a3/nmaxCont "
          "for a3 >> 8,\n    and a3 is proportional to the step's mean loss. So "
          "the alpha convention IS a\n    fixed-energy cut -- one whose energy "
          "is proportional to the step length.\n")
    print(f"    {'model':<24s} {'steps':>6s} {'median E_a [MeV]':>17s} "
          f"{'p05':>10s} {'p95':>10s} {'median a3':>11s}")
    for fn, lab in (("mod_sl10.0.root", "homog toy   L=10mm"),
                    ("mod_sl1.0.root", "homog toy   L= 1mm"),
                    ("realgeo_sl10.0.root", "realgeo pT=3 L=10mm"),
                    ("realgeo_sl1.0.root", "realgeo pT=3 L= 1mm")):
        st = raw_steps(_load(fn))
        Ea = alpha_effective_cut(st, 0.999)
        print(f"    {lab:<24s} {len(st):6d} {np.nanmedian(Ea):17.4g} "
              f"{np.nanpercentile(Ea, 5):10.4g} {np.nanpercentile(Ea, 95):10.4g} "
              f"{np.median(st[:, 6]):11.4g}")
    print("\n    A 10x finer stepping moves the effective cut down by ~10x, and "
          "kappa2 of a\n    1/E^2 spectrum is linear in the cut, which is the "
          "whole non-additivity.\n")


# --------------------------------------------------------------------------
# subcommand: cost
# --------------------------------------------------------------------------

def cmd_cost(args):
    from cf_propagation_test import FUNCTIONALS
    from cgf_saddlepoint import mode, ioni_cgf_derivs
    avec = FUNCTIONALS["qop"]
    legs = _load(args.model)
    k = len(legs) - 1 if args.plane < 0 else args.plane
    st_raw = raw_steps(legs)
    w = qop_weights(legs, k, avec)

    # the fit's own width, in q/p units, under the production convention
    sig999 = float(np.sqrt(np.sum(g4_variance(st_raw, 0.999) * w ** 2)))
    stz = weighted_steps(legs, k, avec, sig999)

    Kinf = tcut_cumulants(st_raw, np.inf)
    var_inf = float(np.sum(Kinf[1] * w ** 2))
    # tail statistics of the delta-ray channel
    m = st_raw[:, 0] != 0
    s = st_raw[m, 9]
    a3, e0, tmax = st_raw[m, 6], st_raw[m, 7] * s, st_raw[m, 8] * s
    print(f"model: {args.model}   plane {k}   {len(st_raw)} ionization steps")
    print(f"  refqop = {legs[k]['refqop']:.6e}   p = {legs[k]['refp']:.3f} GeV")
    print(f"  sigma_qop(alpha=0.999) = {sig999:.6e}   "
          f"sigma_qop(untruncated)  = {np.sqrt(var_inf):.6e}   "
          f"ratio {np.sqrt(var_inf)/sig999:.2f}")
    print(f"  delta-ray channel: sum a3 = {a3.sum():.4g} collisions/track, "
          f"e0 = {e0.min()*1e6:.2f}-{e0.max()*1e6:.2f} eV, "
          f"tmax = {tmax.min():.3g}-{tmax.max():.3g} MeV")
    m0, kap2_0 = mode(stz)
    print(f"  untruncated block: mode = {m0:+.3f} z, kappa2 = {kap2_0:.4g} z^2 "
          f"(NOT a width)\n")

    C = e0 * tmax / (tmax - e0)
    mean_inf = float(np.sum(Kinf[0]))
    Ea = alpha_effective_cut(st_raw, 0.999)
    print(f"  the production alpha = 0.999 convention truncates at an effective "
          f"energy of\n  {np.nanpercentile(Ea, 5):.4g} - "
          f"{np.nanpercentile(Ea, 95):.4g} MeV (median "
          f"{np.nanmedian(Ea):.4g}) on this track's steps\n")

    grid = np.geomspace(args.tmin, args.tmax, args.npoints)
    print(f"{'T_cut':>10s} {'sigma_T/sig999':>14s} {'var frac':>10s} "
          f"{'N(E>T)/trk':>11s} {'mean lost':>10s} {'k3 frac':>9s} "
          f"{'skew z_T':>9s} {'BIAS [z_T]':>11s} {'dp/p [1e-4]':>12s}")
    print("-" * 105)
    # the TRUE (untruncated) block, standardized by each candidate sigma_T:
    # that is the mis-centring a Gaussian block of width sigma_T commits, since
    # the DATA always carry the full spectrum however the model is truncated.
    rows = []
    for T in grid:
        K = tcut_cumulants(st_raw, T)
        vT = float(np.sum(K[1] * w ** 2))
        if vT <= 0:
            continue
        sigT = np.sqrt(vT)
        above = np.where(tmax > T, a3 * C * (1.0 / np.maximum(T, e0) - 1.0 / tmax), 0.0)
        meanlost = 1.0 - float(np.sum(K[0])) / mean_inf
        k3T = float(np.sum(K[2] * w ** 3))
        k3I = float(np.sum(Kinf[2] * w ** 3))
        legsT = [dict(l, ioni=apply_tcut(l["ioni"], T)) for l in legs]
        # (i) the model's own skew: mode of the TRUNCATED block in its own sigma
        mT, _ = mode(weighted_steps(legsT, k, avec, sigT))
        # (ii) the estimator bias: mode of the TRUE block in that same sigma
        mTrue, _ = mode(weighted_steps(legs, k, avec, sigT))
        dpp = -mTrue * sigT / legs[k]["refqop"]
        print(f"{T:10.4g} {sigT/sig999:14.4f} {vT/var_inf:10.5f} "
              f"{above.sum():11.4g} {meanlost:10.5f} {k3T/k3I:9.5f} "
              f"{mT:9.3f} {mTrue:11.3f} {dpp*1e4:12.3f}")
        rows.append((T, sigT / sig999, vT / var_inf, above.sum(),
                     meanlost, k3T / k3I, mT, mTrue))
    print("\ncolumns (all at the outermost plane, q/p functional):")
    print("  sigma_T/sig999 : width of the T_cut Gaussian block relative to the "
          "production one")
    print("  var frac       : kappa2(T) / kappa2(untruncated)")
    print("  N(E>T)/trk     : expected number of delta rays above the cut on "
          "this whole track")
    print("  mean lost      : fraction of the mean ionization loss above the "
          "cut (NOT applied to\n                   the reference, which keeps "
          "the full mean -- this is only how much\n                   of the "
          "energy flow the NOISE model stops describing)")
    print("  skew z_T       : mode of the TRUNCATED model in its own units "
          "(0 = Gaussian)")
    print("  BIAS [z_T]     : mode of the TRUE (untruncated) block in those "
          "same units -- the\n                   mis-centring a Gaussian block "
          "of width sigma_T commits, because the\n                   DATA carry "
          "the full spectrum whatever the model truncates")
    print("  dp/p           : that bias as a relative momentum shift")
    return rows


# --------------------------------------------------------------------------
# subcommand: anchor -- what T_cut should production use?
# --------------------------------------------------------------------------

def solve_tcut(steps, w, target_var, lo=1e-4, hi=1e4):
    """T_cut whose kappa2, transported by w, equals target_var. kappa2(T) is
    strictly increasing in T, so bisection is safe."""
    def f(T):
        return float(np.sum(tcut_cumulants(steps, T)[1] * w ** 2))
    if f(hi) < target_var or f(lo) > target_var:
        return np.nan
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if f(mid) < target_var:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/model"
ANCHOR_MODELS = [
    (f"{CEPH}/model_mu_pt3_eta0.30.root", "mu pT=3   eta0.3"),
    (f"{CEPH}/model_mu_pt10_eta1.00.root", "mu pT=10  eta1.0"),
    (f"{CEPH}/model_mu_pt10_eta1.60.root", "mu pT=10  eta1.6"),
    (f"{CEPH}/model_mu_pt40_eta0.30_phi0.70.root", "mu pT=40  eta0.3"),
    (f"{CEPH}/model_mu_pt100_eta0.30.root", "mu pT=100 eta0.3"),
    ("realgeo_sl10.0.root", "realgeo pT=3 L=10mm"),
    ("realgeo_sl1.0.root", "realgeo pT=3 L= 1mm"),
    ("rg_pt40.root", "realgeo pT=40"),
    ("rg_pt100.root", "realgeo pT=100"),
]


def _one_collision_energy(steps):
    """T at which the block expects exactly one delta ray above the cut:
    N(E>T) = sum a3 C (1/T - 1/tmax) = 1, i.e. T ~ sum a3 C."""
    m = steps[:, 0] != 0
    if not m.any():
        return 0.0
    s = steps[m, 9]
    e0, tm = steps[m, 7] * s, steps[m, 8] * s
    return float(np.sum(steps[m, 6] * e0 * tm / (tm - e0)))


def _fisher_var(steps_z, v_scale):
    """1/I of the block density, returned in the units of v_scale (the
    variance the records were standardized with). RELATIVE floor, per the
    NOTES (XV) lesson."""
    from cgf_fisher import exact_density_fft, fisher_exact
    z, p, dp, _ = exact_density_fft(steps_z, 2000.0, 1 << 19)
    I, _ = fisher_exact(z, p, dp, floor=1e-8)
    return (1.0 / I) * v_scale, 1.0 / I


def cmd_anchor(args):
    from cf_propagation_test import FUNCTIONALS
    avec = FUNCTIONALS["qop"]
    print("CHOOSING T_cut\n")
    print("Three candidate anchors:\n"
          "  (i)   variance-neutral  -- reproduce today's sigma(alpha=0.999), "
          "so the change is\n"
          "        a drop-in and the existing calibration is not invalidated\n"
          "  (ii)  Fisher-optimal    -- sigma^2 = 1/I of the TRUE block "
          "density, the only\n"
          "        variance that gives a Gaussian block the right asymptotic "
          "weight\n"
          "  (iii) one-collision     -- the cut above which the block expects "
          "< 1 delta ray\n")

    print("=" * 100)
    print("A. WHOLE-TRACK block (cumulative to the outermost plane) -- what "
          "the closure test uses")
    print("=" * 100)
    print(f"{'model':<22s} {'p[GeV]':>7s} {'sig999':>11s} {'1/I [z^2]':>10s} "
          f"{'T(i)':>8s} {'T(ii)':>8s} {'T(iii)':>8s} {'T(ii)/T(iii)':>12s}")
    ratios = []
    for fn, lab in ANCHOR_MODELS:
        legs = _load(fn)
        k = len(legs) - 1
        st, w = raw_steps(legs, k), qop_weights(legs, k, avec)
        v999 = float(np.sum(g4_variance(st, 0.999) * w ** 2))
        vF, Iz = _fisher_var(weighted_steps(legs, k, avec, np.sqrt(v999)), v999)
        Ti, Tii = solve_tcut(st, w, v999), solve_tcut(st, w, vF)
        Tiii = _one_collision_energy(st)
        ratios.append(Tii / Tiii)
        print(f"{lab:<22s} {legs[k]['refp']:7.2f} {np.sqrt(v999):11.4e} "
              f"{Iz:10.4f} {Ti:8.4g} {Tii:8.4g} {Tiii:8.4g} {Tii/Tiii:12.3f}")
    r = np.array(ratios)
    print(f"\n  T(ii)/T(iii) = {r.mean():.3f} +- {r.std():.3f} -- the "
          f"Fisher-optimal cut is ~3x the block's\n  own one-collision energy, "
          "to 0.4 %, over the whole set. It is therefore NOT a\n  universal "
          "energy: it scales with the block.")

    print()
    print("=" * 100)
    print("B. PER-LEG blocks -- the granularity the fit's Q matrix actually "
          "has")
    print("=" * 100)
    print(f"{'model':<22s} {'nlegs':>6s} {'T(i) p05':>9s} {'T(i) med':>9s} "
          f"{'T(i) p95':>9s} {'T(ii) med':>10s} {'T(iii) med':>11s} "
          f"{'sqrt(1/I)/sig999 med':>21s}")
    aT, aTf = [], []
    for fn, lab in ANCHOR_MODELS:
        legs = _load(fn)
        Ti, Tii, Tiii, RF = [], [], [], []
        for leg in legs:
            st = leg["ioni"]
            if len(st) < 2 or (st[:, 0] != 0).sum() < 2:
                continue
            q = np.sign(leg["refqop"]) or 1.0
            w = q * st[:, 10] * 1e-3
            v999 = float(np.sum(g4_variance(st, 0.999) * w ** 2))
            if v999 <= 0:
                continue
            stz = st.copy()
            stz[:, 10] = stz[:, 10] * q / np.sqrt(v999)
            vF, _ = _fisher_var(stz, v999)
            t1 = solve_tcut(st, w, v999, lo=1e-8, hi=1e5)
            t2 = solve_tcut(st, w, vF, lo=1e-8, hi=1e5)
            if not (np.isfinite(t1) and np.isfinite(t2)):
                continue
            Ti.append(t1); Tii.append(t2)
            Tiii.append(_one_collision_energy(st))
            RF.append(np.sqrt(vF / v999))
        Ti, Tii = np.array(Ti), np.array(Tii)
        aT += list(Ti); aTf += list(Tii)
        print(f"{lab:<22s} {len(Ti):6d} {np.percentile(Ti, 5):9.4f} "
              f"{np.median(Ti):9.4f} {np.percentile(Ti, 95):9.4f} "
              f"{np.median(Tii):10.4f} {np.median(Tiii):11.5f} "
              f"{np.median(RF):21.3f}")
    aT, aTf = np.array(aT), np.array(aTf)
    print(f"\n  all {len(aT)} legs: T(i) median {np.median(aT):.3f} MeV "
          f"(p05-p95 {np.percentile(aT, 5):.3f}-{np.percentile(aT, 95):.3f}); "
          f"T(ii) median {np.median(aTf):.3f} MeV")
    print("  sqrt(1/I)/sigma999 < 1 per leg: the CURRENT convention is already "
          "1.5-4x WIDER\n  than the Fisher weight on a single leg, the opposite "
          "sign to the whole-track\n  block. No single global T_cut can be "
          "Fisher-optimal at both granularities.")

    print()
    print("C. what a GLOBAL T_cut does to the per-leg block widths")
    print(f"  {'T_cut [MeV]':>11s} {'sigma_T/sigma_999':>18s}  (median, p05, "
          f"p95 over all legs)")
    for T in (0.1, 0.3, 0.5, 1.0, 1.5, 2.5):
        rr = []
        for fn, lab in ANCHOR_MODELS:
            for leg in _load(fn):
                st = leg["ioni"]
                if len(st) < 2:
                    continue
                w = st[:, 10] * 1e-3
                v = float(np.sum(g4_variance(st, 0.999) * w ** 2))
                if v <= 0:
                    continue
                rr.append(np.sqrt(float(np.sum(tcut_cumulants(st, T)[1]
                                               * w ** 2)) / v))
        rr = np.array(rr)
        print(f"  {T:11.2f} {np.median(rr):8.3f} {np.percentile(rr, 5):8.3f} "
              f"{np.percentile(rr, 95):8.3f}")


# --------------------------------------------------------------------------
# subcommand: closure
# --------------------------------------------------------------------------

TOY_PLANES = ("/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
              "Analysis/HitAnalyzer/test/toyPlanes_pt3.py")

CLOSURE_CONFIGS = [
    ("per-step 0.999  L=10mm", "mod_sl10.0.root", None),
    ("per-leg equiv   L=10mm", "perleg_sl10.0.root", None),
    ("no truncation   L=10mm", "notrunc_sl10.0.root", None),
    ("T_cut = 1 MeV   L=10mm", "mod_sl10.0.root", 1.0),
    ("T_cut = 100 keV L=10mm", "mod_sl10.0.root", 0.1),
    ("per-step 0.999  L= 1mm", "mod_sl1.0.root", None),
    ("per-leg equiv   L= 1mm", "perleg_sl1.0.root", None),
    ("no truncation   L= 1mm", "notrunc_sl1.0.root", None),
    ("T_cut = 1 MeV   L= 1mm", "mod_sl1.0.root", 1.0),
    ("T_cut = 100 keV L= 1mm", "mod_sl1.0.root", 0.1),
]


def _config_sigma(legs, T, avec, k):
    """The width of the fit's Gaussian process-noise block at plane k under a
    given convention. T=None is the propagator's own alpha-truncated Q; a float
    replaces ONLY the ionization share by the fixed-T_cut kappa2.

    NOTE the truncation is a convention for the BLOCK WIDTH only. It must not
    be applied to the model's predicted lineshape, which always carries the
    full delta-ray spectrum -- the data do.
    """
    from cf_propagation_test import model_variance
    v, vms, vio = model_variance(legs, k, avec)
    if T is None:
        return float(np.sqrt(v))
    st, w = raw_steps(legs, k), qop_weights(legs, k, avec)
    return float(np.sqrt(vms + np.sum(tcut_cumulants(st, T)[1] * w ** 2)))


_PHI_CACHE = {}


def _closure_row(legs, T, sim, avec, probes, mode_, sigma_fixed=None, tag=""):
    """(<e^{-u z^2}>_data - reference) per plane, then mean and rms over planes.

    mode_ = "cf"    reference is the model's full characteristic function
                    (the physics test: is the transport-fluctuation model right)
    mode_ = "gauss" reference is N(0,1), i.e. the Gaussian block the fit
                    actually assumes (the estimator test)
    sigma_fixed = None -> each configuration's own sigma; array -> common scale.
    """
    from cf_propagation_test import (model_phi, weier_scalar, SIM_BRANCH,
                                     REF_BRANCH, TAU)
    nl = min(sim["valid"].shape[1], len(legs))
    rows, sigs = [], []
    for k in range(nl):
        good = sim["valid"][:, k] & np.isfinite(sim[SIM_BRANCH["qop"]][:, k])
        if good.sum() < 100:
            continue
        sig_own = _config_sigma(legs, T, avec, k)
        sig = sig_own if sigma_fixed is None else float(sigma_fixed[k])
        d = sim[SIM_BRANCH["qop"]][good, k] - legs[k][REF_BRANCH["qop"]]
        z = d / sig
        if mode_ == "gauss":
            ref = [1.0 / np.sqrt(1.0 + 2.0 * u) for u in probes]
        else:
            key = (tag, k, sig)
            if key not in _PHI_CACHE:
                _PHI_CACHE[key] = model_phi(legs, k, avec, sig, TAU)
            ref = [weier_scalar(_PHI_CACHE[key], u, TAU) for u in probes]
        rows.append([float(np.mean(np.exp(-u * z ** 2))) - r
                     for u, r in zip(probes, ref)])
        sigs.append(sig_own)
    a = np.array(rows)
    return a.mean(axis=0), a.std(axis=0), np.array(sigs)


def cmd_closure(args):
    from cf_propagation_test import model_variance, FUNCTIONALS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "cleanprop"))
    from toy_loader import load_toy_sim
    avec = FUNCTIONALS["qop"]
    ns = {}
    exec(open(args.planes).read(), ns)
    sim = load_toy_sim(args.sim, ns["origin"], ns["normal"], ns["uaxis"])
    probes = args.probes

    print("TASK 3 -- the NOTES (XI) three-way comparison, de-confounded\n")
    print(f"toy homogeneous geometry, {sim['ntot']} events, "
          f"q/p residual, mean (rms) over planes\n")

    ref = _load("mod_sl10.0.root")
    SIGREF = np.array([np.sqrt(model_variance(ref, k, avec)[0])
                       for k in range(len(ref))])

    hdr = (f"{'config':24s} {'sigma/sigref':>12s} " +
           "".join(f"{'u=' + str(u):>19s}" for u in probes))
    for title, fixed, mode_ in (
            ("A. data vs the model CF, u in units of EACH configuration's OWN "
             "sigma\n   -- this is what NOTES (XI) quoted", None, "cf"),
            ("B. data vs the model CF, u in units of a COMMON sigma "
             "(alpha=0.999, L=10mm)\n   -- the probe now interrogates the same "
             "PHYSICAL |delta(q/p)| in every row", SIGREF, "cf"),
            ("C. data vs the GAUSSIAN BLOCK the fit actually assumes, N(0,1) "
             "in each\n   configuration's own sigma -- the estimator error, "
             "not the model error", None, "gauss")):
        print(title + "\n")
        print(hdr)
        print("-" * len(hdr))
        for lab, fn, T in CLOSURE_CONFIGS:
            legs = _load(fn)
            m, s, sg = _closure_row(legs, T, sim, avec, probes, mode_, fixed,
                                    tag=fn)
            print(f"{lab:24s} {np.mean(sg / SIGREF[:len(sg)]):12.4f} " +
                  "".join(f"{mi:+11.5f} ({si:5.5f})" for mi, si in zip(m, s)))
        print()


# --------------------------------------------------------------------------
# subcommand: motivations
# --------------------------------------------------------------------------

def cmd_motivations(args):
    from cf_propagation_test import (load_sim, model_phi, model_variance,
                                     weier_scalar, FUNCTIONALS, SIM_BRANCH,
                                     REF_BRANCH, TAU)
    from cgf_saddlepoint import mode
    avec = FUNCTIONALS["qop"]
    print("TASK 4 -- the two motivations, separated\n")

    print("(a) STEP-SIZE DEPENDENCE OF THE FIT'S OWN WIDTH sigma")
    print("    This is a property of the Q matrix alone: no data needed.\n")
    print(f"    {'geometry':<22s} {'convention':<18s} {'sigma(10mm)':>13s} "
          f"{'sigma(1mm)':>13s} {'ratio':>9s}")
    # the per-leg-equivalent alphas actually used for the NOTES (XI) files
    APL = {"mod_sl10.0.root": _fit_alpha(raw_steps(_load("perleg_sl10.0.root"))),
           "mod_sl1.0.root": _fit_alpha(raw_steps(_load("perleg_sl1.0.root")))}
    for lab, fa, fb in (("homogeneous toy", "mod_sl10.0.root", "mod_sl1.0.root"),
                        ("real geometry pT=3", "realgeo_sl10.0.root",
                         "realgeo_sl1.0.root")):
        la, lb = _load(fa), _load(fb)
        k = len(la) - 1
        sa, sb = raw_steps(la), raw_steps(lb)
        wa, wb = qop_weights(la, k, avec), qop_weights(lb, k, avec)
        conv = [("alpha = 0.999", g4_variance(sa, 0.999), g4_variance(sb, 0.999))]
        if fa in APL:
            conv.append(("alpha per-leg equiv",
                         g4_variance(sa, APL[fa]), g4_variance(sb, APL[fb])))
        conv += [("alpha = 1 (G4)", g4_variance(sa, 1.0), g4_variance(sb, 1.0)),
                 ("T_cut = 100 keV", tcut_cumulants(sa, 0.1)[1],
                  tcut_cumulants(sb, 0.1)[1]),
                 ("T_cut = 1 MeV", tcut_cumulants(sa, 1.0)[1],
                  tcut_cumulants(sb, 1.0)[1]),
                 ("T_cut = 2.5 MeV", tcut_cumulants(sa, 2.5)[1],
                  tcut_cumulants(sb, 2.5)[1])]
        for cname, va, vb in conv:
            A = np.sqrt(np.sum(va * wa ** 2))
            B = np.sqrt(np.sum(vb * wb ** 2))
            print(f"    {lab:<22s} {cname:<18s} {A:13.6e} {B:13.6e} "
                  f"{B/A:9.5f}")
    print()

    print("(b) ESTIMATOR INCONSISTENCY UNDER SKEWED NOISE")
    print("    The TRUE block density is the untruncated one whatever the fit "
          "truncates,\n    because the data carry the whole spectrum. Two "
          "mis-specifications follow:\n"
          "      CENTRE  the fit references the MEAN loss; the modal track "
          "loses less\n"
          "      WEIGHT  the fit weights by 1/sigma^2; the correct asymptotic "
          "weight is I\n")
    from cgf_fisher import exact_density_fft, fisher_exact
    print(f"    {'model':<16s} {'convention':<16s} {'sigma_qop':>11s} "
          f"{'mode [sig]':>10s} {'dp/p [1e-4]':>11s} {'sqrt(1/I)/sig':>13s}")
    for fn, lab in (("realgeo_sl10.0.root", "realgeo pT=3"),
                    ("rg_pt40.root", "realgeo pT=40"),
                    ("rg_pt100.root", "realgeo pT=100")):
        legs = _load(fn)
        k = len(legs) - 1
        st = raw_steps(legs, k)
        w = qop_weights(legs, k, avec)
        v999 = float(np.sum(g4_variance(st, 0.999) * w ** 2))
        z, p, dp_, _ = exact_density_fft(weighted_steps(legs, k, avec,
                                                        np.sqrt(v999)),
                                         2000.0, 1 << 19)
        I, _ = fisher_exact(z, p, dp_, floor=1e-8)
        vF = (1.0 / I) * v999
        for cname, T in (("alpha = 0.999", None), ("T_cut = 100 keV", 0.1),
                         ("T_cut = 1 MeV", 1.0), ("T_cut = 2.5 MeV", 2.5)):
            sig = (np.sqrt(v999) if T is None else
                   float(np.sqrt(np.sum(tcut_cumulants(st, T)[1] * w ** 2))))
            m, _ = mode(weighted_steps(legs, k, avec, sig))   # TRUE block
            print(f"    {lab:<16s} {cname:<16s} {sig:11.4e} {m:10.3f} "
                  f"{-m*sig/legs[k]['refqop']*1e4:11.3f} "
                  f"{np.sqrt(vF)/sig:13.3f}")
    print("\n    dp/p is the same in every row of a block by construction: it "
          "is a property\n    of the physics, not of the convention. THAT is "
          "the point -- no choice of\n    truncation moves it, only the CGF "
          "estimator does.\n"
          "    CAVEAT: this is the model's mode. Phase 0 measured the model "
          "mode +4.295\n    against Geant4's +3.298 at this plane, so the "
          "analytic compound Poisson\n    overstates the shift by ~30 % and "
          "these dp/p are upper bounds.\n")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate")

    sub.add_parser("additivity")

    c = sub.add_parser("cost")
    c.add_argument("--model", default="realgeo_sl10.0.root")
    c.add_argument("--plane", type=int, default=-1)
    c.add_argument("--tmin", type=float, default=1e-3)
    c.add_argument("--tmax", type=float, default=1e3)
    c.add_argument("--npoints", type=int, default=19)

    c = sub.add_parser("closure")
    c.add_argument("--sim", default=os.path.join(SCRATCH, "sim_homo_scan.root"))
    c.add_argument("--planes", default=TOY_PLANES)
    c.add_argument("--probes", nargs="+", type=float,
                   default=[0.01, 0.1, 1.0])

    sub.add_parser("anchor")

    sub.add_parser("motivations")

    args = p.parse_args()
    {"validate": cmd_validate, "additivity": cmd_additivity, "cost": cmd_cost,
     "anchor": cmd_anchor, "closure": cmd_closure,
     "motivations": cmd_motivations}[args.cmd](args)


if __name__ == "__main__":
    main()
