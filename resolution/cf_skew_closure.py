#!/usr/bin/env python3
"""ODD-moment closure of the CF track-resolution model.

WHY. `cf_track_resolution.py --closure` compares data against model through
<e^{-u z^2}>, which is EVEN in z: it tests the width and the tails and is
blind to the asymmetry.  The open physics question is odd -- the bare refit's
J/psi mass PEAK sits ~+0.2e-3 high while the MEAN closes, which is the
signature of a skewed pull whose mode and mean disagree, and the suspect is
the Landau mean-vs-mode asymmetry of the ionization energy loss transmitted
through the fit.

The CF model already contains that asymmetry, and contains it in exactly one
place: `model_phi` builds

    S(t) = S_hit(t) + e^{k_ms} S_ms(t) + e^{k_ioni} (S_ioni^re + i S_ioni^im)

and S_hit (gauss) and S_ms are REAL.  SINCE 2026-09-03 there is a SECOND odd
channel: the radiative (brems + pair) block, S = ... + k_rad (Srad^re + i
Srad^im), one-sided in the same direction (a radiated photon can only take
energy AWAY).  `--krad 0` removes it and reproduces every number this script
produced before it existed; `--krad 1` (the default) is the physics model.
`k_skew` still multiplies Im S_ioni ALONE -- it is the ionization lever arm,
and keeping it that way is what makes the two channels separable in the table:
the radiative term is a FIXED prediction and k_skew measures what the
ionization block would have to do on top of it.

THE STATISTIC.  <z e^{-u z^2}>.  It is bounded, |z e^{-uz^2}| <= 1/sqrt(2 e u),
so its sample mean has a finite variance even against a Landau tail -- unlike
<z> or <z^3>, whose sample errors are dominated by whichever single track
happened to radiate.  In transform space, with phi(t) = <e^{itz}> and
phi(-t) = phi(t)*,

    e^{-uz^2} = 1/(2 sqrt(pi u)) Int_R e^{-t^2/4u} e^{-itz} dt
    z e^{-itz} = i d/dt e^{-itz}

    <z e^{-uz^2}> = 1/(2 sqrt(pi u)) Int_R e^{-t^2/4u} [i d/dt phi(-t)] dt
                  = i/(4 u sqrt(pi u)) Int_R t e^{-t^2/4u} phi(t)* dt   [by parts]
                  = 1/sqrt(pi u) Int_0^inf (t/2u) e^{-t^2/4u} Im phi(t) dt

using Re phi even, Im phi odd, so only the Im part survives the symmetric
integral.  This is `weier_odd`, the exact odd twin of `cf_track_resolution.
weier`, on the same t-grid with the same quadrature.  `--validate` checks it
against (i) direct quadrature of the definition, (ii) Monte Carlo of a shifted
exponential, (iii) Monte Carlo of a Gauss (x) compensated-Poisson toy with
1/x^2 jumps -- the analogue of the ionization block itself, (iv) a x64
refinement of the t-grid, (v) the small-t limit of `ioni_step_exponent` on a
synthetic record, which is what makes "the model's mean is zero" a measurement
rather than an assertion, and (vi) the mode estimator itself, on 3.2e5 draws
from a density whose mode is known.

WHAT IT REPORTS.  (0) the bins, with sigma_rel = sigma(q/p) p so a pull shift
in sigma can be read as a relative momentum shift; (1) <z e^{-uz^2}> data minus
model per bin and probe with a bootstrap error; (2) the single scale k on
Im S_ioni that would close it; (2b) the ALTERNATIVE one-parameter hypothesis, a
pure translation of the model density, whose chi2 is directly comparable;
(3) the mode and the trimmed mean of the pull density, data against model, with
the same estimator on both sides; (4) <z> against the trim, which separates a
skew (tight and wide trims move oppositely) from a location shift (they move
together); (5) the model's own mode once its skew is scaled to the fitted k.

SIGN CONVENTION.  z = (q/p_fit - q/p_gen)/sigma (`cf_track_resolution.extract`).
q/p carries the CHARGE, so

    q = +1:  z > 0  <=>  momentum LOWER        q = -1:  z > 0  <=>  momentum HIGHER

An energy-loss bias is charge-EVEN in momentum and therefore charge-ODD in z.
d(q/p) = q cs dE with cs = E/p^3 > 0, so the Landau skew of z has the sign of
q: for mu+ the mode of z sits BELOW zero, for mu- ABOVE, and both mean the same
thing physically -- the typical track is reconstructed with too HIGH a
momentum, which is the sign of the observed J/psi peak shift.  In a charge-
symmetric sample the two cancel in <z>, in <z e^{-uz^2}> and in the mode.

THE CHARGE FACTOR IS AUTOMATIC (see `load`).  The exported cs = E/p^3 is
positive for every track, so the offline CF has to supply the q of
d(q/p) = q cs dE itself.  `cf_track_resolution.extract` does this since
2026-09-03 and marks such caches with the key `ioni_charge_signed`; caches
written before it carry the mu+ skew for BOTH charges while the data cancels,
and `load` repairs them by Sio_im -> q Sio_im, which is exact at cache level
because Re S is even and Im S odd in the block weight.  Either cache gives the
same tables; there is no flag to get wrong.

usage:
  python cf_skew_closure.py --validate
  python cf_skew_closure.py --cache runs/cf_trackres_mugun_ul16_fix.npz \\
      --label "mu pT 20-60 [Aug-8 model+fit]" --tag mugun_ul16_fix
"""
import argparse
import datetime
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.interpolate import CubicSpline

from wums import logging, output_tools, plot_tools

import pubhtml

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cft", os.path.join(_HERE, "cf_track_resolution.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["cft"] = m
spec.loader.exec_module(m)

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

TG = m.TG                       # t conjugate to standardized z, [0,14], 448 pts
PROBES = (0.05, 0.2, 0.5, 1.0, 2.0)


class A:
    """The argument object `model_phi` reads.  `hitmode="gauss"` is the
    historical treatment (one Gaussian of the summed block variance) and is
    what every published closure number was produced with."""
    khit = 0.0
    kms = 0.0
    kioni = 0.0
    # LINEAR scale on the radiative block (cf_track_resolution.model_phi), so
    # that 0 is expressible: krad = 0 is the pre-2026-09-03 model exactly.
    krad = 1.0
    kdel = None
    hitmode = "gauss"


# --------------------------------------------------------------- TRANSFORMS
def weier_odd(phi, u, tg=TG):
    """<z e^{-u z^2}> per track from phi on the t-grid.

    The odd twin of `cf_track_resolution.weier`: same grid, same trapezoid,
    the even Gaussian weight replaced by (t/2u) e^{-t^2/4u} / sqrt(pi u) and
    Re phi by Im phi.  NOT clipped -- the bound is |.| <= 1/sqrt(2 e u) and
    clipping a two-sided statistic at its bound would bias it."""
    w = (tg / (2. * u)) * np.exp(-tg ** 2 / (4. * u)) / np.sqrt(np.pi * u)
    a = np.atleast_2d(phi.imag)
    out = np.trapezoid(a * w[None, :], tg, axis=1)
    return out if np.ndim(phi) == 2 else out[0]


def odd_bound(u):
    return 1.0 / np.sqrt(2.0 * np.e * u)


def skew_phi(phi, Sio_im, kskew, kioni=0.0):
    """phi with Im S_ioni -> kskew * Im S_ioni, real part untouched.

    S = S_hit + e^{kms} S_ms + e^{kioni} S_ioni, and only S_ioni is complex,
    so multiplying by exp(i (k-1) e^{kioni} Im S_ioni) reproduces exactly the
    exponent with its imaginary ionization part scaled.  At kskew = 1 the
    factor is exp(0) = 1, i.e. bit-identical to `model_phi`."""
    if kskew == 1.0:
        return phi
    return phi * np.exp(1j * (kskew - 1.0) * np.exp(kioni) * Sio_im)


def invert_matrices(zg, tg=TG):
    """(Kc, Ks) with p(z) = (Kc @ Re phi + Ks @ Im phi) / pi.

    p(z) = 1/pi Int_0^inf [Re phi cos(tz) + Im phi sin(tz)] dt."""
    w = np.gradient(tg)           # trapezoid weights on a uniform grid
    w[0] *= 0.5
    w[-1] *= 0.5
    ph = tg[None, :] * zg[:, None]
    return np.cos(ph) * w[None, :] / np.pi, np.sin(ph) * w[None, :] / np.pi


def parab_mode(x, y, half=None):
    """Peak position by a least-squares parabola in a window around argmax."""
    i = int(np.argmax(y))
    if half is None:
        return x[i]
    sel = np.abs(x - x[i]) <= half
    if sel.sum() < 3:
        return x[i]
    c = np.polyfit(x[sel] - x[i], y[sel], 2)
    if c[0] >= 0:
        return x[i]
    return x[i] - 0.5 * c[1] / c[0]


def smooth(y, dz, h):
    """Gaussian smoothing of a density sampled on a uniform grid."""
    if h <= 0:
        return y
    n = int(np.ceil(4. * h / dz))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * dz / h) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


# -------------------------------------------------------------- VALIDATION
def _toy_gauss_cpoisson(rng, n, sig=0.9, nu=0.6, x0=0.02, xmax=8.0):
    """Gauss (x) compensated Poisson with dN = nu dx/x^2 on [x0, xmax].

    The analogue of one ionization block: the compensator makes the mean
    exactly zero, the 1/x^2 spectrum makes it right-skewed and heavy-tailed."""
    lam = nu * (1. / x0 - 1. / xmax)                 # total rate
    mean = nu * np.log(xmax / x0)                    # compensator
    k = rng.poisson(lam, n)
    z = rng.normal(0., sig, n) - mean
    tot = int(k.sum())
    # inverse-CDF sample of 1/x^2 on [x0, xmax]
    v = rng.random(tot)
    x = 1.0 / (1. / x0 - v * (1. / x0 - 1. / xmax))
    idx = np.repeat(np.arange(n), k)
    np.add.at(z, idx, x)
    # exponent, by quadrature over the jump measure
    xs = np.geomspace(x0, xmax, 40000)
    th = TG[:, None] * xs[None, :]
    integ = nu * (np.exp(1j * th) - 1. - 1j * th) / xs[None, :] ** 2
    S = np.trapezoid(integ, xs, axis=1) - 0.5 * sig ** 2 * TG ** 2
    return z, np.exp(S)


def validate(rng=None):
    """Three independent checks of `weier_odd`."""
    rng = rng or np.random.default_rng(20260902)
    ok = True
    lines = ["VALIDATION of <z e^{-u z^2}> = 1/sqrt(pi u) Int_0^inf (t/2u) "
             "e^{-t^2/4u} Im phi(t) dt", ""]

    # (i) shifted exponential: exact 1D quadrature of the definition vs the
    #     transform.  Machine-precision test of the identity itself.
    xq = np.linspace(0., 60., 4_000_001)
    fq = np.exp(-xq)
    phi = np.exp(-1j * TG) / (1. - 1j * TG)
    lines.append("(i) shifted exponential (mean 0, var 1, skew 2), "
                 "transform vs direct quadrature of the definition")
    lines.append(f"    {'u':>6} {'quadrature':>14} {'transform':>14} {'diff':>12}")
    for u in PROBES:
        ref = np.trapezoid(fq * (xq - 1.) * np.exp(-u * (xq - 1.) ** 2), xq)
        got = weier_odd(phi, u)
        ok &= abs(got - ref) < 1e-3
        lines.append(f"    {u:6.2f} {ref:+14.8f} {got:+14.8f} {got-ref:+12.2e}")

    # (ii) the same, against Monte Carlo -- checks the ESTIMATOR is unbiased
    z = rng.standard_exponential(4_000_000) - 1.
    lines.append("")
    lines.append("(ii) same distribution, 4e6 Monte Carlo draws")
    lines.append(f"    {'u':>6} {'MC':>14} {'+-':>10} {'transform':>14} {'pull':>8}")
    for u in PROBES:
        s = z * np.exp(-u * z ** 2)
        mc, er = s.mean(), s.std() / np.sqrt(len(s))
        got = weier_odd(phi, u)
        ok &= abs(got - mc) < max(1e-3, 4. * er)
        lines.append(f"    {u:6.2f} {mc:+14.8f} {er:10.2e} {got:+14.8f} "
                     f"{(got-mc)/er:+8.2f}")

    # (iii) Gauss (x) compensated Poisson, 1/x^2 jumps -- the ionization
    #       block's own structure, mean zero by construction
    zt, pt = _toy_gauss_cpoisson(rng, 4_000_000)
    lines.append("")
    lines.append("(iii) Gauss (x) compensated Poisson, dN = nu dx/x^2 "
                 "(the ionization block's structure), 4e6 draws")
    lines.append(f"    {'u':>6} {'MC odd':>14} {'+-':>10} {'transform':>14} "
                 f"{'pull':>8} | {'MC even':>10} {'transform':>10}")
    for u in PROBES:
        s = zt * np.exp(-u * zt ** 2)
        mc, er = s.mean(), s.std() / np.sqrt(len(s))
        got = weier_odd(pt, u)
        ev, evm = np.exp(-u * zt ** 2).mean(), m.weier(pt[None, :], u)[0]
        ok &= abs(got - mc) < max(1e-3, 4. * er)
        lines.append(f"    {u:6.2f} {mc:+14.8f} {er:10.2e} {got:+14.8f} "
                     f"{(got-mc)/er:+8.2f} | {ev:10.6f} {evm:10.6f}")

    # (iv) the t-grid actually resolves the odd integrand: refine x64
    lines.append("")
    lines.append("(iv) quadrature convergence of the SAME integrand on the "
                 "production grid (448 pts on [0,14]) vs x64 refinement")
    lines.append(f"    {'u':>6} {'n=448':>14} {'n=28672':>14} {'diff':>12}")
    for u in PROBES:
        tf = np.linspace(0., 14., 28672)
        pf = np.exp(-1j * tf) / (1. - 1j * tf)
        a, b = weier_odd(phi, u), weier_odd(pf, u, tf)
        ok &= abs(a - b) < 1e-6
        lines.append(f"    {u:6.2f} {a:+14.8f} {b:+14.8f} {a-b:+12.2e}")

    # (v) the model's mean is ZERO BY CONSTRUCTION: every ionization term is
    #     written a(e^{i th} - 1 - i th), so S'(0) = 0 identically.  Shown on a
    #     synthetic Urban record through the production code path, on a t-grid
    #     fine enough to reach the analytic small-t regime.
    lines.append("")
    lines.append("(v) <z>_model = Im S'(0) = 0 by construction -- "
                 "`ioni_step_exponent` on a synthetic record, small-t limit")
    lines.append(f"    {'regime':>8} {'t':>10} {'Im S':>14} {'Im S / t':>14}")
    for reg, ncol in ((1, 11), (2, 13)):
        st = np.zeros((1, ncol))
        st[0, 0] = reg
        st[0, 1] = 1.0e-4      # gsig2
        st[0, 2], st[0, 3] = 30.0, 1.0e-2     # a1, e1 (MeV)
        st[0, 4], st[0, 5] = 10.0, 5.0e-2     # a2, e2
        st[0, 6] = 0.02 if reg == 1 else 0.02  # a3 / xi
        st[0, 7], st[0, 8] = 1.0e-1, 3.0e4    # e0, tmax (MeV)
        st[0, 9] = 1.0                         # gamma scale
        st[0, 10] = 1.0                        # g (qop per GeV -> *1e-3)
        if ncol == 13:
            st[0, 11], st[0, 12] = 0.999, 4.0e4
        tt = np.array([0., 1e-8, 2e-8, 4e-8, 1e-7])
        S = m.ioni_step_exponent(st, 1.0, tt)
        for j in (1, 2, 3, 4):
            lines.append(f"    {reg:8d} {tt[j]:10.1e} {S.imag[j]:+14.4e} "
                         f"{S.imag[j]/tt[j]:+14.4e}")
        slope = S.imag[4] / tt[4]
        ok &= abs(slope) < 1e-6
    lines.append("    -> Im S / t -> 0 as t -> 0 in both regimes: the model "
                 "carries no location shift.")

    # (vi) the MODE pipeline itself: `density_stats` on samples whose true mode
    #      is known, at the production statistics.  Checks that the KDE mode is
    #      unbiased and gives the resolution that limits the measurement.
    zg = np.arange(-8., 8.0001, 0.005)
    lines.append("")
    lines.append("(vi) the mode estimator (`density_stats`) at production "
                 "statistics (3.2e5 draws), on samples with a KNOWN mode")
    lines.append(f"    {'sample':>10} {'h':>5} {'mode_data':>12} {'+-':>9} "
                 f"{'mode_model':>12} {'data-model':>12}")
    zn = rng.standard_normal(320000)
    pn = np.exp(-0.5 * TG ** 2)
    for h in (0.1, 0.2, 0.3):
        st = density_stats(zn, pn, zg, h, 5.0, 120, rng)
        ok &= abs(st["mode_data"] - st["mode_model"]) < 4. * st["mode_data_err"]
        lines.append(f"    {'gauss':>10} {h:5.1f} {st['mode_data']:>+12.4f} "
                     f"{st['mode_data_err']:>9.4f} {st['mode_model']:>+12.4f} "
                     f"{st['mode_data']-st['mode_model']:>+12.4f}")
    zt2, pt2 = _toy_gauss_cpoisson(rng, 320000)
    Kc, Ks = invert_matrices(zg)
    ptrue = Kc @ pt2.real + Ks @ pt2.imag
    lines.append(f"    skewed toy: true (unsmoothed) mode "
                 f"{parab_mode(zg, ptrue, 0.02):+.4f}, true mean 0")
    for h in (0.1, 0.2, 0.3):
        st = density_stats(zt2, pt2, zg, h, 5.0, 120, rng)
        ok &= abs(st["mode_data"] - st["mode_model"]) < 4. * st["mode_data_err"]
        lines.append(f"    {'skew toy':>10} {h:5.1f} {st['mode_data']:>+12.4f} "
                     f"{st['mode_data_err']:>9.4f} {st['mode_model']:>+12.4f} "
                     f"{st['mode_data']-st['mode_model']:>+12.4f}")
    lines.append("    -> unbiased; the +- column is the mode RESOLUTION at "
                 "3.2e5 tracks and is what limits the measurement.")
    lines.append("")
    lines.append("VALIDATION " + ("PASSED" if ok else "FAILED"))
    return ok, lines


# ------------------------------------------------------------------- DRIVER
def load(cache, ptfrom=None, max_tracks=0, seed=1234,
         charge_sel=0, fold_charge=False, sigma_max=0.):
    d = np.load(cache)
    keys = set(d.files)
    out = {k: d[k] for k in ("z", "sigma", "eta", "charge", "vgf", "Sms",
                             "Sio_re", "Sio_im", "Srad_re", "Srad_im")
           if k in keys}
    # provenance, carried through so the caller can report which arm it is on
    out["_rad_model"] = int(d["rad_model"]) if "rad_model" in keys else 0
    for k in ("genpt", "trackpt", "Sdel",
              # SELECTION COLUMNS (2026-09-04 censoring test).  Present on the
              # single-track caches; carried untouched so that an acceptance
              # eps(z) can be measured from the same rows the model is built
              # from.  Nothing here changes any pre-existing number.
              "normchi2", "nvalidhits", "chisqval", "ndof"):
        if k in keys:
            out[k] = d[k]
    n = len(out["z"])
    if "genpt" not in out and ptfrom:
        e = np.load(ptfrom)
        if len(e["z"]) != n or not np.array_equal(e["z"], out["z"]):
            raise SystemExit(f"{ptfrom} is not track-aligned with {cache}")
        out["genpt"] = e["genpt"]
        logger.info(f"genpt taken from the track-aligned sibling {ptfrom} "
                    f"(z arrays bit-identical)")
    # ------------------------------------------------------------- CHARGE
    # THE IONIZATION MAP CARRIES THE CHARGE, AND THE EXPORT DOES NOT.
    # `ioniurbanv` column 10 is `us.cs = E/p^3`
    # (ResidualGlobalCorrectionMakerG4e.cc, "ioniurbanv.push_back(us.cs)"),
    # which Geant4ePropagator.cc:2360 sets to `etotGeV/(pGeV*pGeV*pGeV)` --
    # POSITIVE for every track.  It maps dE -> d(q/p) for a POSITIVE charge
    # only: physically d(q/p) = q cs dE, so for q = -1 the map flips and with
    # it the sign of the ionization skew.  The in-fit CGF block applies it
    # (`const double qsign = (charge >= 0. ? 1. : -1.); s.gs = qsign * wtr *
    # cs * 1e-3;`, Geant4ePropagator.cc ~1524, with the comment that the
    # factor "is not cosmetic ... it cancels in 1/I ... omitting it would be
    # invisible in the WEIGHT").
    #
    # `cf_track_resolution.extract` did not, until 2026-09-03: the pooled
    # block weight `wstd = np.sqrt(vpool/sq2)/sig` was an unsigned square
    # root, so the cached exponent was the mu+ exponent for EVERY track.  It
    # now passes `chg * wstd` and marks the cache with the key
    # `ioni_charge_signed`.  THAT KEY IS THE SWITCH HERE: no flag, no
    # decision by the caller, one behaviour per cache.
    #
    # Repairing a pre-fix cache is exact and needs no re-extraction.  The step
    # exponent depends on the weight only through gs = wstd * g, and every
    # channel enters as a(e^{i gs E t} - 1 - i gs E t) or an integral of the
    # same form, so flipping the sign of gs is t -> -t and phi(-t) = phi(t)*:
    #
    #     Re S(-w) = Re S(+w)      Im S(-w) = -Im S(+w)      EXACTLY
    #
    # (verified to 0.000e+00 in Urban regimes 0/1/2, Kokoulin on and off).
    # Hence `Sio_im -> charge * Sio_im` is the whole correction, Re S is
    # untouched, and the EVEN closure -- <e^{-uz^2}>, k_ms, k_hit -- is
    # bit-identical either way.  A signed and an unsigned cache of the same
    # sample therefore produce IDENTICAL tables in every arm.
    signed = "ioni_charge_signed" in keys
    if charge_sel:
        if "charge" not in out:
            raise SystemExit(f"{cache} has no `charge` branch")
        msk = out["charge"] == charge_sel
        out = {k: (v[msk] if getattr(v, "shape", (0,))[:1] == (n,) else v)
               for k, v in out.items()}
        n = len(out["z"])
        logger.info(f"charge selection q = {charge_sel:+d}: {n} tracks")
    if fold_charge:
        # THE MAXIMALLY SENSITIVE ARM.  Define zhat = q z.  Then
        #   zhat > 0  <=>  momentum LOWER,  for BOTH charges,
        # so an energy-loss effect -- charge-even in momentum -- is charge-EVEN
        # in zhat and the two charges add instead of cancelling.  It is the
        # same measurement as the two charge-split arms combined, at full
        # statistics in one histogram.
        if "charge" not in out:
            raise SystemExit(f"{cache} has no `charge` branch")
        q = np.sign(out["charge"]).astype(np.float64)
        out["z"] = out["z"] * q
        logger.info("CHARGE-FOLDED: z -> q z (zhat > 0 = momentum LOWER for "
                    "both charges)")
    # THE ONE PLACE THE CHARGE FACTOR IS DECIDED.  Im S is odd in the block
    # weight, so
    #   phi_zhat(t) = phi_z(q t)  =>  Im S_zhat = q Im S_z,
    # and a pre-fix cache needs one q to become Im S_z at all.  The model for
    # the z actually stored in out["z"] therefore wants
    #
    #   q^(cache unsigned) * q^(--fold-charge)
    #
    # i.e. exactly one q when precisely one of the two holds, and none when
    # both or neither do.  (The old CLI expressed the same arithmetic as
    # "--charge-sign, and never together with --fold-charge"; making it a
    # property of the cache removes the chance of getting it wrong, and makes
    # the folded arm right on a signed cache -- there it DOES need the factor,
    # which is the one case a literal "signed cache -> never multiply" rule
    # would have broken.)
    #
    # CANDIDATE (MASS) CACHES ARE EXEMPT.  cf_mass_likelihood's caches have no
    # `charge` column because a candidate has none: an energy loss on EITHER
    # muon can only LOWER the mass, so the sign of the ionization block is -1
    # for both legs and both charges, hard-wired at build time (and marked
    # with `ioni_sign_fixed`).  There is nothing to apply here.
    if "charge" not in out:
        out["_ioni_charge_note"] = (
            "candidate cache: Im S_ioni sign fixed at build time"
            + ("" if "ioni_sign_fixed" in keys else " (PRE-FIX build: sign "
               "was sign(sum resinfv), which cancels the two legs)"))
        logger.info(out["_ioni_charge_note"])
    else:
        napply = (0 if signed else 1) + (1 if fold_charge else 0)
        why = ("cache signed at extraction (`ioni_charge_signed`)" if signed
               else "cache UNSIGNED (pre-2026-09-03)") \
            + (", zhat = q z folded" if fold_charge else "")
        if napply % 2:
            q = np.sign(out["charge"]).astype(np.float64)
            if not np.all(np.abs(q) == 1.):
                raise SystemExit("charge branch has entries that are not +-1")
            out["Sio_im"] = (out["Sio_im"].astype(np.float64) * q[:, None])
            # The RADIATIVE block rides the same q/p dof with the same charge
            # map (d(q/p) = q cs dE whatever took the energy), so its odd part
            # is odd in q for exactly the same reason and takes the same
            # factor. It exists only on caches written after 2026-09-03, which
            # are always q-signed, so in practice this fires only under
            # --fold-charge -- but leaving it out would make the folded arm's
            # radiative skew cancel instead of add.
            if "Srad_im" in out:
                out["Srad_im"] = (out["Srad_im"].astype(np.float64)
                                  * q[:, None])
            logger.info(f"IONIZATION CHARGE FACTOR applied in `load`: "
                        f"Sio_im -> q * Sio_im [{why}] "
                        f"({int((q > 0).sum())} mu+, {int((q < 0).sum())} mu-); "
                        f"Re S untouched, so the even closure is unchanged")
        else:
            logger.info(f"IONIZATION CHARGE FACTOR needs no factor in `load` "
                        f"[{why}]; Sio_im used as it stands")
        out["_ioni_charge_note"] = (
            "Im S_ioni model for " + ("zhat = q z (zhat > 0 = momentum LOWER "
                                      "for both charges)" if fold_charge
                                      else "z")
            + "; cache " + ("q-signed" if signed else "unsigned")
            + ", q " + ("applied in `load`" if napply % 2
                        else "needs no factor in `load`"))
    if sigma_max and sigma_max > 0.:
        msk = out["sigma"] < sigma_max
        nd = int((~msk).sum())
        out = {k: (v[msk] if getattr(v, "shape", (0,))[:1] == (n,) else v)
               for k, v in out.items()}
        n = len(out["z"])
        logger.info(f"sigma < {sigma_max}: dropped {nd} entries "
                    f"({100.*nd/(n+nd):.2f} %), {n} left")
    if max_tracks and n > max_tracks:
        # SHUFFLE FIRST.  The first N rows are the first FILES, not a random
        # draw -- a recurring pitfall in this directory.
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(n, max_tracks, replace=False))
        out = {k: (v[sel] if getattr(v, "shape", (0,))[:1] == (n,) else v)
               for k, v in out.items()}
        logger.info(f"random subsample {max_tracks} of {n} tracks (seed {seed})")
    return out


def model_scan(d, args, probes, kskew=1.0, binid=None, nbin=0, chunk=20000,
               dshift=0.0):
    """Per-track model <e^{-uz^2}> and <z e^{-uz^2}>, plus the per-bin mean phi.

    `dshift` translates the model density by delta (phi -> phi e^{i t delta}),
    which is the LOCATION-SHIFT alternative to the skew scale `kskew`.
    Chunked over tracks: phi is (chunk x 448) complex."""
    shf = np.exp(1j * TG * dshift) if dshift else None
    n = len(d["z"])
    Em = np.empty((n, len(probes)))
    Om = np.empty((n, len(probes)))
    phis = np.zeros((max(nbin, 1), len(TG)), dtype=np.complex128)
    cnt = np.zeros(max(nbin, 1))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sub = {"vgf": d["vgf"][lo:hi], "Sms": d["Sms"][lo:hi],
               "Sio_re": d["Sio_re"][lo:hi], "Sio_im": d["Sio_im"][lo:hi]}
        if "Srad_re" in d:
            sub["Srad_re"] = d["Srad_re"][lo:hi]
            sub["Srad_im"] = d["Srad_im"][lo:hi]
        phi = skew_phi(m.model_phi(sub, args), sub["Sio_im"], kskew, args.kioni)
        if shf is not None:
            phi = phi * shf[None, :]
        for iu, u in enumerate(probes):
            Em[lo:hi, iu] = m.weier(phi, u)
            Om[lo:hi, iu] = weier_odd(phi, u)
        if nbin:
            b = binid[lo:hi]
            for k in range(nbin):
                msk = b == k
                if msk.any():
                    phis[k] += phi[msk].sum(axis=0)
                    cnt[k] += msk.sum()
    if nbin:
        phis /= np.maximum(cnt, 1)[:, None]
    return Em, Om, phis, cnt


def boot(s, nboot, rng, chunk=25):
    """Bootstrap standard error of the sample mean of s."""
    n = len(s)
    out = np.empty(nboot)
    for lo in range(0, nboot, chunk):
        hi = min(lo + chunk, nboot)
        idx = rng.integers(0, n, size=(hi - lo, n))
        out[lo:hi] = s[idx].mean(axis=1)
    return out.std(ddof=1)


def make_bins(d, mode, nbin):
    """(binid, labels, edges) for 'sigma' / 'pt' quantiles or '|eta|' bands."""
    if mode == "abseta":
        # FIXED edges, the Z analysis's own leading-muon bands, so a track-level
        # number can be put next to a mass-level one without re-binning either
        v = np.abs(d["eta"])
        e = np.array([0.0, 0.9, 1.6, 2.4]) if nbin == 3 else \
            np.quantile(v, np.linspace(0., 1., nbin + 1))
        lab = [rf"$|\eta|\in[{e[i]:.1f},{e[i+1]:.1f}]$" for i in range(len(e) - 1)]
        b = np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2)
        return b, lab, e
    if mode == "pt":
        v = d["genpt"]
        e = np.quantile(v, np.linspace(0., 1., nbin + 1))
        lab = [f"$p_T$ {e[i]:.1f}-{e[i+1]:.1f} GeV" for i in range(nbin)]
    else:
        v = d["sigma"]
        e = np.quantile(v, np.linspace(0., 1., nbin + 1))
        lab = [rf"$\sigma\in[{1e4*e[i]:.2f},{1e4*e[i+1]:.2f}]\times10^{{-4}}$"
               for i in range(nbin)]
        # the top bin's upper edge is a single outlier track; quoting it makes
        # the legend unreadable and says nothing about the bin
        lab[-1] = rf"$\sigma>{1e4*e[-2]:.2f}\times10^{{-4}}$"
    b = np.clip(np.digitize(v, e[1:-1]), 0, nbin - 1)
    return b, lab, e


def odd_weights(probes, tg=TG):
    """(nt, nu) matrix W with weier_odd(phi, u) = (Im phi) @ W[:, iu]."""
    tw = np.gradient(tg)
    tw[0] *= 0.5
    tw[-1] *= 0.5
    W = np.stack([(tg / (2. * u)) * np.exp(-tg ** 2 / (4. * u))
                  / np.sqrt(np.pi * u) for u in probes], axis=1)
    return W * tw[:, None]


def kscan(d, args, probes, kgrid, cellid, ncell, chunk=20000):
    """<z e^{-uz^2}> per cell for a whole grid of skew scales k, cheaply.

    Writing the exponent as S = ReS + i k B with

        ReS = S_hit + e^{k_ms} S_ms + e^{k_ioni} Re S_ioni ,
        B   = e^{k_ioni} Im S_ioni ,

    the odd statistic is LINEAR in Im phi = e^{ReS} sin(k B), so a whole k grid
    costs one sin() per k instead of a complex exp of the full exponent -- and
    the cell means can be accumulated directly.  Identical to
    `model_scan(..., kskew=k)` at every k (checked in the k=1 column)."""
    W = odd_weights(probes)
    n = len(d["z"])
    out = np.zeros((ncell, len(kgrid), len(probes)))
    cnt = np.zeros(ncell)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        vg = np.exp(args.khit) * d["vgf"][lo:hi][:, None]
        ReS = (-0.5 * vg * TG[None, :] ** 2
               + np.exp(args.kms) * d["Sms"][lo:hi]
               + np.exp(args.kioni) * d["Sio_re"][lo:hi])
        B = np.exp(args.kioni) * d["Sio_im"][lo:hi].astype(np.float64)
        # the radiative block, if the cache has it: k_rad is FIXED (it is not
        # what is being scanned), so it enters ReS as a factor and the phase
        # as an OFFSET -- Im phi = e^{ReS} sin(k B + B_rad). Writing it as an
        # offset rather than folding it into B is what keeps `kskew` the
        # ionization lever arm alone.
        Brad = 0.
        if getattr(args, "krad", 1.0) and "Srad_re" in d:
            ReS = ReS + args.krad * d["Srad_re"][lo:hi]
            Brad = args.krad * d["Srad_im"][lo:hi].astype(np.float64)
        A = np.exp(ReS)
        cid = cellid[lo:hi]
        cnt += np.bincount(cid, minlength=ncell)
        for ik, kk in enumerate(kgrid):
            vals = (A * np.sin(kk * B + Brad)) @ W
            for iu in range(vals.shape[1]):
                out[:, ik, iu] += np.bincount(cid, weights=vals[:, iu],
                                              minlength=ncell)
    return out / np.maximum(cnt, 1)[:, None, None]


def boot_stats(s, nboot, rng, chunk=10):
    """Bootstrap mean, standard error and COVARIANCE across the probes.

    s is (n, nu).  The probes share the tracks, so their errors are strongly
    correlated and a diagonal chi^2 over u would be wrong; the same resampling
    indices are used for every u so the covariance comes out right."""
    n, nu = s.shape
    reps = np.empty((nboot, nu))
    for lo in range(0, nboot, chunk):
        hi = min(lo + chunk, nboot)
        idx = rng.integers(0, n, size=(hi - lo, n))
        reps[lo:hi] = s[idx].mean(axis=1)
    return s.mean(axis=0), reps.std(axis=0, ddof=1), np.cov(reps, rowvar=False)


def kskew_chi2(dat, cov, mod_k):
    """chi^2(k) and its parabolic minimum from a grid of model predictions.

    mod_k is (nk, nu).  Returns (khat, klo, khi, chi2min, chi2_at_1)."""
    C = np.atleast_2d(cov)
    Ci = np.linalg.pinv(C)
    r = dat[None, :] - mod_k
    c2 = np.einsum("ki,ij,kj->k", r, Ci, r)
    return c2


def solve_k(kgrid, c2):
    i = int(np.argmin(c2))
    kh = kgrid[i]
    if 0 < i < len(kgrid) - 1:
        y0, y1, y2 = c2[i - 1], c2[i], c2[i + 1]
        den = y0 - 2. * y1 + y2
        if den > 0.:
            h = 0.5 * (kgrid[i + 1] - kgrid[i - 1])
            kh = kgrid[i] - 0.5 * h * (y2 - y0) / den
    # Delta chi^2 = 1 interval by linear interpolation on the grid
    lo = hi = np.nan
    tgt = c2.min() + 1.0
    for j in range(i, 0, -1):
        if c2[j - 1] >= tgt:
            lo = np.interp(tgt, [c2[j], c2[j - 1]], [kgrid[j], kgrid[j - 1]])
            break
    for j in range(i, len(kgrid) - 1):
        if c2[j + 1] >= tgt:
            hi = np.interp(tgt, [c2[j], c2[j + 1]], [kgrid[j], kgrid[j + 1]])
            break
    return kh, lo, hi, c2.min()


def density_stats(zdat, phibar, zg, hbw, trim=5.0, nboot=200, rng=None):
    """Mode and trimmed mean of the data and of the model density.

    The SAME Gaussian smoothing is applied to both so the two modes are the
    same estimator; the model's unsmoothed mode is reported as well."""
    dz = zg[1] - zg[0]
    Kc, Ks = invert_matrices(zg)
    pm = Kc @ phibar.real + Ks @ phibar.imag
    pms = smooth(pm, dz, hbw)
    edges = np.concatenate((zg - dz / 2., [zg[-1] + dz / 2.]))
    cnt, _ = np.histogram(zdat, bins=edges)
    nin = cnt.sum()
    pd = cnt / (nin * dz)
    pds = smooth(pd, dz, hbw)
    # TWO peak estimators, because on a skewed density they are not the same
    # thing: `narrow` is the argmax of the KDE with a sub-grid parabolic
    # refinement (the mode of the smoothed density), `wide` is a parabolic
    # PEAK FIT over +-2.5h (less noisy, but a symmetric parabola fitted to an
    # asymmetric shoulder is pulled toward the heavy side).  Both are applied
    # identically to data and model, so either is a valid comparison; they are
    # reported side by side so the estimator dependence is visible.
    narrow, half = 4. * (zg[1] - zg[0]), max(2.5 * hbw, 0.12)
    md_d = parab_mode(zg, pds, narrow)
    md_m = parab_mode(zg, pms, narrow)
    md_m_raw = parab_mode(zg, pm, narrow)
    wd_d = parab_mode(zg, pds, half)
    wd_m = parab_mode(zg, pms, half)
    # trimmed means
    t = np.abs(zdat) < trim
    mn_d = zdat[t].mean()
    sel = np.abs(zg) < trim
    mn_m = np.trapezoid(zg[sel] * pm[sel], zg[sel]) / np.trapezoid(pm[sel], zg[sel])
    # bootstrap the data mode by resampling the histogram (multinomial)
    rng = rng or np.random.default_rng(7)
    p = cnt / nin
    reps = np.empty((max(nboot, 2), 2))
    reps[:] = np.nan
    for b in range(nboot):
        c = rng.multinomial(nin, p)
        ps = smooth(c / (nin * dz), dz, hbw)
        reps[b] = (parab_mode(zg, ps, narrow), parab_mode(zg, ps, half))
    return dict(mode_data=md_d, mode_data_err=reps[:, 0].std(ddof=1),
                mode_model=md_m, mode_model_raw=md_m_raw,
                wmode_data=wd_d, wmode_data_err=reps[:, 1].std(ddof=1),
                wmode_model=wd_m,
                mean_data=mn_d, mean_data_err=zdat[t].std() / np.sqrt(t.sum()),
                mean_model=mn_m, p_model=pm, p_model_s=pms, p_data_s=pds,
                p_data=pd, ntrim=int(t.sum()), n=len(zdat))


def trunc_stats(zg, p, trims=(1., 2., 3., 5.)):
    """Truncated means and the median of a density sampled on zg."""
    out = []
    for T in trims:
        sel = np.abs(zg) < T
        out.append(np.trapezoid(zg[sel] * p[sel], zg[sel])
                   / np.trapezoid(p[sel], zg[sel]))
    c = np.concatenate(([0.], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(zg))))
    c /= c[-1]
    return out, float(np.interp(0.5, c, zg))


def accept_eps(z, passing, zg, hbw=0.15, zclip=8.0):
    """Empirical selection efficiency eps(z) on the model's z grid.

    eps(z) = N_pass(z) / N_all(z), both Gaussian-smoothed with the SAME kernel
    (so the ratio is not a ratio of two different estimators), evaluated where
    the denominator is non-empty and held at its edge value outside.  This is a
    MEASUREMENT of the cut's z dependence, not a model of it: it is exactly as
    good as the statistics in the tail, which is where it matters and where it
    is worst.  It is a DIAGNOSTIC -- a first-principles treatment needs a model
    of the fit's chi2 response to a large energy loss, not the data's own
    empirical ratio (that ratio already contains whatever the data did)."""
    dz = zg[1] - zg[0]
    edges = np.concatenate((zg - dz / 2., [zg[-1] + dz / 2.]))
    zc = np.clip(z, zg[0], zg[-1])
    na, _ = np.histogram(zc, bins=edges)
    np_, _ = np.histogram(zc[passing], bins=edges)
    nas = smooth(na.astype(float), dz, hbw)
    nps = smooth(np_.astype(float), dz, hbw)
    eps = np.ones_like(zg)
    ok = nas > 1e-9
    eps[ok] = np.clip(nps[ok] / nas[ok], 0., 1.)
    if ok.any():
        i0, i1 = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])
        eps[:i0] = eps[i0]
        eps[i1 + 1:] = eps[i1]
    return eps, na, np_


def moments_of_density(zg, p, probes, trims=(1., 2., 3., 5.)):
    """<e^{-uz^2}>, <z e^{-uz^2}> and the trimmed means of a density on zg."""
    nrm = np.trapezoid(p, zg)
    ev = [np.trapezoid(np.exp(-u * zg ** 2) * p, zg) / nrm for u in probes]
    od = [np.trapezoid(zg * np.exp(-u * zg ** 2) * p, zg) / nrm for u in probes]
    tm, med = trunc_stats(zg, p, trims)
    return np.array(ev), np.array(od), np.array(tm), med


def accept_block(d, args, zg, phi_incl, L, log):
    """TASK 3: multiply the model z-density by an empirical acceptance eps(z)
    and renormalise, then compare the odd moment against the data WITH the
    same cut applied.

    Read it as a bound, not as a correction: the cut whose eps(z) is measured
    here IS NOT APPLIED anywhere in the production chain (the makers have no
    chi2 or hit-count cut; see NOTES.md 2026-09-04), so the nominal sample's
    eps is identically 1 and the corrected model is the model.  What the table
    measures is how much a hypothetical selection of this shape WOULD move the
    closure -- i.e. how large a hidden cut would have to be to matter."""
    var = args.accept_var
    if var == "none":
        return
    if var not in d:
        L.append(f"## 6. acceptance eps(z): SKIPPED, the cache has no `{var}`")
        return
    v = np.asarray(d[var], dtype=np.float64)
    z = d["z"]
    Kc, Ks = invert_matrices(zg)
    pm = Kc @ phi_incl.real + Ks @ phi_incl.imag
    probes = args.probes
    L.append("## 6. SELECTION ACCEPTANCE eps(z) (DIAGNOSTIC).  eps(z) is "
             f"measured from the data as the fraction of tracks passing a cut "
             f"on `{var}`, and the model density is multiplied by it and "
             "renormalised.  NOTE: no such cut is applied in the production "
             "chain, so this bounds a hypothetical censoring rather than "
             "correcting a real one.")
    head = (f"{'cut':<18}{'n pass':>9}{'frac':>8}"
            + "".join(f"{'<ze^-'+format(u,'g')+'z2> dat':>17}"
                      f"{'mod':>10}{'mod*eps':>10}{'d-m':>9}{'d-m*eps':>9}"
                      for u in probes))
    L.append(head)
    for cut in args.accept_cuts:
        if var == "nvalidhits":
            sel = v >= cut
            lab = f"{var} >= {cut:g}"
        else:
            sel = v < cut
            lab = f"{var} < {cut:g}"
        if sel.sum() < 100:
            continue
        eps, _, _ = accept_eps(z, sel, zg, args.accept_hbw)
        pe = pm * eps
        _, od_m, _, _ = moments_of_density(zg, pm, probes)
        _, od_e, _, _ = moments_of_density(zg, pe, probes)
        row = f"{lab:<18}{int(sel.sum()):>9}{sel.mean():>8.4f}"
        for iu, u in enumerate(probes):
            dat = float(np.mean(z[sel] * np.exp(-u * z[sel] ** 2)))
            row += (f"{dat:>+17.5f}{od_m[iu]:>+10.5f}{od_e[iu]:>+10.5f}"
                    f"{dat-od_m[iu]:>+9.5f}{dat-od_e[iu]:>+9.5f}")
        L.append(row)
    # the uncut reference
    _, od_m, tm_m, med_m = moments_of_density(zg, pm, probes)
    row = f"{'(no cut)':<18}{len(z):>9}{1.0:>8.4f}"
    for iu, u in enumerate(probes):
        dat = float(np.mean(z * np.exp(-u * z ** 2)))
        row += (f"{dat:>+17.5f}{od_m[iu]:>+10.5f}{od_m[iu]:>+10.5f}"
                f"{dat-od_m[iu]:>+9.5f}{dat-od_m[iu]:>+9.5f}")
    L.append(row)
    L.append("")

    # ---- the trim scan under the same cuts ----------------------------
    TR = (1., 2., 3., 5.)
    L.append("   trim scan of <z> under the same cuts (data with the cut "
             "APPLIED against model x eps):")
    L.append(f"   {'cut':<18}" + "".join(
        f"{'|z|<'+format(T,'g'):>26}" for T in TR))
    L.append("   " + " " * 18 + "".join(
        f"{'data':>9}{'model':>9}{'mod*eps':>8}" for _ in TR))
    for cut in list(args.accept_cuts) + [None]:
        if cut is None:
            sel = np.ones(len(z), bool); lab = "(no cut)"; eps = np.ones_like(zg)
        elif var == "nvalidhits":
            sel = v >= cut; lab = f"{var} >= {cut:g}"
        else:
            sel = v < cut; lab = f"{var} < {cut:g}"
        if sel.sum() < 100:
            continue
        if cut is not None:
            eps, _, _ = accept_eps(z, sel, zg, args.accept_hbw)
        _, _, tm_m, _ = moments_of_density(zg, pm, probes, TR)
        _, _, tm_e, _ = moments_of_density(zg, pm * eps, probes, TR)
        row = f"   {lab:<18}"
        zz = z[sel]
        for it, T in enumerate(TR):
            t = np.abs(zz) < T
            row += (f"{zz[t].mean():>+9.4f}{tm_m[it]:>+9.4f}{tm_e[it]:>+8.4f}")
        L.append(row)
    L.append("")
    return


def analyse(args, outdir):
    rng = np.random.default_rng(args.seed)
    d = load(args.cache, args.ptfrom, args.max_tracks, args.seed,
             args.charge, args.fold_charge, args.sigma_max)
    n = len(d["z"])
    z = d["z"]
    a = A()
    a.khit, a.kms, a.kioni, a.hitmode = args.khit, args.kms, args.kioni, args.hitmode
    a.krad = args.krad if "Srad_re" in d else 0.0
    label = args.label or os.path.basename(args.cache)
    L = [f"# ODD-MOMENT CLOSURE   <z e^(-u z^2)>", f"# sample : {label}",
         f"# cache  : {args.cache}",
         f"# tracks : {n}" + (f"  (random subsample, seed {args.seed})"
                              if args.max_tracks and args.max_tracks < n else ""),
         f"# model  : hitmode={a.hitmode} khit={a.khit} kms={a.kms} "
         f"kioni={a.kioni} krad={a.krad}"
         + (f"   (cache rad_model={d['_rad_model']})" if "Srad_re" in d
            else "   (cache has NO radiative block)"),
         f"# charge : " + d["_ioni_charge_note"]
         + (f"   |   selection q = {args.charge:+d}" if args.charge else
            "   |   both charges"),
         "# z = (q/p_fit - q/p_gen)/sigma.  q/p CARRIES THE CHARGE, so",
         "#   q = +1:  z > 0  <=>  1/p_fit > 1/p_gen  <=>  momentum LOWER",
         "#   q = -1:  z > 0  <=> -1/p_fit > -1/p_gen <=>  momentum HIGHER",
         "# A momentum bias is therefore charge-EVEN in p and charge-ODD in z,",
         "# and cancels in the charge-averaged z distribution.",
         ""]

    havept = "genpt" in d
    bineta = bool(getattr(args, "bin_eta", False))
    binnings = []
    if bineta:
        binnings.append(("|eta|", *make_bins(d, "abseta", args.nbins)))
    elif havept:
        binnings.append(("pt", *make_bins(d, "pt", args.nbins)))
    binnings.append(("sigma", *make_bins(d, "sigma", args.nbins)))
    outer = bineta or havept
    ptb, ptlab = (binnings[0][1], binnings[0][2]) if outer else (None, None)

    # ---- model, ONCE, on the OUTER PRODUCT of the two binnings, so that the
    # per-bin mean phi of either binning is a count-weighted sum of cells.
    # <z e^{-uz^2}> is LINEAR in phi, so a bin's model prediction can be read
    # off the bin-mean phi without touching the tracks again -- which is what
    # makes the location-shift scan (2b) free.
    b0, lab0, edg0 = binnings[0][1], binnings[0][2], binnings[0][3]
    nb = args.nbins
    if len(binnings) > 1:
        cb, ncb = b0 * nb + binnings[1][1], nb * nb
        CELLS = {(binnings[0][0], k): np.arange(k * nb, (k + 1) * nb)
                 for k in range(nb)}
        CELLS.update({(binnings[1][0], j): np.arange(j, nb * nb, nb)
                      for j in range(nb)})
    else:
        cb, ncb = b0, nb
        CELLS = {(binnings[0][0], k): np.array([k]) for k in range(nb)}
    for name, _, _, _ in binnings:
        CELLS[(name, "all")] = np.arange(ncb)
    Em, Om, phic, cntc = model_scan(d, a, args.probes, 1.0, cb, ncb)
    PSUM = phic * cntc[:, None]

    def phibar(key):
        c = CELLS[key]
        return PSUM[c].sum(axis=0) / max(cntc[c].sum(), 1)

    phib = np.array([phibar((binnings[0][0], k)) for k in range(nb)])
    phi_incl = phibar((binnings[0][0], "all"))

    # ---- 0. what the bins are, and what one unit of z is worth in momentum
    L.append("## 0. bin inventory.  sigma_rel = sigma(q/p) * p is the "
             "FRACTIONAL momentum resolution, so a pull shift of dz sigma is a "
             "relative momentum shift of dz * sigma_rel.")
    L.append(f"{'bin':<30}{'n':>8}{'<pT>':>9}{'<|eta|>':>9}"
             f"{'<sigma>':>11}{'<sigma_rel>':>13}{'1e-3 rel = dz':>15}")
    pmom = (d["genpt"] * np.cosh(d["eta"])) if havept else None
    for name, bid, labs, edg in binnings[:1]:
        for k in list(range(len(labs))) + ["all"]:
            msk = np.ones(n, bool) if k == "all" else (bid == k)
            lb = "TOTAL" if k == "all" else labs[k]
            if havept:
                srel = (d["sigma"][msk] * pmom[msk]).mean()
            elif args.mass:
                # mass cache: `eta` holds m_gen, so sigma_rel = sigma_m/m
                srel = (d["sigma"][msk] / d["eta"][msk]).mean()
            else:
                srel = np.nan
            L.append(f"{_plain(lb):<30}{msk.sum():>8}"
                     f"{(d['genpt'][msk].mean() if havept else np.nan):>9.1f}"
                     f"{np.abs(d['eta'][msk]).mean():>9.2f}"
                     f"{d['sigma'][msk].mean():>11.3e}{srel:>13.5f}"
                     f"{1e-3/srel:>15.4f}")
    L.append("")

    # ---- 1. the odd closure, per bin and probe
    Sd = np.stack([z * np.exp(-u * z ** 2) for u in args.probes], axis=1)
    Ed = np.stack([np.exp(-u * z ** 2) for u in args.probes], axis=1)
    rows = {}
    for name, bid, labs, edg in binnings:
        L.append(f"## 1. <z e^(-u z^2)>: DATA - MODEL, per {name} bin "
                 f"(bootstrap {args.nboot}); bound |.| <= 1/sqrt(2 e u)")
        hdr = f"{'bin':<34}{'n':>8}" + "".join(
            f"{'u='+format(u,'g'):>28}" for u in args.probes)
        L.append(hdr)
        L.append(" " * 42 + "".join(f"{'data':>10}{'model':>9}{'d-m':>9}"
                                    for _ in args.probes))
        for k in range(len(labs)):
            msk = bid == k
            dm, de, cv = boot_stats(Sd[msk], args.nboot, rng)
            mm = Om[msk].mean(axis=0)
            rows[(name, k)] = dict(dat=dm, err=de, cov=cv, mod=mm,
                                   n=int(msk.sum()), lab=labs[k])
            s = f"{_plain(labs[k]):<34}{msk.sum():>8}"
            for iu in range(len(args.probes)):
                s += f"{dm[iu]:>+10.5f}{mm[iu]:>+9.5f}{dm[iu]-mm[iu]:>+9.4f}"
            L.append(s)
            L.append(" " * 42 + "".join(f"{'+-'+format(de[iu],'.5f'):>10}{'':>9}"
                                        f"{'':>7}" for iu in range(len(args.probes))))
        dm, de, cv = boot_stats(Sd, args.nboot, rng)
        mm = Om.mean(axis=0)
        rows[(name, "all")] = dict(dat=dm, err=de, cov=cv, mod=mm, n=n,
                                   lab="TOTAL")
        s = f"{'TOTAL':<34}{n:>8}"
        for iu in range(len(args.probes)):
            s += f"{dm[iu]:>+10.5f}{mm[iu]:>+9.5f}{dm[iu]-mm[iu]:>+9.4f}"
        L.append(s)
        L.append(" " * 42 + "".join(f"{'+-'+format(de[iu],'.5f'):>10}{'':>9}{'':>9}"
                                    for iu in range(len(args.probes))))
        L.append("")

    # the EVEN closure alongside, for context (this is the published statistic)
    L.append("## 1b. the EVEN statistic <e^(-u z^2)> on the same tracks "
             "(the published closure), TOTAL only")
    L.append(f"{'':<34}" + "".join(f"{'u='+format(u,'g'):>16}" for u in args.probes))
    L.append(f"{'data - model':<34}" + "".join(
        f"{Ed[:,iu].mean()-Em[:,iu].mean():>+16.5f}" for iu in range(len(args.probes))))
    L.append("")

    # ---- 2. a single scale on Im S_ioni
    L.append("## 2. one scale k on the IMAGINARY ionization exponent "
             "(Im S_ioni -> k Im S_ioni, Re untouched); k = 1 is the model")
    kg0 = np.linspace(args.kmin, args.kmax, args.nk)
    OK = kscan(d, a, args.probes, kg0, cb, ncb)      # (ncell, nk, nu)
    NC = cntc
    # O(k) is analytic and, over a 0.1-wide step, very nearly linear, so the
    # chi2 minimisation runs on a spline of the scan rather than on the scan
    # grid itself: the Delta chi2 = 1 interval can be an order of magnitude
    # narrower than the scan step (it is ~0.03 in the low-pT bins) and a
    # grid-interpolated interval would be an artefact of the step.
    kg = np.linspace(args.kmin, args.kmax, 1 + int(round(
        (args.kmax - args.kmin) / 0.002)))
    modk = {}
    for key in rows:
        c = CELLS[key]
        coarse = ((OK[c] * NC[c][:, None, None]).sum(axis=0)
                  / max(NC[c].sum(), 1))
        modk[key] = CubicSpline(kg0, coarse, axis=0)(kg)
    L.append(f"{'bin':<34}{'k_hat':>10}{'-1sig':>9}{'+1sig':>9}"
             f"{'chi2min/ndf':>14}{'chi2(k=1)':>12}")
    kfit = {}
    for key, r in rows.items():
        c2 = kskew_chi2(r["dat"], r["cov"], modk[key])
        kh, klo, khi, c2m = solve_k(kg, c2)
        c21 = float(kskew_chi2(r["dat"], r["cov"],
                               modk[key][np.argmin(np.abs(kg - 1.0))][None, :])[0])
        kfit[key] = (kh, klo, khi)
        L.append(f"{key[0]+' '+_plain(r['lab']):<34}{kh:>10.2f}{klo:>9.2f}"
                 f"{khi:>9.2f}{c2m/max(len(args.probes)-1,1):>14.2f}{c21:>12.1f}")
    L.append("")

    # ---- 2b. the alternative hypothesis: a pure LOCATION SHIFT
    L.append("## 2b. the ALTERNATIVE one-parameter hypothesis: the model "
             "density translated by delta (phi -> phi e^{i t delta}), no shape "
             "change.  Same probes, same covariance, so chi2min is directly "
             "comparable with section 2.")
    dg = np.linspace(-0.04, 0.04, 801)
    modd = {}
    for key in rows:
        pb = phibar(key)
        modd[key] = np.array([[weier_odd(pb[None, :] * np.exp(1j * TG * dl), u)[0]
                               for u in args.probes] for dl in dg])
    L.append(f"{'bin':<34}{'delta_hat':>11}{'-1sig':>9}{'+1sig':>9}"
             f"{'chi2min/ndf':>13}{'chi2(d=0)':>11}{'chi2min(k)':>12}"
             f"{'k_hat':>8}")
    for key, r in rows.items():
        c2 = kskew_chi2(r["dat"], r["cov"], modd[key])
        dh, dlo, dhi, c2m = solve_k(dg, c2)
        c20 = float(kskew_chi2(r["dat"], r["cov"],
                               modd[key][np.argmin(np.abs(dg))][None, :])[0])
        c2k = kskew_chi2(r["dat"], r["cov"], modk[key]).min()
        L.append(f"{key[0]+' '+_plain(r['lab']):<34}{dh:>+11.4f}{dlo:>+9.4f}"
                 f"{dhi:>+9.4f}{c2m/max(len(args.probes)-1,1):>13.2f}"
                 f"{c20:>11.1f}{c2k:>12.2f}{kfit[key][0]:>8.2f}")
    L.append("   delta is in units of sigma_pred, SAME sign convention as z: "
             "delta > 0 means the model must be moved toward LOWER fitted "
             "momentum to match the data.")
    L.append("")

    # ---- 3. mode vs mean
    L.append("## 3. pull density: MODE and TRIMMED MEAN (|z| < 5), in units of "
             "sigma_pred")
    zg = np.arange(-args.zmax, args.zmax + 1e-9, args.dz)
    dens = {}
    order = ([(k, ptlab[k]) for k in range(args.nbins)] if outer
             else [(k, binnings[0][2][k]) for k in range(args.nbins)])
    for ih, hb in enumerate(args.hbw):
        L.append(f"   -- KDE bandwidth h = {hb} --")
        L.append(f"{'bin':<30}{'n':>8}{'mode_dat':>10}{'+-':>8}{'mode_mod':>10}"
                 f"{'mode_mod_raw':>13}{'wmode_dat':>11}{'+-':>8}{'wmode_mod':>11}"
                 f"{'mean_dat':>10}{'+-':>8}{'mean_mod':>10}"
                 f"{'(mo-me)dat':>12}{'(mo-me)mod':>12}{'d-m':>9}")
        for k, lb in order + [("all", "TOTAL")]:
            msk = np.ones(n, bool) if k == "all" else (b0 == k)
            pb = phi_incl if k == "all" else phib[k]
            st = density_stats(z[msk], pb, zg, hb, 5.0, args.nboot, rng)
            st["lab"] = lb
            if ih == args.hplot:
                dens[k] = st
            dd = st["mode_data"] - st["mean_data"]
            mm = st["mode_model"] - st["mean_model"]
            L.append(f"{_plain(lb):<30}{msk.sum():>8}{st['mode_data']:>+10.4f}"
                     f"{st['mode_data_err']:>8.4f}{st['mode_model']:>+10.4f}"
                     f"{st['mode_model_raw']:>+13.4f}"
                     f"{st['wmode_data']:>+11.4f}{st['wmode_data_err']:>8.4f}"
                     f"{st['wmode_model']:>+11.4f}"
                     f"{st['mean_data']:>+10.4f}{st['mean_data_err']:>8.4f}"
                     f"{st['mean_model']:>+10.4f}{dd:>+12.4f}{mm:>+12.4f}"
                     f"{dd-mm:>+9.4f}")
        L.append("")
    L.append("   mode_* = argmax of the KDE (sub-grid parabolic refinement); "
             "wmode_* = parabolic PEAK FIT over +-2.5h.  mode_mod/wmode_mod are "
             "the MODEL density smoothed with the same kernel, so each pair is "
             "the same estimator on both sides; mode_mod_raw is the unsmoothed "
             "model density's mode.  The (mo-me) columns use mode_*.")
    L.append("   The model's UNTRIMMED mean is 0 exactly (every ionization term "
             "is a(e^{i th} - 1 - i th)); see --validate check (v).  mean_mod is "
             "the |z|<5 TRUNCATED mean, the matched estimator to mean_dat.")
    L.append("")

    # ---- 4. where the core sits: the trim scan of <z>, and the median
    TRIMS = (1., 2., 3., 5.)
    L.append("## 4. LOCATION of the pull core: <z> as a function of the trim, "
             "and the median.  A pure SKEW moves the tight-trim mean and the "
             "wide-trim mean in OPPOSITE directions; a location SHIFT moves "
             "them together.")
    L.append(f"{'bin':<30}{'n':>8}" + "".join(
        f"{'<z>|z|<'+format(T,'g'):>27}" for T in TRIMS)
        + f"{'median':>20}")
    L.append(" " * 38 + "".join(f"{'data':>11}{'+-':>8}{'model':>8}" for _ in TRIMS)
             + f"{'data':>10}{'model':>10}")
    Kc, Ks = invert_matrices(zg)
    for k, lb in order + [("all", "TOTAL")]:
        msk = np.ones(n, bool) if k == "all" else (b0 == k)
        pb = phi_incl if k == "all" else phib[k]
        pm = Kc @ pb.real + Ks @ pb.imag
        mt, med_m = trunc_stats(zg, pm, TRIMS)
        row = f"{_plain(lb):<30}{msk.sum():>8}"
        zz = z[msk]
        for it, T in enumerate(TRIMS):
            t = np.abs(zz) < T
            row += (f"{zz[t].mean():>+11.4f}{zz[t].std()/np.sqrt(t.sum()):>8.4f}"
                    f"{mt[it]:>+8.4f}")
        row += f"{np.median(zz):>+10.4f}{med_m:>+10.4f}"
        L.append(row)
    L.append("")

    # ---- 5. the density with the ionization skew scaled to its fitted value
    L.append("## 5. the model density with Im S_ioni -> k_hat Im S_ioni "
             "(k_hat from section 2, the SAME bin's fit): does the odd part, "
             "once scaled to the data, reproduce the observed mode?")
    L.append(f"{'bin':<30}{'k_hat':>8}{'mode_mod(k)':>13}{'mean_mod(k)':>13}"
             f"{'(mo-me)mod(k)':>15}{'mode_dat':>10}{'+-':>8}{'(mo-me)dat':>12}")
    hb = args.hbw[args.hplot]
    for k, lb in order + [("all", "TOTAL")]:
        key = (binnings[0][0], k)
        if key not in kfit:
            continue
        kh = float(np.clip(kfit[key][0], args.kmin, args.kmax))
        _, _, pk, ck = model_scan(d, a, args.probes[:1], kh, b0, args.nbins)
        pbk = ((pk * ck[:, None]).sum(axis=0) / ck.sum()) if k == "all" else pk[k]
        msk = np.ones(n, bool) if k == "all" else (b0 == k)
        stk = density_stats(z[msk], pbk, zg, hb, 5.0, 0, rng)
        st = dens[k]
        L.append(f"{_plain(lb):<30}{kh:>8.2f}{stk['mode_model']:>+13.4f}"
                 f"{stk['mean_model']:>+13.4f}"
                 f"{stk['mode_model']-stk['mean_model']:>+15.4f}"
                 f"{st['mode_data']:>+10.4f}{st['mode_data_err']:>8.4f}"
                 f"{st['mode_data']-st['mean_data']:>+12.4f}")
    L.append("")

    # ---- 6. selection acceptance eps(z)  (2026-09-04 censoring test)
    accept_block(d, args, zg, phi_incl, L, logger.info)

    # ---------------------------------------------------------------- FIGURES
    uu = np.geomspace(0.03, 3.0, 15)
    Sdu = np.stack([z * np.exp(-u * z ** 2) for u in uu], axis=1)
    _, Ou, _, _ = model_scan(d, a, uu)
    fig, axs = plt.subplots(1, 2 if len(binnings) > 1 else 1,
                            figsize=(9 * (2 if len(binnings) > 1 else 1), 7),
                            squeeze=False, constrained_layout=True)
    for ia, (name, bid, labs, edg) in enumerate(binnings):
        ax = axs[0][ia]
        for k in range(len(labs)):
            msk = bid == k
            dd = Sdu[msk].mean(axis=0) - Ou[msk].mean(axis=0)
            ee = Sdu[msk].std(axis=0) / np.sqrt(msk.sum())
            ax.errorbar(uu, dd, ee, marker="o", markersize=4, lw=1.4, label=labs[k])
        dd = Sdu.mean(axis=0) - Ou.mean(axis=0)
        ee = Sdu.std(axis=0) / np.sqrt(n)
        ax.errorbar(uu, dd, ee, marker="s", markersize=5, lw=2.0, color="black",
                    label="all")
        ax.axhline(0., color="gray", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"probe $u$")
        ax.set_ylabel(r"$\langle z e^{-uz^2}\rangle_{\mathrm{data}}"
                      r"-\langle z e^{-uz^2}\rangle_{\mathrm{model}}$")
        ax.set_title(label + ("  [$p_T$ bins]" if name == "pt" else
                              r"  [$\sigma$ bins]"), fontsize="small")
        ax.legend(fontsize="x-small")
    name1 = f"skew_closure_{args.tag}"
    plot_tools.save_pdf_and_png(outdir, name1, fig)
    plt.close(fig)

    nb = len(order)
    ncol = 2
    nrow = int(np.ceil((nb + 1) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(8 * ncol, 6 * nrow),
                            squeeze=False, constrained_layout=True)
    sel = np.abs(zg) < 3.0
    zin = np.abs(zg) < 0.6
    for i, (k, lb) in enumerate(order + [("all", "TOTAL")]):
        ax = axs[i // ncol][i % ncol]
        st = dens[k]
        ax.plot(zg[sel], st["p_data_s"][sel], color="black", lw=1.6,
                label="pulls (KDE)")
        ax.plot(zg[sel], st["p_model_s"][sel], color="crimson", lw=1.6,
                label="CF-product model")
        ax.axvline(0., color="gray", lw=1)
        ax.set_xlim(-3., 3.)
        ax.set_ylim(0., 1.32 * st["p_data_s"][sel].max())
        ax.set_xlabel(r"$z=\Delta(q/p)/\sigma_{\mathrm{pred}}$")
        ax.set_ylabel("density")
        ax.set_title(f"{label}   {_plain(lb)}   "
                     rf"($h={args.hbw[args.hplot]}$)", fontsize="small")
        ax.legend(fontsize="x-small", loc="center left")
        # THE POINT OF THE FIGURE IS A 0.01-0.03 SIGMA SHIFT, which is one
        # pixel on the |z| < 3 axis.  The inset is the measurement; the outer
        # panel is only there to show that nothing else is going on.
        ins = ax.inset_axes([0.64, 0.46, 0.34, 0.34])
        # each curve normalised to its OWN peak: the inset is about WHERE the
        # peak is, and a residual width/normalisation difference between data
        # and model would otherwise hide the 0.01-sigma shift being measured.
        ins.plot(zg[zin], st["p_data_s"][zin] / st["p_data_s"][zin].max(),
                 color="black", lw=1.4)
        ins.plot(zg[zin], st["p_model_s"][zin] / st["p_model_s"][zin].max(),
                 color="crimson", lw=1.4)
        ins.axvline(st["mode_data"], color="black", ls="--", lw=1.2)
        ins.axvline(st["mode_model"], color="crimson", ls="--", lw=1.2)
        ins.axvline(0., color="gray", lw=0.8)
        ins.set_xlim(-0.6, 0.6)
        ins.set_ylim(0.982, 1.0015)
        ins.set_yticks([0.985, 0.99, 0.995, 1.0])
        ins.tick_params(labelsize="xx-small")
        ax.text(0.03, 0.97,
                f"mode  data  {st['mode_data']:+.4f} $\\pm$ "
                f"{st['mode_data_err']:.4f}\n"
                f"mode  model {st['mode_model']:+.4f}\n"
                f"mean  data  {st['mean_data']:+.4f} $\\pm$ "
                f"{st['mean_data_err']:.4f}\n"
                f"mean  model {st['mean_model']:+.4f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize="x-small", family="monospace")
    for j in range(nb + 1, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    name2 = f"skew_density_{args.tag}"
    plot_tools.save_pdf_and_png(outdir, name2, fig)
    plt.close(fig)

    with open(os.path.join(outdir, f"{name1}.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    output_tools.write_logfile(outdir, name1, args=args, wd=_HERE)
    logger.info("\n".join(L))
    logger.info(f"wrote {outdir}/{name1}.txt")
    return L


def _plain(s):
    for a, b in ((r"$", ""), (r"\sigma", "sigma"), (r"\in", " in "),
                 (r"\times", "x"), ("p_T", "pT"), ("{", ""), ("}", ""),
                 ("^", "e"), ("10e-4", "1e-4"), ("\\", "")):
        s = s.replace(a, b)
    return s.strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="runs/cf_trackres_mugun_ul16_fix.npz")
    p.add_argument("--ptfrom", default=None,
                   help="track-aligned sibling cache carrying genpt "
                        "(the _sel caches are bit-identical extensions of _fix)")
    p.add_argument("--label", default=None)
    p.add_argument("--tag", default="run")
    p.add_argument("--probes", type=float, nargs="+", default=list(PROBES))
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--bin-eta", dest="bin_eta", action="store_true",
                   help="bin in |eta| instead of pT, with the Z analysis's own "
                        "leading-muon band edges (0, 0.9, 1.6, 2.4) when "
                        "--nbins 3. The inclusive odd-moment ratio was never "
                        "checked per eta, and if the TRACK-level skew closure "
                        "is eta-dependent then the CF family model (Moliere / "
                        "Urban / radiative content against material and "
                        "E = pT cosh eta) is wrong per region -- a "
                        "first-principles fix in the model tables, not in the "
                        "likelihood.")
    p.add_argument("--nboot", type=int, default=500)
    p.add_argument("--max-tracks", type=int, default=0)
    p.add_argument("--charge", type=int, default=0, choices=[-1, 0, 1],
                   help="select one charge (0 = both)")
    p.add_argument("--sigma-max", type=float, default=0.,
                   help="drop entries with sigma >= this (0 = keep all). The "
                        "candidate MASS caches carry a per-mille tail of "
                        "pathological sigma (up to 8e5 GeV) from failed "
                        "vertex fits; those candidates have z ~ 0 by "
                        "construction and are a spike the model cannot have.")
    p.add_argument("--mass", action="store_true",
                   help="the cache is a candidate MASS cache "
                        "(cf_mass_likelihood --pairs/--pairs-tt): z = "
                        "(m_reco - m_gen)/sigma_m and the `eta` column holds "
                        "m_gen, so the fractional scale is sigma/m_gen.")
    p.add_argument("--fold-charge", action="store_true",
                   help="use zhat = q z, for which an energy-loss effect is "
                        "charge-EVEN and the two charges ADD -- the sharpest "
                        "test of the ionization skew. The model's charge "
                        "factor follows automatically (Im S_zhat = q Im S_z, "
                        "so a signed cache needs one q here and an unsigned "
                        "one needs none).")
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--khit", type=float, default=0.0)
    p.add_argument("--kms", type=float, default=0.0)
    p.add_argument("--kioni", type=float, default=0.0)
    p.add_argument("--krad", type=float, default=1.0,
                   help="LINEAR scale on the radiative (brems+pair) block; "
                        "0 is the pre-2026-09-03 model exactly. Ignored on a "
                        "cache that has no Srad arrays")
    p.add_argument("--hitmode", choices=["gauss", "class"], default="gauss")
    p.add_argument("--kmin", type=float, default=-2.0)
    p.add_argument("--kmax", type=float, default=6.0)
    p.add_argument("--nk", type=int, default=81)
    p.add_argument("--hbw", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    # ---- selection-acceptance diagnostic (2026-09-04 censoring test) ----
    p.add_argument("--accept-var", default="none",
                   choices=["none", "normchi2", "nvalidhits"],
                   help="measure eps(z) for a cut on this cached column and "
                        "apply it to the model density (section 6). DIAGNOSTIC: "
                        "no such cut is applied in the production chain.")
    p.add_argument("--accept-cuts", type=float, nargs="+",
                   default=[2., 3., 5., 10.],
                   help="cut values scanned (normchi2 < c ; nvalidhits >= c)")
    p.add_argument("--accept-hbw", type=float, default=0.15,
                   help="Gaussian bandwidth used to smooth eps(z)")
    p.add_argument("--hplot", type=int, default=1,
                   help="index into --hbw used for the density figure")
    p.add_argument("--zmax", type=float, default=8.0)
    p.add_argument("--dz", type=float, default=0.005)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--outpath", default=None)
    return p.parse_args()


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_skew/")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    if args.validate:
        ok, lines = validate()
        txt = "\n".join(lines)
        logger.info("\n" + txt)
        with open(os.path.join(outdir, "skew_validate.txt"), "w") as fh:
            fh.write(txt + "\n")
        logger.info(f"wrote {outdir}/skew_validate.txt")
        if not ok:
            raise SystemExit("validation FAILED")
        return
    analyse(args, outdir)


if __name__ == "__main__":
    main()
