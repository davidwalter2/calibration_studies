#!/usr/bin/env python3
"""The J/psi final-state-radiation kernel for the unbinned CVH mass likelihood.

The J/psi mass term of the joint card carries a ``DeltaKernel`` at ``MJPSI``:
the modelled pre-resolution mass is the PDG value with no spread.  The MC it is
fitted against radiates, so the candidate's true (post-FSR, bare) dimuon mass is
not ``MJPSI``.  This script builds the object that repairs it -- the kernel
characteristic function ``phi_K(t)`` that ``rabbit.unbinned.MassCFTerm`` takes as
``phik = (phik_t, phik_re, phik_im)`` and multiplies into the resolution CF.

For a **delta** lineshape the kernel CF is exact and additive.  With
``m_pre == M`` the post-FSR density is ``p(m') = K(m'/M)/M`` and

    phi_K(t) = Int K(r) exp(i t M (r - 1)) dr = < exp(i t dm) > ,  dm = m' - M .

(The Z channel cannot use this form -- there ``m_pre`` varies and the kernel is
multiplicative -- which is why ``zfsr_kernel.py`` carries the caveat and this
file does not.)

TWO KERNELS, exactly as for the Z (``zchannel/fsr_config.py``):

``mc``
    the sample's own, measured from the production's gen record.  ``dm`` is
    ``Jpsigen_mass - MJPSI`` per candidate, taken from a ``cf_inmaker.py pairs``
    cache (where the post-FSR gen mass is stored under the key ``eta``).  This
    is what an MC closure fit must use: it is a measurement of the sample, not
    a model of it.
``analytic``
    exponentiated exact QED (``fsr_analytic.FSRKernel``), the kernel a DATA fit
    must use.  ``--mass-exact`` replaces the massless O(alpha) spectrum by the
    exact-mass matrix element, which at the J/psi is a 1e-3 effect and at the Z
    is not (``fsr_analytic.r1_fast``).

WHAT THE MC KERNEL MUST AND MUST NOT CONTAIN
    * **not** the observed-mass window.  ``MassCFTerm(norm_window=...)``
      renormalises the density over it; applying it here too double-counts it.
    * **yes** the pairs cache's own gen-mass acceptance.  ``cf_inmaker`` cuts
      ``|m_gen - 3.0969| <= 0.35`` when it writes the cache, and the likelihood
      cannot see that cut, so the kernel it is given has to carry it.  The
      truncated kernel renormalised to one makes the model the conditional
      density given that cut, and ``Z`` then integrates the reco window
      correctly: the factorisation is exact.
    * **yes** the card's reco quality cuts (chi2/ndof, sigma/m).  Those select
      the candidates the term is built from, so the kernel is their conditional
      kernel.

DISCRETISATION
    ``phi_K`` is tabulated on a UNIFORM ``t`` grid, which is what
    ``MassCFTerm._interp_phik`` and ``_build_norm`` both require.  The grid must
    reach ``max(tgrid)/min(sigma)`` -- 7.8926 / 0.0103 = 767 1/GeV on the J/psi
    cache -- or ``_build_norm`` refuses the card.  The step follows from
    ``|phi_K''| <= E[dm^2] = 1.2e-3``: linear interpolation at ``dt = 0.02``
    costs 6e-8, four orders below the 3.6e-4 statistical noise of a 7.8 M-sample
    empirical CF.

    The empirical CF is evaluated by ONE real FFT of a fine ``dm`` histogram
    rather than by ``exp`` over 7.8 M x 50 k (`cf_from_samples`), and the result
    is checked against the direct sum over every sample at six values of ``t``
    (`validate_cf`): the measured worst discrepancy is 9e-7, i.e. 400x below the
    kernel's own statistical noise.

usage::

    # the sample's own kernel, from the J/psi v2 pairs cache
    python jpsi_fsr_kernel.py mc --pairs ../fullscale/runs/jpairs_v2_n600.npz \\
        -o data/jpsi_kern_mc.npz --report

    # exact QED, the data kernel
    python jpsi_fsr_kernel.py analytic --variant exp2nll --pair e mu \\
        --mass-exact -o data/jpsi_kern_analytic.npz --report
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MJPSI = 3.0969
#: the pairs cache's own gen-mass acceptance (``cf_inmaker.py --mass-window``)
GEN_HALFWIDTH = 0.35


# --------------------------------------------------------------------------
# the MC kernel
# --------------------------------------------------------------------------
def load_mc_dm(path, mass, max_chi2_ndof, max_sigma_rel, obs_window, log):
    """``dm = m_gen - mass`` over the card's candidates, minus the mass window.

    Returns ``(dm, info)``.  The selection is `make_card.select`'s, with the
    observed-mass window OPTIONAL (and off by default -- see the module
    docstring).
    """
    d = np.load(path, allow_pickle=True)
    z = d["z"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    mgen = d["eta"].astype(np.float64)          # `eta` IS the post-FSR gen mass
    m = z * sig + mgen
    n0 = len(z)
    keep = np.isfinite(m) & np.isfinite(sig) & (sig > 0.0)
    steps = [("finite m, sigma > 0", int(keep.sum()))]
    if max_chi2_ndof > 0 and "chisqval" in d.files and "ndof" in d.files:
        c2 = (d["chisqval"].astype(np.float64)
              / np.maximum(d["ndof"].astype(np.float64), 1.0))
        keep &= c2 < max_chi2_ndof
        steps.append((f"chi2/ndof < {max_chi2_ndof:g}", int(keep.sum())))
    if max_sigma_rel > 0:
        keep &= sig / np.maximum(np.abs(m), 1e-9) < max_sigma_rel
        steps.append((f"sigma_m/m < {max_sigma_rel:g}", int(keep.sum())))
    if obs_window > 0:
        keep &= np.abs(m - mass) <= obs_window
        steps.append((f"|m_obs - M| <= {obs_window:g}  [NOT the default]",
                      int(keep.sum())))
    log(f"  {path}")
    for name, n in steps:
        log(f"    {name:44s} {n:9d}  ({100.0 * n / n0:6.2f} %)")
    dm = mgen[keep] - mass
    return dm, {"n_cache": n0, "n": int(keep.sum()), "steps": steps}


def cf_from_samples(dm, tmax, npoints, nbins_pow=25, dt_dft=0.25, log=print):
    """Empirical ``<exp(i t dm)>`` on ``linspace(0, tmax, npoints)``, by FFT.

    Histogram ``dm`` on a uniform grid of ``N = 2^nbins_pow`` bins whose period
    is ``P = 2 pi / dt_dft``, take one real FFT, and divide out the exact
    ``sinc(t dbin/2)`` of the binning.  The DFT frequencies are
    ``t_k = 2 pi k/P = k dt_dft``; the output grid is then a cubic spline of
    that.

    THE TWO ERRORS PULL AGAINST EACH OTHER and ``dt_dft`` is where they are
    balanced.  The DFT step is ``2 pi/P``, so a fine ``t`` grid needs a LONG
    period, and at fixed ``N`` a long period means a coarse ``dm`` bin -- while
    ``dm`` has a 0.1 MeV-wide core that has to be resolved.  Splining a coarse
    ``t`` grid costs ``dt^4 E[dm^4]/384`` (5e-9 at ``dt_dft = 0.25``) and a
    coarse ``dm`` bin costs ``(t dbin)^2/24`` beyond the sinc (4e-9 at
    ``N = 2^25``, ``t = 1000``), so a short period with a spline is the cheap
    side of the trade.  `validate_cf` checks the result against the direct sum.
    """
    from scipy.interpolate import CubicSpline

    t_out = np.linspace(0.0, tmax, npoints)
    n = 1 << int(nbins_pow)
    period = 2.0 * np.pi / float(dt_dft)
    lo, hi = float(dm.min()), float(dm.max())
    if hi - lo >= period:
        raise SystemExit(f"dm spans {hi - lo:g} GeV, more than the FFT period "
                         f"{period:g}; lower --dt-dft")
    dbin = period / n
    b = np.floor((dm - lo) / dbin).astype(np.int64)
    np.clip(b, 0, n - 1, out=b)
    w = np.bincount(b, minlength=n).astype(np.float64)
    w /= w.sum()
    # sum_b w_b e^{+i t_k (lo + (b+1/2) dbin)}
    #   = conj(rfft(w)[k]) e^{i t_k (lo + dbin/2)}
    F = np.conjugate(np.fft.rfft(w))
    tk = 2.0 * np.pi * np.arange(len(F)) / period
    F *= np.exp(1j * tk * (lo + 0.5 * dbin))
    F /= np.sinc(0.5 * tk * dbin / np.pi)       # exact uniform-bin deconvolution
    if tk[-1] < tmax:
        raise SystemExit(f"the DFT reaches t = {tk[-1]:.1f} < tmax {tmax}")
    log(f"  empirical CF: {n} bins of {dbin:.3e} GeV over a period of "
        f"{period:.3f} GeV; DFT step {dt_dft}, output step "
        f"{t_out[1] - t_out[0]:.5f} 1/GeV")
    keep = tk <= tmax + 4.0 * dt_dft
    re = CubicSpline(tk[keep], F.real[keep])(t_out)
    im = CubicSpline(tk[keep], F.imag[keep])(t_out)
    return t_out, re, im


def validate_cf(dm, t, re, im, tprobe=(1.0, 10.0, 100.0, 400.0, 767.0, 1000.0),
                log=print):
    """The tabulation against the DIRECT sum over every sample, at a few ``t``."""
    log("  CF validation against the direct sum over all samples:")
    worst = 0.0
    for tp in tprobe:
        if tp > t[-1]:
            continue
        ex = np.mean(np.exp(1j * tp * dm))
        ap = complex(np.interp(tp, t, re), np.interp(tp, t, im))
        d = abs(ap - ex)
        worst = max(worst, d)
        log(f"    t = {tp:7.1f}  exact {ex.real:+.9f}{ex.imag:+.9f}i   "
            f"table - exact = {ap.real - ex.real:+.2e}{ap.imag - ex.imag:+.2e}i"
            f"   |d| = {d:.2e}")
    log(f"    worst |table - exact| = {worst:.3e}  (statistical noise on the "
        f"CF itself is 1/sqrt(N) = {1.0 / math.sqrt(len(dm)):.2e})")
    return worst


# --------------------------------------------------------------------------
# the analytic kernel
# --------------------------------------------------------------------------
def analytic_cells(mass, variant, pair, mass_exact, n_fine, ng, log):
    """Fine cells ``(w_j, ubar_j, var_j)`` of the analytic kernel in ``u``."""
    import fsr_analytic as fa

    k = fa.FSRKernel(mass, variant=variant, pair=tuple(pair),
                     mass_exact=mass_exact)
    c = k.cells(n_fine=n_fine, ng=ng)
    w, m1, m2 = c[0], c[1], c[2]
    good = w > 0
    w, m1, m2 = w[good], m1[good], m2[good]
    ub = m1 / w
    var = np.maximum(m2 / w - ub * ub, 0.0)
    tot = w.sum()
    log(f"  FSRKernel(m={mass}, variant={variant}, pair={tuple(pair)}, "
        f"mass_exact={mass_exact}): beta = {k.beta:.6f}, L = {k.L:.6f}, "
        f"pair rate = {k.pair_rate:.4e}, {len(w)} cells, captured {tot:.6f}")
    return k, w / tot, ub, var


def cf_from_cells(w, ub, var, mass, tmax, npoints, log=print, chunk=4096):
    """``phi_K(t) = Int K(u) exp(i t m (e^{-u} - 1)) du`` from the fine cells.

    Each cell contributes ``w_j <f(u)>_j`` with
    ``f(u) = exp(i t m (e^{-u} - 1))``; the conditional mean is expanded to
    second order in the cell, ``<f> = f(ubar)(1 + (1/2) (f''/f)(ubar) var)``,
    which is exact to ``O(var^2)`` and converges with ``--n-fine``.
    """
    t = np.linspace(0.0, tmax, npoints)
    g = np.expm1(-ub)                    # e^{-u} - 1, the relative mass shift
    dg = -(1.0 + g)                      # d/du
    d2g = (1.0 + g)
    out = np.zeros(npoints, np.complex128)
    for i in range(0, len(w), chunk):
        sl = slice(i, i + chunk)
        ph = mass * np.outer(t, g[sl])               # (nt, nc)
        f = np.exp(1j * ph)
        # (d^2 f/du^2)/f = i t m g'' - (t m g')^2
        corr = 1.0 + 0.5 * var[None, sl] * (
            1j * t[:, None] * mass * d2g[None, sl]
            - (t[:, None] * mass * dg[None, sl]) ** 2)
        out += (f * corr) @ w[sl]
    log(f"  analytic CF on {npoints} points to t = {tmax} 1/GeV from "
        f"{len(w)} cells")
    return t, out.real.copy(), out.imag.copy()


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
U_GRID = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.0566, 0.113, 0.2)


def report_mc(dm, mass, window, log):
    u = -np.log1p(dm / mass)
    out = {}
    log(f"\n  === MC kernel, {len(dm)} candidates, M = {mass} GeV ===")
    log(f"  <dm>      = {dm.mean() * 1e3:+.4f} MeV    "
        f"median {np.median(dm) * 1e3:+.4f} MeV")
    log(f"  <u>       = {u.mean():.6e}    rms(dm) = {dm.std() * 1e3:.3f} MeV")
    out["mean_dm"] = float(dm.mean())
    out["mean_u"] = float(u.mean())
    out["rms_dm"] = float(dm.std())
    for c in (1e-6, 1e-5, 1e-4, 1e-3):
        log(f"  P(|dm| < {c:.0e} GeV) = {np.mean(np.abs(dm) < c):.5f}")
        out[f"p_core_{c:g}"] = float(np.mean(np.abs(dm) < c))
    log("  P(u > u0):")
    for u0 in U_GRID:
        p = float(np.mean(u > u0))
        log(f"    u0 = {u0:<8g}  {p:.6f}")
        out[f"tail_{u0:g}"] = p
    qs = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
    log("  dm quantiles [MeV] "
        + "  ".join(f"{q}%:{v * 1e3:+.4f}"
                    for q, v in zip(qs, np.percentile(dm, qs))))
    if window > 0:
        m = np.abs(dm) <= window
        log(f"  in |dm| <= {window} GeV: P = {m.mean():.6f}, "
            f"<dm> = {dm[m].mean() * 1e3:+.4f} MeV "
            f"= {dm[m].mean() / mass:+.4e} relative")
        out["window"] = float(window)
        out["p_in_window"] = float(m.mean())
        out["mean_dm_window"] = float(dm[m].mean())
    return out


def report_analytic(k, w, ub, mass, window, log):
    out = {}
    u = ub
    log(f"\n  === analytic kernel, M = {mass} GeV ===")
    mom = k.moments()
    log(f"  beta = {k.beta:.6f}   <u> (moments) = {mom['u']:.6e}   "
        f"<1-z> = {mom['x']:.6e}")
    out["beta"] = float(k.beta)
    out["mean_u"] = float(mom["u"])
    dm = mass * np.expm1(-u)
    log(f"  <dm> = {float(np.sum(w * dm)) * 1e3:+.4f} MeV "
        f"(cells, normalised to 1)")
    out["mean_dm"] = float(np.sum(w * dm))
    log("  P(u > u0):")
    for u0 in U_GRID:
        p = float(np.sum(w[u > u0]))
        log(f"    u0 = {u0:<8g}  {p:.6f}")
        out[f"tail_{u0:g}"] = p
    if window > 0:
        m = np.abs(dm) <= window
        pw = float(np.sum(w[m]))
        mw = float(np.sum(w[m] * dm[m]) / pw)
        log(f"  in |dm| <= {window} GeV: P = {pw:.6f}, <dm> = {mw * 1e3:+.4f} "
            f"MeV = {mw / mass:+.4e} relative")
        out["window"] = float(window)
        out["p_in_window"] = pw
        out["mean_dm_window"] = mw
    return out


# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["mc", "analytic"])
    p.add_argument("--pairs", default=None, help="mc: the pairs cache")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--mass", type=float, default=MJPSI,
                   help="the term's m_ref, i.e. the delta the kernel replaces")
    p.add_argument("--tmax", type=float, default=1000.0,
                   help="upper end of the CF tabulation [1/GeV]; must exceed "
                        "max(tgrid)/min(sigma) of the term (767 on the J/psi "
                        "v2 cache) or `_build_norm` refuses the card")
    p.add_argument("--npoints", type=int, default=50001,
                   help="uniform tabulation points on [0, tmax]")
    p.add_argument("--nbins-pow", type=int, default=25,
                   help="mc: log2 of the dm histogram size behind the FFT")
    p.add_argument("--dt-dft", type=float, default=0.25,
                   help="mc: the DFT frequency step; the FFT period is "
                        "2 pi / dt_dft and the dm bin is period / 2^nbins_pow")
    # the card's selection, so the kernel is the selected candidates' kernel
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--max-sigma-rel", type=float, default=0.10)
    p.add_argument("--obs-window", type=float, default=0.0,
                   help="mc: ALSO cut |m_obs - M|. Default 0 = OFF; the "
                        "likelihood's norm_window is the one place the "
                        "observed-mass selection may be applied")
    p.add_argument("--window", type=float, default=GEN_HALFWIDTH,
                   help="the window the in-window mean shift is quoted in")
    # the analytic kernel
    p.add_argument("--variant", default="exp2nll",
                   choices=["exp1", "exp2", "exp2nll", "oalpha", "born"])
    p.add_argument("--pair", nargs="*", default=["e", "mu"])
    p.add_argument("--mass-exact", action="store_true", default=True,
                   help="use the exact-mass O(alpha) matrix element for the "
                        "hard remainder (default ON at the J/psi, where it is "
                        "a 1e-3 effect)")
    p.add_argument("--no-mass-exact", dest="mass_exact", action="store_false")
    p.add_argument("--truncate", type=float, default=0.0,
                   help="analytic: restrict the kernel to |dm| <= this and "
                        "renormalise, to match a cache built with a gen-mass "
                        "acceptance. 0 = off")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="analytic: also convolve with a Breit-Wigner of this "
                        "width [GeV] -- the resonance's own lineshape, which "
                        "the MC kernel carries automatically (the J/psi total "
                        "width is 9.26e-5 GeV). It is symmetric, so it changes "
                        "the CF but not the mean shift.")
    p.add_argument("--n-fine", type=int, default=20000)
    p.add_argument("--ng", type=int, default=16)
    p.add_argument("--save-dm", action="store_true",
                   help="mc: also store the dm sample itself (float32), for "
                        "`fullscale/gate_jpsi_fsr.py` and the figures")
    p.add_argument("--report", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    log = print
    meta = {"mode": a.mode, "mass": a.mass, "tmax": a.tmax,
            "npoints": a.npoints, "window": a.window}

    if a.mode == "mc":
        if not a.pairs:
            raise SystemExit("mode mc needs --pairs")
        dm, info = load_mc_dm(a.pairs, a.mass, a.max_chi2_ndof,
                              a.max_sigma_rel, a.obs_window, log)
        meta.update(info)
        meta["source"] = os.path.abspath(a.pairs)
        stats = report_mc(dm, a.mass, a.window, log) if a.report else {}
        t, re, im = cf_from_samples(dm, a.tmax, a.npoints, a.nbins_pow,
                                    a.dt_dft, log)
        meta["cf_validation"] = validate_cf(dm, t, re, im, log=log)
        extra = {"n": len(dm)}
        if a.save_dm:
            extra["dm"] = dm.astype(np.float32)
    else:
        k, w, ub, var = analytic_cells(a.mass, a.variant, a.pair,
                                       a.mass_exact, a.n_fine, a.ng, log)
        if a.truncate > 0:
            dmc = a.mass * np.expm1(-ub)
            m = np.abs(dmc) <= a.truncate
            w = w[m] / w[m].sum()
            ub, var = ub[m], var[m]
            log(f"  truncated to |dm| <= {a.truncate} GeV and renormalised: "
                f"{m.sum()} of {len(m)} cells")
        stats = report_analytic(k, w, ub, a.mass, a.window, log) if a.report \
            else {}
        t, re, im = cf_from_cells(w, ub, var, a.mass, a.tmax, a.npoints, log)
        if a.gamma > 0:
            # phi_BW(t) = exp(-Gamma |t| / 2); the resonance's own lineshape
            damp = np.exp(-0.5 * a.gamma * t)
            re, im = re * damp, im * damp
            log(f"  x Breit-Wigner, Gamma = {a.gamma:g} GeV: "
                f"|phi_BW(tmax)| = {damp[-1]:.6f}")
        meta.update({"variant": a.variant, "pair": list(a.pair),
                     "gamma": a.gamma,
                     "mass_exact": bool(a.mass_exact),
                     "truncate": a.truncate, "beta": float(k.beta),
                     "L": float(k.L), "pair_rate": float(k.pair_rate)})
        extra = {"u": ub, "w": w, "var": var}

    log(f"\n  phi_K(0) = {re[0]:.12f}{im[0]:+.3e}i   "
        f"|phi_K(tmax)| = {math.hypot(re[-1], im[-1]):.6f}")
    # the mean shift the CF implies, as a cross-check of the tabulation
    dphi = (im[1] - im[0]) / (t[1] - t[0])
    log(f"  d Im phi_K/dt at 0 = <dm> = {dphi * 1e3:+.4f} MeV "
        f"= {dphi / a.mass:+.4e} relative")
    meta["mean_dm_cf"] = float(dphi)
    meta.update({f"stat_{k_}": v for k_, v in stats.items()})

    if a.output:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".",
                    exist_ok=True)
        np.savez_compressed(a.output, phik_t=t, phik_re=re, phik_im=im,
                            meta=json.dumps(meta), **extra)
        log(f"  -> {a.output}")


if __name__ == "__main__":
    main()
