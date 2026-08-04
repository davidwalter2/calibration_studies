"""A3: exact compound-Poisson CF fit of the ionization straggling scale,
using the Urban-model export (ioniurbanidx / ioniurbanv branches, 2026-07-24)
and the direct nu storage (gradllv).

Model per ionization block b (parmtype 11), standardized exactly as the
track fit standardized it (sigma_ref^2 = sum_s gsig2_s * (cs_s*1e-3)^2, the
alpha-truncated per-step variances the Q matrix carried):

  z_b = sqrt(h) * z_true + sqrt(1-h) * z_est
  - h = nu (gradllv entry) is the hat value: the fraction of the block's
    noise variance actually measured; the rest is Gaussian estimation noise
    from the other (hit/MS-dominated) degrees of freedom.
  - z_true: centered compound-Poisson straggling / sigma_ref, with per-step
    exponent (energies MeV -> qop via g = cs*1e-3, gamma = scaling):
      excitations: a_j * (e^{i tau g gamma e_j} - 1 - i tau g gamma e_j)
      delta rays:  a3 * <e^{i tau g gamma E} - 1 - i tau g gamma E>_{p(E)}
                   with p(E) ~ 1/E^2 on [e0, tmax] (closed form via E1)
      Gaussian-regime steps: -tau^2 g^2 gsig2 / 2
    All terms are linear in the collision rates, so the material scale k
    (rates x e^k) enters as phi_true(tau; k) = exp(e^k * S_b(tau)).

  E[e^{-u z^2}] = (4 pi u)^{-1/2} Int dt e^{-t^2/(4u)}
                  Re[ exp(e^k S_b(sqrt(h) t)) ] * e^{-(1-h) t^2 / 2}
  (Weierstrass transform; the imaginary part integrates to zero.)

Fit: solve mean_b(data e^{-u z2}) = mean_b(model_b(k)) for k at each probe
u. k is the PHYSICAL material-density scale with the delta-electron tail
included exactly and untruncated -- flatness of k-hat(u) is the test that
the Urban compound-Poisson model describes the simulated straggling.
"""

import argparse
import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from scipy.optimize import brentq
from scipy.special import exp1

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

NPARS = 11  # floats per exported Urban step
# t-grid for the Weierstrass integral. Must resolve the weight e^{-t^2/4u}
# at the smallest probe (width ~ 2 sqrt(u) ~ 0.45 at u=0.05): 512 points on
# [0, 12] give dt = 0.023, ~20 points across it. The Gaussian check
# F_model(k=0) -> (1+2u)^{-1/2} in the h->0 limit validates the quadrature.
TGRID = np.linspace(0.0, 12.0, 512)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_closure_260724_urban_alpha999_fb01e7b7741/"
                           "task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=10,
                   help="number of task files to process (blocks are plentiful)")
    p.add_argument("--max-blocks", type=int, default=60000)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf_ioni_cache.npz"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--outpath", default=None)
    p.add_argument("--max-chi2ndof", type=float, default=0.,
                   help="track preselection chisq/ndof < cut (0 = off). "
                   "Needed for hadrons: decay-in-flight kinks produce a "
                   "structured non-straggling z2 population")
    p.add_argument("--postfix", default="")
    return p.parse_args()


def delta_term(a):
    """<e^{iaE} - 1 - iaE> for p(E) = C/E^2 on [e0=1, tmax=w] is computed by
    the caller in units of e0; here the raw integrals on [1, w]:
      I(a, w) = int_1^w (e^{iaE} - 1 - iaE) / E^2 dE / int_1^w dE/E^2
    with int dE/E^2 = 1 - 1/w. Uses
      int e^{iaE}/E^2 dE = -e^{iaE}/E + ia * int e^{iaE}/E dE
      int_x^y e^{iaE}/E dE = E1(-iax) - E1(-iay)   (principal branch)
    a: real array (tau * g * gamma * e0), w: scalar tmax/e0 per step.
    """
    a = np.asarray(a, dtype=np.complex128)
    w = np.float64(delta_term.w)
    ia = 1j * a
    # int (e^{iaE}-1)/E^2 on [1,w]
    with np.errstate(all="ignore"):
        e1_lo = exp1(-ia)         # E1(-ia*1)
        e1_hi = exp1(-ia * w)     # E1(-ia*w)
        t1 = (-np.exp(ia * w) / w + np.exp(ia)) + ia * (e1_lo - e1_hi)
        t1 -= (1.0 - 1.0 / w)                  # the -1 part
        t1 -= ia * np.log(w)                   # the -iaE part: int dE/E = ln w
    norm = 1.0 - 1.0 / w
    out = t1 / norm
    # small-a fallback (E1 cancellation): quadratic term with
    # <E^2>/e0^2 = w (for the 1/E^2 spectrum, <E^2> = e0 tmax) -- relative
    # to the mean-subtracted quadratic: -(a^2/2) * <E^2>/norm-ish
    small = np.abs(a) < 1e-6
    if small.any():
        # raw quadratic + CUBIC IMAGINARY term (the transmitted skew --
        # <E^3>_raw = (w^2-1)/2 for the 1/E^2 spectrum; dropping it zeroed
        # the muon ionization skew, every muon step has |a| < 1e-6)
        ar = a[small].real
        out[small] = (-0.5 * ar ** 2 * (w - 1.)
                      - 1j * ar ** 3 / 6. * (w * w - 1.) / 2.) / norm
    return out


def block_exponent(steps, tau):
    """S_b(tau): centered log-CF exponent of the block's straggling in qop
    units standardized by sigma_ref (rates at k=0). steps: (n, NPARS)."""
    reg = steps[:, 0]
    gsig2 = steps[:, 1].astype(np.float64)
    g = steps[:, 10].astype(np.float64) * 1e-3  # qop per MeV
    sigref2 = np.sum(gsig2 * g * g)
    if sigref2 <= 0.:
        return None, 0.
    sigref = np.sqrt(sigref2)
    S = np.zeros(len(tau), dtype=np.complex128)
    for s in range(len(steps)):
        gs = g[s] / sigref
        if reg[s] == 0:
            # Gaussian-regime step: variance gsig2 * g^2 in qop units
            S += -0.5 * tau ** 2 * gsig2[s] * gs ** 2
            continue
        gam = steps[s, 9]
        a1, e1, a2, e2 = steps[s, 2], steps[s, 3] * gam, steps[s, 4], steps[s, 5] * gam
        a3, e0, tmax = steps[s, 6], steps[s, 7] * gam, steps[s, 8] * gam
        for aj, ej in ((a1, e1), (a2, e2)):
            if aj > 0. and ej > 0.:
                th = tau * gs * ej
                S += aj * (np.exp(1j * th) - 1. - 1j * th)
        if a3 > 0. and tmax > e0 > 0.:
            delta_term.w = tmax / e0
            S += a3 * delta_term(tau * gs * e0)
    return S, sigref


def extract(files, args):
    """Per selected ioni block: z2, h (=nu), and S_b on TGRID."""
    z2s, hs, Ss, c2ns = [], [], [], []
    pt = None
    nblocks = 0
    for fn in files[:args.ntasks]:
        f = uproot.open(fn)
        if "tree" not in f:
            continue
        if pt is None:
            pt = f["runtree"]["parmtype"].array(library="np")
        t = f["tree"]
        a = t.arrays(["globalidxv", "gradchisqv", "gradllv", "chisqval",
                      "ndof", "ioniurbanidx", "ioniurbanv"], library="np")
        for ic in range(len(a["globalidxv"])):
            gi = np.asarray(a["globalidxv"][ic])
            tp = pt[gi]
            m11 = np.where(tp == 11)[0]
            if not len(m11):
                continue
            nu = np.asarray(a["gradllv"][ic], dtype=np.float64)
            q = -np.asarray(a["gradchisqv"][ic], dtype=np.float64)
            uidx = np.asarray(a["ioniurbanidx"][ic])
            uv = np.asarray(a["ioniurbanv"][ic], dtype=np.float64).reshape(-1, NPARS)
            for j in m11:
                gidx = gi[j]
                h = nu[j]
                if h <= 0. or q[j] < 0.:
                    continue
                steps = uv[uidx == gidx]
                if not len(steps):
                    continue
                # evaluate the exponent directly at this block's arguments
                # sqrt(h)*t -- no later interpolation (h spans decades and
                # the S structure lives at tau ~ sqrt(h), unresolvable on a
                # shared linear grid)
                S, sigref = block_exponent(steps, np.sqrt(h) * TGRID)
                if S is None:
                    continue
                z2s.append(q[j] / h)
                hs.append(h)
                Ss.append(S)
                c2ns.append(a["chisqval"][ic] / max(a["ndof"][ic], 1))
                nblocks += 1
            if nblocks >= args.max_blocks:
                break
        logger.info(f"{fn.split('/')[-2]}: cumulative {nblocks} blocks")
        if nblocks >= args.max_blocks:
            break
    np.savez_compressed(args.cache,
                        z2=np.array(z2s), h=np.array(hs),
                        S=np.array(Ss), c2n=np.array(c2ns), tgrid=TGRID)
    logger.info(f"wrote {args.cache} ({nblocks} blocks)")


def model_stat(u, k, S, h, tgrid):
    """mean_b E[e^{-u z^2}] under the model at material scale k.
    S: (nb, nt) complex exponents at k=0, PRE-EVALUATED at sqrt(h_b)*tgrid
    per block (see extract); h: (nb,).
    phi_b(t) = exp(e^k * S_b(sqrt(h) t)) * exp(-(1-h) t^2/2).
    """
    ek = np.exp(k)
    phi = np.exp(ek * S) * np.exp(-0.5 * (1. - h)[:, None] * tgrid[None, :] ** 2)
    integ = np.real(phi)
    w = np.exp(-tgrid ** 2 / (4. * u))
    # half-line integral, trapezoid (the t=0 endpoint carries weight dt/2)
    vals = (np.pi * u) ** -0.5 * np.trapz(integ * w[None, :], tgrid, axis=1)
    return np.mean(np.clip(vals, 0., 1.))


def model_stat2(u, k, sest, S, h, tgrid):
    """Two-parameter model: material scale k on the block's own straggling
    plus a variance scale sest on the (1-h) Gaussian estimation-noise part.
    With h ~ 1e-6 the block residual is dominated by estimation noise from
    the OTHER families (hits, MS), whose model variances are themselves off
    at the several-percent level -- fixing sest = 1 confounds the k fit.
    """
    ek = np.exp(k)
    phi = np.exp(ek * S) * np.exp(-0.5 * sest * (1. - h)[:, None] * tgrid[None, :] ** 2)
    w = np.exp(-tgrid ** 2 / (4. * u))
    vals = (np.pi * u) ** -0.5 * np.trapz(np.real(phi) * w[None, :], tgrid, axis=1)
    return np.mean(np.clip(vals, 0., 1.))


def joint_fit(z2, S, h, tgrid, probes):
    """Joint (k, sest) least squares over all probe statistics."""
    from scipy.optimize import minimize
    Fd = np.array([np.mean(np.exp(-u * z2)) for u in probes])
    sF = np.array([np.std(np.exp(-u * z2)) / np.sqrt(len(z2)) for u in probes])

    def chi2(p):
        k, sest = p
        Fm = np.array([model_stat2(u, k, sest, S, h, tgrid) for u in probes])
        return np.sum(((Fd - Fm) / sF) ** 2)

    res = minimize(chi2, [0., 0.92], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3})
    k, sest = res.x
    # numerical Hessian for errors
    eps = np.array([2e-2, 2e-3])
    H = np.zeros((2, 2))
    f0 = chi2(res.x)
    for i in range(2):
        for j in range(i, 2):
            dp_i = np.zeros(2); dp_i[i] = eps[i]
            dp_j = np.zeros(2); dp_j[j] = eps[j]
            H[i, j] = H[j, i] = (chi2(res.x + dp_i + dp_j) - chi2(res.x + dp_i - dp_j)
                                 - chi2(res.x - dp_i + dp_j) + chi2(res.x - dp_i - dp_j)) \
                                / (4 * eps[i] * eps[j])
    cov = 2. * np.linalg.inv(H)
    return k, sest, np.sqrt(np.diag(cov)), f0, res


def fit(args, outdir):
    d = np.load(args.cache)
    z2, h, S, tgrid = d["z2"], d["h"], d["S"], d["tgrid"]
    if args.max_chi2ndof > 0. and "c2n" in d.files:
        keep = d["c2n"] < args.max_chi2ndof
        logger.info(f"preselection chisq/ndof<{args.max_chi2ndof}: "
                    f"keep {keep.sum()}/{len(z2)}")
        z2, h, S = z2[keep], h[keep], S[keep]
    logger.info(f"{len(z2)} blocks; h median {np.median(h):.3g} "
                f"[{np.percentile(h,16):.3g}, {np.percentile(h,84):.3g}]")
    ks, kerrs = [], []
    print(f"{'u':>6s} {'F_data':>10s} {'F_model(0)':>10s} {'k-hat':>9s} {'err':>8s}")
    for u in args.probes:
        F = np.mean(np.exp(-u * z2))
        sigF = np.std(np.exp(-u * z2)) / np.sqrt(len(z2))
        F0 = model_stat(u, 0., S, h, tgrid)

        def eqn(k):
            return model_stat(u, k, S, h, tgrid) - F

        try:
            k = brentq(eqn, -2., 2., xtol=1e-6)
            eps = 5e-3
            dm = (eqn(k + eps) - eqn(k - eps)) / (2 * eps)
            kerr = sigF / abs(dm)
        except ValueError:
            k, kerr = np.nan, np.nan
        ks.append(k)
        kerrs.append(kerr)
        print(f"{u:6.2f} {F:10.6f} {F0:10.6f} {k:9.4f} {kerr:8.4f}")

    k2, sest, errs2, chi2min, res = joint_fit(z2, S, h, tgrid, args.probes)
    ndf = len(args.probes) - 2
    print(f"\njoint fit: k = {k2:.4f} +- {errs2[0]:.4f}   "
          f"s_est = {sest:.4f} +- {errs2[1]:.4f}   chi2/ndf = {chi2min:.1f}/{ndf}")
    Fm = [model_stat2(u, k2, sest, S, h, tgrid) for u in args.probes]
    for u, fd, fm in zip(args.probes,
                         [np.mean(np.exp(-u * z2)) for u in args.probes], Fm):
        print(f"   u={u:5.2f}  F_data={fd:.6f}  F_joint={fm:.6f}")

    np.savez(os.path.join(outdir, f"cf_ioni_results_{args.postfix or 'nominal'}.npz"),
             probes=np.array(args.probes), k=np.array(ks), kerr=np.array(kerrs),
             kjoint=k2, sest=sest, jointerrs=errs2, chi2=chi2min)

    finite = np.isfinite(ks)
    if not np.any(finite):
        logger.warning("all single-u k-solves NaN; skipping scan plot")
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(np.array(args.probes)[finite], np.array(ks)[finite],
                yerr=np.array(kerrs)[finite], marker="o", color="#9467bd",
                label="exact compound-Poisson CF fit (untruncated tail)")
    ax.axhline(0., color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"ECF probe $u$")
    ax.set_ylabel(r"fitted material scale $\hat{k}$ (ionization)")
    ax.legend(fontsize=15)
    plot_tools.add_decor(ax, "CMS", "Work in progress (B→J/ψ+X MC gen closure)",
                         data=False, lumi=None, loc=2)
    name = "_".join(filter(None, ["cf_ioni_kscan", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    np.savez(os.path.join(outdir, "cf_ioni_results.npz"),
             probes=np.array(args.probes), k=np.array(ks), kerr=np.array(kerrs))


def main():
    args = parse_args()
    if args.extract:
        files = sorted(glob.glob(args.input))
        if not files:
            sys.exit(f"no files match {args.input}")
        extract(files, args)
    if args.fit:
        today = datetime.date.today().strftime("%y%m%d")
        outpath = args.outpath or os.path.expanduser(
            f"~/public_html/calibration_studies/{today}_cf_ioni_exact/")
        outdir = output_tools.make_plot_dir(outpath)
        fit(args, outdir)


if __name__ == "__main__":
    main()
