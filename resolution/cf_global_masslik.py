"""Prototype of the unified calibration objective: the exact mass
likelihood as the scale anchor of the global fit.

Per J/psi candidate the mass responds linearly to the global calibration
parameters theta (B-field modes, material scales, ...) via the stored
reference-state Jacobians:

    m_i(theta) = m0_i + D_i . delta_theta,
    D_i = a1^T J1 + a2^T J2   (a = mass Jacobian per leg, J = jacrefv)

and the exact per-candidate likelihood L_i(m) = [resonance x FSR x
rho_i](m) is built from the candidate CF product (+ uniform mixture
floor). The global fit maximizes sum_i ln L_i(m_i(theta)) by Newton
iterations with the analytic score/Fisher -- each iteration is a spline
lookup, no track refits.

The same machinery with the Gaussian-constraint score ((m-M)/sigma_eff^2)
quantifies the FSR bias of the chi2 constraint on identical inputs.

Stages:
  --collect  pair muons, store m0, CF exponents, and the dense D_i
             restricted to the chosen parmtypes (default: 14 = the 50
             scalar-potential field modes)
  --fit      Newton fit of delta_theta (NLL and Gauss variants)
  --inject   linear injection of a known delta_theta into m0, then both
             fits: recovered-vs-injected pulls and the bias comparison

usage:
  python cf_global_masslik.py --collect [--files GLOB] [--parmtypes 14]
  python cf_global_masslik.py --fit [--ridge 1e2]
  python cf_global_masslik.py --inject --inject-modes 0 2 7 --inject-size 2e-4
"""

import argparse
import datetime
import glob
import os

import numpy as np
import uproot

from wums import logging, output_tools
from cf_mass_likelihood import (MJPSI, TG, collect_pairs, mass_jacobian,
                                leg_exponents, FBKG)

logger = logging.child_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_trackres_jpsi260803a/task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--parmtypes", type=int, nargs="+", default=[14],
                   help="global parmtypes to fit (14 = field modes, 15 = material groups)")
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf_globmass_cache.npz"))
    p.add_argument("--kernel-cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs/fsr_kernel_jpsi.npz"))
    p.add_argument("--collect", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--inject", action="store_true")
    p.add_argument("--inject-modes", type=int, nargs="+", default=[0, 2, 7])
    p.add_argument("--inject-size", type=float, default=2e-4)
    p.add_argument("--ridge", type=float, default=0.,
                   help="ridge prior sigma^-2 added to the Fisher diagonal")
    p.add_argument("--covtol", type=float, default=5e-3)
    return p.parse_args()


def collect(args):
    files = sorted(glob.glob(args.files))[:args.ntasks]
    logger.info(f"{len(files)} files")
    frun = uproot.open(files[0])
    pt = frun["runtree"]["parmtype"].array(library="np")
    sel_reg = np.where(np.isin(pt, args.parmtypes))[0]
    colmap = {int(g): i for i, g in enumerate(sel_reg)}
    npar = len(sel_reg)
    logger.info(f"fitting {npar} global params (parmtypes {args.parmtypes})")

    m0s, mgens, sigs, vgf = [], [], [], []
    Sms_l, Sio_re_l, Sio_im_l, D_l = [], [], [], []
    nsel = ndrop = 0
    want = ("refParms", "refCov", "reseigidx", "resinfbv", "msmoliidx",
            "msmoliv", "ioniurbanidx", "ioniurbanv", "jacrefv", "globalidxv")
    for fn, a, gp, pairs in collect_pairs(files, want=want):
        for i1, i2 in pairs:
            from cf_mass_likelihood import pair_mass
            mg = pair_mass(gp[i1][None, :], gp[i2][None, :])[0]
            if abs(mg - MJPSI) > 0.35:
                continue
            r1 = np.asarray(a["refParms"][i1], dtype=np.float64)
            r2 = np.asarray(a["refParms"][i2], dtype=np.float64)
            mr, a1, a2 = mass_jacobian(r1, r2)
            legs, ok = [], True
            for av, ic in ((a1, i1), (a2, i2)):
                out = leg_exponents(av, ic, a, pt)
                if out is None:
                    ok = False
                    break
                legs.append(out)
            if not ok:
                continue
            var = sum(l[2] for l in legs)
            exact = 0.
            for av, ic in ((a1, i1), (a2, i2)):
                Cu = np.asarray(a["refCov"][ic], dtype=np.float64).reshape(5, 5)
                exact += av @ (np.triu(Cu) + np.triu(Cu, 1).T) @ av
            if not (exact > 0.) or abs(var / exact - 1.) > args.covtol:
                ndrop += 1
                continue
            sig = np.sqrt(var)
            # dense D over the fitted subset from both legs' jacrefv
            D = np.zeros(npar, dtype=np.float64)
            for av, ic in ((a1, i1), (a2, i2)):
                gi = np.asarray(a["globalidxv"][ic])
                J = np.asarray(a["jacrefv"][ic], dtype=np.float64).reshape(5, len(gi))
                dm = av @ J
                for j, g in enumerate(gi):
                    c = colmap.get(int(g))
                    if c is not None:
                        D[c] += dm[j]
            from cf_mass_likelihood import ms_step_exponent, ioni_step_exponent
            Sms = np.zeros(len(TG))
            Sio = np.zeros(len(TG), dtype=np.complex128)
            vg = 0.
            for vgauss, groups, _ in legs:
                vg += vgauss
                for famcode, weff, steps in groups:
                    if famcode == 10:
                        Sms += ms_step_exponent(steps, abs(weff) / sig, TG)
                    else:
                        Sio += ioni_step_exponent(steps, weff / sig, TG)
            m0s.append(mr)
            mgens.append(mg)
            sigs.append(sig)
            vgf.append(vg / var)
            Sms_l.append(Sms.astype(np.float32))
            Sio_re_l.append(Sio.real.astype(np.float32))
            Sio_im_l.append(Sio.imag.astype(np.float32))
            D_l.append(D.astype(np.float32))
            nsel += 1
        logger.info(f"{fn.split('/')[-2]}: cumulative {nsel} candidates (drop {ndrop})")
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez_compressed(args.cache, m0=np.array(m0s), mgen=np.array(mgens),
                        sigma=np.array(sigs), vgf=np.array(vgf),
                        Sms=np.array(Sms_l), Sio_re=np.array(Sio_re_l),
                        Sio_im=np.array(Sio_im_l), D=np.array(D_l),
                        regidx=sel_reg, tgrid=TG)
    logger.info(f"wrote {args.cache} ({nsel} candidates, {npar} params)")


def candidate_splines(d, kernel):
    """Per-candidate ln L(m) and derivatives on a local mass grid:
    L_i(m) = (1-f) * [K x rho_i](m - MJPSI) + f/W via the CF product."""
    dm = kernel["dm"]
    sig = d["sigma"]
    n = len(sig)
    tmax_abs = TG[-1] / sig.min()
    tabs = np.linspace(0., tmax_abs, 8192)
    phiK_tab = np.zeros(len(tabs), dtype=np.complex128)
    for i0 in range(0, len(tabs), 1024):
        phiK_tab[i0:i0+1024] = np.mean(
            np.exp(1j * np.outer(tabs[i0:i0+1024], dm)), axis=1)
    tgi = TG[None, :] / sig[:, None]
    phiK = (np.interp(tgi, tabs, phiK_tab.real)
            + 1j * np.interp(tgi, tabs, phiK_tab.imag))
    S = (-0.5 * d["vgf"][:, None] * TG[None, :] ** 2
         + d["Sms"] + 1j * d["Sio_im"] + d["Sio_re"])
    Phi = np.exp(S) * phiK
    mgrid = np.linspace(-0.15, 0.15, 301)
    lnL = np.empty((n, len(mgrid)), dtype=np.float32)
    for im, m in enumerate(mgrid):
        integ = (Phi * np.exp(-1j * tgi * m)).real
        Li = np.trapezoid(integ, TG, axis=1) / (np.pi * sig)
        lnL[:, im] = np.log((1. - FBKG) * np.clip(Li, 0., None) + FBKG / 0.7)
    return mgrid, lnL


def newton_fit(dm0, D, mgrid, lnL, ridge, gauss_sigma=None, nit=25):
    """Maximize sum_i lnL_i(dm0_i + D_i.x) over x. gauss_sigma given ->
    Gaussian score instead (chi2 constraint emulation). Damped Newton:
    the per-candidate mass shift is trust-region-capped to stay on the
    lnL spline support (the mass-only Fisher has flat mode directions,
    so ridge > 0 is required for a meaningful 50-mode demo)."""
    npar = D.shape[1]
    n = len(dm0)
    x = np.zeros(npar)
    h = mgrid[1] - mgrid[0]
    for it in range(nit):
        m = dm0 + D @ x
        if gauss_sigma is None:
            im = np.clip(((m - mgrid[0]) / h).astype(int), 1, len(mgrid) - 3)
            frac = (m - mgrid[im]) / h
            l0 = np.take_along_axis(lnL, im[:, None], 1)[:, 0]
            lp = np.take_along_axis(lnL, im[:, None] + 1, 1)[:, 0]
            lm = np.take_along_axis(lnL, im[:, None] - 1, 1)[:, 0]
            s = (lp - lm) / (2 * h) + frac * (lp + lm - 2 * l0) / h ** 2
            c = -(lp + lm - 2 * l0) / h ** 2
            c = np.clip(c, 1., None)
        else:
            s = -m / gauss_sigma ** 2
            c = np.full(n, 1. / gauss_sigma ** 2)
        g = D.T @ s - ridge * x
        H = (D * c[:, None]).T @ D
        if ridge > 0.:
            H = H + ridge * np.eye(npar)
        dx = np.linalg.solve(H, g)
        # trust region: cap the largest per-candidate mass shift this step
        # to 20 MeV so the spline lookups stay in their support
        dmmax = np.abs(D @ dx).max()
        if dmmax > 0.02:
            dx *= 0.02 / dmmax
        x = x + dx
        if np.abs(D @ dx).max() < 1e-7:
            break
    cov = np.linalg.inv(H)
    return x, cov, it + 1


def run_fit(args, inject=None):
    d = np.load(args.cache)
    k = np.load(args.kernel_cache)
    D = d["D"].astype(np.float64)
    n, npar = D.shape
    logger.info(f"{n} candidates, {npar} params; building candidate splines")
    mgrid, lnL = candidate_splines(d, k)
    dm0 = d["m0"] - MJPSI
    if inject is not None:
        dm0 = dm0 + D @ inject
        logger.info(f"injected |dtheta| = {np.abs(inject).max():g} on "
                    f"{int((inject != 0).sum())} modes")
    res = {}
    for lab in ("nll", "gauss"):
        gs = d["sigma"].mean() if lab == "gauss" else None
        # DIFFERENTIAL closure: the sample carries real residual offsets
        # (the alpha scale, the core-width surplus) that the fit rightly
        # absorbs into the modes; the injection test is the DIFFERENCE
        # between the injected and baseline fits, which cancels them.
        xb, covb, nb_ = newton_fit(d["m0"] - MJPSI, D, mgrid, lnL,
                                   args.ridge, gauss_sigma=gs)
        x, cov, nit = newton_fit(dm0, D, mgrid, lnL, args.ridge, gauss_sigma=gs)
        err = np.sqrt(np.diag(cov))
        res[lab] = (x, xb, err)
        logger.info(f"[{lab}] baseline |x|_max = {np.abs(xb).max():.3e}, "
                    f"injected fit converged in {nit} iters")
        if inject is not None:
            dx = x - xb
            pull = (dx - inject) / err
            logger.info(f"[{lab}] DIFFERENTIAL recovery: " + "  ".join(
                f"m{j}: {dx[j]:+.2e}/{inject[j]:+.2e} (pull {pull[j]:+.2f})"
                for j in np.nonzero(inject)[0]))
            logger.info(f"[{lab}] non-injected: max|dx| = "
                        f"{np.abs(dx[inject == 0.]).max():.2e}, "
                        f"max|pull| = {np.abs(pull[inject == 0.]).max():.2f}")
    if inject is not None:
        dd = (res["gauss"][0] - res["gauss"][1]) - (res["nll"][0] - res["nll"][1])
        logger.info(f"gauss - nll differential difference: max |delta| = "
                    f"{np.abs(dd).max():.3e}")
        # the absolute-fit difference IS the constraint bias on the real sample
        db = res["gauss"][1] - res["nll"][1]
        logger.info(f"gauss - nll BASELINE difference (constraint bias in "
                    f"theta space on the real sample): max |delta| = "
                    f"{np.abs(db).max():.3e}")
    return res


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    if args.collect:
        collect(args)
    if args.fit:
        run_fit(args)
    if args.inject:
        d = np.load(args.cache)
        npar = d["D"].shape[1]
        inj = np.zeros(npar)
        for m in args.inject_modes:
            inj[m] = args.inject_size
        run_fit(args, inject=inj)


if __name__ == "__main__":
    main()
