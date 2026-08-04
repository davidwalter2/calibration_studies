"""Candidate-level mass machinery for the CF likelihood program:
replace the Gaussian chi2 mass constraint by the exact likelihood

  L(m_obs) = int dmu p_FSR(mu) p_res(m_obs - mu)

assembled in transform space: the resonance (delta at m_Jpsi; Gamma = 93
keV negligible), the FSR kernel (post-FSR gen dimuon mass from Photos --
extracted from genParms pairs), and the per-candidate resolution CF (the
Level-2 CF product of both legs projected on the mass Jacobian) MULTIPLY
as characteristic functions; one inverse transform gives the exact
likelihood, no Gaussian assumption anywhere.

Stages:
  --kernel   pair gen-matched muons by event, build p_FSR(m_gen_mumu)
             around the J/psi pole, save histogram + CF
  --pairs    build reco candidate list: per pair, m_reco (from refParms),
             m_gen, mass Jacobian a_leg = dm/d(qop,lam,phi) per leg, and
             the per-block signed weights u_b = a^T B_b (needs resinfbv,
             production 260802b); saves the per-candidate mass-CF
             exponents on the shared t-grid
  --closure  mass pull closure: (m_reco - m_gen)/sigma_pred vs the
             predicted candidate lineshape (candidate-level analogue of
             the track closure)
  --demo     unbinned scale fit: NLL(alpha) with m_obs -> m_obs(1+alpha)
             on MC, demonstrating the chi2-constraint replacement

usage: python cf_mass_likelihood.py --kernel [--files GLOB]
"""

import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools, plot_tools
from cf_track_resolution import ms_step_exponent, ioni_step_exponent, TG

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

MMU = 0.1056584
MJPSI = 3.0969
FBKG = 0.005


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_trackres_260802/task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--kernel", action="store_true")
    p.add_argument("--pairs", action="store_true")
    p.add_argument("--closure", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--kernel-cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs/fsr_kernel_jpsi.npz"))
    p.add_argument("--pairs-cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs/cf_masspairs_cache.npz"))
    p.add_argument("--covtol", type=float, default=5e-3)
    p.add_argument("--vtx-tol", type=float, default=1e-3,
                   help="gen production-vertex match tolerance [cm] for true "
                        "J/psi pairs; <=0 keeps all opposite-charge combinations "
                        "(the old behaviour, which leaks combinatorics)")
    # --demo scan grids; the r range must BRACKET the minimum, otherwise the
    # profile rails at an edge and the reported alpha is pulled by the
    # alpha-r correlation
    p.add_argument("--r-min", type=float, default=0.30)
    p.add_argument("--r-max", type=float, default=1.15)
    p.add_argument("--r-step", type=float, default=0.05)
    p.add_argument("--alpha-min", type=float, default=-1e-3)
    p.add_argument("--alpha-max", type=float, default=4e-3)
    p.add_argument("--alpha-n", type=int, default=41)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def p4(parms):
    """(E, px, py, pz) from (qop, lam, phi) rows: parms (n, >=3)."""
    p = 1. / np.abs(parms[:, 0])
    cl = np.cos(parms[:, 1])
    px = p * cl * np.cos(parms[:, 2])
    py = p * cl * np.sin(parms[:, 2])
    pz = p * np.sin(parms[:, 1])
    E = np.sqrt(p * p + MMU * MMU)
    return np.stack([E, px, py, pz], axis=1)


def pair_mass(g1, g2):
    s = p4(g1) + p4(g2)
    m2 = s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2 - s[:, 3] ** 2
    return np.sqrt(np.maximum(m2, 0.))


def collect_pairs(files, want=("genParms",), vtxtol=None):
    """Group tracks by (run, lumi, event); yield per file the index pairs
    of opposite-charge gen-matched muons (all combinations; the J/psi
    window cut is applied downstream on the gen mass).

    vtxtol (cm), if set, additionally requires the two muons to share a gen
    production vertex -- i.e. to be the SAME J/psi. Without it every
    opposite-charge combination is kept, and the samples reach 12 gen-matched
    muons per event, so mispaired combinations leak in: they populate
    m_gen ABOVE the pole (up to +295 MeV), which FSR cannot produce. The
    separation distribution is sharply bimodal (0 vs cm-scale) so any
    tolerance in 1e-4..1e-2 cm gives the same answer."""
    for fn in files:
        try:
            f = uproot.open(fn)
            if "tree" not in f:
                continue
            t = f["tree"]
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        vtxb = ("genX", "genY", "genZ") if vtxtol is not None else ()
        branches = sorted(set(("run", "lumi", "event", "genParms", "genCharge")
                              + want + vtxb))
        a = t.arrays(branches, library="np")
        gp = np.stack(a["genParms"]) if len(a["genParms"]) else np.zeros((0, 5))
        ok = np.abs(gp[:, 0]) > 0.
        evid = (a["run"].astype(np.int64) << 40) ^ (a["lumi"].astype(np.int64) << 24) ^ a["event"].astype(np.int64)
        order = np.argsort(evid, kind="stable")
        pairs = []
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and evid[order[j]] == evid[order[i]]:
                j += 1
            idx = [k for k in order[i:j] if ok[k]]
            for ia in range(len(idx)):
                for ib in range(ia + 1, len(idx)):
                    k1, k2 = idx[ia], idx[ib]
                    if a["genCharge"][k1] * a["genCharge"][k2] >= 0:
                        continue
                    if vtxtol is not None:
                        d2 = ((a["genX"][k1] - a["genX"][k2]) ** 2
                              + (a["genY"][k1] - a["genY"][k2]) ** 2
                              + (a["genZ"][k1] - a["genZ"][k2]) ** 2)
                        if d2 > vtxtol ** 2:
                            continue
                    pairs.append((k1, k2))
            i = j
        yield fn, a, gp, pairs


def mass_jacobian(parms1, parms2):
    """Numerical d m / d(qop, lam, phi) per leg (central differences),
    parms: (5,) refParms rows. Returns m, a1 (5,), a2 (5,)."""
    def m_of(x1, x2):
        return pair_mass(x1[None, :], x2[None, :])[0]
    m0 = m_of(parms1, parms2)
    a1 = np.zeros(5)
    a2 = np.zeros(5)
    for i in range(3):
        h = max(1e-7, 1e-6 * abs(parms1[i]))
        xp = parms1.copy(); xm = parms1.copy()
        xp[i] += h; xm[i] -= h
        a1[i] = (m_of(xp, parms2) - m_of(xm, parms2)) / (2 * h)
        h = max(1e-7, 1e-6 * abs(parms2[i]))
        xp = parms2.copy(); xm = parms2.copy()
        xp[i] += h; xm[i] -= h
        a2[i] = (m_of(parms1, xp) - m_of(parms1, xm)) / (2 * h)
    return m0, a1, a2


def leg_exponents(av, ic, a, pt):
    """One leg's contribution to the candidate mass CF, in ABSOLUTE mass
    units (standardization to sigma_pred happens at candidate level).
    Returns (vgauss, Sms_unnorm, Sio_unnorm, vtot) where the S arrays are
    evaluated on TG/sigma later -- to keep one shared grid we instead
    return the per-group (weff, steps) lists; simpler: evaluate on a
    provisional unit grid and rescale afterwards is wrong for nonlinear
    CFs, so we evaluate AFTER sigma is known. Hence this returns the raw
    ingredients."""
    gi = np.asarray(a["reseigidx"][ic])
    B = np.asarray(a["resinfbv"][ic]).reshape(-1, 5, 5)
    u = np.einsum("p,bpj->bj", av, B)          # (nent, 5) signed dof weights
    v = (u ** 2).sum(axis=1)                   # per-entry variance contribution
    fam = pt[gi]
    vgauss = v[(fam == 8) | (fam == 9)].sum()
    uvm = np.asarray(a["msmoliv"][ic], dtype=np.float64)
    uim = np.asarray(a["msmoliidx"][ic])
    uvm = uvm.reshape(-1, len(uvm) // max(len(uim), 1)) if len(uim) else uvm.reshape(0, 8)
    uvi = np.asarray(a["ioniurbanv"][ic], dtype=np.float64)
    uii = np.asarray(a["ioniurbanidx"][ic])
    uvi = uvi.reshape(-1, len(uvi) // max(len(uii), 1)) if len(uii) else uvi.reshape(0, 11)
    groups = []
    for famcode, (uidx, uv) in ((10, (uim, uvm)), (11, (uii, uvi))):
        sel = fam == famcode
        for g in np.unique(gi[sel]):
            m = sel & (gi == g)
            vpool = v[m].sum()
            if vpool <= 0.:
                continue
            steps = uv[uidx == g]
            if not len(steps):
                return None
            if famcode == 10:
                sq2 = steps[:, 5].sum()
            else:
                gq = steps[:, 10] * 1e-3
                sq2 = float(np.sum(steps[:, 1] * gq * gq))
            if sq2 <= 0.:
                continue
            # signed effective weight (sign only matters for ioni skew)
            sgn = np.sign(u[m].sum()) or 1.
            groups.append((famcode, sgn * np.sqrt(vpool / sq2), steps))
    return vgauss, groups, v.sum()


def build_pairs(args, outdir):
    """Candidate cache in the cf_track_resolution.closure format:
    z = (m_reco - m_gen)/sigma_pred, sigma = sigma_pred [GeV], plus the
    exponent components on TG."""
    files = sorted(glob.glob(args.files))[:args.ntasks]
    logger.info(f"{len(files)} files")
    zs, sigs, mgen, vgf = [], [], [], []
    Sms_l, Sio_re_l, Sio_im_l = [], [], []
    nsel = ndropid = 0
    pt = None
    want = ("refParms", "refCov", "reseigidx", "resinfbv",
            "msmoliidx", "msmoliv", "ioniurbanidx", "ioniurbanv")
    vtxtol = args.vtx_tol if args.vtx_tol > 0 else None
    for fn, a, gp, pairs in collect_pairs(files, want=want, vtxtol=vtxtol):
        if pt is None:
            f = uproot.open(fn)
            pt = f["runtree"]["parmtype"].array(library="np")
        for i1, i2 in pairs:
            g1, g2 = gp[i1], gp[i2]
            mg = pair_mass(g1[None, :], g2[None, :])[0]
            if abs(mg - MJPSI) > 0.35:
                continue
            r1 = np.asarray(a["refParms"][i1], dtype=np.float64)
            r2 = np.asarray(a["refParms"][i2], dtype=np.float64)
            mr, a1, a2 = mass_jacobian(r1, r2)
            legs = []
            okleg = True
            for av, ic in ((a1, i1), (a2, i2)):
                out = leg_exponents(av, ic, a, pt)
                if out is None:
                    okleg = False
                    break
                legs.append(out)
            if not okleg:
                continue
            var = sum(l[2] for l in legs)
            # identity guard: sum over blocks vs a^T C a from refCov
            exact = 0.
            for av, ic in ((a1, i1), (a2, i2)):
                Cu = np.asarray(a["refCov"][ic], dtype=np.float64).reshape(5, 5)
                C = np.triu(Cu) + np.triu(Cu, 1).T
                exact += av @ C @ av
            if not (exact > 0.) or abs(var / exact - 1.) > args.covtol:
                ndropid += 1
                continue
            sig = np.sqrt(var)
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
            zs.append((mr - mg) / sig)
            sigs.append(sig)
            mgen.append(mg)
            vgf.append(vg / var)
            Sms_l.append(Sms.astype(np.float32))
            Sio_re_l.append(Sio.real.astype(np.float32))
            Sio_im_l.append(Sio.imag.astype(np.float32))
            nsel += 1
        logger.info(f"{fn.split('/')[-2]}: cumulative {nsel} candidates "
                    f"(drop identity {ndropid})")
    os.makedirs(os.path.dirname(args.pairs_cache), exist_ok=True)
    np.savez_compressed(args.pairs_cache, z=np.array(zs), sigma=np.array(sigs),
                        eta=np.array(mgen), vgf=np.array(vgf),
                        Sms=np.array(Sms_l), Sio_re=np.array(Sio_re_l),
                        Sio_im=np.array(Sio_im_l), tgrid=TG)
    logger.info(f"wrote {args.pairs_cache} ({nsel} candidates, "
                f"{ndropid} dropped by identity guard)")


def build_kernel(args, outdir):
    files = sorted(glob.glob(args.files))[:args.ntasks]
    logger.info(f"{len(files)} files")
    masses = []
    vtxtol = args.vtx_tol if args.vtx_tol > 0 else None
    for fn, a, gp, pairs in collect_pairs(files, vtxtol=vtxtol):
        if not pairs:
            continue
        i1 = np.array([p[0] for p in pairs])
        i2 = np.array([p[1] for p in pairs])
        m = pair_mass(gp[i1], gp[i2])
        masses.append(m[np.abs(m - MJPSI) < 0.35])
        logger.info(f"{fn.split('/')[-2]}: {len(pairs)} pairs, "
                    f"cumulative {sum(len(x) for x in masses)} in window")
    m = np.concatenate(masses)
    logger.info(f"total {len(m)} J/psi-window gen pairs")
    # FSR kernel: distribution of dm = m_gen - M_JPSI (<= 0 up to gen
    # smearing/width; radiative tail below)
    dm = m - MJPSI
    hist, edges = np.histogram(dm, bins=700, range=(-0.35, 0.005), density=True)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    frac_tail = float((dm < -0.005).mean())
    logger.info(f"FSR tail fraction (dm < -5 MeV): {frac_tail:.4f}")
    os.makedirs(os.path.dirname(args.kernel_cache), exist_ok=True)
    np.savez_compressed(args.kernel_cache, dm=dm, hist=hist, ctr=ctr,
                        frac_tail=frac_tail)
    logger.info(f"wrote {args.kernel_cache}")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(1e3 * dm, bins=280, range=(-350, 5), density=True, histtype="step",
            color="black")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m^{\mathrm{gen}}_{\mu\mu} - m_{J/\psi}$ [MeV]")
    ax.set_ylabel("density [1/GeV]")
    ax.text(0.05, 0.95, f"FSR tail (< $-5$ MeV): {100*frac_tail:.1f}%",
            transform=ax.transAxes, va="top")
    name = f"fsr_kernel_jpsi{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"wrote {outdir}/{name}")


def demo(args, outdir):
    """Unbinned momentum-scale fit on MC: per candidate the likelihood

      L_i(alpha, r) = 1/pi Int_0^inf Re[ phi_K(t) phi_res,i(t; r)
                                          e^{-i t (m_i - M(1+alpha))} ] dt

    with phi_K the empirical CF of the FSR kernel (dm samples), phi_res,i
    the candidate CF-product (exponent scaled by r = resolution variance
    factor), evaluated on the candidate's standardized grid t = TG/sigma_i.
    Everything multiplies in transform space: no binning, no templates,
    no Gaussian assumption. NLL profiled on an (alpha, r) grid."""
    d = np.load(args.pairs_cache)
    k = np.load(args.kernel_cache)
    dm = k["dm"]
    z, sig = d["z"], d["sigma"]
    n = len(z)
    logger.info(f"{n} candidates, {len(dm)} kernel samples")
    # empirical kernel CF on the union of candidate grids: interpolate from
    # a fine absolute-t table
    tmax_abs = TG[-1] / sig.min()
    tabs = np.linspace(0., tmax_abs, 8192)
    phiK_tab = np.mean(np.exp(1j * np.outer(tabs[:1024], dm)), axis=1)
    # extend with second block to limit memory
    phiK_tab = np.concatenate([phiK_tab,
        np.mean(np.exp(1j * np.outer(tabs[1024:], dm)), axis=1)])
    Sexp = d["Sms"] + 1j * 0. + d["Sio_re"] + 1j * d["Sio_im"]

    mobs_minus_M = z * sig + (d["eta"] - MJPSI)   # m_reco - M_JPSI per candidate

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_n)
    rs = np.arange(args.r_min, args.r_max + 1e-9, args.r_step)
    tgi = TG[None, :] / sig[:, None]              # absolute t grid per candidate
    phiK = np.interp(tgi, tabs, phiK_tab.real) + 1j * np.interp(tgi, tabs, phiK_tab.imag)
    nll = np.zeros((len(rs), len(alphas)))
    for ir, r in enumerate(rs):
        S = (-0.5 * r * d["vgf"][:, None] * TG[None, :] ** 2 + r * Sexp)
        Phi = np.exp(S) * phiK
        for ia, al in enumerate(alphas):
            delta = (mobs_minus_M - MJPSI * al)[:, None]
            integrand = (Phi * np.exp(-1j * tgi * delta)).real
            Li = np.trapezoid(integrand, TG, axis=1) / (np.pi * sig)
            # small uniform mixture over the mass window: regularizes the
            # oscillation-cancelled far-tail likelihoods AND models the
            # true mispair/combinatoric floor visible at |z| > 5
            Li = (1. - FBKG) * np.clip(Li, 0., None) + FBKG / 0.7
            nll[ir, ia] = -np.sum(np.log(Li))
        logger.info(f"r={r:.2f}: min NLL at alpha={alphas[np.argmin(nll[ir])]*1e3:.3f}e-3")
    irbest, iabest = np.unravel_index(np.argmin(nll), nll.shape)
    if irbest in (0, len(rs) - 1):
        logger.warning(f"r profile RAILED at grid edge r={rs[irbest]:.2f} "
                       f"(scanned {rs[0]:.2f}-{rs[-1]:.2f}) -- widen --r-min/--r-max")
    if iabest in (0, len(alphas) - 1):
        logger.warning(f"alpha profile RAILED at grid edge "
                       f"alpha={alphas[iabest]:.3e} -- widen --alpha-min/--alpha-max")
    # parabolic alpha error at best r
    y = nll[irbest]
    ia = np.clip(iabest, 1, len(alphas) - 2)
    c = (y[ia+1] + y[ia-1] - 2*y[ia]) / (alphas[1]-alphas[0])**2
    alpha_hat = alphas[ia] - (y[ia+1]-y[ia-1])/(2*c*(alphas[1]-alphas[0]))
    err = 1./np.sqrt(c)
    logger.info(f"BEST: alpha = ({alpha_hat*1e3:.4f} +- {err*1e3:.4f})e-3, "
                f"r = {rs[irbest]:.2f}  (Dm = {alpha_hat*MJPSI*1e3:.2f} MeV)")
    # naive Gaussian-constraint estimate for comparison (kernel ignored)
    wg = 1./sig**2
    alpha_gauss = np.sum(wg*mobs_minus_M)/np.sum(wg)/MJPSI
    logger.info(f"naive Gaussian-constraint scale (FSR ignored): "
                f"alpha = {alpha_gauss*1e3:.4f}e-3 "
                f"(bias {1e3*(alpha_gauss-alpha_hat):+.4f}e-3)")

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    ax = axs[0]
    for ir, r in enumerate(rs):
        ax.plot(alphas*1e3, nll[ir]-nll.min(), label=f"r = {r:.2f}")
    ax.axvline(alpha_gauss*1e3, color="gray", ls="--", label="Gaussian constraint")
    ax.set_xlabel(r"momentum scale $\alpha$ [$10^{-3}$]")
    ax.set_ylabel(r"$\Delta$NLL")
    ax.set_ylim(0, 50)
    ax.legend(fontsize="x-small")
    ax = axs[1]
    # observed spectrum vs best-fit prediction
    mg = np.linspace(-0.25, 0.2, 120)
    S = (-0.5*rs[irbest]*d["vgf"][:, None]*TG[None, :]**2 + rs[irbest]*Sexp)
    Phi = np.exp(S) * phiK
    sub = np.random.default_rng(3).choice(n, size=min(n, 4000), replace=False)
    pm = np.zeros(len(mg))
    for im, m in enumerate(mg):
        integrand = (Phi[sub]*np.exp(-1j*tgi[sub]*(m - MJPSI*alpha_hat))).real
        pm[im] = np.mean(np.trapezoid(integrand, TG, axis=1)/(np.pi*sig[sub]))
    ax.hist(np.clip(mobs_minus_M, mg[0], mg[-1]), bins=mg, density=True,
            histtype="step", color="black", label=r"$m_{\mu\mu}^{\rm reco} - m_{J/\psi}$")
    ax.plot(0.5*(mg[1:]+mg[:-1]), 0.5*(pm[1:]+pm[:-1]), color="crimson",
            label="resonance $\\otimes$ FSR $\\otimes$ resolution")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m - m_{J/\psi}$ [GeV]")
    ax.set_ylabel("density")
    ax.legend(fontsize="x-small")
    name = f"masslik_demo{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"runs/masslik_demo_scan{args.postfix}.npz"),
             alphas=alphas, rs=rs, nll=nll, alpha_hat=alpha_hat, err=err,
             alpha_gauss=alpha_gauss)
    logger.info(f"wrote {outdir}/{name}")


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_masslik/")
    os.makedirs(outdir, exist_ok=True)
    if args.kernel:
        build_kernel(args, outdir)
    if args.pairs:
        build_pairs(args, outdir)
    if args.closure:
        import cf_track_resolution as ctr
        ns = argparse.Namespace(**vars(args))
        ns.cache = args.pairs_cache
        ns.khit = ns.kms = ns.kioni = 0.0
        ns.nsigma_bins = 4
        ns.probes = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        ns.postfix = "_mass" + args.postfix
        ctr.closure(ns, outdir)
    if args.demo:
        demo(args, outdir)


if __name__ == "__main__":
    main()
