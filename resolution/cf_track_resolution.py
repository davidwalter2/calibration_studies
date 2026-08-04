"""Per-track resolution prediction from the CF product ("Level 2").

The linearized CVH fit maps the noise vector n to the momentum error via
delta(q/p) = w^T n with w = (C e_qop)^T F^T V^{-1}. The refit exports, per
resolution entry b (aligned with reseigidx), the variance contribution
v_b = w_b^T V_b w_b (resinfvarv) and their total resinfcov, which equals
refCov(0,0) EXACTLY (validated: ratio 1.0000 on every smoke track) -- every
noise dof feeding q/p carries an entry.

Independence of the blocks then gives the full non-Gaussian CF of the
standardized momentum error z = (qop_reco - qop_gen)/sigma, sigma^2 =
refCov(0,0):

  log phi_z(t) = -(Vg/sigma^2) t^2/2                        [hits, Gaussian]
    + sum_MS  e^{k_ms}  sum_steps (chic2/chia2) G(t w_std sqrt(chia2); FF)
    + sum_ion e^{k_ion} sum_steps S_urban(t w_std g_s ...)  [centered]

where per pooled block w_std = sqrt(v_pool / sigma_Q^2) / sigma is the
effective scalar weight (exact under per-collision azimuthal isotropy:
any linear functional of the two projected angles is again projected-
Moliere with |weight|), and sigma_Q^2 is the fit-assumed block variance
(sum thp2 for MS, sum gsig2 g^2 for ionization). Blocks are pooled by
global parameter index (steps are matched the same way as in cf_ms_exact /
cf_ioni_exact); pooling merges same-family crossings with a shared w --
exact for the common one-crossing case.

Closure ("--closure"): on gen-matched MC with the NOMINAL fit
(fitFromGenParms=False), compare per track
  data  e^{-u z_obs^2}          vs   model  E[e^{-u z^2}] (Weierstrass)
averaged in bins of predicted sigma and probe u, plus the averaged
predicted lineshape p(z) against the pull histogram.

usage:
  python cf_track_resolution.py --extract [--files GLOB] [--ntasks N]
  python cf_track_resolution.py --closure [--probes ...]
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
from cf_ms_exact import moliere_params, gshape
import cf_ioni_exact

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

TG = np.linspace(0.0, 14.0, 448)  # t conjugate to standardized z
COVTOL = 5e-3                     # |resinfcov/refCov00 - 1| guard


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_trackres_260802/task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--max-tracks", type=int, default=100000)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf_trackres_cache.npz"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--closure", action="store_true")
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--khit", type=float, default=0.0,
                   help="log-scale applied to the hit (Gaussian) variance share")
    p.add_argument("--kms", type=float, default=0.0,
                   help="log-scale applied to the MS exponent (collision counts)")
    p.add_argument("--kioni", type=float, default=0.0,
                   help="log-scale applied to the ionization exponent")
    p.add_argument("--nsigma-bins", type=int, default=4)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def _delta_term_2d(a, w):
    """Vectorized <(e^{iaE} - 1 - iaE)/E^2> for the 1/E^2 spectrum on
    [1, w] (units of e0): a (ns, nt) real, w (ns,) scalar per step.
    Same math as cf_ioni_exact.delta_term."""
    from scipy.special import exp1
    a = np.asarray(a, dtype=np.complex128)
    w = np.asarray(w, dtype=np.float64)[:, None]
    ia = 1j * a
    with np.errstate(all="ignore"):
        t1 = (-np.exp(ia * w) / w + np.exp(ia)) + ia * (exp1(-ia) - exp1(-ia * w))
        t1 -= (1.0 - 1.0 / w)
        t1 -= ia * np.log(w)
    norm = 1.0 - 1.0 / w
    out = t1 / norm
    wb = np.broadcast_to(w, a.shape)
    # The series expansion of <e^{iaE}-1-iaE> requires a*E << 1 over the WHOLE
    # support, and the support reaches E = w, so the expansion parameter is
    # a*w -- NOT a. For muons w = tmax/e0 ~ 1e8-1e10 (keV-scale e0 against a
    # multi-GeV kinematic tmax), so a guard on |a| alone selects the series in
    # a regime where it is wrong by orders of magnitude: at w = 1e9, a = 1e-6
    # the series gives Im = -8.3e-2 against the true -6.5e-6 (verified against
    # direct sampling of the 1/E^2 spectrum). That spurious phase, stacked over
    # ~350 steps, turns the model CF into a pure oscillation at small t.
    # Below a*w ~ 1e-2 the closed form loses precision to cancellation and the
    # series is the accurate one, so switch there; the two agree to ~3e-4
    # across a*w = 1e-2 .. 1e-1.
    small = np.abs(a) * wb < 5e-2
    if small.any():
        # The series MUST keep the cubic imaginary term: it is the transmitted
        # Landau skew (mean-vs-mode). For the 1/E^2 spectrum on [1, w]:
        # <E^2>_raw = w-1, <E^3>_raw = (w^2-1)/2.
        nb = np.broadcast_to(norm, a.shape)
        ar = a.real
        quad = (-0.5 * ar ** 2 * (wb - 1.)
                - 1j * ar ** 3 / 6. * (wb * wb - 1.) / 2.) / nb
        out[small] = quad[small]
    return out


def ioni_step_exponent(steps, wstd, tau):
    """Centered Urban log-CF exponent of one pooled ionization block,
    in standardized-z units: per step the raw qop-noise CF evaluated at
    wstd*tau (wstd = w/sigma includes the fit-unit conversion). steps:
    (n, 11) ioniurbanv records. Vectorized over steps. Returns complex
    array on tau."""
    reg = steps[:, 0]
    gsig2 = steps[:, 1].astype(np.float64)
    g = steps[:, 10].astype(np.float64) * 1e-3  # qop per MeV
    gs = wstd * g                               # (ns,)
    S = np.zeros(len(tau), dtype=np.complex128)

    mg = reg == 0
    if mg.any():
        S += -0.5 * tau ** 2 * float(np.sum(gsig2[mg] * gs[mg] ** 2))

    mu = ~mg
    if mu.any():
        gam = steps[mu, 9]
        gsu = gs[mu]
        for ja, je in ((2, 3), (4, 5)):
            aj = steps[mu, ja]
            ej = steps[mu, je] * gam
            act = (aj > 0.) & (ej > 0.)
            if act.any():
                th = (gsu[act] * ej[act])[:, None] * tau[None, :]
                S += np.sum(aj[act][:, None] * (np.exp(1j * th) - 1. - 1j * th), axis=0)
        a3 = steps[mu, 6]
        e0 = steps[mu, 7] * gam
        tmax = steps[mu, 8] * gam
        act = (a3 > 0.) & (tmax > e0) & (e0 > 0.)
        if act.any():
            aarg = (gsu[act] * e0[act])[:, None] * tau[None, :]
            S += np.sum(a3[act][:, None] * _delta_term_2d(aarg, tmax[act] / e0[act]),
                        axis=0)
    return S


def ms_step_exponent(steps, wstd, tau):
    """Moliere log-CF exponent of one pooled MS block in standardized-z
    units, FF-cut per step, vectorized over steps with the ymax bucketing
    of cf_ms_moliere._slot_S. steps: (n, 8) msmoliv records [effZ, effA,
    xg, pGeV, beta, thp2, dOverX0, stepGroup]."""
    ok = steps[:, 5] > 0.
    if not ok.any():
        return np.zeros(len(tau))
    prm = np.array([moliere_params(*s) for s in steps[ok, :5]])  # (ns, 3)
    chic2, chia2, thff2 = prm[:, 0], prm[:, 1], prm[:, 2]
    act = (chic2 > 0.) & (chia2 > 0.)
    if not act.any():
        return np.zeros(len(tau))
    chic2, chia2, thff2 = chic2[act], chia2[act], thff2[act]
    args = np.sqrt(chia2)[:, None] * (wstd * tau)[None, :]
    ym = np.sqrt(thff2 / chia2)
    lg = np.log(np.clip(ym, 1e1, 1e7))
    rows = np.round((lg - np.log(1e1)) / (np.log(1e7 / 1e1) / 12)).astype(int)
    S = np.zeros(len(tau))
    for r in np.unique(rows):
        m = rows == r
        ymr = float(np.exp(np.log(1e1) + r * (np.log(1e7 / 1e1) / 12)))
        gsh = gshape(args[m].ravel(), ymax=ymr).reshape(int(m.sum()), len(tau))
        S += np.sum((chic2[m] / chia2[m])[:, None] * gsh, axis=0)
    return S


def extract(args):
    """Per selected track: z_obs, sigma, eta, and the three exponent
    components on TG (Gaussian variance share Vg_frac; MS real exponent;
    ionization complex exponent), stored separately so per-family k
    scalings can be applied at closure time."""
    files = sorted(glob.glob(args.files))[:args.ntasks]
    logger.info(f"{len(files)} files")
    zs, sigs, etas, vgf = [], [], [], []
    Sms_l, Sio_re_l, Sio_im_l = [], [], []
    nsel = ndropcov = ndropgen = 0
    pt = None
    for fn in files:
        try:
            f = uproot.open(fn)
            if "tree" not in f:
                continue
            if pt is None:
                pt = f["runtree"]["parmtype"].array(library="np")
            t = f["tree"]
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        a = t.arrays(["refParms", "refCov", "genParms", "resinfcov",
                      "resinfvarv", "reseigidx", "msmoliidx", "msmoliv",
                      "ioniurbanidx", "ioniurbanv"], library="np")
        for ic in range(len(a["resinfcov"])):
            qg = a["genParms"][ic][0]
            if qg == 0.:
                ndropgen += 1
                continue
            c00 = float(a["refCov"][ic][0])
            cov = float(a["resinfcov"][ic])
            if not (c00 > 0.) or abs(cov / c00 - 1.) > COVTOL:
                ndropcov += 1
                continue
            sig = np.sqrt(c00)
            gi = np.asarray(a["reseigidx"][ic])
            vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
            fam = pt[gi]
            vgauss = vb[(fam == 8) | (fam == 9)].sum()

            uvm = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            uim = np.asarray(a["msmoliidx"][ic])
            uvm = uvm.reshape(-1, len(uvm) // max(len(uim), 1)) if len(uim) else uvm.reshape(0, 8)
            uvi = np.asarray(a["ioniurbanv"][ic], dtype=np.float64)
            uii = np.asarray(a["ioniurbanidx"][ic])
            uvi = uvi.reshape(-1, len(uvi) // max(len(uii), 1)) if len(uii) else uvi.reshape(0, 11)

            Sms = np.zeros(len(TG))
            Sio = np.zeros(len(TG), dtype=np.complex128)
            ok = True
            for famcode, (uidx, uv) in ((10, (uim, uvm)), (11, (uii, uvi))):
                sel = fam == famcode
                for g in np.unique(gi[sel]):
                    vpool = vb[sel & (gi == g)].sum()
                    if vpool <= 0.:
                        continue
                    steps = uv[uidx == g]
                    if not len(steps):
                        ok = False
                        break
                    if famcode == 10:
                        sq2 = steps[:, 5].sum()
                        if sq2 <= 0.:
                            continue
                        wstd = np.sqrt(vpool / sq2) / sig
                        Sms += ms_step_exponent(steps, wstd, TG)
                    else:
                        gq = steps[:, 10] * 1e-3
                        sq2 = float(np.sum(steps[:, 1] * gq * gq))
                        if sq2 <= 0.:
                            continue
                        wstd = np.sqrt(vpool / sq2) / sig
                        Sio += ioni_step_exponent(steps, wstd, TG)
                if not ok:
                    break
            if not ok:
                continue

            zs.append((a["refParms"][ic][0] - qg) / sig)
            sigs.append(sig)
            etas.append(-np.log(np.tan((np.pi / 2. - a["refParms"][ic][1]) / 2.)))
            vgf.append(vgauss / c00)
            Sms_l.append(Sms.astype(np.float32))
            Sio_re_l.append(Sio.real.astype(np.float32))
            Sio_im_l.append(Sio.imag.astype(np.float32))
            nsel += 1
        logger.info(f"{fn.split('/')[-2]}: cumulative {nsel} tracks "
                    f"(drop gen {ndropgen}, cov {ndropcov})")
        if nsel >= args.max_tracks:
            break
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez_compressed(args.cache, z=np.array(zs), sigma=np.array(sigs),
                        eta=np.array(etas), vgf=np.array(vgf),
                        Sms=np.array(Sms_l), Sio_re=np.array(Sio_re_l),
                        Sio_im=np.array(Sio_im_l), tgrid=TG)
    logger.info(f"wrote {args.cache} ({nsel} tracks)")


def model_phi(d, args):
    """Complex phi_z(t) per track on TG with the per-family k applied.
    Total variance is renormalized so z stays standardized to the
    *scaled* model: sigma_model^2/sigma^2 = e^khit*vgf + scaled tails --
    handled by evaluating phi of the scaled physics and letting closure
    compare against z_obs built with the unscaled sigma."""
    vg = np.exp(args.khit) * d["vgf"][:, None]
    S = (-0.5 * vg * TG[None, :] ** 2
         + np.exp(args.kms) * d["Sms"]
         + np.exp(args.kioni) * (d["Sio_re"] + 1j * d["Sio_im"]))
    return np.exp(S)


def weier(phi, u):
    """E[e^{-u z^2}] per track from phi on TG (even weight, Re phi)."""
    w = np.exp(-TG ** 2 / (4. * u)) / np.sqrt(np.pi * u)
    return np.clip(np.trapezoid(phi.real * w[None, :], TG, axis=1), 0., 1.)


def closure(args, outdir):
    d = np.load(args.cache)
    n = len(d["z"])
    logger.info(f"{n} tracks from {args.cache}")
    phi = model_phi(d, args)
    z = d["z"]

    # per-probe closure, inclusive and in predicted-sigma quartiles
    qs = np.quantile(d["sigma"], np.linspace(0., 1., args.nsigma_bins + 1))
    lines = []
    for u in args.probes:
        Em = weier(phi, u)
        Ed = np.exp(-u * z ** 2)
        dm, sm = Ed.mean() - Em.mean(), Ed.std() / np.sqrt(n)
        lines.append(f"u={u:5.2f}  <data>-<model> = {dm:+.5f} +- {sm:.5f}  "
                     f"(<model> = {Em.mean():.4f})")
    logger.info("inclusive closure:\n" + "\n".join("  " + s for s in lines))

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    # left: pull histogram vs averaged predicted lineshape
    ax = axs[0]
    zg = np.linspace(-8., 8., 161)
    # p(z) = 1/pi Int_0^inf Re[phi(t) e^{-itz}] dt, averaged over tracks
    ph = phi.mean(axis=0)
    pz = np.array([np.trapezoid((ph * np.exp(-1j * TG * zz)).real, TG) / np.pi
                   for zz in zg])
    ax.hist(np.clip(z, zg[0], zg[-1]), bins=zg, density=True,
            histtype="step", color="black", label="pulls (gen-matched MC)")
    ax.plot(zg, pz, color="crimson", label="CF-product prediction")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2.)
    ax.set_xlabel(r"$z = \Delta(q/p)/\sigma_{\mathrm{pred}}$")
    ax.set_ylabel("density")
    ax.legend()
    # right: data-model difference of the bounded statistic vs u, by sigma bin
    ax = axs[1]
    uu = np.geomspace(0.02, 4., 25)
    for ib in range(args.nsigma_bins):
        m = (d["sigma"] >= qs[ib]) & (d["sigma"] <= qs[ib + 1])
        dd = np.array([np.exp(-u * z[m] ** 2).mean() - weier(phi[m], u).mean()
                       for u in uu])
        ss = np.array([np.exp(-u * z[m] ** 2).std() / np.sqrt(m.sum()) for u in uu])
        ax.errorbar(uu, dd, ss, marker="o", markersize=4, linestyle="-",
                    label=rf"$\sigma \in [{1e2*qs[ib]:.2f}, {1e2*qs[ib+1]:.2f}]\times 10^{{-2}}$")
    ax.axhline(0., color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"probe $u$")
    ax.set_ylabel(r"$\langle e^{-uz^2}\rangle_{\mathrm{data}} - \langle e^{-uz^2}\rangle_{\mathrm{model}}$")
    ax.legend(fontsize="x-small")
    name = f"trackres_closure{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(outdir, f"{name}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info(f"wrote {outdir}/{name}")


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_trackres/")
    os.makedirs(outdir, exist_ok=True)
    if args.extract:
        extract(args)
    if args.closure:
        closure(args, outdir)


if __name__ == "__main__":
    main()
