"""Phase B: exact screened-Rutherford compound-Poisson CF fit of the
multiple-scattering scale, using the raw per-step material export
(msmoliidx / msmoliv branches) and gradllv.

Model per MS block b (parmtype 10), mirroring cf_ioni_exact.py:

  z_b = sqrt(h) * z_true + sqrt(1-h) * s_est * z_gauss,  h = nu (hat value)

  z_true: weighted sum of per-step projected Moliere deflections,
  standardized by the model (sum_s thp2_s = the Rossi variances actually in
  Q). Per step, the single-scattering density in theta^2 is
      n(t2) = chi_c^2 / (t2 + chi_a^2)^2      (N_scat = chi_c^2/chi_a^2)
  with (PDG/Bethe conventions, p in GeV, x in g/cm^2):
      chi_c^2 = 0.157e-6 * Z(Z+1)/A * x / (p^2 beta^2)         [rad^2]
      chi_a^2 = chi_0^2 * (1.13 + 3.76 (alpha Z / beta)^2)
      chi_0   = 2.007e-5 * Z^(1/3) * (1 + 3.34 (alpha Z / beta)^2)^0 / p
                -- classic Moliere screening angle m_e alpha Z^(1/3)/(0.885 p)
  The projected-angle CF of one step's compound is
      S_step(t) = Int_0^inf (J0(t*theta) - 1) n(theta^2) dtheta^2
  (J0 from the azimuthal average; even -> exactly centered). The material
  scale k multiplies chi_c^2 (collision rate) only, so the block exponent
  is again linear in e^k: phi_true = exp(e^k * S_b(tau)).

  Per-step weights into the block scalar: w_s^2 = thp2_s / sum thp2 (the
  R-projection lever arms are not exported; variance-matched uniform
  weighting is the v1 approximation, same spirit as the ioni fit).

Fit: joint (k, s_est) over the probe statistics E[e^{-u z^2}] via the
Weierstrass transform. With h(ms) ~ 0.3 the signal is O(30%) of the block
residual (vs 1e-6 for ioni), so the joint fit is expected well-conditioned.
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
from scipy.optimize import minimize
from scipy.special import j0

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

NPARS = 7
TGRID = np.linspace(0.0, 12.0, 512)
ALPHA_EM = 1.0 / 137.036


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_closure_260724_moli_alpha999_fb01e7b7741/"
                           "task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=10)
    p.add_argument("--max-blocks", type=int, default=40000)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf_ms_cache.npz"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def moliere_params(effZ, effA, xg, pGeV, beta):
    """chi_c^2, chi_a^2 and the nuclear form-factor cutoff theta_FF^2
    [rad^2] for one step (WentzelVI-consistent single-scattering inputs).

    chi_0 = m_e alpha / 0.885 * Z^(1/3) / p  (Thomas-Fermi screening;
    4.214e-6 GeV -- NOTE: an earlier version used 2.007e-5, conflating the
    Lynch-Dahl chi_a^2 constant with a linear chi_0 formula: chi_a^2 was
    ~23x too large, N_scat ~23x too small. Fixed 2026-07-24.)
    theta_FF = hbar c / (p R_N), R_N = 1.27 A^0.27 fm: dipole/nuclear
    form-factor scale that terminates the single-Rutherford tail (G4
    WentzelVI's FF); implemented as a hard cutoff of the y^2 integral.
    Mott (McKinley-Feshbach) factor not yet included (few-% tail shape).
    """
    chic2 = 0.157e-6 * effZ * (effZ + 1.) / effA * xg / (pGeV ** 2 * beta ** 2)
    az = ALPHA_EM * effZ / beta
    chi0 = 4.214e-6 * effZ ** (1. / 3.) / pGeV
    chia2 = chi0 ** 2 * (1.13 + 3.76 * az ** 2)
    rn_fm = 1.27 * max(effA, 1.) ** 0.27
    # dipole form-factor characteristic angle^2 (G4WentzelOKandVIxSection
    # convention: FF = 1/(1 + q^2 R^2/12)^2, i.e. theta_c^2 = 12 (hbarc/pR)^2
    # -- the earlier hard cutoff at (hbarc/pR)^2 was 12x too tight and the
    # wrong shape)
    thff2 = 12. * (0.19733 / (pGeV * rn_fm)) ** 2
    return chic2, chia2, thff2


# Universal Moliere shape: with theta = sqrt(chia2)*y,
#   S_step(t) = (chic2/chia2) * G(t*sqrt(chia2)),
#   G(tau) = Int_0^inf (J0(tau*y) - 1) / (1+y^2)^2 dy^2
# Precomputed once on a log-tau grid; every step is then an interpolation.
# G(0) = 0, G(inf) -> -1 (one unit per scatter). The y^2 grid upper end is
# the kinematic-scale cutoff of the single-scattering spectrum; the CF
# converges (unlike the variance) so the exact value is uncritical.
_Y2 = np.logspace(-4, 13, 360)
_DY2 = np.gradient(_Y2)
_GTAU = np.logspace(-8, 4, 1600)
# 2D shape table: G(tau; ymax) with the y^2 integral cut at ymax^2
# (nuclear form-factor cutoff, ymax = theta_FF/chi_a per step).
_YMAXG = np.logspace(1, 7, 13)
_G2D = np.empty((len(_YMAXG), len(_GTAU)))
_k0 = j0(np.outer(_GTAU, np.sqrt(_Y2))) - 1.
for _iy, _ym in enumerate(_YMAXG):
    # smooth dipole form factor 1/(1+y^2/ymax^2)^2 (G4 convention), not a
    # hard cutoff
    _w = _DY2 / (1. + _Y2) ** 2 / (1. + _Y2 / _ym ** 2) ** 2
    _G2D[_iy] = _k0 @ _w
_G = _G2D[-1]  # backwards-compatible: effectively uncut


def gshape(tau, ymax=None):
    """Universal exponent shape, optionally with the FF cutoff ymax;
    linear interpolation in log(ymax) between precomputed rows."""
    tau = np.abs(np.asarray(tau, dtype=np.float64))
    if ymax is None:
        row = _G
    else:
        ly = np.clip(np.log(ymax), np.log(_YMAXG[0]), np.log(_YMAXG[-1]))
        fi = (ly - np.log(_YMAXG[0])) / (np.log(_YMAXG[1]) - np.log(_YMAXG[0]))
        i0 = int(np.clip(np.floor(fi), 0, len(_YMAXG) - 2))
        f = fi - i0
        row = (1. - f) * _G2D[i0] + f * _G2D[i0 + 1]
    out = np.interp(tau, _GTAU, row, left=np.nan, right=row[-1])
    tiny = tau < _GTAU[0]
    if np.any(tiny):
        out[tiny] = row[0] * (tau[tiny] / _GTAU[0]) ** 2
    return out


def block_exponent(steps, tau):
    """S_b(tau) = sum_s (chic2/chia2) G(tau/sigma_ref * sqrt(chia2)),
    absolute-chic2 normalization: sigma_ref from the exported thp2 (the Q
    matrix), chic2/chia2 from first principles. k=0 closure then tests
    Geant4's realized scattering against Moliere theory directly.
    steps: (n, NPARS) = [effZ, effA, xg, pGeV, beta, thp2, dOverX0]."""
    thp2 = steps[:, 5].astype(np.float64)
    tot = thp2.sum()
    if tot <= 0.:
        return None
    sig = np.sqrt(tot)
    S = np.zeros(len(tau))
    for s in range(len(steps)):
        if thp2[s] <= 0.:
            continue
        chic2, chia2, thff2 = moliere_params(*steps[s, :5])
        if chic2 <= 0. or chia2 <= 0.:
            continue
        S += (chic2 / chia2) * gshape(tau / sig * np.sqrt(chia2),
                                      ymax=np.sqrt(thff2 / chia2))
    return S


def extract(files, args):
    z2s, hs, Ss, reffs = [], [], [], []
    pt = None
    nblocks = 0
    for fn in files[:args.ntasks]:
        f = uproot.open(fn)
        if "tree" not in f:
            continue
        if pt is None:
            pt = f["runtree"]["parmtype"].array(library="np")
        t = f["tree"]
        a = t.arrays(["globalidxv", "gradchisqv", "gradllv", "hesspackedv",
                      "msmoliidx", "msmoliv"], library="np")
        for ic in range(len(a["globalidxv"])):
            gi = np.asarray(a["globalidxv"][ic])
            tp = pt[gi]
            m10 = np.where(tp == 10)[0]
            if not len(m10):
                continue
            nu = np.asarray(a["gradllv"][ic], dtype=np.float64)
            q = -np.asarray(a["gradchisqv"][ic], dtype=np.float64)
            hp = np.asarray(a["hesspackedv"][ic], dtype=np.float64)
            n = len(gi)
            rr = np.arange(n)
            hdiag = hp[(rr * n - rr * (rr - 1) // 2).astype(np.int64)]
            uidx = np.asarray(a["msmoliidx"][ic])
            uv = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            # stride detection: 7 floats/step (260724_moli) or 8 (+stepGroup)
            stride = len(uv) // max(len(uidx), 1)
            uv = uv.reshape(-1, stride)
            for j in m10:
                h = nu[j]
                if not (0. < h < 1.) or q[j] < 0.:
                    continue
                steps = uv[uidx == gi[j]]
                if not len(steps):
                    continue
                # effective rank of the block quadratic form from the
                # Fisher diagonal (c = 1 exactly, validated in cf0)
                reff = max(h * h / hdiag[j], 1.0) if hdiag[j] > 0. else 1.0
                # per-dof variance is 1/r (E[z^2]=1 pooled over r dofs), so
                # every per-dof CF argument carries 1/sqrt(r)
                S = block_exponent(steps, np.sqrt(h / reff) * TGRID)
                if S is None:
                    continue
                z2s.append(q[j] / h)
                hs.append(h)
                Ss.append(S)
                reffs.append(reff)
                nblocks += 1
            if nblocks >= args.max_blocks:
                break
        logger.info(f"{fn.split('/')[-2]}: cumulative {nblocks} blocks")
        if nblocks >= args.max_blocks:
            break
    np.savez_compressed(args.cache, z2=np.array(z2s), h=np.array(hs),
                        S=np.array(Ss), reff=np.array(reffs), tgrid=TGRID)
    logger.info(f"wrote {args.cache} ({nblocks} blocks)")


def model_stat(u, k, sest, S, h, tgrid, reff=None):
    """Dimension-r radial Weierstrass transform per entry:
      E[e^{-u z^2}]_b = Int_0^inf w_r(t) phi_b(t) dt,
      w_r(t) = 2 t^{r-1} e^{-t^2/4u} / ((4u)^{r/2} Gamma(r/2)),
    exact for any r in the Gaussian limit (gives (1+2u)^{-r/2}); the block's
    isotropic per-dof exponent S is shared across its r_eff dofs (J0 is the
    2D deflection CF, so r=2 is the exact two-projection case and non-integer
    r is the two-moment interpolation of the eigenvalue spectrum)."""
    from scipy.special import gammaln
    ek = np.exp(k)
    if reff is None:
        r = np.ones(len(h))
    else:
        r = reff
    # per-dof Gaussian estimation part also carries variance 1/r
    phi = np.exp(ek * S) * np.exp(-0.5 * sest * ((1. - h) / r)[:, None] * tgrid[None, :] ** 2)
    # avoid t=0 singularity for r<1 is not needed (r >= 1); t^{r-1} at t=0
    # is 1 for r=1 and 0 for r>1 -- handle t=0 node explicitly
    t = tgrid.copy()
    t[0] = 1e-12
    logw = (np.log(2.) + (r[:, None] - 1.) * np.log(t[None, :])
            - t[None, :] ** 2 / (4. * u)
            - 0.5 * r[:, None] * np.log(4. * u)
            - gammaln(0.5 * r)[:, None])
    vals = np.trapz(np.exp(logw) * phi, tgrid, axis=1)
    return np.mean(np.clip(vals, 0., 1.))


def fit(args, outdir):
    d = np.load(args.cache)
    z2, h, S, tgrid = d["z2"], d["h"], d["S"], d["tgrid"]
    reff = d["reff"] if "reff" in d.files else None
    logger.info(f"{len(z2)} blocks; h median {np.median(h):.3g} "
                f"[{np.percentile(h,16):.3g}, {np.percentile(h,84):.3g}]")
    Fd = np.array([np.mean(np.exp(-u * z2)) for u in args.probes])
    sF = np.array([np.std(np.exp(-u * z2)) / np.sqrt(len(z2)) for u in args.probes])

    print(f"{'u':>6s} {'F_data':>10s} {'F_model(0,1)':>12s}")
    for u, fd in zip(args.probes, Fd):
        print(f"{u:6.2f} {fd:10.6f} {model_stat(u, 0., 1., S, h, tgrid, reff):12.6f}")

    def chi2(p):
        return np.sum(((Fd - np.array([model_stat(u, p[0], p[1], S, h, tgrid, reff)
                                       for u in args.probes])) / sF) ** 2)

    res = minimize(chi2, [0., 0.95], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3})
    k, sest = res.x
    eps = np.array([1e-2, 2e-3])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(i, 2):
            di = np.zeros(2); di[i] = eps[i]
            dj = np.zeros(2); dj[j] = eps[j]
            H[i, j] = H[j, i] = (chi2(res.x + di + dj) - chi2(res.x + di - dj)
                                 - chi2(res.x - di + dj) + chi2(res.x - di - dj)) \
                                / (4 * eps[i] * eps[j])
    cov = 2. * np.linalg.inv(H)
    errs = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (errs[0] * errs[1])
    ndf = len(args.probes) - 2
    print(f"\njoint fit: k = {k:.4f} +- {errs[0]:.4f}   "
          f"s_est = {sest:.4f} +- {errs[1]:.4f}   rho = {rho:.2f}   "
          f"chi2/ndf = {res.fun:.1f}/{ndf}")
    for u, fd in zip(args.probes, Fd):
        print(f"   u={u:5.2f}  F_data={fd:.6f}  "
              f"F_joint={model_stat(u, k, sest, S, h, tgrid, reff):.6f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    kk = np.linspace(k - 0.3, k + 0.3, 25)
    prof = []
    for kv in kk:
        r1 = minimize(lambda s: chi2([kv, s[0]]), [sest], method="Nelder-Mead",
                      options={"xatol": 1e-4})
        prof.append(r1.fun)
    ax.plot(kk, np.array(prof) - res.fun, color="#2ca02c", lw=2)
    ax.axhline(1., color="k", ls="--", lw=1)
    ax.set_xlabel(r"MS material scale $k$")
    ax.set_ylabel(r"$\Delta\chi^2$ (s$_{est}$ profiled)")
    plot_tools.add_decor(ax, "CMS", "Work in progress (B→J/ψ+X MC gen closure)",
                         data=False, lumi=None, loc=2)
    name = "_".join(filter(None, ["cf_ms_profile", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    np.savez(os.path.join(outdir, "cf_ms_results.npz"),
             k=k, sest=sest, errs=errs, rho=rho, chi2=res.fun,
             probes=np.array(args.probes), Fd=Fd)


def main():
    args = parse_args()
    if args.extract:
        files = sorted(glob.glob(args.input))
        if not files:
            sys.exit(f"no files match {args.input}")
        extract(files, args)
    if args.fit:
        today = datetime.date.today().strftime("%y%m%d")
        outdir = output_tools.make_plot_dir(
            args.outpath or os.path.expanduser(
                f"~/public_html/calibration_studies/{today}_cf_ms_exact/"))
        fit(args, outdir)


if __name__ == "__main__":
    main()
