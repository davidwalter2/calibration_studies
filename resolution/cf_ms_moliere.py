"""Per-group Moliere-CF material fit: the absolute version of
fit_ms_material.py. The Gaussian statistic is replaced by the exact
screened-Rutherford compound model per eigen-dof, so the fitted k_g are
material scales with the tail included from first principles -- and the
test of the model is CONVENTION-INDEPENDENCE: fit at probe u = 1, then the
fitted model must reproduce the data at other u (the Gaussian-convention
fit provably drifts by ~x2 across u = 0.5..2).

Model per MS parameter entry b:
  z^2 = sum_j (lam_j / nu) zeta_j^2,  lam_j = exact block eigenvalues
  zeta_j = sqrt(lam_j) zeta_true + sqrt(1-lam_j) zeta_est   (per-dof hat)
  E[e^{-u z^2}] = prod_j E_j   (independence across eigen-dofs; exact in
                                the Gaussian limit)
  E_j = Int_0^inf dt e^{-t^2/(4 u_j)}/sqrt(pi u_j) * phi_j(t),
        u_j = u lam_j / nu
  phi_j(t) = exp( sum_g e^{k_g} S_{b,g}(sqrt(lam_j) t) )
             * exp(-0.5 s_est (1-lam_j) t^2)
  S_{b,g}(tau) = N_{b,g} * G(tau * abar_{b,g})    (universal Moliere shape,
        N = sum chic2/chia2 over the group's steps, abar^2 = N-weighted
        mean of chia2/sigma_ref^2; sigma_ref^2 = sum thp2 = the Q model)

Gaussian-limit check: S -> -lam t^2/2 recovers prod_j (1+2u lam~_j)^{-1/2},
exactly the closed-form statistic of fit_ms_material.py.

Dofs with lam < LAM_MOLI use the closed-form Gaussian factor (their
tail content enters as lam^{3/2} -- negligible); only the leading dofs get
the full Weierstrass integral.

Estimating equations at u = 1: per group f-weighted moments + one
unweighted moment for s_est; damped Newton with the ANALYTIC Jacobian
(the exponent is linear in e^{k_g}).
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

from wums import logging, output_tools, plot_tools
from cf_ms_exact import moliere_params, gshape
from fit_ms_material import group_names

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

NDOF = 6      # eigen-dofs kept per entry (descending lambda)
NGRP = 20     # group slots per entry
LAM_MOLI = 0.05  # below this, dof treated as pure Gaussian
TG = np.linspace(0.0, 12.0, 384)   # shared t-grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_closure_260724_eig_alpha999_fb01e7b7741/"
                           "task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=10)
    p.add_argument("--max-blocks", type=int, default=40000)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/ms_moliere_cache.npz"))
    p.add_argument("--groups-file",
                   default="/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
                           "Analysis/HitAnalyzer/data/materialGroups50.txt")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--probe", type=float, default=1.0)
    p.add_argument("--check-probes", type=float, nargs="+",
                   default=[0.2, 0.5, 2.0])
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def extract(files, args):
    z2s, nus = [], []
    lams = []          # (n, NDOF) descending, zero-padded
    gids = []          # (n, NGRP) group ids, -1 padded
    gN, gA2, gF = [], [], []   # (n, NGRP): N, abar^2 (sigma_ref units), thp2 frac
    gY = []                    # (n, NGRP): N-weighted FF cutoff ymax
    pt = None
    nblocks = 0
    for fn in files[:args.ntasks]:
        f = uproot.open(fn)
        if "tree" not in f:
            continue
        if pt is None:
            pt = f["runtree"]["parmtype"].array(library="np")
        t = f["tree"]
        a = t.arrays(["globalidxv", "gradchisqv", "gradllv",
                      "reseigidx", "reseigv", "msmoliidx", "msmoliv"],
                     library="np")
        for ic in range(len(a["globalidxv"])):
            gi = np.asarray(a["globalidxv"][ic])
            tp = pt[gi]
            m10 = np.where(tp == 10)[0]
            if not len(m10):
                continue
            nu = np.asarray(a["gradllv"][ic], dtype=np.float64)
            q = -np.asarray(a["gradchisqv"][ic], dtype=np.float64)
            eidx = np.asarray(a["reseigidx"][ic])
            ev = np.asarray(a["reseigv"][ic], dtype=np.float64).reshape(-1, 5)
            uidx = np.asarray(a["msmoliidx"][ic])
            uv = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            stride = len(uv) // max(len(uidx), 1)
            uv = uv.reshape(-1, stride)
            for j in m10:
                h = nu[j]
                if not (0. < h < 1.) or q[j] < 0.:
                    continue
                lam = np.sort(ev[eidx == gi[j]].ravel())[::-1]
                lam = lam[lam > 1e-6][:NDOF]
                if not len(lam):
                    continue
                steps = uv[uidx == gi[j]]
                if not len(steps):
                    continue
                thp2 = steps[:, 5]
                grp = steps[:, 7].astype(int)
                sig2 = thp2.sum()
                if sig2 <= 0.:
                    continue
                gl = np.full(NGRP, -1, dtype=np.int32)
                gn = np.zeros(NGRP)
                ga2 = np.zeros(NGRP)
                gy = np.zeros(NGRP)
                gf = np.zeros(NGRP)
                slots = {}
                for s in range(len(steps)):
                    if thp2[s] <= 0. or grp[s] < 0:
                        continue
                    chic2, chia2, thff2 = moliere_params(*steps[s, :5])
                    if chic2 <= 0. or chia2 <= 0.:
                        continue
                    g = grp[s]
                    if g not in slots:
                        if len(slots) >= NGRP:
                            continue
                        slots[g] = len(slots)
                        gl[slots[g]] = g
                    sl = slots[g]
                    N = chic2 / chia2
                    gn[sl] += N
                    ga2[sl] += N * (chia2 / sig2)
                    gy[sl] += N * np.sqrt(thff2 / chia2)   # FF cutoff (y units)
                    gf[sl] += thp2[s] / sig2
                sel = gn > 0.
                ga2[sel] /= gn[sel]
                gy[sel] /= gn[sel]
                lpad = np.zeros(NDOF)
                lpad[:len(lam)] = lam
                z2s.append(q[j] / h)
                nus.append(h)
                lams.append(lpad)
                gids.append(gl)
                gN.append(gn)
                gA2.append(ga2)
                gY.append(gy)
                gF.append(gf)
                nblocks += 1
            if nblocks >= args.max_blocks:
                break
        logger.info(f"{fn.split('/')[-2]}: cumulative {nblocks}")
        if nblocks >= args.max_blocks:
            break
    np.savez_compressed(args.cache,
                        z2=np.array(z2s), nu=np.array(nus),
                        lam=np.array(lams), gid=np.array(gids),
                        gN=np.array(gN), gA2=np.array(gA2), gY=np.array(gY),
                        gF=np.array(gF))
    logger.info(f"wrote {args.cache} ({nblocks} blocks)")


class MoliereModel:
    """Vectorized model + analytic k-Jacobian; probe u and the global
    screening-angle scale gamc passed per call. S is evaluated on the fly
    per group slot (no precomputed (nd, NGRP, nt) table -- that would be
    ~4 GB at full statistics and would freeze gamc)."""

    def __init__(self, d, ngroups):
        self.z2 = d["z2"]
        self.nu = d["nu"]
        self.lam = d["lam"]
        self.gid = d["gid"]
        self.gN = d["gN"]
        self.gA2 = d["gA2"]
        self.gY = d["gY"] if "gY" in d.files else None
        self.gF = d["gF"]
        self.n = len(self.z2)
        self.ngroups = ngroups
        self.moli_dofs = self.lam >= LAM_MOLI
        ie, ij = np.where(self.moli_dofs)
        self.idx_e, self.idx_j = ie, ij
        nd = len(ie)
        logger.info(f"{self.n} entries, {nd} Moliere dofs")
        # scalar G-argument base per (moli-dof, slot): sqrt(lam_j) * abar
        abar = np.sqrt(np.maximum(self.gA2[ie], 1e-30))     # (nd, NGRP)
        self.argbase = np.sqrt(self.lam[ie, ij])[:, None] * abar
        self.gidd = self.gid[ie]                            # (nd, NGRP)
        self.gNd = self.gN[ie]
        self.gYd = self.gY[ie] if self.gY is not None else None

    def _slot_S(self, sl, act, gamc):
        arg = (self.argbase[act, sl, None] * gamc) * TG[None, :]
        if self.gYd is None:
            return self.gNd[act, sl, None] * gshape(arg.ravel()).reshape(-1, len(TG))
        # bucket entries by their FF-cutoff table row (ymax varies per
        # entry; gshape's ymax argument is scalar per call). The cutoff is
        # in y units, so the gamc screening rescale divides it.
        out = np.empty((int(act.sum()), len(TG)))
        ym = self.gYd[act, sl] / gamc
        lg = np.log(np.clip(ym, 1e1, 1e7))
        rows = np.round((lg - np.log(1e1)) / (np.log(1e7 / 1e1) / 12)).astype(int)
        argact = arg
        for r in np.unique(rows):
            m = rows == r
            ymr = float(np.exp(np.log(1e1) + r * (np.log(1e7 / 1e1) / 12)))
            out[m] = gshape(argact[m].ravel(), ymax=ymr).reshape(-1, len(TG))
        return self.gNd[act, sl, None] * out

    def _phi_gauss_scale(self, k):
        ek = np.zeros(self.ngroups + 1)
        ek[:self.ngroups] = np.exp(k)
        ek[-1] = 1.
        gsafe = np.where(self.gid >= 0, self.gid, self.ngroups)
        return np.sum(self.gF * ek[gsafe], axis=1) + (1. - self.gF.sum(axis=1))

    def stat(self, k, sest, u, gamc=1., want_jac=False):
        n, ngroups = self.n, self.ngroups
        lam, nu = self.lam, self.nu
        sb = self._phi_gauss_scale(k)
        lamt = lam / nu[:, None]
        sd = lam * sb[:, None] + (1. - lam) * sest
        gmask = (lam > 0) & ~self.moli_dofs
        args = 1. + 2. * u * lamt * sd
        logM = -0.5 * np.sum(np.where(gmask, np.log(np.maximum(args, 1e-12)), 0.),
                             axis=1)
        ie = self.idx_e
        lamd = lam[ie, self.idx_j]
        uj = u * lamd / nu[ie]
        ek = np.exp(k)
        # first pass: accumulate the summed exponent
        nd = len(ie)
        Ssum = np.zeros((nd, len(TG)))
        for sl in range(NGRP):
            act = self.gidd[:, sl] >= 0
            if not act.any():
                continue
            Ssum[act] += ek[self.gidd[act, sl]][:, None] * self._slot_S(sl, act, gamc)
        phi = np.exp(Ssum - 0.5 * sest * (1. - lamd)[:, None] * TG[None, :] ** 2)
        w = np.exp(-TG[None, :] ** 2 / (4. * uj[:, None])) / np.sqrt(np.pi * uj)[:, None]
        Ej = np.clip(np.trapz(phi * w, TG, axis=1), 1e-12, 1.)
        np.add.at(logM, ie, np.log(Ej))
        F = np.exp(logM)
        if not want_jac:
            return F, None, None
        dF_dk = np.zeros((n, ngroups))
        dlog_dsd = -u * lamt * lam / np.maximum(args, 1e-12)
        dsb = np.sum(np.where(gmask, dlog_dsd, 0.), axis=1)
        for sl in range(NGRP):
            act = self.gid[:, sl] >= 0
            if act.any():
                g = self.gid[act, sl]
                np.add.at(dF_dk, (np.where(act)[0], g),
                          dsb[act] * self.gF[act, sl] * ek[g])
        # second pass for the Moliere k-Jacobian (recompute slot S)
        for sl in range(NGRP):
            act = self.gidd[:, sl] >= 0
            if not act.any():
                continue
            g = self.gidd[act, sl]
            dEj = np.trapz(phi[act] * w[act] * self._slot_S(sl, act, gamc),
                           TG, axis=1)
            np.add.at(dF_dk, (ie[act], g), ek[g] * dEj / Ej[act])
        dlog_dsest = np.sum(np.where(gmask,
                                     -u * lamt * (1. - lam)
                                     / np.maximum(args, 1e-12), 0.), axis=1)
        dEs = np.trapz(phi * w * (-0.5 * (1. - lamd)[:, None] * TG[None, :] ** 2),
                       TG, axis=1)
        np.add.at(dlog_dsest, ie, dEs / Ej)
        return F, dF_dk * F[:, None], dlog_dsest * F


def fit(args, outdir):
    d = np.load(args.cache)
    names = group_names(args.groups_file)
    ngroups = int(max(d["gid"].max() + 1, 42))
    mdl = MoliereModel(d, ngroups)
    z2 = d["z2"]
    W = np.zeros((mdl.n, ngroups))
    for sl in range(NGRP):
        act = d["gid"][:, sl] >= 0
        W[np.where(act)[0], d["gid"][act, sl]] = d["gF"][act, sl]
    fmean = W.mean(axis=0)
    active = fmean > 1e-4
    na = int(active.sum())
    logger.info(f"{na}/{ngroups} groups active")

    # multi-u moment conditions: the tail-vs-core (k vs s_est) separation
    # IS the u-dependence, so a single-u fit is degenerate by construction
    # (verified: it runs away exactly like the ioni k/s_est valley).
    ufit = [0.2, 1.0, 2.0]
    Fdat = {u: np.exp(-u * z2) for u in ufit}
    # moment errors for the weighting
    sigE = {}
    for u in ufit:
        var = np.var(Fdat[u])
        sigE[u] = np.sqrt(np.maximum((W ** 2).sum(axis=0) * var, 1e-30))
        sigE[u] = np.concatenate([sigE[u], [np.sqrt(var / mdl.n)]])

    k = np.zeros(ngroups)
    sest = 1.0
    for u in ufit:
        F0, _, _ = mdl.stat(k, sest, u)
        print(f"pre-fit u={u}: <F_data>={Fdat[u].mean():.6f} "
              f"<F_model(0,1)>={F0.mean():.6f}")

    # ---------- stage 1: global shape (kbar, sest, gamc) on mean stats ----
    from scipy.optimize import minimize as _minimize
    uglob = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    Fbar = {u: float(np.exp(-u * z2).mean()) for u in uglob}
    sbar = {u: float(np.std(np.exp(-u * z2)) / np.sqrt(mdl.n)) for u in uglob}

    def chi2glob(p):
        kb, se, gc = p
        if not (0.2 < se < 5. and 0.3 < gc < 3.):
            return 1e12
        c = 0.
        for u in uglob:
            Fm = float(mdl.stat(np.full(ngroups, kb), se, u, gc)[0].mean())
            c += ((Fbar[u] - Fm) / sbar[u]) ** 2
        return c

    res1 = _minimize(chi2glob, [0., 0.95, 1.0], method="Nelder-Mead",
                     options={"xatol": 2e-3, "fatol": 0.5, "maxfev": 250})
    kbar, sest, gamc = res1.x
    print(f"stage 1 (global shape, {len(uglob)} probes): kbar = {kbar:.4f}  "
          f"s_est = {sest:.4f}  gamma_c = {gamc:.4f}  "
          f"chi2 = {res1.fun:.1f}/{len(uglob) - 3}")

    # ---------- stage 2: per-group deviations, shape frozen ---------------
    sel = active
    nsel = int(sel.sum())
    k = np.full(ngroups, kbar)
    PRIOR = 1.0  # N(kbar, PRIOR) conditioning ridge on the deviations
    for it in range(40):
        Es, Js = [], []
        for u in ufit:
            F, dFk, _ = mdl.stat(k, sest, u, gamc, want_jac=True)
            r = Fdat[u] - F
            E = (W.T @ r) / sigE[u][:ngroups]
            J = (-W.T @ dFk) / sigE[u][:ngroups, None]
            Es.append(E[sel])
            Js.append(J[np.ix_(sel, sel)])
        Estack = np.concatenate(Es + [(k[sel] - kbar) / PRIOR])
        Jstack = np.concatenate(Js + [np.eye(nsel) / PRIOR], axis=0)
        H = Jstack.T @ Jstack
        g = Jstack.T @ Estack
        ridge = 1e-8 * np.max(np.abs(np.diag(H)))
        dxa = np.clip(np.linalg.solve(H + ridge * np.eye(nsel), -g), -0.3, 0.3)
        k[sel] += dxa
        k = np.clip(k, kbar - 3., kbar + 3.)
        if np.max(np.abs(dxa)) < 1e-5:
            break

    chi2 = float(Estack @ Estack)
    ndf = 3 * nsel - nsel
    cov = np.linalg.inv(H + ridge * np.eye(nsel))
    errs = np.zeros(ngroups)
    errs[sel] = np.sqrt(np.diag(cov))
    serr = gerr = 0.

    print(f"\nstage 2 converged (iter {it}): moment chi2/ndf = {chi2:.1f}/{ndf}")
    print("per-group deviations from kbar (material closure; global shape "
          "absorbed in stage 1; prior sigma = 1):")
    print(f"{'group':32s} {'k_g-kbar':>9s} {'err':>8s} {'pull':>7s} {'<f>':>8s}")
    dk = k - kbar
    order = [g for g in np.argsort(-np.abs(dk / np.maximum(errs, 1e-12)))
             if active[g]]
    for g in order[:18]:
        print(f"{names.get(g, f'group{g}')[:32]:32s} {dk[g]:9.4f} "
              f"{errs[g]:8.4f} {dk[g]/max(errs[g],1e-12):7.1f} {fmean[g]:8.4f}")

    # convention-independence check on HELD-OUT probes (not in the fit)
    print(f"\nconvention check (fit probes {ufit}):")
    print(f"{'u':>6s} {'F_data':>10s} {'F_model(k-hat)':>14s} {'F_model(0)':>11s} {'held-out':>9s}")
    for uc in sorted(set(args.check_probes + ufit)):
        Fd = np.exp(-uc * z2).mean()
        Fm = mdl.stat(k, sest, uc, gamc)[0].mean()
        Fm0 = mdl.stat(np.zeros(ngroups), 1., uc, 1.)[0].mean()
        held = "  *" if uc not in ufit else ""
        print(f"{uc:6.2f} {Fd:10.6f} {Fm:14.6f} {Fm0:11.6f} {held:>9s}")

    np.savez(os.path.join(outdir, "ms_moliere_results.npz"),
             k=k, err=errs, sest=sest, serr=serr, gamc=gamc, gerr=gerr,
             kbar=kbar, active=active)
    output_tools.write_logfile(outdir, "ms_moliere_fit", args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))


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
                f"~/public_html/calibration_studies/{today}_ms_moliere/"))
        fit(args, outdir)


if __name__ == "__main__":
    main()
