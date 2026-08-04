"""MS-tied material fit with the ECF convention: per-material-group MS
variance scales k_g fitted from the MS block residuals with the
tail-robust bounded-influence statistic E[e^{-u z^2}] at fixed probe u
(the convention), using
  - the EXACT per-entry eigenvalue spectra (reseigidx/reseigv export):
    Gaussian model statistic in closed form,
      E[e^{-u z^2}]_b = prod_j (1 + 2 u lam~_j * s_b(k))^{-1/2},
    lam~_j = lambda_j / sum(lambda) (union over the parameter's legs);
  - per-block group fractions f_{b,g} from the Moliere export's stepGroup
    + thp2 shares (variance-additive);
  - the shrinkage response of the observable variance to a truth-side
    scale: s_b(k) = 1 + h_b * (sum_g f_{b,g} e^{k_g} - 1)   (h = nu).

Estimating equations (one per group, f-weighted moments):
  E_g({k}) = sum_b f_{b,g} [ e^{-u z2_b} - model_b({k}) ] = 0
solved by damped Newton with a numeric Jacobian. Sandwich errors.

This is the deliverable of the original 2022 program: material constrained
by its multiple-scattering effect, with the non-Gaussian tails controlled
by the estimator convention instead of corrupting the fit.
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

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

NEIG = 5

# Material classes for the offline group refinement: effZ is exported per
# step, so a structural group can be split by CONSTITUENT MATERIAL without
# any C++ change or re-production. Each (structural group, Z-class) pair
# that carries enough MS weight becomes its own parameter -> single-material
# subgroups, where the one-scale treatment (mean loss, straggling and MS all
# linear in the same rho*d) is EXACT rather than approximate.
ZEDGES = np.array([0., 7., 10., 13., 15., 17., 21., 1e9])
NZ = len(ZEDGES) - 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_closure_260724_eig_alpha999_fb01e7b7741/"
                           "task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=25)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/ms_material_cache.npz"))
    p.add_argument("--groups-file",
                   default="/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
                           "Analysis/HitAnalyzer/data/materialGroups50.txt")
    p.add_argument("--split-materials", action="store_true",
                   help="refine the structural groups by constituent material "
                        "(effZ class): each (group, Z-class) becomes its own "
                        "parameter. Tests whether the group deviations track "
                        "MATERIAL or STEP REGIME.")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--probe", type=float, default=1.0,
                   help="ECF probe u (THE convention; report 0.5/1/2 for stability)")
    p.add_argument("--probes-scan", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def group_names(groups_file):
    names = {}
    if groups_file and os.path.exists(groups_file):
        for line in open(groups_file):
            if line.startswith("RULE"):
                f = line.split("\t")
                names[int(f[1])] = f[2]
    return names


def group_priors(groups_file):
    """prior sigma per group id (uncertainty-class priors, axis 3)."""
    pr = {}
    if groups_file and os.path.exists(groups_file):
        for line in open(groups_file):
            if line.startswith("RULE"):
                f = line.split("\t")
                pr[int(f[1])] = float(f[10])
    return pr


def extract(files, args):
    """Per MS parameter entry: z2, h, eigen spectrum (union over legs,
    normalized), group fractions (from thp2 shares)."""
    z2s, hs = [], []
    lams = []   # (n, NEIG*2) padded normalized spectra (legs can double it)
    fracs = []  # (n, ngroups)
    pt = None
    ngroups = None
    nblocks = 0
    NLAM = 2 * NEIG
    for fn in files[:args.ntasks]:
        f = uproot.open(fn)
        if "tree" not in f:
            continue
        if pt is None:
            pt = f["runtree"]["parmtype"].array(library="np")
            ngroups = int(np.sum(pt == 15))
            nfit_groups = ngroups * NZ if args.split_materials else ngroups
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
            ev = np.asarray(a["reseigv"][ic], dtype=np.float64).reshape(-1, NEIG)
            uidx = np.asarray(a["msmoliidx"][ic])
            uv = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            stride = len(uv) // max(len(uidx), 1)
            uv = uv.reshape(-1, stride)
            for j in m10:
                h = nu[j]
                if not (0. < h < 1.) or q[j] < 0.:
                    continue
                lam = ev[eidx == gi[j]].ravel()
                lam = lam[lam > 0.]
                if not len(lam) or len(lam) > NLAM:
                    lam = lam[:NLAM] if len(lam) else lam
                if not len(lam):
                    continue
                steps = uv[uidx == gi[j]]
                if not len(steps):
                    continue
                thp2 = steps[:, 5]
                grp = steps[:, 7].astype(int)
                tot = thp2.sum()
                if tot <= 0.:
                    continue
                fr = np.zeros(nfit_groups)
                ok = (grp >= 0) & (grp < ngroups)
                if args.split_materials:
                    zc = np.clip(np.searchsorted(ZEDGES, steps[:, 0], side="right") - 1,
                                 0, NZ - 1)
                    ids = grp[ok] * NZ + zc[ok]
                else:
                    ids = grp[ok]
                np.add.at(fr, ids, thp2[ok])
                fr /= tot
                lamn = np.zeros(NLAM)
                lamn[:len(lam)] = lam / lam.sum()
                z2s.append(q[j] / h)
                hs.append(h)
                lams.append(lamn)
                fracs.append(fr)
                nblocks += 1
        logger.info(f"{fn.split('/')[-2]}: cumulative {nblocks} blocks")
    gn = group_names(args.groups_file)
    if args.split_materials:
        labels = [f"{gn.get(g, f'grp{g}')}|Z{ZEDGES[c]:.0f}-{ZEDGES[c+1]:.0f}"
                  .replace("-1000000000", "+")
                  for g in range(ngroups) for c in range(NZ)]
    else:
        labels = [gn.get(g, f"grp{g}") for g in range(ngroups)]
    np.savez_compressed(args.cache, z2=np.array(z2s), h=np.array(hs),
                        lam=np.array(lams), frac=np.array(fracs),
                        labels=np.array(labels))
    logger.info(f"wrote {args.cache} ({nblocks} blocks, {ngroups} groups)")


def model_F(u, k, lam, h, frac):
    """Closed-form Gaussian statistic per entry with shrinkage response."""
    s = 1. + h * (frac @ np.exp(k) - 1.)          # (nb,)
    args = 1. + 2. * u * lam * s[:, None]          # (nb, NLAM); lam zero-padded
    args = np.where(lam > 0., args, 1.)
    return np.exp(-0.5 * np.sum(np.log(args), axis=1))


def fit(args, outdir):
    d = np.load(args.cache)
    z2, h, lam, frac = d["z2"], d["h"], d["lam"], d["frac"]
    ngroups = frac.shape[1]
    names = ({i: str(l) for i, l in enumerate(d["labels"])} if "labels" in d.files
             else group_names(args.groups_file))
    logger.info(f"{len(z2)} blocks, {ngroups} groups; h median {np.median(h):.3f}")

    # groups with (near-)zero MS occupancy are unconstrained -- freeze them
    # at zero instead of letting them run away
    fmean = frac.mean(axis=0)
    active = fmean > 1e-4
    logger.info(f"{active.sum()}/{ngroups} groups active "
                f"(frozen: {[g for g in range(ngroups) if not active[g]]})")

    # uncertainty-class priors (axis 3): k ~ N(0, sigma_g) from the groups
    # file -- pins known-budget classes (air), regularizes sparse ones
    prmap = group_priors(args.groups_file)
    prior_w = np.array([1. / prmap.get(g, 0.2) ** 2 for g in range(ngroups)])

    results = {}
    for u in args.probes_scan:
        F = np.exp(-u * z2)
        k = np.zeros(ngroups)
        W = frac  # (nb, ng) estimating-equation weights
        for it in range(60):
            r = F - model_F(u, k, lam, h, frac)
            E = W.T @ r
            # numeric Jacobian dE/dk
            J = np.zeros((ngroups, ngroups))
            eps = 1e-3
            for g in range(ngroups):
                kp = k.copy(); kp[g] += eps
                J[:, g] = W.T @ (-(model_F(u, kp, lam, h, frac)
                                   - model_F(u, k, lam, h, frac)) / eps)
            # damped Newton on the active subspace, with the class priors
            ridge = 1e-6 * np.max(np.abs(np.diag(J)))
            Ja = (J + np.diag(prior_w))[np.ix_(active, active)] \
                 + ridge * np.eye(int(active.sum()))
            dk = np.zeros(ngroups)
            dk[active] = np.linalg.solve(Ja, -(E + prior_w * k)[active])
            dk = np.clip(dk, -0.5, 0.5)
            k += dk
            if np.max(np.abs(dk)) < 1e-5:
                break
        # sandwich errors: Var(E) = W^T diag(Var(F)) W (block correlations
        # within a track neglected -- v1 caveat)
        Sm = (W * ((F - model_F(u, k, lam, h, frac)) ** 2)[:, None]).T @ W
        na = int(active.sum())
        Ja = (J + np.diag(prior_w))[np.ix_(active, active)] + ridge * np.eye(na)
        Jinv = np.linalg.inv(Ja)
        cov = Jinv @ Sm[np.ix_(active, active)] @ Jinv.T
        errs = np.zeros(ngroups)
        errs[active] = np.sqrt(np.diag(cov))
        results[u] = (k.copy(), errs)
        print(f"\n=== u = {u} (iter {it}) ===")
        order = [g for g in np.argsort(-np.abs(k / np.maximum(errs, 1e-12)))
                 if active[g]]
        print(f"{'group':32s} {'k_g':>9s} {'err':>8s} {'pull':>7s} {'<f>':>8s}")
        for g in order[:15]:
            nm = names.get(g, f"group{g}")[:32]
            print(f"{nm:32s} {k[g]:9.4f} {errs[g]:8.4f} "
                  f"{k[g]/max(errs[g],1e-12):7.1f} {frac[:,g].mean():8.4f}")

    # stability plot: k_g vs probe, only groups constrained to err < 0.5
    u0 = args.probe
    k0, e0 = results[u0]
    good = active & (e0 < 0.5) & (e0 > 0.)
    xg = np.where(good)[0]
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(xg))
    for u, (k, e) in results.items():
        ax.errorbar(x + 0.12 * (u - u0), k[xg], yerr=e[xg], ls="", marker="o",
                    markersize=5, label=f"u = {u}")
    ax.axhline(0., color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(g, str(g))[:16] for g in xg],
                       rotation=75, ha="right", fontsize=11)
    ax.set_ylabel(r"MS variance scale $k_g$ (ECF convention)")
    ax.legend(fontsize=15)
    plot_tools.add_decor(ax, "CMS", "Work in progress (B→J/ψ+X MC gen closure)",
                         data=False, lumi=None, loc=2)
    name = "_".join(filter(None, ["ms_material_kg", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    np.savez(os.path.join(outdir, "ms_material_results.npz"),
             **{f"k_u{u}": v[0] for u, v in results.items()},
             **{f"err_u{u}": v[1] for u, v in results.items()})


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
                f"~/public_html/calibration_studies/{today}_ms_material/"))
        fit(args, outdir)


if __name__ == "__main__":
    main()
