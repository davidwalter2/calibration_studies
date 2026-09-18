#!/usr/bin/env python3
"""Does the ANGULAR dependence of the track covariance belong in `a_res`?

`MASSCFTERM_SPEC`'s sigma-artefact coefficient is a regression slope,

    a = Cov(dm, d sigma_m)/Var(dm) = (J^T Sigma G)/(J^T Sigma J),
    J = grad_u m,  G = grad_u sigma_m,  u = (plus, minus) x (q/p, lambda, phi)

and the closed form in use, `a = (1 + f_hit) sigma/m`, is the special case in
which BOTH gradients lie along the two momenta.  The K_S -> pi pi channel
breaks that: 70 % of the mass variance is the opening angle.  Three candidate
forms and the truth are compared here, all as the dimensionless prefactor
`A = a m/sigma` (the quantity the spec predicts to be `1 + f_hit`):

  mom   A = 1 + f_hit                         the J/psi form, as shipped
  ang   A = (1-f_ang)[(1+f_hit)(1-f_ang) + f_ang]
                                              momentum gradients only:
                                              an angular fluctuation is
                                              assumed to move m but not sigma
  full  A = m (J^T Sigma G)/sigma^3           G from the exact mass Hessian
                                              plus a model of dSigma/du
  truth measured on the MC, `ares_truth`

`full` is built in `ares_grad`: `d(sigma^2)/du = 2 (dJ/du)^T Sigma J +
J^T (dSigma/du) J`.  The first term is EXACT and needs no detector model --
it is the reason `ang` fails, because `J` itself depends on the angles, so
`sigma_m` responds to an angular fluctuation even at perfectly fixed `Sigma`.
The second term is the detector's own response and is taken either from the
path-length scaling (`msmodel`) or from the population (`population`).

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 ares_angles.py --cov runs/kscov_all.npz --pairs runs/kspairs_all.npz \\
      --masses pion --tag ares_angles --write runs/kspairs_ares.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import ares_kin as AK     # noqa: E402
import ares_grad as AG    # noqa: E402
import ares_truth as AT   # noqa: E402

MASSES = {"pion": (AK.M_PI, AK.M_PI), "muon": (AK.M_MU, AK.M_MU)}


# ----------------------------------------------------------------- loading
def load(cov, pairs, masses, log=print):
    c = dict(np.load(cov))
    p = dict(np.load(pairs)) if pairs else None
    if p is not None:
        for k in ("run", "lumi", "event"):
            if not np.array_equal(p[k], c[k]):
                raise SystemExit(f"the two caches are not row-aligned on {k}")
        if not np.array_equal(p["sigma"], c["Jpsi_sigmamass"].astype(np.float64)):
            raise SystemExit("the two caches are not row-aligned on sigma")
    u = AK.state_from_cache(c)
    C = AK.unpack_cov(c["Jpsi_covrefmom"])
    sig = c["Jpsi_sigmamass"].astype(np.float64)
    m = c["Jpsi_mass"].astype(np.float64)
    m0, J, H = AK.jac_hess(u, masses)
    Jex = c["Jpsi_jacrefmom"].astype(np.float64)
    # ---- the three export gates -------------------------------------
    r = np.abs(J / np.where(Jex != 0, Jex, np.nan) - 1.0)
    log(f"  gate 1  analytic dm/du vs Jpsi_jacrefmom: max |ratio-1| "
        f"{np.nanmax(np.nanmedian(r, axis=0)):.2e} (median over candidates, "
        f"per component)")
    s2 = np.einsum("ni,nij,nj->n", Jex, C, Jex)
    log(f"  gate 2  sqrt(J C J^T)/Jpsi_sigmamass: med {np.median(np.sqrt(s2)/sig):.6f}"
        f" q01 {np.quantile(np.sqrt(s2)/sig, 0.01):.6f} "
        f"q99 {np.quantile(np.sqrt(s2)/sig, 0.99):.6f}")
    Jk = Jex.copy()
    Jk[:, [1, 2, 4, 5]] = 0.0
    fang = 1.0 - np.einsum("ni,nij,nj->n", Jk, C, Jk) / sig ** 2
    if p is not None and "fang" in p:
        log(f"  gate 3  f_ang rebuilt from the 6x6 vs the exported Jpsi_fang: "
            f"max |diff| {np.abs(fang - p['fang']).max():.2e}")
    log(f"  mass from the {list(MASSES)[0] if masses == MASSES['pion'] else 'muon'} "
        f"hypothesis vs Jpsi_mass: median |rel diff| "
        f"{np.median(np.abs(m0 - m)/m):.2e}")
    return c, p, u, C, J, H, sig, m, fang


# ------------------------------------------------------------- the forms
def forms(c, p, u, C, J, H, sig, m, fang, fhit, args, log=print):
    """Per-candidate `A = a m/sigma` for every form, plus diagnostics."""
    out = {}
    out["mom"] = 1.0 + fhit
    out["ang"] = (1.0 - fang) * ((1.0 + fhit) * (1.0 - fang) + fang)
    out["ang_simple"] = (1.0 - fang) * (1.0 + fhit)

    aux = dict(phivtx=np.arctan2(c["Jpsi_y"], c["Jpsi_x"]),
               nvp=c["Muplus_nvalid"].astype(np.float64),
               nvm=c["Muminus_nvalid"].astype(np.float64),
               rvtx=np.hypot(c["Jpsi_x"], c["Jpsi_y"]),
               zvtx=c["Jpsi_z"].astype(np.float64))
    L = AG.leg_table(u, C, aux)
    n = len(sig)

    # ---- variant 1: the analytic path-length scaling --------------------
    glnp_ms, glam_ms = AG.ms_model(u)
    # ---- variant 2: the population's own log-derivatives ----------------
    ctrl = [L["nv"], L["lnr"], L["zv"], np.abs(L["zv"]), L["nv"] ** 2,
            L["lnr"] ** 2, L["dphi"], L["dphi"] ** 2]
    phiabs = np.concatenate([u[:, 2], u[:, 5]])
    glnp_pop = np.empty((n, 6))
    glam_pop = np.empty((n, 6))
    log("\n  population log-derivatives of the covariance diagonal "
        "(legs pooled, controls partialled out):")
    log(f"    {'element':10s} {'dlnS/dlnp':>11s} {'dlnS/dlam / tan(lam)':>22s} "
        f"{'|azimuth coef| q95':>20s}")
    for k, a in enumerate(("qop", "lam", "phi")):
        d1, d2, info = AG.local_linear(
            L["lnS_" + a], L["lnp"], L["lam"], ctrl, nb1=args.nb_p,
            nb2=args.nb_lam, minn=args.minn,
            extra=[np.cos(phiabs), np.sin(phiabs)])
        # ENFORCE the detector's z-symmetry: dlnSigma/dlambda must be ODD in
        # lambda.  The local slopes are fitted to `c tan(lambda)` through the
        # origin, on the legs where `tan(lambda)` has leverage; the fit is what
        # is used, so a cell straddling lambda = 0 cannot contribute an even
        # component that the symmetry forbids.
        t = np.tan(L["lam"])
        sel = np.isfinite(d2) & (np.abs(L["lam"]) > args.lam_min)
        cc = float((d2[sel] * t[sel]).sum() / (t[sel] ** 2).sum())
        bb = np.where(np.isfinite(d1), d1, np.nanmedian(d1))
        az = np.abs(np.array([i[7] for i in info]))
        log(f"    Sigma_{a:5s} {np.nanmedian(d1):11.3f} {cc:22.3f} "
            f"{np.quantile(az, 0.95):20.3f}")
        for l in (0, 1):
            glnp_pop[:, 3 * l + k] = bb[l * n:(l + 1) * n]
            glam_pop[:, 3 * l + k] = cc * np.tan(u[:, 3 * l + 1])
    out["_glnp_ms"], out["_glam_ms"] = glnp_ms, glam_ms
    out["_glnp_pop"], out["_glam_pop"] = glnp_pop, glam_pop

    def A_of(glnp, glam, tag):
        g, Kv, Rv = AG.grad_sigma(u, C, J, H, glnp, glam, sig)
        a = AG.slope(J, C, g, sig)
        out["_grad_" + tag] = g
        return a * m / sig

    out["kin"] = A_of(np.zeros((n, 6)), np.zeros((n, 6)), "kin")
    out["full_ms"] = A_of(glnp_ms, glam_ms, "full_ms")
    out["full_pop"] = A_of(glnp_pop, glam_pop, "full_pop")
    # the angular-covariance piece ISOLATED: the same model with the lambda
    # response of Sigma switched off (the kinematic angular term stays -- it is
    # not a model and cannot be switched off)
    out["full_ms_nolam"] = A_of(glnp_ms, np.zeros((n, 6)), "full_ms_nolam")
    out["full_pop_nolam"] = A_of(glnp_pop, np.zeros((n, 6)), "full_pop_nolam")
    # and with the whole ANGULAR SECTOR of grad(sigma) removed, which is what
    # form `ang` assumes
    g = out["_grad_full_ms"].copy()
    g[:, [1, 2, 4, 5]] = 0.0
    out["full_ms_momonly"] = AG.slope(J, C, g, sig) * m / sig
    return out


# ---------------------------------------------------------------- truth
def toy(args, ctrl, tv, sig, mgen):
    """Closure of the estimator itself: a sample with a KNOWN slope.

    `sigma_bar` is the real per-candidate width times a log-normal factor that
    stands in for the hit pattern (the residual scatter the controls cannot
    remove, 0.33-0.39 in the data); the mass fluctuation is Gaussian at
    `sigma_bar`; and `sigma_hat = sigma_bar + a dm` by construction.  The
    estimator is then run exactly as on the data.
    """
    n = len(sig)
    rng = np.random.default_rng(1234)
    A = float(args.toy)
    cuts = (("no cut", lambda dm, sh, sb: np.ones(n, bool)),
            (f"|m-m_gen| < {args.max_resid} GeV",
             lambda dm, sh, sb: np.abs(dm) < args.max_resid),
            ("|m-m_gen| < 3 sigma_bar (truth width)",
             lambda dm, sh, sb: np.abs(dm) < 3 * sb),
            ("|m-m_gen| < 3 sigma (PULL cut, biased)",
             lambda dm, sh, sb: np.abs(dm) < 3 * sh))
    print(f"\n--- estimator closure, A_true = {A:.3f}, {args.toy_nrep} "
          f"replicas, pattern noise {args.toy_noise:g} ---")
    for tag, cut in cuts:
        vals = []
        keep = 1.0
        for _ in range(args.toy_nrep):
            sb = sig * np.exp(rng.normal(0.0, args.toy_noise, n))
            dm = rng.normal(0.0, 1.0, n) * sb
            sh = sb + A * sb / mgen * dm
            ok = cut(dm, sh, sb)
            keep = float(ok.mean())
            cl = AT.cells(tv["lp"][ok], tv["lm"][ok], *args.ncell)
            vals.append(AT.measure(np.log(sh)[ok], (dm / mgen)[ok],
                                   [v[ok] for v in ctrl], cl, nboot=0)["slope"])
        v = np.array(vals)
        e = v.std(ddof=1) / np.sqrt(len(v))
        print(f"  {tag:40s} keep {keep:.4f}  A_meas {v.mean():+.4f} +- {e:.4f} "
              f" bias {v.mean()-A:+.4f}")


def truth_kinematics(c):
    """True per-leg (ln p, lambda) and the production point, either layout.

    The K_S cache carries the offline Geant4 join (`t_*`); a J/psi production
    carries the maker's own gen matching (`Mu*gen_pt/eta/phi`) and has no
    displaced vertex, so the decay point degenerates to the candidate's own.
    """
    if "t_pabsp" in c:
        return (np.log(c["t_pabsp"]), np.log(c["t_pabsm"]),
                c["t_lamp"], c["t_lamm"],
                np.hypot(c["t_vx"], c["t_vy"]), c["t_vz"].astype(np.float64))
    lap = np.arctan(np.sinh(c["Muplusgen_eta"].astype(np.float64)))
    lam_ = np.arctan(np.sinh(c["Muminusgen_eta"].astype(np.float64)))
    pp = c["Muplusgen_pt"].astype(np.float64) / np.maximum(np.cos(lap), 1e-6)
    pm = c["Muminusgen_pt"].astype(np.float64) / np.maximum(np.cos(lam_), 1e-6)
    return (np.log(np.maximum(pp, 1e-3)), np.log(np.maximum(pm, 1e-3)),
            lap, lam_, np.hypot(c["Jpsi_x"], c["Jpsi_y"]),
            c["Jpsi_z"].astype(np.float64))


def truth_controls(c, p):
    lp, lm, lap, lam_, rdec, zdec = truth_kinematics(c)
    nvp = c["Muplus_nvalid"].astype(np.float64)
    nvm = c["Muminus_nvalid"].astype(np.float64)
    npp = c["Muplus_nvalidpixel"].astype(np.float64)
    npm = c["Muminus_nvalidpixel"].astype(np.float64)
    lr = np.log(rdec + 0.5)
    ctrl = [lp, lm, lap, lam_, lap ** 2, lam_ ** 2, lr, zdec, np.abs(zdec),
            nvp, nvm, lp * lm, lp ** 2, lm ** 2, npp, npm, nvp ** 2, nvm ** 2,
            nvp * nvm, lp * lap, lm * lam_, lr ** 2, lp * lr, lm * lr,
            lap * lam_, np.abs(lap), np.abs(lam_), npp * npm, zdec ** 2]
    return ctrl, dict(lp=lp, lm=lm, lap=lap, lam=lam_, rdec=rdec, zdec=zdec)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--masses", choices=list(MASSES), default="pion")
    ap.add_argument("--max-resid", type=float, default=0.06,
                   help="tail control, as an ABSOLUTE |m_reco - m_gen| in GeV "
                        "-- the same window `make_card.py --residual-mode` "
                        "applies, so the measurement is made on the population "
                        "the fit uses. It must NOT be a pull cut: a threshold "
                        "built from the candidate's own sigma manufactures the "
                        "very correlation being measured (see ares_truth).")
    ap.add_argument("--toy", type=float, default=None,
                   help="replace the data by a toy with this known A and "
                        "report the estimator's bias instead")
    ap.add_argument("--toy-nrep", type=int, default=20)
    ap.add_argument("--toy-noise", type=float, default=0.35)
    ap.add_argument("--nb-p", type=int, default=6)
    ap.add_argument("--nb-lam", type=int, default=12)
    ap.add_argument("--minn", type=int, default=300)
    ap.add_argument("--lam-min", type=float, default=0.25,
                    help="legs below this |lambda| carry no leverage on the "
                         "path-length term and are left out of the `c tan` fit")
    ap.add_argument("--ncell", type=int, nargs=2, default=(8, 8),
                    help="truth cells in (ln p_plus, ln p_minus)")
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--write", default=None,
                    help="write a copy of the pairs cache with one a_res "
                         "column per form (`ares_<form>`)")
    ap.add_argument("--tag", default="ares_angles")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args(argv)

    masses = MASSES[args.masses]
    print(f"=== {args.cov} ({args.masses} hypothesis) ===")
    c, p, u, C, J, H, sig, m, fang = load(args.cov, args.pairs, masses)
    n = len(sig)
    fhit = (p["vgf"] if p is not None and "vgf" in p
            else c["cfmass_vgf"]).astype(np.float64)
    mgen = p["eta"].astype(np.float64) if p is not None else c["t_mgen"]
    z = (m - mgen) / sig
    print(f"  {n} candidates; sigma/m med {np.median(sig/m):.5f}, "
          f"f_hit med {np.median(fhit):.4f}, f_ang med {np.median(fang):.4f}")

    F = forms(c, p, u, C, J, H, sig, m, fang, fhit, args)

    srel = sig / m
    print("\n--- A = a m/sigma, per candidate ---")
    print(f"  {'form':22s} {'median':>9s} {'mean':>9s} {'q10':>9s} {'q90':>9s}")
    order = ["mom", "ang", "ang_simple", "kin", "full_ms", "full_ms_nolam",
             "full_ms_momonly", "full_pop", "full_pop_nolam"]
    for k in order:
        v = F[k]
        print(f"  {k:22s} {np.median(v):9.4f} {v.mean():9.4f} "
              f"{np.quantile(v,0.1):9.4f} {np.quantile(v,0.9):9.4f}")

    # ---- the truth slope ------------------------------------------------
    ctrl, tv = truth_controls(c, p)
    if args.toy is not None:
        toy(args, ctrl, tv, sig, mgen)
        return 0
    y = np.log(sig)
    x = (m - mgen) / mgen
    ok = np.abs(m - mgen) < args.max_resid
    cell = AT.cells(tv["lp"][ok], tv["lm"][ok], *args.ncell)
    print(f"\n--- the measured slope, |m_reco - m_gen| < {args.max_resid} GeV "
          f"({int(ok.sum())} candidates, {100.0*ok.mean():.2f} %, "
          f"{args.ncell[0]}x{args.ncell[1]} truth cells) ---")
    res = AT.measure(y[ok], x[ok], [v[ok] for v in ctrl], cell,
                     nboot=args.nboot)
    Ameas, Aerr = res["slope"], res["err"]
    print(f"  A_measured = {Ameas:+.4f} +- {Aerr:.4f}")
    print(f"\n  {'form':22s} {'A_pred':>9s} {'meas-pred':>12s} {'pull':>7s}")
    preds = {}
    for k in order:
        r = AT.measure(y[ok], x[ok], [v[ok] for v in ctrl], cell,
                       pred=F[k][ok], nboot=0)
        preds[k] = r["pred"]
        d = Ameas - r["pred"]
        print(f"  {k:22s} {r['pred']:9.4f} {d:+12.4f} {d/Aerr:7.2f}")

    # ---- bins -----------------------------------------------------------
    def binned(name, var, edges, use_quantiles=False):
        print(f"\n--- in bins of {name} ---")
        if use_quantiles:
            edges = np.quantile(var, edges)
        hdr = (f"  {'bin':>18s} {'n':>7s} {'A_meas':>16s} "
               + " ".join(f"{k:>9s}" for k in ("mom", "ang", "full_ms",
                                               "full_pop")))
        print(hdr)
        rows = []
        for i in range(len(edges) - 1):
            s = ok & (var >= edges[i]) & (var < edges[i + 1])
            if s.sum() < 2000:
                continue
            cl = AT.cells(tv["lp"][s], tv["lm"][s], 5, 5)
            r = AT.measure(y[s], x[s], [v[s] for v in ctrl], cl,
                           nboot=args.nboot)
            pr = {k: AT.measure(y[s], x[s], [v[s] for v in ctrl], cl,
                                pred=F[k][s], nboot=0)["pred"]
                  for k in ("mom", "ang", "full_ms", "full_pop")}
            print(f"  {edges[i]:8.3g}-{edges[i+1]:<8.3g} {int(s.sum()):7d} "
                  f"{r['slope']:+9.4f} +-{r['err']:5.4f} "
                  + " ".join(f"{pr[k]:9.4f}" for k in
                             ("mom", "ang", "full_ms", "full_pop")))
            rows.append((0.5 * (edges[i] + edges[i + 1]), r["slope"], r["err"],
                         pr["mom"], pr["ang"], pr["full_ms"], pr["full_pop"]))
        rows = np.array(rows) if rows else np.zeros((0, 7))
        if len(rows) > 1:
            nb = len(rows)
            msg = []
            for j, k in ((3, "mom"), (4, "ang"), (5, "full_ms"),
                         (6, "full_pop")):
                chi2 = float(np.sum(((rows[:, 1] - rows[:, j]) / rows[:, 2]) ** 2))
                msg.append(f"{k} {chi2:.1f}/{nb}")
            print("    chi2 of the measurement against each form: "
                  + ",  ".join(msg))
        return rows

    # f_ang is RECONSTRUCTED -- it is built from the fitted state and the
    # fitted covariance, so it moves with the very fluctuation being measured
    # and binning on it is the trap the controls exist to avoid.  The primary
    # binning is on its TRUTH-ONLY projection: f_ang regressed on the true
    # kinematics, which keeps the physics (the angular share is a kinematic
    # property) and drops the fluctuation.  The reco binning is run too, and
    # printed as the demonstration that it manufactures structure.
    X = np.stack([np.ones(n)] + list(ctrl), axis=1)
    beta = np.linalg.lstsq(X[ok], fang[ok], rcond=None)[0]
    fang_true = X @ beta
    print(f"\n  f_ang projected on the true kinematics: "
          f"corr {np.corrcoef(fang[ok], fang_true[ok])[0,1]:.3f}, "
          f"residual rms {np.std(fang[ok]-fang_true[ok]):.3f}")
    b_fang = binned("f_ang PROJECTED ON TRUTH", fang_true,
                    list(np.quantile(fang_true[ok], np.linspace(0, 1, 8))))
    binned("f_ang RECONSTRUCTED -- a reco-correlated binning, shown as the "
           "artefact it is", fang,
           list(np.quantile(fang[ok], np.linspace(0, 1, 8))))
    pmin = np.exp(np.minimum(tv["lp"], tv["lm"]))
    b_p = binned("the softer leg's TRUE |p| [GeV]", pmin,
                 list(np.quantile(pmin, np.linspace(0, 1, 7))))
    lmax = np.maximum(np.abs(tv["lap"]), np.abs(tv["lam"]))
    b_lam = binned("max |lambda| (TRUE)", lmax,
                   list(np.quantile(lmax, np.linspace(0, 1, 6))))

    # ---- the consequence, in a_res units --------------------------------
    print("\n--- a_res = A sigma/m, the number the card carries ---")
    print(f"  {'form':22s} {'median a_res':>14s} {'ratio to mom':>14s}")
    wref = srel ** 2
    for k in order + ["truth"]:
        v = (np.full(n, Ameas) if k == "truth" else F[k]) * srel
        rat = np.sum(v * srel) / np.sum(F["mom"] * srel * srel)
        print(f"  {k:22s} {np.median(v):14.5f} {rat:14.4f}")

    if args.write:
        out = dict(p) if p is not None else {}
        for k in order:
            out["ares_" + k] = F[k] * srel
        out["ares_truth"] = np.full(n, Ameas) * srel
        out["ares_A_measured"] = np.array([Ameas, Aerr])
        np.savez_compressed(args.write, **out)
        print(f"\nwrote {args.write} with per-candidate a_res columns "
              f"{['ares_' + k for k in order] + ['ares_truth']}")

    if not args.no_figs:
        figs(args, F, fang, srel, Ameas, Aerr, preds, b_fang, b_p, b_lam,
             order, u, C, J, sig, m)
    return 0


def figs(args, F, fang, srel, Ameas, Aerr, preds, b_fang, b_p, b_lam, order,
         u, C, J, sig, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    import pubhtml
    hep.style.use(hep.style.ROOT)
    outdir = args.figdir or pubhtml.figdir(args.tag)
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir)

    lab = {"mom": r"(i) $1+f_{\rm hit}$  (J/$\psi$ form)",
           "ang": r"(ii) momentum-gradient only",
           "full_ms": r"(iii) full, path-length $\Sigma$",
           "full_pop": r"(iii) full, population $\Sigma$",
           "kin": r"kinematic term only ($\Sigma$ fixed)"}
    col = {"mom": "C0", "ang": "C3", "full_ms": "C2", "full_pop": "C1",
           "kin": "C4"}

    # 1. the distributions of A
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bins = np.linspace(-0.5, 3.0, 141)
    for k in ("mom", "ang", "full_ms", "full_pop", "kin"):
        ax.hist(np.clip(F[k], bins[0], bins[-1]), bins=bins, histtype="step",
                lw=2, color=col[k], label=lab[k])
    ax.axvspan(Ameas - Aerr, Ameas + Aerr, color="0.5", alpha=0.35, zorder=0)
    ax.axvline(Ameas, color="k", lw=2, ls="--",
               label=f"measured from truth  {Ameas:+.2f}$\\pm${Aerr:.2f}")
    ax.set_xlabel(r"$A = a\,m/\sigma_m$")
    ax.set_ylabel("candidates")
    ax.legend(fontsize=13, loc="upper right")
    pubhtml.savefig(fig, os.path.join(outdir, "ares_A_distributions.pdf"))
    plt.close(fig)

    # 2. A vs f_ang: the measurement against the forms
    for tag, rows, xl in (("fang", b_fang,
                           r"$f_{\rm ang}$ projected on the true kinematics"),
                          ("pmin", b_p, r"softer pion true $|p|$ [GeV]"),
                          ("lam", b_lam, r"max $|\lambda|$ (true)")):
        if not len(rows):
            continue
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="ko",
                    ms=7, lw=2, capsize=4, label="measured (MC truth)")
        for j, k in ((3, "mom"), (4, "ang"), (5, "full_ms"), (6, "full_pop")):
            ax.plot(rows[:, 0], rows[:, j], "-", lw=2, color=col[k],
                    marker="s", ms=5, label=lab[k])
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$A = a\,m/\sigma_m$")
        ax.axhline(0.0, color="0.7", lw=1)
        ax.legend(fontsize=13)
        pubhtml.savefig(fig, os.path.join(outdir, f"ares_A_vs_{tag}.pdf"))
        plt.close(fig)

    # 3. where the slope comes from: the six components of J^T Sigma grad(sigma)
    SJ = np.einsum("nij,nj->ni", C, J)
    g = F["_grad_full_ms"]
    comp = SJ * g / sig[:, None] ** 2 * (m / sig)[:, None]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    names = [r"$q/p^{+}$", r"$\lambda^{+}$", r"$\phi^{+}$",
             r"$q/p^{-}$", r"$\lambda^{-}$", r"$\phi^{-}$"]
    vals = comp.mean(axis=0)
    ax.bar(range(6), vals, color=["C0", "C2", "C2", "C0", "C2", "C2"])
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(names)
    ax.set_ylabel(r"mean contribution to $A$")
    ax.set_title(r"$A = \sum_i (\Sigma J)_i\, \partial_i\sigma_m \cdot m/\sigma_m^3$",
                 fontsize=15)
    tot = vals.sum()
    ax.text(0.03, 0.95, f"sum {tot:.3f};  angular share "
                        f"{100*vals[[1,2,4,5]].sum()/tot:.0f} %",
            transform=ax.transAxes, va="top", fontsize=14)
    pubhtml.savefig(fig, os.path.join(outdir, "ares_A_components.pdf"))
    plt.close(fig)

    # 4. the verdict in one panel: every form's population slope against the
    #    measurement, on the SAME weights and the SAME realised fluctuations
    show = ["ang", "ang_simple", "full_ms_momonly", "full_pop", "full_ms",
            "mom", "kin"]
    nice = {"mom": r"(i) $1+f_{\rm hit}$",
            "ang": r"(ii) $(1-f_{\rm ang})[\ldots]$",
            "ang_simple": r"(ii$^\prime$) $(1-f_{\rm ang})(1+f_{\rm hit})$",
            "full_ms_momonly": r"(iii) momentum sector only",
            "full_ms": r"(iii) full, path-length $\Sigma$",
            "full_pop": r"(iii) full, population $\Sigma$",
            "kin": r"kinematic only ($\Sigma$ frozen)"}
    have = [k for k in show if k in preds]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    yy = np.arange(len(have))
    ax.barh(yy, [preds[k] for k in have], color="C0", height=0.6, alpha=0.85)
    ax.axvspan(Ameas - Aerr, Ameas + Aerr, color="C3", alpha=0.3, zorder=0)
    ax.axvline(Ameas, color="C3", lw=2.5,
               label=f"measured from truth  {Ameas:.3f} $\pm$ {Aerr:.3f}")
    for i, k in enumerate(have):
        ax.text(preds[k] + 0.03, yy[i], f"{preds[k]:.3f}  "
                f"({(Ameas-preds[k])/Aerr:+.1f}$\sigma$)",
                va="center", fontsize=12)
    ax.set_yticks(yy)
    ax.set_yticklabels([nice.get(k, k) for k in have], fontsize=13)
    ax.set_xlabel(r"$A = a\,m/\sigma_m$")
    ax.set_xlim(0, max(max(preds[k] for k in have), Ameas) * 1.45)
    ax.set_title(f"{args.masses} pair, {len(sig)} candidates, "
                 rf"$f_{{\rm ang}}$ median {np.median(fang):.3f}", fontsize=14)
    ax.legend(fontsize=13, loc="lower right")
    pubhtml.savefig(fig, os.path.join(outdir, "ares_forms_summary.pdf"))
    plt.close(fig)
    print(f"\nfigures in {outdir}")


if __name__ == "__main__":
    sys.exit(main())
