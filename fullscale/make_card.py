#!/usr/bin/env python3
"""Assemble the FULL-SCALE Z -> mumu card of the unbinned CVH mass likelihood.

This is `zchannel/make_z_card.py` taken to production statistics and to the
model the group-leader numbers require. What is different, and why:

1. **Weights.** The sample is POWHEG MiNNLO: 7.7 % of the events carry a
   negative weight and a handful carry |w| ~ 1e19 (unweighting failures). The
   smoke card was unweighted, which is only harmless at 459 candidates. Here
   `genweight` is clipped at `--wclip` times the modal |w|, rescaled to mean 1
   (so the inverse Hessian is the naive covariance and the weight variance
   shows up where it belongs, in the sandwich), and passed as `weights=`.

2. **The FSR fold is MULTIPLICATIVE and lives in the PROVIDER**, not in
   `MassCFTerm`'s additive `phi_K`. The empirical kernel is exactly a rescaling
   (`zchannel/README.md`, "Step 4"), and treating it as additive costs 25 MeV
   on `m_Z`. So no `phik` is passed at all; `ZGammaLineshape(fsr=...)` folds it,
   BANDED in `m_pre` at `sigma_cap <= 3.3e-4`, and `acceptance=` carries A(m).

3. **A floated smooth K(m).** The provider is LO in the hard ME and in the
   parton luminosity while the sample is MiNNLO; without a smooth
   multiplicative shape the generator-level fit is +76 MeV on `Gamma_Z`. Five
   Legendre terms close it and cost 1.25x on sigma(m_Z).

4. **Both corrections of MASSCFTERM_SPEC, in the FLUCTUATION form.** `a_res`
   (sec. 2-3, the self-consistent resolution) and `jensen_s2` (sec. 4b, the
   second-order map). Each alone is 15-43 MeV at Z resolution and they
   partially cancel; both must be in explicitly. Both are expansions in the
   RESOLUTION fluctuation, so at the Z they must be applied as a deterministic
   map of it INSIDE the convolution (`--corr-form fluctuation`), not evaluated
   at `delta_i = m_i - M(theta)`, which over a +-27 sigma window is the
   Breit-Wigner tail and FSR rather than resolution. `--corr-form residual`
   plus `--corr-clip` is the historical stopgap, kept for the J/psi gate.

5. **Quality cuts**, documented and scannable: the observed-mass window, the
   fit quality `chi2/ndof`, and a `sigma_m/m` cut. The last is not cosmetic:
   `Jpsi_sigmamass` runs to O(1e2-1e5) GeV on a per-mille tail, where the
   second-order expansion the Jensen term inverts has no meaning.

usage:
  python make_card.py --pairs runs/zpairs_dyv2.npz \\
      --fsr  ../zchannel/data/kern_loose_band3.3e-4.npz \\
      --acc  ../zchannel/data/acc_loose_d8.json \\
      -o cards/zcard_dyv2.hdf5 --shape 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_Z = os.path.join(os.path.dirname(HERE), "zchannel")
for _p in (HERE, _Z):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MZ_REF = 91.1876
FAMILY_ORDER = ["ms", "ioni", "rad"]
CACHE_KEYS = {"ms": ("Sms", None), "ioni": ("Sio_re", "Sio_im"),
              "rad": ("Srad_re", "Srad_im"), "del": ("Sdel", None)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="cf_inmaker.py pairs cache")
    p.add_argument("--fsr", default=None,
                   help="banded multiplicative FSR kernel npz (fit_gen.py kernel)")
    p.add_argument("--acc", default=None, help="acceptance json (fit_gen.py acceptance)")
    p.add_argument("-o", "--output", default=None, help="datacard (.hdf5)")
    p.add_argument("--dump", default=None, help="also write the assembled arrays")
    p.add_argument("--name", default="zmass")
    p.add_argument("--channel", default="z")
    p.add_argument("--mref", type=float, default=MZ_REF)
    p.add_argument("--window", type=float, nargs=2, default=[60.0, 120.0],
                   metavar=("LO", "HI"),
                   help="the OBSERVED-mass selection window. The production "
                        "cuts 60 < m < 120 on the PRE-REFIT track mass "
                        "(Jpsitrk_mass); this cut is on the refit mass the "
                        "likelihood models, and --report-selection measures "
                        "the difference.")
    p.add_argument("--born-window", type=float, nargs=2, default=[50.0, 130.0])
    p.add_argument("--shape", type=int, default=5,
                   help="Legendre terms of the floated smooth K(m); 0 = none")
    p.add_argument("--shape-window", type=float, nargs=2, default=None,
                   help="window the Legendre basis is orthogonal over "
                        "(default: --window)")
    p.add_argument("--no-window-norm", action="store_true")
    p.add_argument("--norm-classes", type=int, default=64)
    p.add_argument("--norm-tpoints", type=int, default=8192)
    p.add_argument("--fit-upsample", type=int, default=4)
    # ---- selection -------------------------------------------------------
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--max-sigma-rel", type=float, default=0.10,
                   help="drop candidates with sigma_m/m above this")
    p.add_argument("--min-sigma-rel", type=float, default=0.0,
                   help="drop candidates with sigma_m/m BELOW this. With "
                        "--max-sigma-rel this cuts a resolution slice, which "
                        "is the differential test of whether a residual bias "
                        "is the resolution model: both corrections and any "
                        "error in the per-candidate CF scale as sigma_rel^2, "
                        "so the bias must grow across the slices. It is the "
                        "Z analogue of MASSCFTERM_SPEC's gate J4.")
    p.add_argument("--max-sigma", type=float, default=0.0,
                   help="absolute sigma_m cut [GeV]; 0 = off")
    p.add_argument("--maxn", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234,
                   help="seed for the --maxn subsample (a HEAD slice would be "
                        "the first tasks, i.e. one contiguous run range)")
    # ---- weights ---------------------------------------------------------
    p.add_argument("--wclip", type=float, default=100.0,
                   help="clip |genweight| at this multiple of the modal |w|; "
                        "0 = no clip; -1 = unweighted")
    # ---- the two corrections --------------------------------------------
    p.add_argument("--ares", choices=["on", "write-off", "off"], default="on",
                   help="on = the self-consistent resolution is active; "
                        "write-off = a_i is stored but the correction is off, "
                        "so one card serves both fits; off = no a_i at all")
    p.add_argument("--max-ares", type=float, default=0.5)
    p.add_argument("--jensen", choices=["exact", "shift", "off"], default="exact")
    p.add_argument("--corr-form", choices=["fluctuation", "residual"],
                   default="fluctuation",
                   help="WHERE the two corrections act. `fluctuation` (the "
                        "default, and the only form defined at the Z) applies "
                        "them as one deterministic map of the resolution "
                        "fluctuation INSIDE the convolution: no clip, no "
                        "log-Jacobian, no dependence on delta_i. `residual` is "
                        "the historical form, exact at a delta kernel and "
                        "measured on the J/psi, which needs --corr-clip at the "
                        "Z and is kept only as the reference for that gate.")
    p.add_argument("--corr-coeff-max", type=float, default=0.08,
                   help="bound on the fluctuation form's quadratic coefficient "
                        "|c_i/sigma_i| -- the expansion parameter itself. A "
                        "per-candidate CONSTANT, so it cannot deform the "
                        "likelihood's dependence on the parameters. Below it "
                        "the first-order truncation can put the modelled "
                        "density negative in the tail of a large-sigma_m/m "
                        "candidate. 0 disables it.")
    p.add_argument("--corr-clip", type=float, default=0.0,
                   help="the domain of BOTH corrections, in units of sigma_i. "
                        "They are expansions in the resolution fluctuation, "
                        "and at the Z the deviation from the pole is FSR and "
                        "the Breit-Wigner tail out to 27 sigma -- fed that, "
                        "the exact Jensen map moves the residual by a median "
                        "57.7 MeV against the 20.6 MeV it exists to apply. "
                        "0 = no clip (the J/psi behaviour the spec's gates "
                        "were measured with). IGNORED, and required to be 0, "
                        "when --corr-form fluctuation: that form has no "
                        "argument to clip.")
    p.add_argument("--jensen-fang", action="store_true", default=True,
                   help="fold the per-candidate angular share into s^2")
    p.add_argument("--no-jensen-fang", dest="jensen_fang", action="store_false")
    # ---- resolution knobs ------------------------------------------------
    p.add_argument("--k-prior", type=float, default=0.0,
                   help="Gaussian prior on the k_* resolution scales; 0 = free. "
                        "They are FIXED at the MC truth (1) by the fit driver's "
                        "--fix, so this is only for a variant.")
    p.add_argument("--gz-prior", type=float, default=0.0)
    p.add_argument("--mz-prior", type=float, default=0.0)
    p.add_argument("--shape-prior", type=float, default=0.0)
    p.add_argument("--with-alpha", action="store_true")
    p.add_argument("--background", choices=["none", "uniform", "bernstein"],
                   default="none",
                   help="MC has no background; the default is therefore none, "
                        "not the smoke card's uniform")
    p.add_argument("--bernstein-degree", type=int, default=2)
    p.add_argument("--fbkg", type=float, default=0.0)
    p.add_argument("--float-bkg", action="store_true")
    p.add_argument("--del-family", action="store_true")
    p.add_argument("--width-scheme", choices=["fixed", "running"], default="fixed")
    p.add_argument("--nm", type=int, default=8192,
                   help="lineshape mass grid. NOT the provider's 32768 default: "
                        "with `fsr=` the fold is a DENSE (nm, n_born) matrix, "
                        "and n_born ~ nm (m_hi_born - lo)/(hi - lo), so "
                        "nm = 32768 over a 50-130 GeV Born window with the "
                        "grid extended to 200 GeV is a 16 GB constant. At 8192 "
                        "it is 1.0 GB and dm = 9.8 MeV, i.e. 1/256 of Gamma_Z; "
                        "the generator-level closure found 4096/8192/16384 "
                        "agree to 0.06 MeV.")
    p.add_argument("--fsr-mmax", type=float, default=0.0,
                   help="cap the extended Born grid the FSR fold needs [GeV]; "
                        "0 = the luminosity table's own upper edge (200)")
    p.add_argument("--tau-max", type=float, default=40.0)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--report-selection", action="store_true", default=True)
    return p.parse_args(argv)


def discover_families(keys, want_del=False):
    fams = []
    for name in FAMILY_ORDER + (["del"] if want_del else []):
        re_k, im_k = CACHE_KEYS[name]
        if re_k in keys:
            fams.append((name, re_k, im_k if (im_k and im_k in keys) else None))
    return fams


def select(d, args, log=print):
    """The candidate selection. Returns (index array, a table of the steps)."""
    z = d["z"].astype(np.float64)
    sigma = d["sigma"].astype(np.float64)
    mgen = d["eta"].astype(np.float64)
    m = z * sigma + mgen
    n0 = len(z)
    w = (np.asarray(d["w"], dtype=np.float64) if "w" in d.files
         else np.ones(n0))
    srel = sigma / np.maximum(np.abs(m), 1e-9)

    steps = []
    keep = np.isfinite(m) & np.isfinite(sigma) & (sigma > 0.0)
    steps.append(("finite m, sigma > 0", keep.copy()))
    lo, hi = args.window
    keep &= (m >= lo) & (m <= hi)
    steps.append((f"m_obs in [{lo:g}, {hi:g}]", keep.copy()))
    if args.max_chi2_ndof > 0 and "chisqval" in d.files:
        keep &= (np.asarray(d["chisqval"], dtype=np.float64)
                 / np.maximum(np.asarray(d["ndof"], dtype=np.float64), 1.0)
                 ) < args.max_chi2_ndof
        steps.append((f"chi2/ndof < {args.max_chi2_ndof:g}", keep.copy()))
    if args.max_sigma_rel > 0:
        keep &= srel < args.max_sigma_rel
        steps.append((f"sigma_m/m < {args.max_sigma_rel:g}", keep.copy()))
    if args.min_sigma_rel > 0:
        keep &= srel >= args.min_sigma_rel
        steps.append((f"sigma_m/m >= {args.min_sigma_rel:g}", keep.copy()))
    if args.max_sigma > 0:
        keep &= sigma < args.max_sigma
        steps.append((f"sigma_m < {args.max_sigma:g} GeV", keep.copy()))

    log(f"  selection on {n0} cached candidates")
    prev = n0
    for label, k in steps:
        nk = int(k.sum())
        log(f"    {label:28s} {nk:9d}  ({100.0*nk/n0:6.2f} %, -{prev-nk})")
        prev = nk
    idx = np.flatnonzero(keep)
    if args.maxn and args.maxn < len(idx):
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, args.maxn, replace=False))
        log(f"    --maxn {args.maxn} (seed {args.seed}, random not head)")
    # the selection-variable mismatch, if the cache records it
    if args.report_selection and "mtrk" in d.files:
        mt = np.asarray(d["mtrk"], dtype=np.float64)
        base = np.isfinite(m) & np.isfinite(sigma) & (sigma > 0.0)
        insel = base & (mt >= lo) & (mt <= hi)
        inobs = base & (m >= lo) & (m <= hi)
        log(f"    SELECTION VARIABLE: the production cut 60 < Jpsitrk_mass < 120. "
            f"in-sel {int(insel.sum())}, in-obs {int(inobs.sum())}, "
            f"sel&~obs {int((insel & ~inobs).sum())}, "
            f"obs&~sel {int((inobs & ~insel).sum())} "
            f"({100.0*(insel ^ inobs).sum()/max(int(base.sum()),1):.3f} % disagree)")
    return idx, m, sigma, w, srel


def build_weights(w, idx, args, log=print):
    """Clipped, mean-1 MiNNLO weights, and the effective statistics."""
    if args.wclip < 0:
        log("  UNWEIGHTED (--wclip -1)")
        return None, {"neff_frac": 1.0, "nneg": 0}
    ww = w[idx].copy()
    modal = float(np.median(np.abs(ww)))
    nclip = 0
    if args.wclip > 0 and modal > 0:
        cap = args.wclip * modal
        nclip = int(np.sum(np.abs(ww) > cap))
        ww = np.clip(ww, -cap, cap)
    mean = float(ww.mean())
    if mean == 0.0:
        raise SystemExit("the weights average to zero")
    ww = ww / mean
    neff = ww.sum() ** 2 / (len(ww) * (ww ** 2).sum())
    info = {"modal_abs_w": modal, "nclip": nclip,
            "nneg": int((ww < 0).sum()), "neff_frac": float(neff)}
    log(f"  weights: modal |w| {modal:.5g}, clipped {nclip} above "
        f"{args.wclip:g}x, {info['nneg']} negative "
        f"({100.0*info['nneg']/len(ww):.2f} %), N_eff/N = {neff:.4f} "
        f"-> sqrt(N/N_eff) = {1.0/np.sqrt(neff):.4f} on every error")
    return ww, info


def build(args, log=print):
    from rabbit import unbinned
    from rabbit.lineshapes import ZGammaLineshape

    d = np.load(args.pairs, allow_pickle=True)
    tgrid = np.asarray(d["tgrid"], dtype=np.float64)
    nt = len(tgrid)

    idx, m_all, sig_all, w_all, srel_all = select(d, args, log)
    n = len(idx)
    if not n:
        raise SystemExit("the selection kept nothing")
    sigma = sig_all[idx]
    mreco = m_all[idx]
    mgen = d["eta"].astype(np.float64)[idx]
    vgf = d["vgf"].astype(np.float64)[idx]
    mobs = mreco - args.mref
    lo, hi = args.window

    weights, winfo = build_weights(w_all, idx, args, log)

    # ---- the two corrections --------------------------------------------
    # a_i = (1 + f_hit,i) sigma_i / m_i, from d ln sigma_m/d ln m = 1 + f_hit
    # (- f_ioni, which is 1.1e-3 and is dropped). Truth-free: m_i is observed.
    a_res = None
    if args.ares != "off":
        a_res = (1.0 + vgf) * sigma / np.maximum(np.abs(mreco), 1e-9)
        if args.corr_clip and args.corr_form == "residual":
            log(f"  corrections clipped to |delta| < {args.corr_clip:g} sigma "
                f"(that is {100.0*np.mean(np.abs(mobs) < args.corr_clip*sigma):.2f} % "
                f"of candidates inside the clip)")
        nclip = int(np.sum(np.abs(a_res) > args.max_ares))
        a_res = np.clip(a_res, -args.max_ares, args.max_ares)
        log(f"  a_res: median {np.median(a_res):.5f}, "
            f"q99 {np.quantile(np.abs(a_res), 0.99):.5f}, "
            f"{nclip} clipped at {args.max_ares}")
    jensen_s2 = None
    if args.jensen != "off":
        s2 = (sigma / np.maximum(np.abs(mreco), 1e-9)) ** 2
        if args.jensen_fang and "fang" in d.files:
            fang = np.asarray(d["fang"], dtype=np.float64)[idx]
            fang = np.clip(fang, -0.5, 1.0)
            s2 = s2 * (1.5 - fang) / 1.5
            log(f"  jensen: f_ang median {np.median(fang):.3e} "
                f"(q01 {np.quantile(fang,0.01):.3e}, "
                f"q99 {np.quantile(fang,0.99):.3e}) folded into s^2")
        jensen_s2 = s2
        log(f"  jensen ({args.jensen}): median 1.5 s^2 = "
            f"{1.5*np.median(s2):.4e} relative "
            f"-> {1.5*np.median(s2)*args.mref*1e3:.2f} MeV")

    # ---- families --------------------------------------------------------
    fams = discover_families(set(d.files), args.del_family)
    log(f"  {n} candidates, nt = {nt}, tau [0, {tgrid[-1]:.4f}], "
        f"families {[f[0] for f in fams]} + hit, upsample {args.fit_upsample}")
    log(f"  sigma: min {sigma.min():.4f} med {np.median(sigma):.4f} "
        f"max {sigma.max():.4f} GeV -> tau needed "
        f"{tgrid[-1]/sigma.min():.2f} 1/GeV")

    families = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    datasets = {"sigma": sigma, "mobs": mobs, "vgf": vgf, "tgrid": tgrid}
    if weights is not None:
        datasets["weights"] = weights
    if a_res is not None:
        datasets["a_res"] = a_res
    if jensen_s2 is not None:
        datasets["jensen_s2"] = jensen_s2
    arrays = {}
    for name, re_k, im_k in fams:
        families.append({"name": name, "param": f"k_{name}", "kind": "tab"})
        arrays[name] = {"re": np.asarray(d[re_k])[idx]}
        datasets[f"S_re_{name}"] = arrays[name]["re"]
        if im_k:
            arrays[name]["im"] = np.asarray(d[im_k])[idx]
            datasets[f"S_im_{name}"] = arrays[name]["im"]

    # ---- truncation normalisation classes --------------------------------
    norm = None
    if not args.no_window_norm:
        K = n if args.norm_classes <= 0 else min(args.norm_classes, n)
        if K == n:
            cls = np.arange(n)
            sig_c, vgf_c = sigma.copy(), vgf.copy()
            fam_c = {nm: {c: a.astype(np.float64) for c, a in arr.items()}
                     for nm, arr in arrays.items()}
        else:
            edges = np.quantile(sigma, np.linspace(0, 1, K + 1))
            cls = np.clip(np.searchsorted(edges[1:-1], sigma, "right"), 0, K - 1)
            sig_c = np.empty(K)
            vgf_c = np.empty(K)
            fam_c = {nm: {c: np.empty((K, nt)) for c in arr}
                     for nm, arr in arrays.items()}
            for c in range(K):
                sel = cls == c
                if not sel.any():
                    sel = np.ones(n, bool)
                sig_c[c] = np.median(sigma[sel])
                vgf_c[c] = np.mean(vgf[sel])
                for nm, arr in arrays.items():
                    for comp, a in arr.items():
                        fam_c[nm][comp][c] = a[sel].mean(axis=0)
        norm = {"sigma": sig_c, "vgf": vgf_c, "class": cls,
                "families": [dict(name=nm, **fam_c[nm]) for nm in fam_c]}
        datasets["norm_sigma"] = sig_c
        datasets["norm_vgf"] = vgf_c
        datasets["norm_class"] = cls.astype(np.int64)
        for nm, arr in fam_c.items():
            for comp, a in arr.items():
                datasets[f"S_{comp}_{nm}_norm"] = a.astype(np.float32)
        log(f"  truncation normalisation on [{lo}, {hi}], {K} class(es), "
            f"{args.norm_tpoints} t points")
    else:
        log("  NO window normalisation (biased likelihood; diagnostic only)")

    # ---- provider --------------------------------------------------------
    t0 = time.time()
    acc = None
    if args.acc:
        with open(args.acc) as fh:
            acc = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
        log(f"  acceptance: {args.acc} ({acc.get('kind','bernstein')}, "
            f"{len(acc.get('coef', []))} coefficients)")
    kw = {}
    if args.shape_prior and not args.shape:
        pass
    if args.fsr_mmax:
        kw["fsr_mmax"] = args.fsr_mmax
    if args.shape:
        kw["shape"] = args.shape
        kw["shape_window"] = tuple(args.shape_window or args.window)
    provider = ZGammaLineshape(
        m_ref=args.mref, window=tuple(args.born_window), nm=args.nm,
        tau_max=args.tau_max, width_scheme=args.width_scheme,
        fsr=args.fsr, acceptance=acc, **kw)
    need, ok = provider.check_tau_range(tgrid, sigma)
    log(f"  {provider} in {time.time()-t0:.1f} s; tau needed {need:.2f} 1/GeV "
        f"-> {'ok' if ok else 'TOO SMALL'}")
    if not ok:
        raise SystemExit("lineshape tau_max too small; raise --tau-max or "
                         "tighten --max-sigma-rel")

    if args.background == "none":
        background = None
    elif args.background == "uniform":
        background = unbinned.UniformBackground((lo, hi))
    else:
        background = unbinned.BernsteinBackground(
            (lo, hi), [f"bkg_c{i}" for i in range(args.bernstein_degree + 1)])

    # The two corrections live on `material-resolution` (a_res) and this
    # file's Jensen patch; the provider lives on `z-lineshape-kernel`. A rabbit
    # that has only one of them must FAIL here, not silently write a card
    # without the correction: an unmarked missing 20 MeV is the worst possible
    # outcome.
    import inspect
    sig = set(inspect.signature(unbinned.MassCFTerm.__init__).parameters)
    kw2 = {}
    if a_res is not None or args.ares != "off":
        if "a_res" not in sig:
            raise SystemExit(
                "this rabbit's MassCFTerm has no `a_res`: it is not the "
                "material-resolution branch (or its merge). Re-run with "
                "--ares off only if you really want the uncorrected card.")
        kw2.update(a_res=a_res, self_consistent_sigma=(args.ares == "on"))
    if args.jensen != "off":
        if "jensen_mode" not in sig:
            raise SystemExit(
                "this rabbit's MassCFTerm has no `jensen_mode`: apply "
                "fullscale/patches/unbinned_jensen.py first, or pass "
                "--jensen off.")
        kw2.update(jensen_s2=jensen_s2, jensen_mode=args.jensen)
    if args.corr_form != "residual":
        if "corr_form" not in sig:
            raise SystemExit(
                "this rabbit's MassCFTerm has no `corr_form`: it predates the "
                "fluctuation reformulation. Pass --corr-form residual "
                "--corr-clip 5 to reproduce the clipped stopgap.")
        if args.corr_clip:
            raise SystemExit(
                "--corr-clip has no meaning with --corr-form fluctuation "
                "(there is no residual-valued argument to clip); pass "
                "--corr-clip 0")
        kw2["corr_form"] = args.corr_form
        if "corr_coeff_max" in sig:
            kw2["corr_coeff_max"] = args.corr_coeff_max
    if args.corr_clip and (a_res is not None or args.jensen != "off"):
        if "corr_clip" not in sig:
            raise SystemExit(
                "this rabbit's MassCFTerm has no `corr_clip`: it predates the "
                "domain fix. Pass --corr-clip 0 only if you mean to feed the "
                "corrections the full residual.")
        kw2["corr_clip"] = args.corr_clip
    if args.shape and "shape" not in inspect.signature(
            ZGammaLineshape.__init__).parameters:
        raise SystemExit(
            "this rabbit's ZGammaLineshape has no `shape`: apply "
            "fullscale/patches/zgamma_shape.py first, or pass --shape 0. "
            "Without a floated K(m) the LO->MiNNLO ratio biases Gamma_Z by "
            "+76 MeV.")
    term = unbinned.MassCFTerm(
        args.name, sigma=sigma, mobs=mobs, tgrid=tgrid,
        families=[dict(f, **arrays.get(f["name"], {})) for f in families],
        vgf=vgf, phik=None,
        kernel=unbinned.TabulatedLineshapeKernel(provider=provider),
        background=background, m_ref=args.mref,
        scale_param="alpha" if args.with_alpha else None,
        bkg_frac_param="f_bkg" if args.float_bkg else None,
        bkg_frac=args.fbkg,
        weights=weights,
        norm_window=None if args.no_window_norm else (lo, hi),
        norm_tpoints=args.norm_tpoints, norm=norm,
        upsample=args.fit_upsample,
        chunk=args.chunk, channel=args.channel, **kw2)

    pdkw = {"mz_prior": args.mz_prior or None,
            "gz_prior": args.gz_prior or None}
    if "shape_prior" in inspect.signature(provider.param_declarations).parameters:
        pdkw["shape_prior"] = args.shape_prior or None
    decl = dict(provider.param_declarations(**pdkw))
    for f in families:
        decl[f["param"]] = (1.0, args.k_prior or np.nan, 1.0, 0)
    if args.with_alpha:
        decl["alpha"] = (0.0, np.nan, 0.0, 1)
    if args.float_bkg:
        decl["f_bkg"] = (args.fbkg / unbinned.FBKG_UNIT, np.nan,
                         args.fbkg / unbinned.FBKG_UNIT, 0)
    if args.background == "bernstein":
        for i in range(args.bernstein_degree + 1):
            decl[f"bkg_c{i}"] = (float(np.log(np.expm1(1.0))), np.nan,
                                 float(np.log(np.expm1(1.0))), 0)
    decl = unbinned.declare_params(term, decl)
    log(f"  parameters {list(term.param_names)}")
    log(f"    POIs {[p for p, f in zip(term.param_names, decl['param_is_poi']) if f]}")

    info = {"n": n, "n_cache": int(len(d['z'])), "window": [lo, hi],
            "born_window": list(args.born_window), "mreco": mreco,
            "mgen": mgen, "weights_info": winfo,
            "provider_config": provider.config()}
    return term, datasets, decl, info


def main():
    args = parse_args()
    print(f"[make_card] {args.pairs}")
    term, datasets, decl, info = build(args)
    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)) or ".",
                    exist_ok=True)
        np.savez_compressed(
            args.dump, config=json.dumps(term.config()),
            params=np.array(list(term.param_names)), mreco=info["mreco"],
            mgen=info["mgen"], argv=np.array(sys.argv[1:], dtype=object),
            **datasets, **decl)
        print(f"  -> {args.dump}")
    if args.output:
        from rabbit import tensorwriter
        writer = tensorwriter.TensorWriter()
        writer.add_dummy_channel(name=f"{args.channel}_dummy")
        writer.add_unbinned_term(args.name, term.config(), term.param_names,
                                 datasets, **decl)
        out = os.path.abspath(args.output)
        name = os.path.basename(out)
        if name.endswith(".hdf5"):
            name = name[:-5]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        t0 = time.time()
        writer.write(outfolder=os.path.dirname(out) or ".", outfilename=name)
        p = os.path.join(os.path.dirname(out), name) + ".hdf5"
        print(f"  -> {p} ({os.path.getsize(p)/1e9:.2f} GB) in "
              f"{time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
