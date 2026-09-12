#!/usr/bin/env python3
"""Assemble the FULL-SCALE Z -> mumu card of the unbinned CVH mass likelihood.

This is `zchannel/make_z_card.py` taken to production statistics and to the
full model. What is different, and why:

1. **Weights.** The sample is POWHEG MiNNLO: 7.7 % of the events carry a
   negative weight and a handful carry |w| ~ 1e19 (unweighting failures).
   Running unweighted is only harmless at a few hundred candidates. Here
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
   plus `--corr-clip` evaluates them at `delta_i` instead and is kept as the
   reference for the J/psi gate.

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
    p.add_argument("--residual-mode", action="store_true",
                   help="THE PURE DETECTOR TEST. Model `m_reco - m_gen` "
                        "directly against the per-candidate resolution CF: the "
                        "observable becomes the residual (offset by --mref so "
                        "`alpha` still has units), the physics kernel becomes a "
                        "DELTA, and the FSR fold, the acceptance and K(m) are "
                        "all dropped. Nothing of the mass model survives, so a "
                        "width or shift measured here is the DETECTOR half "
                        "alone. `corr_mass` carries the real per-candidate mass "
                        "so the two corrections stay evaluated where they "
                        "belong.")
    p.add_argument("--set", dest="set_params", nargs="*", default=[],
                   metavar="NAME=VALUE",
                   help="override a parameter's DEFAULT (and its prior mean) "
                        "in the card. With `fit.py --fix NAME` -- which fixes "
                        "at the default -- this is how a family is switched "
                        "off: `--set k_rad=0` removes the radiative block, "
                        "since S(t) carries it as `+ k_rad (Srad_re + i "
                        "Srad_im)`. Without it there is no way to ask for a "
                        "non-default fixed value: --fix fixes at the default "
                        "and --start-from only seeds FREE parameters.")
    p.add_argument("--floor-scale", type=float, default=1e-7,
                   help="softness of the positivity floor on L_i. A single "
                        "candidate whose Fourier-reconstructed density "
                        "undershoots to a small negative value underflows "
                        "MassCFTerm's own default (1e-9) to exactly 0 and takes "
                        "the whole NLL to -inf. Against a peak density of ~0.4 "
                        "the 1e-7 default here is a 2.5e-7 relative bias and "
                        "keeps log finite; 1e-4 is too aggressive (it moved "
                        "alpha to +71 +- 25 MeV). Pass 0 to fall back to "
                        "MassCFTerm's 1e-9. "
                        "This is NOT residual-mode-only: full physics-kernel "
                        "cards need it just as much (`gate_nanstep.py` "
                        "diagnoses it). A 3 000 000-candidate J/psi leg can "
                        "have candidates already negative AT THE START POINT "
                        "(loss inf, gradient non-finite in most components), "
                        "and a trust-region step that drives a high shape term "
                        "to ~-1 puts hundreds of Z densities negative. Always "
                        "set it explicitly.")
    p.add_argument("--max-resid", type=float, default=10.0,
                   help="residual-mode only: keep |m_reco - m_gen| below this "
                        "[GeV]. A delta kernel convolved with the resolution "
                        "has no support many sigma out, so without a cut the "
                        "far tail underflows the positivity floor and the NLL "
                        "is -inf. The cut is applied AND normalised (the "
                        "truncated likelihood), so it costs nothing.")
    p.add_argument("--eta-lead", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="keep only candidates whose LEADING muon (higher pT) "
                        "has LO <= |eta| < HI. A bias localised in eta points "
                        "at the field/alignment-like effects the joint fit can "
                        "absorb; one that is flat in eta does not.")
    p.add_argument("--eta-max", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="keep only LO <= max(|eta_p|, |eta_m|) < HI. THE SAFE "
                        "BAND VARIABLE, and the one to use: --eta-lead picks "
                        "the leg with the larger RECO pT, so when the legs "
                        "have similar pT which one 'leads' is decided by which "
                        "one fluctuated up and the band edge becomes a cut on "
                        "the residual -- corr(|eta| lead, z) = +0.0203 against "
                        "+0.0025 here and +0.0004 for the gen definition. "
                        "max(|eta_p|, |eta_m|) does not "
                        "depend on which leg leads and needs no truth. The two "
                        "select DIFFERENT candidates (the max-band barrel "
                        "requires BOTH legs central), so cards built with the "
                        "two are not cell-by-cell comparable -- compare the "
                        "SPREAD across bands.")
    p.add_argument("--lead-charge", choices=["any", "plus", "minus"],
                   default="any",
                   help="keep only candidates whose leading muon has this "
                        "charge. A charge-ordered split isolates the "
                        "charge-ODD (misalignment-like) part of a bias.")
    p.add_argument("--vgf-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="keep only LO <= vgf < HI. `vgf` is the GAUSSIAN "
                        "(hit-resolution) share of sigma_m^2, so this splits "
                        "hit-dominated from process-noise-dominated "
                        "candidates. It is the proxy available in the FLAT "
                        "cache; the true per-hit-class shares live in the "
                        "per-group caches (`cf_inmaker.py --groups`).")
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
    p.add_argument("--sigma-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="keep only `LO <= sigma_m < HI` [GeV]. The ABSOLUTE "
                        "sigma, not sigma/m: the likelihood assumes the Born "
                        "spectrum is the same for every candidate whatever its "
                        "sigma_i, and on this sample it is not -- <m_gen> runs "
                        "from 84.94 GeV in the lowest sigma octile to 91.28 in "
                        "the highest, rho(sigma, m_gen) = 0.168. Slicing on "
                        "sigma makes that mis-specification measurable: inside "
                        "a narrow slice the class-conditional Born spectrum is "
                        "a SMOOTH function of m away from the marginal one, "
                        "which the floated K(m) can absorb; inclusively one "
                        "K(m) has to serve every class at once and cannot.")
    p.add_argument("--maxn", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234,
                   help="seed for the --maxn subsample (a HEAD slice would be "
                        "the first tasks, i.e. one contiguous run range)")
    # ---- weights ---------------------------------------------------------
    p.add_argument("--decorrelate-sigma", type=int, default=0,
                   metavar="NCLASS",
                   help="DIAGNOSTIC. Reweight the candidates so that the true "
                        "mass is INDEPENDENT of the per-candidate resolution, "
                        "using NCLASS quantile classes of the absolute "
                        "sigma_m: w_i *= p(m_gen_i) / p(m_gen_i | class_i). "
                        "The likelihood assumes exactly that independence and "
                        "it is badly violated (<m_gen> runs 84.9 -> 91.3 GeV "
                        "across sigma octiles), so if the -11 MeV closure "
                        "failure is this mis-specification the reweighted fit "
                        "must close. It uses the MC truth and is a diagnostic, "
                        "not a correction.")
    p.add_argument("--decorr-clip", type=float, default=5.0,
                   help="cap on the --decorrelate-sigma weight ratio")
    p.add_argument("--decorr-var", choices=["sigma", "srel"], default="sigma",
                   help="which resolution variable the classes are quantiles "
                        "of. `sigma` is the absolute width the term stores; "
                        "`srel` = sigma/m is the EFFECTIVE conditioning "
                        "variable of the fluctuation form, whose "
                        "c_i = sigma_i^2/m_i term makes the map the "
                        "multiplicative kernel m_obs = m'(1 + s_i x) to first "
                        "order. rho(m_gen, .) is 0.168 for the first and 0.035 "
                        "for the second, so which one the fit responds to says "
                        "which conditioning it is really doing.")
    p.add_argument("--toy-shuffle", type=int, default=0, metavar="NCLASS",
                   help="DIAGNOSTIC TOY: replace the observed mass by "
                        "`m_gen_i + sigma_i z_j`, `z_j` the standardized "
                        "residual of a random OTHER candidate in the same "
                        "sigma/m class (NCLASS quantiles). The toy then "
                        "satisfies the likelihood's own assumption exactly -- "
                        "the fluctuation is independent of the true mass -- "
                        "while keeping the empirical per-class residual SHAPE, "
                        "which is already known to be right (pull width 0.996, "
                        "kernel-free closure +0.9 +- 2.1 MeV). It separates the "
                        "two remaining possibilities: if the toy returns the "
                        "same -11 MeV the ASSEMBLY of the chain is wrong; if it "
                        "returns zero the data differ from the model in the "
                        "convolution itself.")
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
                        "log-Jacobian, no dependence on delta_i. `residual` "
                        "applies them to the residual delta_i instead: exact "
                        "at a delta kernel, so it is the form the J/psi gate "
                        "is defined in, but it needs --corr-clip at the Z.")
    p.add_argument("--vpow", type=float, default=0.0,
                   help="THE v FORMULATION. Convolve in "
                        "`v(m) = Int dm/m^p` instead of in `m`, condition each "
                        "candidate on the resolution CONSTANT "
                        "`k_i = sigma_i/m_i^p` instead of on `sigma_i`, and "
                        "give the provider the lineshape's density in `v`. "
                        "0 = off (the m formulation). The mass resolution of a "
                        "track pair scales as `m^p` with `p = 1 + f_hit` "
                        "because the CURVATURE resolution is what is constant, "
                        "so a fixed-width convolution in `m` is wrong at the "
                        "true mass AND makes `sigma_i` a conditioning label "
                        "that carries information about it: measured here, "
                        "`<m_gen>` runs 84.94 -> 91.28 GeV across sigma "
                        "octiles and `rho(sigma, m_gen) = +0.168`, against "
                        "`rho(k, m_gen) = -0.011` at p = 1.25. That is the "
                        "whole of the -11.06 +- 2.27 MeV closure failure "
                        "(reweighting it away gives -0.12 +- 2.42). The "
                        "measured optimum is p = 1.235 and the a-correction's "
                        "own exponent is 1 + <vgf> = 1.264; the predicted bias "
                        "is within +-0.35 MeV over p in [1.20, 1.264].")
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
                   help="MC has no background, so the default is none")
    p.add_argument("--bernstein-degree", type=int, default=2)
    p.add_argument("--fbkg", type=float, default=0.0)
    p.add_argument("--float-bkg", action="store_true")
    p.add_argument("--add-del-family", "--del-family", dest="add_del_family",
                   action="store_true",
                   help="ADD the delta-ray family. `--del-family` is "
                        "accepted as an alias, but it reads as a family "
                        "REMOVER and is the opposite of what it does, so "
                        "prefer the long spelling. To "
                        "remove a family, build from a cache with that "
                        "family's S_* keys dropped -- `discover_families` only "
                        "appends families present in the cache -- or set its "
                        "coefficient to zero with --set.")
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
    if args.eta_max is not None:
        etam_ = np.maximum(np.abs(np.asarray(d["etap"], np.float64)),
                           np.abs(np.asarray(d["etam"], np.float64)))
        lo_e, hi_e = args.eta_max
        keep &= (etam_ >= lo_e) & (etam_ < hi_e)
        steps.append((f"{lo_e:g} <= max(|eta_p|,|eta_m|) < {hi_e:g}",
                      keep.copy()))
    if args.eta_lead is not None or args.lead_charge != "any" or \
            args.vgf_range is not None:
        ptp = np.asarray(d["ptp"], dtype=np.float64)
        ptm = np.asarray(d["ptm"], dtype=np.float64)
        plus_leads = ptp >= ptm
        if args.eta_lead is not None:
            etal = np.abs(np.where(plus_leads, np.asarray(d["etap"], np.float64),
                                   np.asarray(d["etam"], np.float64)))
            lo_e, hi_e = args.eta_lead
            keep &= (etal >= lo_e) & (etal < hi_e)
            steps.append((f"{lo_e:g} <= |eta| lead < {hi_e:g}", keep.copy()))
        if args.lead_charge != "any":
            keep &= plus_leads if args.lead_charge == "plus" else ~plus_leads
            steps.append((f"leading muon is mu{args.lead_charge[0]}",
                          keep.copy()))
        if args.vgf_range is not None:
            v = np.asarray(d["vgf"], dtype=np.float64)
            lo_v, hi_v = args.vgf_range
            keep &= (v >= lo_v) & (v < hi_v)
            steps.append((f"{lo_v:g} <= vgf < {hi_v:g}", keep.copy()))
    if args.sigma_range is not None:
        lo_s, hi_s = args.sigma_range
        keep &= (sigma >= lo_s) & (sigma < hi_s)
        steps.append((f"{lo_s:g} <= sigma_m < {hi_s:g} GeV", keep.copy()))
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


def decorrelation_weights(mgen, sigma, w, nclass, clip, log=print):
    """`p(m_gen) / p(m_gen | sigma class)`, per candidate.

    Breaks the correlation between the per-candidate resolution and the true
    mass, which the likelihood assumes is absent.  The marginal spectrum is
    untouched by construction -- the class-conditional densities average to it
    -- so the model does not have to change; only the pairing of kernel width
    with mass does.
    """
    e = np.linspace(60.0, 120.0, 61)
    q = np.quantile(sigma, np.linspace(0.0, 1.0, nclass + 1))
    q[0], q[-1] = -np.inf, np.inf
    cls = np.clip(np.searchsorted(q, sigma, "right") - 1, 0, nclass - 1)
    b = np.clip(np.searchsorted(e, mgen, "right") - 1, 0, len(e) - 2)
    tot = np.bincount(b, w, len(e) - 1)
    tot = tot / max(tot.sum(), 1e-30)
    r = np.ones(len(mgen))
    for c in range(nclass):
        s_ = cls == c
        h = np.bincount(b[s_], w[s_], len(e) - 1)
        h = h / max(h.sum(), 1e-30)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(h > 0, tot / np.maximum(h, 1e-30), 1.0)
        r[s_] = np.clip(ratio[b[s_]], 1.0 / clip, clip)
    log(f"  --decorrelate-sigma {nclass}: weight ratio median "
        f"{np.median(r):.3f}, p1 {np.percentile(r, 1):.3f}, "
        f"p99 {np.percentile(r, 99):.3f}, {100.0 * np.mean((r <= 1.0 / clip) | (r >= clip)):.2f} % at the cap")
    return r


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
    if args.decorrelate_sigma:
        mg_all = d["eta"].astype(np.float64)
        w_all = w_all.copy()
        rvar = (sig_all if args.decorr_var == "sigma" else srel_all)[idx]
        log(f"  --decorr-var {args.decorr_var}")
        w_all[idx] = w_all[idx] * decorrelation_weights(
            mg_all[idx], rvar, w_all[idx],
            args.decorrelate_sigma, args.decorr_clip, log)
    n = len(idx)
    if not n:
        raise SystemExit("the selection kept nothing")
    sigma = sig_all[idx]
    mreco = m_all[idx]
    mgen = d["eta"].astype(np.float64)[idx]
    vgf = d["vgf"].astype(np.float64)[idx]
    if args.toy_shuffle:
        rng = np.random.default_rng(args.seed + 7)
        srel_sel = sigma / np.maximum(np.abs(mreco), 1e-9)
        zres = (mreco - mgen) / sigma
        qc = np.quantile(srel_sel, np.linspace(0.0, 1.0, args.toy_shuffle + 1))
        qc[0], qc[-1] = -np.inf, np.inf
        cls = np.clip(np.searchsorted(qc, srel_sel, "right") - 1, 0,
                      args.toy_shuffle - 1)
        zperm = zres.copy()
        for c in range(args.toy_shuffle):
            w_ = np.where(cls == c)[0]
            zperm[w_] = zres[rng.permutation(w_)]
        log(f"  TOY SHUFFLE: residuals permuted inside {args.toy_shuffle} "
            f"sigma/m classes. corr(z, m_gen) "
            f"{np.corrcoef(zres, mgen)[0, 1]:+.4f} -> "
            f"{np.corrcoef(zperm, mgen)[0, 1]:+.4f}")
        mreco = mgen + sigma * zperm
        # the window is a cut on the OBSERVED mass and the toy has new ones
        lo_w, hi_w = args.window
        keep_t = (mreco >= lo_w) & (mreco <= hi_w)
        log(f"    re-applying the {lo_w:g}-{hi_w:g} window to the toy: "
            f"{int(keep_t.sum())} of {len(mreco)} "
            f"({100.0 * keep_t.mean():.3f} %)")
        idx = idx[keep_t]
        sigma, mreco, mgen, vgf = (a[keep_t] for a in
                                   (sigma, mreco, mgen, vgf))
        n = len(idx)
    if args.residual_mode:
        # the observable is the RESIDUAL; `alpha` enters as m_ref * alpha, so it
        # keeps its units of 1e-3 of the Z mass
        mobs = (mreco - mgen)
        keep_r = np.abs(mobs) < args.max_resid
        log(f"  |m_reco - m_gen| < {args.max_resid:g} GeV keeps "
            f"{int(keep_r.sum())} of {len(mobs)} "
            f"({100.0*keep_r.mean():.3f} %)")
        idx = idx[keep_r]
        sigma, mreco, mgen, vgf = (a[keep_r] for a in
                                   (sigma, mreco, mgen, vgf))
        mobs = mobs[keep_r]
        n = len(idx)
        # `weights` is built from `idx` further down, so it needs no slicing
        log(f"  RESIDUAL MODE: modelling m_reco - m_gen. "
            f"median {1e3*np.median(mobs):+.2f} MeV, "
            f"RMS {1e3*np.std(mobs):.0f} MeV; the kernel is a delta and the "
            f"FSR fold, the acceptance and K(m) are all OFF")
    else:
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
    fams = discover_families(set(d.files), args.add_del_family)
    log(f"  {n} candidates, nt = {nt}, tau [0, {tgrid[-1]:.4f}], "
        f"families {[f[0] for f in fams]} + hit, upsample {args.fit_upsample}")
    # ---- the v formulation ----------------------------------------------
    vkw = {}
    if args.vpow:
        if args.residual_mode:
            raise SystemExit("--vpow and --residual-mode are exclusive: the "
                             "residual has no mass to raise to a power")
        pw = float(args.vpow)
        vmap = (lambda x: np.log(x)) if pw == 1.0 else \
            (lambda x: x ** (1.0 - pw) / (1.0 - pw))
        v_ref = float(vmap(np.array(args.mref)))
        # the term's `m_ref` is now only the lever of the `alpha` scale: a
        # physical shift `m_ref alpha` is a shift `m_ref alpha / m_ref^p` in v,
        # so the lever is `m_ref^{1-p}`. It is NOT `v(m_ref)`, which is
        # `m_ref^{1-p}/(1-p)`; the two differ by `1/(1-p)` and confusing them
        # rescales alpha silently.
        mref_term = float(args.mref) ** (1.0 - pw)
        k = sigma / mreco ** pw
        vobs = vmap(mreco) - v_ref
        log(f"  --vpow {pw:g}: convolving in v = Int dm/m^p")
        log(f"    sigma -> k = sigma/m^p : median {np.median(sigma):.4f} GeV "
            f"-> {np.median(k):.6g}")
        log(f"    mobs  -> v(m) - v(m_ref): [{vobs.min():.6g}, {vobs.max():.6g}]")
        log(f"    m_ref -> the alpha lever m_ref^(1-p) = {mref_term:.6g} "
            f"(v(m_ref) = {v_ref:.6g}, NOT the same thing)")
        # the window, in the term's own variable `mobs + m_ref`
        lo_v = float(vmap(np.array(lo))) - v_ref + mref_term
        hi_v = float(vmap(np.array(hi))) - v_ref + mref_term
        vkw = dict(vpow=pw, sigma_v=k, mobs_v=vobs, mref_term=mref_term,
                   window_v=(lo_v, hi_v))
        sigma, mobs = k, vobs

    log(f"  sigma: min {sigma.min():.4f} med {np.median(sigma):.4f} "
        f"max {sigma.max():.4f} GeV -> tau needed "
        f"{tgrid[-1]/sigma.min():.2f} 1/GeV")

    families = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    datasets = {"sigma": sigma, "mobs": mobs, "vgf": vgf, "tgrid": tgrid}
    if args.residual_mode or vkw:
        # the corrections must still see the candidate's PHYSICAL mass: in
        # residual mode `mobs` is `m_reco - m_gen`, and under `--vpow` it is
        # `v(m) - v(m_ref)`. Neither can supply it.
        datasets["corr_mass"] = mreco
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
    if args.residual_mode:
        args.shape = 0
        args.fsr = None
        args.acc = None
        args.with_alpha = True
        lo, hi = args.mref - args.max_resid, args.mref + args.max_resid
    t0 = time.time()
    acc = None
    if args.acc:
        with open(args.acc) as fh:
            acc = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
        log(f"  acceptance: {args.acc} ({acc.get('kind','bernstein')}, "
            f"{len(acc.get('coef', []))} coefficients)")
    kw = {}
    if args.fsr_mmax:
        kw["fsr_mmax"] = args.fsr_mmax
    if args.shape:
        kw["shape"] = args.shape
        kw["shape_window"] = tuple(args.shape_window or args.window)
    if vkw:
        kw["vpow"] = vkw["vpow"]
    provider = ZGammaLineshape(
        m_ref=args.mref, window=tuple(args.born_window), nm=args.nm,
        tau_max=args.tau_max, width_scheme=args.width_scheme,
        fsr=args.fsr, acceptance=acc, **kw)
    if vkw:
        # `check_tau_range` compares against the provider's tau_max in MASS
        # units; in v the same physical range is `tau_max * window_hi^p`, which
        # `_build_v` has already applied, so compare against the widths in v.
        need = float(np.max(tgrid) / sigma.min())
        ok = need <= provider.vtau_max
        log(f"  v tau: needed {need:.1f}, tabulated to {provider.vtau_max:.1f} "
            f"-> {'ok' if ok else 'TOO SMALL'}")
    else:
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
                "this rabbit's MassCFTerm has no `corr_form`: it does not "
                "implement the fluctuation form. Pass --corr-form residual "
                "--corr-clip 5 for the clipped residual form.")
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
                "this rabbit's MassCFTerm has no `corr_clip`. Pass "
                "--corr-clip 0 only if you mean to feed the corrections the "
                "full residual.")
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
        kernel=(unbinned.DeltaKernel() if args.residual_mode
                else unbinned.TabulatedLineshapeKernel(provider=provider)),
        background=background,
        m_ref=(vkw["mref_term"] if vkw else args.mref),
        corr_mass=(mreco if (args.residual_mode or vkw) else None),
        **({"vpow": vkw["vpow"]} if vkw else {}),
        scale_param="alpha" if args.with_alpha else None,
        bkg_frac_param="f_bkg" if args.float_bkg else None,
        bkg_frac=args.fbkg,
        weights=weights,
        **({"floor_scale": args.floor_scale} if args.floor_scale else {}),
        norm_window=(None if args.no_window_norm else
                     (vkw["window_v"] if vkw else (lo, hi))),
        norm_tpoints=args.norm_tpoints, norm=norm,
        upsample=args.fit_upsample,
        chunk=args.chunk, channel=args.channel, **kw2)

    if args.residual_mode:
        decl = {}
    else:
        pdkw = {"mz_prior": args.mz_prior or None,
                "gz_prior": args.gz_prior or None}
        if "shape_prior" in inspect.signature(
                provider.param_declarations).parameters:
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
    for spec in getattr(args, "set_params", []) or []:
        nm, eq, val = spec.partition("=")
        if not eq:
            raise SystemExit(f"--set wants NAME=VALUE, got {spec!r}")
        if nm not in decl:
            raise SystemExit(f"--set {nm}: not a parameter of this card; "
                             f"have {sorted(decl)}")
        d0, ps, pm, poi = decl[nm]
        v = float(val)
        # the prior MEAN moves with the default: a parameter set to a value is
        # being asserted, not pulled back towards the card's own default
        decl[nm] = (v, ps, v, poi)
        log(f"  --set {nm} = {v:g}  (was {d0:g})")
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
