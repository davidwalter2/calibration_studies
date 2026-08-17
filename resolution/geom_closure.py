#!/usr/bin/env python3
"""Clean-propagation closure under the FISHER normalization, across THREE
geometries, to decide whether the surviving `locx` non-closure is intrinsic to
the multiple-scattering model or a property of how material is distributed.

THE QUESTION
------------
NOTES_FISHERNORM section 5 records a `+0.045` locx residual at u = 1 on the
HOMOGENEOUS toy that survives every knob tried (100x step limit, 100x
intersection precision, the j0 guard, and the renormalization itself).  It was
measured in ONE geometry.  If it is the same size in the LAYERED toy and in the
REAL tracker, it is a property of the MS model and the next move is Moliere
transform vs what Geant4's MSC samples.  If it tracks the geometry, it is about
material distribution / step structure and the toy conclusion does not
transfer.

WHAT IS COMPUTED
----------------
For every geometry, for `qop` and `locx`, the closure
`<e^{-u z^2}>_data - <e^{-u z^2}>_model` with `z = (data - reference)/s_F`,
`s_F = sigma sqrt(1/I)` and `1/I` from the EXACT FFT inversion of the model CF
(never the saddlepoint).  Quoted as a curve over u, with the plane-to-plane rms
and the statistical error on the plane mean.

Machinery is `fisher_norm.plane_scales` / `cgf_channels.exact_density` +
`fisher_exact`, i.e. exactly the pipeline that produced the reference values;
`closure --control` re-derives the homogeneous numbers before anything else is
believed.

SUBCOMMANDS
-----------
    pairs      model/sim correspondence + stepper provenance.  RUN THIS FIRST:
               a mis-targeted pair produces plausible garbage silently
               (NOTES_TCUT section 0).
    closure    the three-way comparison
    scan       the NSUB and thickness trends inside the layered toy
"""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

import cf_propagation_test as cpt                                # noqa: E402
import fisher_norm as fn                                         # noqa: E402
from cf_propagation_test import (FUNCTIONALS, REF_BRANCH,        # noqa: E402
                                 SIM_BRANCH, load_model, model_phi,
                                 model_variance, weier_scalar)

SCRATCH = fn.SCRATCH
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop"
PLANES = fn.PLANES

UCURVE = fn.UCURVE
UHEAD = fn.UHEAD
IHEAD = fn.IHEAD

# --------------------------------------------------------------------------
# the geometries
#
# `real` is the pair the deck's closure results use (cleanprop/make_slide_figs.py
# header): sim_260808tight_pt3_eta0.30_phi0.70 + model_mu_pt3_eta0.30.  It is NOT
# the realgeo_*/rg_* scratchpad files, which are a different configuration
# (29.33 MeV of reference loss against the sim's 24.99) and are not sim-matched.
# --------------------------------------------------------------------------
GEOMS = {
    "homo":    dict(kind="toy", model="mod_sl10.0.root", sim="sim_homo_scan.root",
                    label="homogeneous toy (rho=0.1073, Z=8/A=16)"),
    "layK1":   dict(kind="toy", model="mK1.root", sim="sK1.root",
                    label="layered toy T=0.10 NSUB=1"),
    "layK4":   dict(kind="toy", model="mK4.root", sim="sK4.root",
                    label="layered toy T=0.10 NSUB=4"),
    "layK16":  dict(kind="toy", model="mK16.root", sim="sK16.root",
                    label="layered toy T=0.10 NSUB=16"),
    "layT0.10": dict(kind="toy", model="mT0.10.root", sim="sT0.10.root",
                     label="layered toy T=0.10 (rho=9.0)"),
    "layT0.50": dict(kind="toy", model="mT0.50.root", sim="sT0.50.root",
                     label="layered toy T=0.50 (rho=1.8)"),
    "layT2.00": dict(kind="toy", model="mT2.00.root", sim="sT2.00.root",
                     label="layered toy T=2.00 (rho=0.45)"),
    # 50x statistics (100k events), regenerated 2026-08-14 as MATCHED sim+model
    # sets by scratchpad/hs_regen.sh.  The 2000-event versions above leave a
    # +-0.006 correlated error on the plane-mean locx closure at u = 1, which is
    # larger than the whole layered-toy signal and larger than the NSUB trend
    # that was previously reported without an error bar.
    "H_homo":   dict(kind="toy", model="hmHOMO.root", sim="hsHOMO.root",
                     label="homogeneous toy, 100k events"),
    "H_layK1":  dict(kind="toy", model="hmK1.root", sim="hsK1.root",
                     label="layered toy T=0.10 NSUB=1, 100k events"),
    "H_layK4":  dict(kind="toy", model="hmK4.root", sim="hsK4.root",
                     label="layered toy T=0.10 NSUB=4, 100k events"),
    "H_layK16": dict(kind="toy", model="hmK16.root", sim="hsK16.root",
                     label="layered toy T=0.10 NSUB=16, 100k events"),
    "H_layT0.50": dict(kind="toy", model="hmT0.50.root", sim="hsT0.50.root",
                       label="layered toy T=0.50, 100k events"),
    "H_layT2.00": dict(kind="toy", model="hmT2.00.root", sim="hsT2.00.root",
                       label="layered toy T=2.00, 100k events"),
    "real":    dict(kind="real",
                    model=f"{CEPH}/model/model_mu_pt3_eta0.30.root",
                    sim=f"{CEPH}/sim_260808tight_pt3_eta0.30_phi0.70/"
                        f"simstates_*.root",
                    label="real tracker, pT=3 eta=0.30 phi=0.70"),
}
THREE = ("homo", "layK16", "real")

_SIMC = {}


def geom_model(g):
    p = GEOMS[g]["model"]
    return fn.load(p) if GEOMS[g]["kind"] == "toy" else fn.load(p)


def geom_sim(g, acceptance="perplane"):
    """The sim, in the same dict layout on both sides.

    Toy: `toy_loader.load_toy_sim` applies the plane frames here (the watcher
    writes global coordinates on purpose).  Real: `cf_propagation_test.load_sim`
    with PER-PLANE acceptance -- the modal-sequence cut drops 5.6 % of pT=3
    muons TAIL-FIRST and biases the closure by +0.0022 at u=1 (NOTES 2026-08-08,
    cleanprop/acceptance_bias.py), which is 5 % of the effect being measured
    here and would land entirely on the `real` column.
    """
    key = (g, acceptance)
    if key in _SIMC:
        return _SIMC[key]
    if GEOMS[g]["kind"] == "toy":
        _SIMC[key] = fn.load_sim(GEOMS[g]["sim"])
    else:
        _SIMC[key] = cpt.load_sim(GEOMS[g]["sim"], acceptance=acceptance)
    return _SIMC[key]


# ==========================================================================
# closure with a statistical error
# ==========================================================================

def closure_rows(legs, sim, func, scale, probes=UCURVE):
    """(rows, err, ks, n, err_mean): (data - model) per plane, its per-plane
    statistical error, and the error on the PLANE MEAN.

    The model term is deterministic, so the error is the error on the sample
    mean of e^{-u z^2}.

    THE PLANE MEAN'S ERROR IS *NOT* err_plane/sqrt(nplanes).  Every plane is
    evaluated on the SAME events, and a ray that scattered early is displaced
    at every later plane, so the per-plane statistics are strongly positively
    correlated -- the naive sqrt(14) division under-states the error by up to
    that factor.  `err_mean` is instead the error of the per-EVENT plane
    average

        c_e = (1/K) sum_k m_ek E_ek (N/n_k),      err = std(c_e)/sqrt(N)

    which carries the full covariance and reduces to the naive form only when
    the planes are independent.  (m_ek is the per-plane acceptance mask; the
    N/n_k factor makes each plane's term an unbiased estimate of its own mean
    when some rays are missing.)
    """
    nl = min(sim["valid"].shape[1], len(legs), len(scale))
    rows, errs, ks, ns = [], [], [], []
    tau = fn.closure_tau(float(np.max(probes)))
    nev = sim["valid"].shape[0]
    acc = np.zeros((len(probes), nev))          # sum_k over kept planes
    goods = [(sim["valid"][:, k] & np.isfinite(sim[SIM_BRANCH[func]][:, k]))
             for k in range(nl)]
    keep = [k for k in range(nl) if goods[k].sum() >= 100]
    # The MODEL half of every plane's row -- <e^{-uz^2}>_model, i.e. the
    # Weierstrass transform of the block CF -- is a function of the legs only
    # and is where all the time goes; the DATA half is one vectorized pass over
    # the events. So only the model half is farmed out, which keeps the big
    # `sim` arrays out of the pickling path entirely (see fisher_norm.pmap).
    # Order is preserved and each worker runs the identical serial code, so
    # this is a scheduling change and not a numerical one.
    global _CR_CTX
    _CR_CTX = (legs, func, scale, probes, tau)
    # kernels built in the parent; the forked workers inherit them
    import cf_nucel_exact as _cnu
    _cnu.warm(legs)
    models = dict(zip(keep, fn.pmap(_closure_model_one, keep)))
    for k in keep:
        good = goods[k]
        s = float(scale[k])
        z = (sim[SIM_BRANCH[func]][good, k] - legs[k][REF_BRANCH[func]]) / s
        e = np.exp(-np.asarray(probes)[:, None] * z[None, :] ** 2)
        rows.append(e.mean(axis=1) - models[k])
        errs.append(e.std(axis=1) / np.sqrt(good.sum()))
        acc[:, good] += e * (nev / good.sum())
        ks.append(k)
        ns.append(int(good.sum()))
    acc /= max(len(ks), 1)
    err_mean = acc.std(axis=1) / np.sqrt(nev)
    return (np.array(rows), np.array(errs), np.array(ks), np.array(ns),
            err_mean)


_CR_CTX = None


def _closure_model_one(k):
    """<e^{-u z^2}>_model on plane k, for every probe. See `closure_rows`."""
    legs, func, scale, probes, tau = _CR_CTX
    s = float(scale[k])
    phi = model_phi(legs, k, FUNCTIONALS[func], s, tau)
    return np.array([weier_scalar(phi, u, tau) for u in probes])


def geom_closure(g, func, norm="Fisher", acceptance="perplane"):
    legs = geom_model(g)
    sim = geom_sim(g, acceptance)
    sc = fn.plane_scales(legs, func, tag=g)
    scale = sc["sF"] if norm == "Fisher" else sc["sigma"]
    rows, errs, ks, ns, err_mean = closure_rows(legs, sim, func, scale)
    return dict(rows=rows, errs=errs, ks=ks, ns=ns, err=err_mean, scales=sc,
                legs=legs, sim=sim, scale=scale)


# ==========================================================================
# subcommand: pairs
# ==========================================================================

def _model_meta(path):
    f = uproot.open(path)
    tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
    a = f[tk].arrays(["detid", "ok", "refglobr", "refp", "reflocx", "reflocz",
                      "refqop"], library="np")
    return {k: np.asarray(v) for k, v in a.items()}


def _toy_sim_meta(path):
    t = uproot.open(fn._path(path))["simstates"]
    a = t.arrays(["detid", "globr", "pabs", "qop"], library="np")
    nev = len(a["detid"])
    npl = 1 + max(int(np.max(d)) for d in a["detid"] if len(d))
    r = [[] for _ in range(npl)]
    for i in range(nev):
        for d, rr in zip(np.asarray(a["detid"][i]), np.asarray(a["globr"][i])):
            r[int(d)].append(float(rr))
    full = [i for i in range(nev) if len(a["detid"][i]) == npl]
    p0 = np.array([a["pabs"][i][0] for i in full])
    p1 = np.array([a["pabs"][i][-1] for i in full])
    return dict(nev=nev, npl=npl, r=np.array([np.median(x) for x in r]),
                nfull=len(full), dE=float(np.median(p0 - p1)) * 1e3,
                seq=np.arange(npl))


def _real_sim_meta(path):
    files = sorted(glob.glob(path))
    t = uproot.open(files[0])["simstates/simstates"]
    a = t.arrays(["detid", "locz", "globr", "pabs"], library="np")
    seqs = [tuple(zip(d.tolist(), np.round(np.asarray(z), 4).tolist()))
            for d, z in zip(a["detid"], a["locz"])]
    modal, n = Counter(seqs).most_common(1)[0]
    sel = [i for i, s in enumerate(seqs) if s == modal]
    p0 = np.array([a["pabs"][i][0] for i in sel])
    p1 = np.array([a["pabs"][i][-1] for i in sel])
    rr = np.median(np.stack([np.asarray(a["globr"][i]) for i in sel]), axis=0)
    ntot = sum(uproot.open(f)["simstates/simstates"].num_entries for f in files)
    return dict(nev=ntot, nfiles=len(files), npl=len(modal),
                modalfrac=n / len(seqs), r=rr,
                seq=np.array([m[0] for m in modal]),
                face=np.array([m[1] for m in modal]),
                dE=float(np.median(p0 - p1)) * 1e3)


def cmd_pairs(args):
    print("=" * 78)
    print("MODEL / SIM PAIR VALIDATION")
    print("=" * 78)
    print("A mis-targeted pair is silent: it produces a plausible closure "
          "number that\nmeans nothing (NOTES_TCUT section 0, "
          "model_pt10_eta0.30_phi0.20).  Four tests:\n"
          "  (1) same number of scoring surfaces\n"
          "  (2) identical (module, entry-face) sequence -- for the toy, the\n"
          "      plane index and the radius\n"
          "  (3) the model's reference sits at the sim's radius on every plane\n"
          "  (4) the reference's energy loss is compatible with the sim's, i.e.\n"
          "      the two saw the SAME material.  The model quotes a MEAN and the\n"
          "      sim a MEDIAN, so the two differ by the Landau mean-mode gap;\n"
          "      what would flag a mis-pair is a gap far outside the 2.4-3.8 MeV\n"
          "      band the matched pairs share.\n")

    for g in args.geoms:
        d = GEOMS[g]
        mp = d["model"] if os.path.sep in d["model"] else os.path.join(
            SCRATCH, d["model"])
        m = _model_meta(mp)
        s = (_toy_sim_meta(d["sim"]) if d["kind"] == "toy"
             else _real_sim_meta(d["sim"]))
        dE_m = 1e3 * float(m["refp"][0] - m["refp"][-1])

        nok = (len(m["detid"]) == s["npl"])
        seqok = nok and bool(np.array_equal(np.asarray(m["detid"]),
                                            s["seq"]))
        dr = (np.abs(np.asarray(m["refglobr"]) - s["r"]).max() if nok
              else np.nan)
        print(f"--- {g:<9s} {d['label']}")
        print(f"    model {os.path.basename(mp)}   "
              f"sim {os.path.basename(d['sim'])}")
        print(f"    (1) legs {len(m['detid'])}  vs  sim planes {s['npl']}"
              f"        {'OK' if nok else 'MISMATCH'}")
        print(f"    (2) detid sequence identical: {seqok}"
              f"{'' if seqok else '   <-- MISMATCH'}")
        if d["kind"] == "real":
            fok = np.abs(np.asarray(m["reflocz"]) - s["face"]).max()
            print(f"        entry-face (locz) max |diff| = {fok:.5f} cm "
                  f"(float32 rounding is ~1e-4)")
            print(f"        modal sequence covers {100*s['modalfrac']:.2f} % "
                  f"of rays (first file); {s['nfiles']} sim files")
        print(f"    (3) max |refglobr - sim median r| = {dr:.4f} cm")
        print(f"    (4) dE  model(mean) {dE_m:7.3f} MeV   "
              f"sim(median) {s['dE']:7.3f} MeV   "
              f"gap {dE_m - s['dE']:+6.3f} MeV")
        print(f"        sim events {s['nev']}"
              + (f", complete-sequence {s['nfull']}"
                 if d["kind"] == "toy" else ""))
        print()

    print("STEPPER PROVENANCE (standing rule: DeltaOneStep=1e-5, "
          "DeltaIntersection=1e-6)")
    print("  toy  : runToyGeomCheck.py prints '[toy] TIGHT stepper' unless "
          "loosestepper=1;")
    print("         grep of the production logs is done by "
          "`geom_closure.py pairs --logs`.")
    print("  real : produced by runCleanPropSim.py, whose TIGHT branch is the "
          "default and")
    print("         whose LOOSE branch is opt-in via CLEANPROP_LOOSE_STEPPER "
          "and prints")
    print("         '>>> LOOSE stepper'.  The absence of that line in the "
          "task logs is the")
    print("         positive evidence (the tight branch prints nothing).")
    rl = sorted(glob.glob(os.path.dirname(GEOMS["real"]["sim"]) + "/task_*.log"))
    if rl:
        nloose = sum(1 for f in rl
                     if "LOOSE stepper" in open(f, errors="ignore").read())
        print(f"    real      {len(rl)} task logs, "
              f"{nloose} contain '>>> LOOSE stepper'  -> "
              f"{'TIGHT' if nloose == 0 else 'LOOSE PRESENT'}")
    if args.logs:
        print()
        for lg, tag in ((f"{SCRATCH}/lsim_homo_scan.log", "homo"),
                        (f"{SCRATCH}/lsK1.log", "layK1"),
                        (f"{SCRATCH}/lsK4.log", "layK4"),
                        (f"{SCRATCH}/lsK16.log", "layK16"),
                        (f"{SCRATCH}/ls0.10.log", "layT0.10"),
                        (f"{SCRATCH}/ls0.50.log", "layT0.50"),
                        (f"{SCRATCH}/ls2.00.log", "layT2.00")):
            if not os.path.exists(lg):
                print(f"    {tag:<9s} LOG MISSING {lg}")
                continue
            txt = open(lg, errors="ignore").read(4000)
            state = ("TIGHT" if "TIGHT stepper" in txt else
                     ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
            print(f"    {tag:<9s} {os.path.basename(lg):<22s} {state}")

    # plane definitions: the toy planes are a function of RADII alone
    print("\nPLANE DEFINITIONS (toy)")
    ns = {}
    exec(open(PLANES).read(), ns)
    print(f"    {PLANES}")
    print(f"    header says '{open(PLANES).readline().strip()}'")
    print(f"    radii = {ns['radii']}")
    print("    gen_toy_config.py builds origin/normal/uaxis from RADII and the\n"
          "    hard-coded (pt, eta, phi0, B, q) ONLY -- T and NSUB do not enter,\n"
          "    so the same file serves both toys.  Verified numerically below\n"
          "    against each sim's own recorded radii.")
    for g in args.geoms:
        if GEOMS[g]["kind"] != "toy":
            continue
        s = _toy_sim_meta(GEOMS[g]["sim"])
        print(f"    {g:<9s} max |sim median r - toyPlanes radii| = "
              f"{np.abs(s['r'] - np.array(ns['radii'])).max():.4f} cm")


# ==========================================================================
# subcommand: closure
# ==========================================================================

def _print_curve(tag, rows, err, extra=""):
    m = rows.mean(axis=0)
    print(f"  {tag:<22s} " + "".join(f"{v:+10.5f}" for v in m) + extra)
    print(f"  {'   +- (stat, corr)':<22s} " + "".join(f"{v:10.5f}" for v in err))
    print(f"  {'   rms over planes':<22s} " +
          "".join(f"{v:10.5f}" for v in rows.std(axis=0)))


def cmd_closure(args):
    if args.control:
        print("=" * 78)
        print("CONTROL -- reproduce the published homogeneous-toy numbers "
              "before trusting anything")
        print("=" * 78)
        print("targets (NOTES_FISHERNORM section 3, mean over 14 planes):")
        print("   Fisher  qop  +0.00118 / +0.02292 / +0.01074   at "
              "u = 0.01 / 0.1 / 1")
        print("   Fisher  locx +0.00219 / +0.01417 / +0.04465")
        print("   sigma   qop  +0.00445 / +0.01893 / +0.00817")
        print("   sigma   locx +0.00181 / +0.01188 / +0.04299\n")
        for func in ("qop", "locx"):
            for norm in ("sigma", "Fisher"):
                r = geom_closure("homo", func, norm=norm)
                m = r["rows"].mean(axis=0)[IHEAD]
                print(f"   {norm:<7s} {func:<5s} " +
                      "".join(f"{v:+11.5f}" for v in m))
        print()

        # Independent check of the REAL-geometry chain: the sigma-normalized
        # locx closure must reproduce the npz that cleanprop's own
        # cf_propagation_test --compare wrote for the same pair.  Different
        # sim loader entry point, different t grid (TAU vs closure_tau), same
        # answer.
        ref = f"{SCRATCH}/accfix/cleanprop_rows_perplane.npz"
        if os.path.exists(ref):
            d = np.load(ref, allow_pickle=True)
            m = d["func"] == "locx"
            tgt = (d["fdata"] - d["fmodel"])[m].mean(axis=0)
            r = geom_closure("real", "locx", norm="sigma")
            print("REAL-GEOMETRY CHAIN CHECK  (sigma normalization, "
                  "perplane acceptance, locx)")
            print(f"   cf_propagation_test npz  " +
                  "".join(f"{v:+11.5f}" for v in tgt))
            print(f"   geom_closure.py          " +
                  "".join(f"{v:+11.5f}"
                          for v in r["rows"].mean(axis=0)[IHEAD]))
            print()

    print("=" * 78)
    print("CLOSURE UNDER THE FISHER NORMALIZATION, THREE GEOMETRIES")
    print("=" * 78)
    print("s_F = sigma sqrt(1/I), 1/I by exact FFT inversion of the model CF.\n"
          "u is in units of THIS normalization for every row, so the columns "
          "are\ncommensurable across geometries.\n")
    print(step_structure(args.geoms))

    # Campaign-level parallelism: build EVERY (geometry, functional) plane-scale
    # set in one flat pool before the per-geometry loop runs, so the machine
    # sees geometries x functionals x planes at once rather than one geometry's
    # planes at a time. The loop below then hits the cache.
    fn.prewarm_scales([(geom_model(g), func, g)
                       for g in args.geoms for func in args.funcs])

    res = {}
    for g in args.geoms:
        for func in args.funcs:
            res[(g, func)] = geom_closure(g, func, norm=args.norm)

    for func in args.funcs:
        print(f"### {func}   (mean over planes; u in units of s_F)")
        print(f"  {'geometry':<22s} " +
              "".join(f"{u:>10.3g}" for u in UCURVE))
        for g in args.geoms:
            r = res[(g, func)]
            _print_curve(g, r["rows"], r["err"],
                         f"   [{len(r['ks'])} planes, "
                         f"{int(np.median(r['ns']))} ev/plane]")
        print()

    print("### headline probes, both functionals")
    print(f"  {'geometry':<22s} " +
          "".join(f"{f + ' u=' + str(u):>18s}"
                  for f in args.funcs for u in UHEAD))
    for g in args.geoms:
        cells = []
        for f in args.funcs:
            r = res[(g, f)]
            m = r["rows"].mean(axis=0)[IHEAD]
            e = r["err"][IHEAD]
            cells += [f"{mi:+.5f}+-{ei:.5f}" for mi, ei in zip(m, e)]
        print(f"  {g:<22s} " + "".join(f"{c:>18s}" for c in cells))
    print()

    if args.perplane:
        for func in args.funcs:
            print(f"### {func}: per-plane closure at u = 1 "
                  f"(radius-ordered)")
            for g in args.geoms:
                r = res[(g, func)]
                legs = r["legs"]
                print(f"  {g}")
                print(f"    {'k':>3} {'1/I':>8} {'s_F':>11} "
                      f"{'u=0.01':>10} {'u=0.1':>10} {'u=1':>10}")
                for i, k in enumerate(r["ks"]):
                    print(f"    {k:3d} {r['scales']['invI'][k]:8.4f} "
                          f"{r['scale'][k]:11.4e} " +
                          "".join(f"{v:+10.5f}" for v in r["rows"][i, IHEAD]))
            print()

    if args.dump:
        out = {}
        for (g, f), r in res.items():
            out[f"{g}|{f}|rows"] = r["rows"]
            out[f"{g}|{f}|errs"] = r["errs"]
            out[f"{g}|{f}|err"] = r["err"]
            out[f"{g}|{f}|ks"] = r["ks"]
            out[f"{g}|{f}|invI"] = r["scales"]["invI"]
            out[f"{g}|{f}|sF"] = r["scales"]["sF"]
            out[f"{g}|{f}|sigma"] = r["scales"]["sigma"]
        np.savez(args.dump, u=UCURVE, **out)
        print(f"wrote {args.dump}")


# ==========================================================================
# subcommand: scan
# ==========================================================================

def cmd_scan(args):
    print("=" * 78)
    print("LAYERED TOY: does the residual track the STEP STRUCTURE?")
    print("=" * 78)
    print("NSUB subdivides each 1 mm station into NSUB sub-shells of the SAME\n"
          "material: rho, total material, radial distribution and scoring\n"
          "surfaces are all identical, so ONLY the forced-boundary count (and\n"
          "hence the model's step length, which is boundary-limited) changes.\n"
          "A trend here is evidence for a step-structure origin.\n")
    print(step_structure(args.geoms))
    print(near_plane_share(args.geoms))
    for func in args.funcs:
        print(f"### {func}, Fisher normalization, mean over planes "
              f"(correlated stat error in brackets)")
        print(f"  {'geometry':<22s} {'ms/leg':>7} {'ioni/leg':>9} "
              f"{'dE[MeV]':>8} " + "".join(f"{'u=' + str(u):>17s}"
                                           for u in UHEAD)
              + f"{'rms(u=1)':>11}")
        for g in args.geoms:
            r = geom_closure(g, func, norm="Fisher")
            legs = r["legs"]
            nms = sum(len(l["ms"]) for l in legs) / len(legs)
            nio = sum(len(l["ioni"]) for l in legs) / len(legs)
            dE = 1e3 * (1.0 / abs(legs[0]["refqop"])
                        - 1.0 / abs(legs[-1]["refqop"]))
            m = r["rows"].mean(axis=0)[IHEAD]
            e = r["err"][IHEAD]
            print(f"  {g:<22s} {nms:7.1f} {nio:9.1f} {dE:8.2f} " +
                  "".join(f"{mi:+9.5f}({ei:.5f})" for mi, ei in zip(m, e)) +
                  f"{r['rows'].std(axis=0)[IHEAD][-1]:11.5f}")
        print()


def near_plane_share(geoms, func="locx"):
    """How much of the MS variance is generated in material the ray is still
    INSIDE when it reaches the scoring plane.

    A step of length L whose far end sits a lever arm D from the plane
    contributes to a POSITION functional with weight proportional to D at the
    end of the step and D+L at its start, so `w_end/w_start = D/(D+L)`.  Steps
    with that ratio below 1/2 have D < L: the plane is inside, or barely
    beyond, the material that scattered the ray.  Those are exactly the steps
    for which the end-of-step point-kick bookkeeping is worst and for which the
    model has to get the WITHIN-step displacement right (the
    `D^2 -> D^2 + DL + L^2/3` correction that `step_transports` documents and
    that `MS_NSUB = 4` approximates by quadrature).

    Two numbers per plane, both weighted by each step's own chi_c^2 (the MS
    variance coefficient), not by step count:

      near   = share of the MS variance from steps with w_end/w_start < 0.9,
               i.e. from material within ~10 % of the remaining lever arm of
               the plane;
      q/e    = (sub-step quadrature) / (end-of-step point kick), the size of
               the within-step displacement correction itself.

    This is a MEASUREMENT of how the geometries differ, offered as the obvious
    thing to test next.  It is not by itself evidence that the within-step
    displacement is the cause of anything.
    """
    from cf_ms_exact import moliere_params
    avec = FUNCTIONALS[func]
    out = ["### where the MS variance is generated, relative to the scoring "
           "plane",
           f"  ({func}; chi_c^2-weighted.  near = share from steps with "
           f"w_end/w_start < 0.9,",
           "   q/e = sub-step quadrature over end-of-step point kick)",
           f"  {'geometry':<22s} {'<near>':>8} {'max near':>9} "
           f"{'<q/e>':>8} {'max q/e':>8}   per-plane near"]
    for g in geoms:
        legs = geom_model(g)
        near, qe = [], []
        for k in range(len(legs)):
            A_ms, _, A_ms0 = cpt.step_transports(legs, k)
            tot, low, vq, ve = 0.0, 0.0, 0.0, 0.0
            for j in range(k + 1):
                leg = legs[j]
                if not len(leg["ms"]):
                    continue
                wv = np.einsum("i,sij->sj", avec, A_ms[j])
                wv0 = np.einsum("i,sij->sj", avec, A_ms0[j])
                coslam = leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0

                def _w(v):
                    return np.sqrt(v[:, 1] ** 2
                                   + (v[:, 2] / max(coslam, 1e-3)) ** 2)
                we, w0 = _w(wv), _w(wv0)
                st = leg["ms"]
                c = np.array([moliere_params(*s[:5], s[7], s[8])[0]
                              if st.shape[1] >= 10
                              else moliere_params(*s[:5])[0] for s in st])
                f = ((1 + np.arange(cpt.MS_NSUB)) / cpt.MS_NSUB
                     - 0.5 / cpt.MS_NSUB)
                wq2 = np.mean((w0[:, None] + f[None, :]
                               * (we - w0)[:, None]) ** 2, axis=1)
                r = np.where(w0 > 0, we / np.maximum(w0, 1e-300), 1.0)
                tot += float((c * wq2).sum())
                low += float((c * wq2)[r < 0.9].sum())
                vq += float((c * wq2).sum())
                ve += float((c * we ** 2).sum())
            near.append(low / tot if tot > 0 else np.nan)
            qe.append(vq / ve if ve > 0 else np.nan)
        near, qe = np.array(near), np.array(qe)
        out.append(f"  {g:<22s} {near.mean():8.4f} {near.max():9.4f} "
                   f"{qe.mean():8.4f} {qe.max():8.4f}   " +
                   " ".join(f"{v:.3f}" for v in near[:6]) + " ...")
    return "\n".join(out) + "\n"


def step_structure(geoms):
    """The model's per-step material, in RADIATION LENGTHS.

    d/X0 per step is the variable Moliere's log term is non-additive in, and
    unlike a step LENGTH it is comparable across geometries of different
    density and composition.  The material-weighted mean sum(d^2)/sum(d) is
    the step size the SCATTERING actually sees (a step contributes to the
    variance in proportion to its own d), rather than the unweighted median,
    which the vacuum gaps dominate in the layered toy.
    """
    out = ["### model step structure, per-step d/X0 (msmoliv column 6)",
           f"  {'geometry':<22s} {'nms':>6} {'sum d/X0':>9} "
           f"{'<d/X0>_w':>10} {'max':>10}"]
    for g in geoms:
        p = GEOMS[g]["model"]
        p = p if os.path.sep in p else os.path.join(SCRATCH, p)
        f = uproot.open(p)
        tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
        a = f[tk].arrays(["ileg", "msmoliv"], library="np")
        st = np.vstack([np.asarray(v, dtype=float).reshape(
            -1, 10 if np.asarray(v).size % 10 == 0 else 8)
            for v in a["msmoliv"]])
        d = st[:, 6]
        out.append(f"  {g:<22s} {len(d):6d} {d.sum():9.4f} "
                   f"{(d ** 2).sum() / d.sum():10.3e} {d.max():10.3e}")
    return "\n".join(out) + "\n"


# ==========================================================================

def cmd_acceptance(args):
    """Does the real geometry's radial `qop` growth survive per-plane
    acceptance -- and can the acceptance and material-sampling contaminations
    be told apart?

    NOTE ON THE PREMISE.  `geom_closure.py` has used `acceptance="perplane"`
    for the real geometry since the first run (`geom_sim` default), so the
    +0.0004 -> +0.0676 profile ALREADY carries the remedy.  This subcommand
    measures the other mode as well, so the size of the acceptance effect is a
    number rather than an assumption.

    WHAT THIS COMPARISON DOES AND DOES NOT ISOLATE.  It was tempting to argue
    that modal (common module sequence) suppresses material-sampling variation
    while perplane suppresses tail selection, so that the pair separates the
    two contaminations.  MEASURED, THAT IS FALSE: the spread of median energy
    loss across wander quintiles is 15.21 % under perplane and 13.59 % under
    modal, i.e. requiring a common module SEQUENCE barely constrains the path
    WITHIN those modules.  So this comparison isolates ACCEPTANCE only, and
    material sampling has to be probed separately (`cmd_wander`).

      modal     tail-selective, drops 5.6 % of rays, same 19 modules and entry
                faces for every kept ray;
      perplane  every ray that crossed THIS plane's (detid, face) is used on
                this plane whatever it did elsewhere.

    `s_F` is a property of the MODEL alone and is identical in the two, so the
    closures are directly comparable.
    """
    g = args.geom
    legs = geom_model(g)
    print("=" * 78)
    print(f"ACCEPTANCE DEPENDENCE OF THE PER-PLANE PROFILE -- {g}")
    print("=" * 78)
    print(cmd_acceptance.__doc__.split("NOTE ON THE PREMISE.")[1]
          .split("So an acceptance")[0].strip())
    print()

    res = {}
    for mode in ("modal", "perplane"):
        sim = geom_sim(g, mode)
        ntot = int(sim["ntot"])
        nper = sim["valid"].sum(axis=0)
        res[mode] = dict(ntot=ntot, nper=nper)
        for func in args.funcs:
            sc = fn.plane_scales(legs, func, tag=g)
            rows, errs, ks, ns, err = closure_rows(legs, sim, func, sc["sF"])
            res[mode][func] = dict(rows=rows, errs=errs, ks=ks, ns=ns,
                                   err=err, sF=sc["sF"])

    ntot = res["perplane"]["ntot"]
    print(f"### sample kept, per plane   (sim has {ntot} rays)")
    print(f"  {'k':>3} {'r[cm]':>8} {'modal':>10} {'perplane':>10} "
          f"{'modal lost':>11} {'perplane lost':>14} {'recovered':>10}")
    nmodal = int(res["modal"]["nper"][0])
    rr = np.nanmedian(geom_sim(g, "perplane")["globr"], axis=0)
    for k in range(len(legs)):
        npp = int(res["perplane"]["nper"][k])
        fm, fp = 1 - nmodal / ntot, 1 - npp / ntot
        print(f"  {k:3d} {rr[k]:8.2f} "
              f"{nmodal:10d} {npp:10d} {100*fm:10.3f}% {100*fp:13.3f}% "
              f"{100*(fm - fp)/max(fm, 1e-12):9.1f}%")
    print()

    for func in args.funcs:
        A, B = res["modal"][func], res["perplane"][func]
        print(f"### {func}: per-plane closure at the three probes, "
              f"BOTH acceptances (Fisher)")
        print(f"  {'k':>3} " + "".join(f"{'modal u=' + str(u):>16s}"
                                       for u in UHEAD)
              + "".join(f"{'pplane u=' + str(u):>16s}" for u in UHEAD)
              + f"{'extrap u=1':>13s}")
        for i, k in enumerate(B["ks"]):
            j = int(np.where(A["ks"] == k)[0][0]) if k in A["ks"] else None
            cells = ("".join(f"{A['rows'][j, m]:+16.5f}" for m in IHEAD)
                     if j is not None else " " * 48)
            npp = int(res["perplane"]["nper"][k])
            fm, fp = 1 - nmodal / ntot, 1 - npp / ntot
            ex = np.nan
            if j is not None and fm > fp:
                # linear in the LOST fraction, extrapolated to zero loss.
                # Two points only, so this is a bound-style estimate, not a fit.
                ex = (B["rows"][i, IHEAD[-1]] - fp
                      * (A["rows"][j, IHEAD[-1]] - B["rows"][i, IHEAD[-1]])
                      / (fm - fp))
            print(f"  {k:3d} " + cells +
                  "".join(f"{B['rows'][i, m]:+16.5f}" for m in IHEAD) +
                  f"{ex:+13.5f}")
        print(f"  {'mean':>3} " +
              "".join(f"{A['rows'].mean(axis=0)[m]:+16.5f}" for m in IHEAD) +
              "".join(f"{B['rows'].mean(axis=0)[m]:+16.5f}" for m in IHEAD))
        print(f"  {'+-':>3} " +
              "".join(f"{A['err'][m]:16.5f}" for m in IHEAD) +
              "".join(f"{B['err'][m]:16.5f}" for m in IHEAD))
        gm = (A["rows"][-1, IHEAD[-1]], A["rows"][0, IHEAD[-1]])
        gp = (B["rows"][-1, IHEAD[-1]], B["rows"][0, IHEAD[-1]])
        print(f"  growth at u=1, innermost -> outermost:  "
              f"modal {gm[1]:+.5f} -> {gm[0]:+.5f}   "
              f"perplane {gp[1]:+.5f} -> {gp[0]:+.5f}")
        print()


def cmd_wander(args):
    """Direct measurement of MATERIAL-SAMPLING variation, independent of the
    closure test.

    The model propagates ONE reference path; a real ray that wanders
    transversely crosses different material.  In a cylindrically symmetric toy
    this cannot happen by construction -- wander moves the ray to an equivalent
    point of the same shell -- so the toy is the null control for the effect.

    Measured as: bin rays by their transverse wander at the outermost plane,
    then take the MEDIAN energy loss to that plane per bin.  The median is used
    rather than the mean so that Landau straggling (which is not material
    sampling) does not drive the spread.  A pure straggling effect gives a flat
    profile; a path-length/material effect gives a monotone rise.
    """
    print("=" * 78)
    print("MATERIAL-SAMPLING PROBE: median energy loss vs transverse wander")
    print("=" * 78)
    print("  Wander = the ray's local transverse offset at the outermost plane\n"
          "  (locx in the bending direction, locy along z).  Quintiles of\n"
          "  |wander|; median (p_first - p_last) in each.  A cylindrically\n"
          "  symmetric toy MUST be flat here -- that is the control.\n")
    for g in args.geoms:
        sim = geom_sim(g, args.acceptance)
        legs = geom_model(g)
        kl = len(legs) - 1
        m = sim["valid"][:, kl] & sim["valid"][:, 0]
        for key in ("locx", "locy"):
            if key not in sim:
                continue
            m = m & np.isfinite(sim[key][:, kl])
        dE = 1e3 * (sim["pabs"][m, 0] - sim["pabs"][m, kl])
        w = np.hypot(sim["locx"][m, kl] - np.median(sim["locx"][m, kl]),
                     sim["locy"][m, kl] - np.median(sim["locy"][m, kl]))
        q = np.quantile(np.abs(w), np.linspace(0, 1, 6))
        meds, ns = [], []
        for i in range(5):
            s = (np.abs(w) >= q[i]) & (np.abs(w) <= q[i + 1])
            meds.append(float(np.median(dE[s])))
            ns.append(int(s.sum()))
        meds = np.array(meds)
        # error on a median ~ 1.253 sigma/sqrt(n)
        emed = 1.253 * float(np.std(dE)) / np.sqrt(np.mean(ns))
        print(f"  {g:<12s} n={m.sum():7d}  wander quintile medians [MeV]: " +
              " ".join(f"{v:7.3f}" for v in meds) +
              f"   spread {100*(meds[-1]/meds[0] - 1):+6.2f} %"
              f"   (+-{100*emed/meds[0]:.2f} % per bin)")
    print()

    if not args.closure:
        return

    # ---- does the growth live on the OFF-REFERENCE paths? -----------------
    # Restrict to the LOWEST wander quintile: the sub-ensemble for which the
    # model's single-reference-path assumption is best.  If material sampling
    # carries the radial growth, the growth must shrink here.
    #
    # THE TEST IS ONE-SIDED, and that is what makes it usable.  Cutting on
    # wander also truncates the MS tail, which the model DOES include, so the
    # data look narrower than the model and the closure is pushed MORE
    # positive.  A SMALLER growth in the low-wander quintile is therefore a
    # robust statement; a larger one is inconclusive.
    print("### per-plane closure restricted by wander quintile "
          "(one-sided test, see code comment)")
    for g in args.geoms:
        if GEOMS[g]["kind"] != "real":
            continue
        sim, legs = geom_sim(g, args.acceptance), geom_model(g)
        kl = len(legs) - 1
        # WHICH wander variable, and why it matters.  Built from locx, the test
        # is CIRCULAR for the locx functional -- the selection variable IS the
        # residual being tested, and the low/high quintiles come out at +0.32 /
        # -0.32 at the outermost plane, which is the cut talking, not physics.
        # `locy` is the NON-BENDING direction: it does not enter qop at all and
        # enters locx only through the (tiny) frame coupling, while still
        # deciding which material the ray crosses in a barrel.  It is the
        # honest proxy; `both` is kept only to display the artefact.
        if args.wandervar == "locy":
            w = np.abs(sim["locy"][:, kl]
                       - np.nanmedian(sim["locy"][:, kl]))
        else:
            w = np.hypot(sim["locx"][:, kl] - np.nanmedian(sim["locx"][:, kl]),
                         sim["locy"][:, kl] - np.nanmedian(sim["locy"][:, kl]))
        ok = np.isfinite(w)
        qlo, qhi = np.nanquantile(w[ok], [0.2, 0.8])
        for func in args.funcs:
            sc = fn.plane_scales(legs, func, tag=g)
            print(f"  --- {g}  {func}   (u = 1)")
            print(f"  {'k':>3} {'r[cm]':>8} {'all':>11} {'low wander':>12} "
                  f"{'high wander':>13}")
            rr = np.nanmedian(sim["globr"], axis=0)
            for k in (0, kl // 3, 2 * kl // 3, kl - 1, kl):
                row = []
                for sel in (np.ones(len(w), bool), ok & (w <= qlo),
                            ok & (w >= qhi)):
                    sub = {kk: (vv[sel] if isinstance(vv, np.ndarray)
                                and vv.ndim == 2 else vv)
                           for kk, vv in sim.items()}
                    sub["valid"] = sim["valid"][sel]
                    sub["ntot"] = int(sel.sum())
                    r = fn.closure_plane(legs, sub, func, k, sc["sF"][k],
                                         [1.0])
                    row.append(np.nan if r is None else float(r[0]))
                print(f"  {k:3d} {rr[k]:8.2f} " +
                      "".join(f"{v:+12.5f}" for v in row))
        print()


def cmd_diag(args):
    """Structural differences between the geometries, and the one toy-only
    approximation that could have faked them."""
    print(step_structure(args.geoms))
    print(near_plane_share(args.geoms, func=args.func))
    print("### the toy tangent-plane approximation, in units of s_F")
    print("  The toy watcher scores at the CYLINDER crossing while the model\n"
          "  propagates to a TANGENT PLANE, so the sim state sits a normal\n"
          "  offset locz off the plane and its locx is wrong by locz*dx/dz.\n"
          "  Both toys carry this identically; the real geometry does not carry\n"
          "  it at all (its planes ARE the module surfaces).  If it were large\n"
          "  it would be a toy-only artefact masquerading as a geometry effect.")
    print(f"  {'geometry':<12s} {'k':>3} {'rms locz [um]':>14} "
          f"{'rms locz dx/dz [um]':>20} {'s_F [um]':>10} {'ratio':>9}")
    for g in args.geoms:
        if GEOMS[g]["kind"] != "toy":
            continue
        sim, legs = geom_sim(g), geom_model(g)
        sc = fn.plane_scales(legs, "locx", tag=g)
        for k in (0, 3, len(legs) - 1):
            m = sim["valid"][:, k]
            c = sim["locz"][m, k] * sim["dxdz"][m, k]
            print(f"  {g:<12s} {k:3d} {1e4 * np.std(sim['locz'][m, k]):14.2f} "
                  f"{1e4 * np.std(c):20.4f} {1e4 * sc['sF'][k]:10.2f} "
                  f"{np.std(c) / sc['sF'][k]:9.5f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("pairs")
    q.add_argument("--geoms", nargs="+", default=list(GEOMS))
    q.add_argument("--logs", action="store_true", default=True)
    q.set_defaults(fn=cmd_pairs)

    q = sub.add_parser("closure")
    q.add_argument("--geoms", nargs="+", default=list(THREE))
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--norm", default="Fisher", choices=("Fisher", "sigma"))
    q.add_argument("--control", action="store_true")
    q.add_argument("--perplane", action="store_true")
    q.add_argument("--dump", default="")
    q.set_defaults(fn=cmd_closure)

    q = sub.add_parser("scan")
    q.add_argument("--geoms", nargs="+",
                   default=["layK1", "layK4", "layK16", "layT0.10",
                            "layT0.50", "layT2.00", "homo"])
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.set_defaults(fn=cmd_scan)

    q = sub.add_parser("acceptance")
    q.add_argument("--geom", default="real")
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.set_defaults(fn=cmd_acceptance)

    q = sub.add_parser("wander")
    q.add_argument("--geoms", nargs="+",
                   default=["real", "H_homo", "H_layK1", "H_layK16"])
    q.add_argument("--acceptance", default="perplane")
    q.add_argument("--closure", action="store_true")
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--wandervar", default="locy", choices=("locy", "both"))
    q.set_defaults(fn=cmd_wander)

    q = sub.add_parser("diag")
    q.add_argument("--geoms", nargs="+",
                   default=["homo", "layK1", "layK4", "layK16", "layT0.50",
                            "layT2.00", "real"])
    q.add_argument("--func", default="locx")
    q.set_defaults(fn=cmd_diag)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
