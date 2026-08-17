#!/usr/bin/env python3
"""What Geant4's MSC contains that the analytic Moliere transform does not.

CONTEXT
-------
NOTES_CLOSURE_FINAL closed `qop` at its statistical floor and left `locx`
3.5-14 sigma from closure for all eight species, 70/72 probes positive
(the model OVER-predicts), with MS carrying 85-95 % of that functional's
kappa2.  This module is the MS-channel analogue of NOTES_DELTASPEC /
NOTES_SAMPLERGAP: name what the simulation's scattering model contains that
the model's transform does not, and size each term.

WHAT THE SIMULATION ACTUALLY RUNS  (established, not inferred -- see NOTES)
--------------------------------------------------------------------------
FTFP_BERT_EMM -> CMSEmStandardPhysics -> G4EmBuilder::ConstructCharged(..,
isWVI = true).  For mu+-, pi+-, K+-, p, pbar:

    msc process : G4MuMultipleScattering / G4hMultipleScattering
    msc model   : G4WentzelVIModel   (NOT G4UrbanMscModel)
    plus        : G4CoulombScattering ("CoulombScat"), combined mode

with the Geant4 defaults CMS never overrides for mu/hadrons:
`MscThetaLimit = pi` (so the msc<->CoulombScat handover is the DYNAMIC
nuclear-size angle 1-cos = factorA2*A^-2/3/p^2), `mscStepLimitMuHad =
fMinimal`, `rangeFactorMuHad = 0.2`, `muhadLateralDisplacement = FALSE`,
`useMottCorrection = false`, `nucFormfactor = fExponentialNF`.

SUBCOMMANDS
    sim       the `nocs` arm: CoulombScat deactivated, everything else as in
              the published `off` arm.  The single-scattering half of the
              WentzelVI/CoulombScat split, removed with the one switch this
              study has proven live (ProcessActivationWatcher INSIDE the
              Simulation biglib).
    live      the step census for that arm -- CoulombScat must be ABSENT
    closure   the locx/qop closure of the `nocs` arm against the published
              `off` arm
    gauge     controlled perturbations of the model's MS channel through the
              registered knobs (KMS_SCALE, MS_NSUB, theta_cut), reported as
              SIZE and SHAPE (cosine against the measured residual)
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("RES_NO_PHI_CACHE", "1")

import cf_propagation_test as cpt                                # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402
import cf_ms_exact as cme                                        # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
import hadron_probe as hp                                        # noqa: E402
import speciesdedx as sd                                         # noqa: E402
import barkas_probe as bp                                        # noqa: E402
import radoff_species as rs                                      # noqa: E402

OUT = hp.OUT
LOGD = os.path.join(hp.SCRATCH, "msterms")
os.makedirs(LOGD, exist_ok=True)

UCURVE = fn.UCURVE
UROW = "".join(f"{u:>9g}" for u in UCURVE)

# ---------------------------------------------------------------- the new arm
#
# `nocs` = the published `off` arm (nuclear + Decay off, radiation ON) PLUS
# CoulombScat off.  It is deliberately a ONE-SIDED switch: the model has no
# notion of the WentzelVI/CoulombScat split at all, so removing CoulombScat
# makes the SIMULATION lose a channel the model still carries.  The size of
# the closure move is therefore the size of that channel in `locx`, which is
# exactly the census question ("what populates the tail") answered with a
# switch instead of a watcher field.
hp.ARMS["nocs"] = ("nuclear + Decay off, radiation ON, CoulombScat OFF")

_ORIG_INACT = hp.inact_of


def _inact_of(pdg, arm):
    if arm == "nocs":
        return _ORIG_INACT(pdg, "off") + ["CoulombScat"]
    return _ORIG_INACT(pdg, arm)


hp.inact_of = _inact_of

# The seed pin of NOTES_PION s1 must be installed (radoff_species asserts it;
# importing it here is what installs it, via speciesdedx -> pion_probe).
assert "_s1" in hp.sim_glob(13, "off"), "the seed pin is not installed"


def _fmt(v, w=9, p=5):
    return "".join(f"{x:{w}.{p}f}" for x in v)


def _rms(v):
    return float(np.sqrt((np.asarray(v) ** 2).mean()))


# =========================================================================
# 1.  the simulations
# =========================================================================

def cmd_sim(args):
    hp.cmd_sim(argparse.Namespace(pdg=args.pdg, arms=["nocs"],
                                  events=args.events, jobs=args.jobs,
                                  seed0=args.seed0, workers=args.workers,
                                  force=args.force))


def cmd_live(args):
    """CoulombScat must define EXACTLY zero steps in the `nocs` arm, and the
    `off` arm is printed beside it as the positive control."""
    print("=" * 110)
    print("SWITCH LIVENESS.  A deactivated process must define EXACTLY ZERO "
          "steps; absence from the census table IS zero.")
    print("=" * 110)
    print(f"  {'species':<8}{'arm':<8}{'logs':>6}{'CoulombScat':>14}"
          f"{'msc':>10}{'ioni':>14}{'accept':>10}")
    for pdg in args.pdg:
        for arm in args.arms:
            files = sorted(glob.glob(hp.sim_glob(pdg, arm)))
            logs = [f[:-5] + ".log" for f in files]
            tot = {}
            for lg in logs:
                for k, (pri, _all) in hp.steps_from_log(lg).items():
                    tot[k] = tot.get(k, 0) + pri
            ion = sum(v for k, v in tot.items()
                      if k in ("muIoni", "hIoni"))
            try:
                _c, acc = hp._acc(pdg, arm)
            except Exception:
                acc = float("nan")
            cs = tot.get("CoulombScat")
            print(f"  {hp.SPECIES[pdg]['label']:<8}{arm:<8}{len(logs):>6}"
                  f"{(str(cs) if cs is not None else '(absent=0)'):>14}"
                  f"{tot.get('msc', 0):>10}{ion:>14}{acc:>10.4f}")


# =========================================================================
# 2.  the closure of the new arm against the published one
# =========================================================================

def _rows(pdg, arm, func, model_rad=True):
    """One closure cell, with the three-cache discipline of
    `radoff_species._rows`.  `arm` selects the SIMULATION; the model is
    always the published all-four-on, radiation-ON export."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    kok = hp.kok_for(pdg)
    cpt.RAD_CHANNEL = bool(model_rad)
    ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        path = rs.mp(pdg, model_rad)
        legs = cpt.load_model(path)
        sim = hp.sim_of(pdg, arm)
        sc = rs._scale(legs, func, path, rs.chans(model_rad))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(m=rows.mean(axis=0), err=err, nev=int(sim["valid"].shape[0]),
                sF=float(np.mean(sc["sF"])))


def cmd_closure(args):
    print("=" * 152)
    print("THE WentzelVI/CoulombScat SPLIT, MEASURED: the `off` arm (published) "
          "against `nocs` (CoulombScat deactivated).")
    print("=" * 152)
    print("Same model in both columns (all four corrections ON, radiation ON), "
          "same seeds, same geometry; only the")
    print("SIMULATION differs.  A move here is the size of the explicit "
          "single-scattering channel in that functional.")
    print("=" * 152)
    res = {}
    for func in args.funcs:
        print()
        print(f"# FUNCTIONAL: {func}")
        print(f"  {'species':<8}{'sim arm':<8}" + UROW + "      rms       nev")
        for pdg in args.pdg:
            for arm in args.arms:
                if not glob.glob(hp.sim_glob(pdg, arm)):
                    print(f"  {hp.SPECIES[pdg]['label']:<8}{arm:<8}(sim missing)")
                    continue
                r = _rows(pdg, arm, func)
                res[(func, pdg, arm)] = r
                print(f"  {hp.SPECIES[pdg]['label']:<8}{arm:<8}" + _fmt(r["m"])
                      + f" {_rms(r['m']):9.5f} {r['nev']:>9d}")
                print(f"  {'':<8}{'+-':<8}" + _fmt(r["err"]))
            if all((func, pdg, a) in res for a in ("off", "nocs")):
                a, b = res[(func, pdg, "off")], res[(func, pdg, "nocs")]
                d = b["m"] - a["m"]
                de = np.hypot(a["err"], b["err"])
                sig = d / np.where(de > 0, de, np.nan)
                print(f"  {'':<8}{'nocs-off':<8}" + _fmt(d))
                print(f"  {'':<8}{'sigma':<8}" + _fmt(sig, 9, 2))
                print(f"  {'':<8}{'':<8}rms {_rms(a['m']):.5f} -> "
                      f"{_rms(b['m']):.5f}   max |sigma| "
                      f"{np.nanmax(np.abs(sig)):.2f}")
            print()
    if args.npz:
        np.savez(args.npz, u=UCURVE,
                 **{f"{f}_{hp.SPECIES[p]['label']}_{a}_{k}": v
                    for (f, p, a), r in res.items() for k, v in
                    (("m", r["m"]), ("err", r["err"]))})
        print(f"wrote {args.npz}")


# =========================================================================
# 3.  the gauge: perturb the model's MS channel, report SIZE and SHAPE
# =========================================================================

def _base_rows(pdg, func, arm="off", rad=True):
    return _rows(pdg, arm, func, model_rad=rad)


def cmd_gauge(args):
    """Inject a controlled change into the model's MS channel and ask whether
    it reproduces the SHAPE of the measured residual.

    `KMS_SCALE` multiplies the whole MS log-CF exponent, i.e. it is a pure
    RELATIVE change of every MS cumulant -- the response to it is the gauge
    that converts "the MS variance is x % too big" into closure units, and
    its u-profile is the profile of ANY pure-magnitude MS defect.

    `MS_NSUB = 1` removes the within-step lateral displacement from the model
    (the D*L + L^2/3 term).  Geant4 runs mu/hadrons with
    `muhadLateralDisplacement = false`, so the SIMULATION has no within-step
    MSC displacement at all; this is the size and shape of that term.
    """
    print("=" * 152)
    print("THE MS GAUGE.  Each row is a controlled change of the MODEL's MS "
          "channel, at fixed simulation.")
    print("=" * 152)
    print(f"  {'species':<8}{'knob':<22}" + UROW + "      rms   cos(resid)")
    for pdg in args.pdg:
        base = _base_rows(pdg, args.func)
        print(f"  {hp.SPECIES[pdg]['label']:<8}{'BASE (residual)':<22}"
              + _fmt(base["m"]) + f" {_rms(base['m']):9.5f}      1.000")
        print(f"  {'':<8}{'+-':<22}" + _fmt(base["err"]))
        for knob, val in args.knobs:
            old = None
            if knob == "KMS_SCALE":
                old, cpt.KMS_SCALE = cpt.KMS_SCALE, float(val)
            elif knob == "MS_NSUB":
                old, cpt.MS_NSUB = cpt.MS_NSUB, int(val)
            elif knob == "G4_FF_SQUARED":
                old, cme.G4_FF_SQUARED = cme.G4_FF_SQUARED, bool(int(val))
                cme.set_j0_guard(cme.J0M1_GUARD)      # rebuilds the table
            elif knob == "MS_ELEC_TMAX":
                old, ctr.MS_ELEC_TMAX = ctr.MS_ELEC_TMAX, float(val)
            elif knob == "MS_SNAP_YMAX":
                old, ctr.MS_SNAP_YMAX = ctr.MS_SNAP_YMAX, float(val)
            else:
                raise SystemExit(f"unknown knob {knob}")
            try:
                r = _rows(pdg, "off", args.func)
            finally:
                if knob == "KMS_SCALE":
                    cpt.KMS_SCALE = old
                elif knob == "MS_NSUB":
                    cpt.MS_NSUB = old
                elif knob == "G4_FF_SQUARED":
                    cme.G4_FF_SQUARED = old
                    cme.set_j0_guard(cme.J0M1_GUARD)
                elif knob == "MS_ELEC_TMAX":
                    ctr.MS_ELEC_TMAX = old
                elif knob == "MS_SNAP_YMAX":
                    ctr.MS_SNAP_YMAX = old
            d = r["m"] - base["m"]
            nb, nd = np.linalg.norm(base["m"]), np.linalg.norm(d)
            cos = float(d @ base["m"] / (nb * nd)) if nb * nd > 0 else np.nan
            print(f"  {'':<8}{knob + '=' + str(val):<22}" + _fmt(d)
                  + f" {_rms(d):9.5f} {cos:10.3f}")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("sim")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, 2212])
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=10)
    q.add_argument("--seed0", type=int, default=101)
    q.add_argument("--workers", type=int, default=20)
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_sim)

    q = s.add_parser("live")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, 2212])
    q.add_argument("--arms", nargs="+", default=["off", "nocs"])
    q.set_defaults(f=cmd_live)

    q = s.add_parser("closure")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, 2212])
    q.add_argument("--arms", nargs="+", default=["off", "nocs"])
    q.add_argument("--funcs", nargs="+", default=["locx"])
    q.add_argument("--npz", default=None)
    q.set_defaults(f=cmd_closure)

    q = s.add_parser("gauge")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, 2212])
    q.add_argument("--func", default="locx")
    q.add_argument("--knobs", nargs="+", default=None)
    q.set_defaults(f=cmd_gauge)

    a = p.parse_args()
    if getattr(a, "knobs", None) is not None:
        a.knobs = [tuple(k.split("=", 1)) for k in a.knobs]
    elif a.cmd == "gauge":
        a.knobs = [("KMS_SCALE", 1.02), ("KMS_SCALE", 0.98),
                   ("MS_NSUB", 1)]
    a.f(a)


if __name__ == "__main__":
    main()
