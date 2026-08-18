#!/usr/bin/env python3
"""Kokoulin in the EXTRAPOLATOR's variance, and the downstream validation of Q.

Two things live here, in the order the task asks for them.

TASK 1 -- the wiring.  `CVH_IONI_KOKOULIN` is now read by ONE C++ function,
`cvhcgf::ioniKokoulinEnabled()`, which the fluctuation model calls when it
builds the exact-delta channel's second moment; the offline consumer
`cf_track_resolution.IONI_KOKOULIN` takes its DEFAULT from the same variable.
So "on" is on in the fit's Q and in the model the closure is measured with.
The subcommands here export the three models (`off` / `on` = exact delta /
`onk` = exact delta + Kokoulin), prove bit-identity when the new switch is off,
show it is LIVE when on, and re-run the closure.

TASK 2 -- downstream of Q.  Two corrections now move `Q` and `dQI` and nothing
downstream had been re-validated for either.  `cvh` runs the real-tracker CVH
refit on real J/psi ALCARECO tracks in the four configurations (nominal /
exact-delta / exact-delta+Kokoulin / Kokoulin alone) and `cvhcmp` compares the
fitted momenta, the reported covariances, the material/eloss Jacobians that
feed the Millepede marginalisation, and the convergence.

Every cmsRun goes through Analysis/HitAnalyzer/test/cmsswlock.sh run.
"""

import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_propagation_test as cpt                                # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402
import deltaspec as ds                                           # noqa: E402
import fisher_norm as fn                                         # noqa: E402
from wums import logging as _wums_logging                        # noqa: E402

logger = _wums_logging.child_logger(__name__)

SCRATCH = fn.SCRATCH
OUT = os.path.join(SCRATCH, "qv")
CMSSW = ds.CMSSW
SRCTEST = ds.SRCTEST

# The three export configurations.  `on` is what NOTES_DELTASPEC shipped;
# `onk` is that plus the new switch.  `kokonly` exists only to show that the
# new code is confined to the exact-delta branch (regime 1 has no `xi`, so the
# correction cannot be applied there and the export must be identical to
# `off`).
ENVS = {
    "off":     {},
    "on":      {"CVH_IONI_EXACTDELTA": "1"},
    "onk":     {"CVH_IONI_EXACTDELTA": "1", "CVH_IONI_KOKOULIN": "1"},
    "kokonly": {"CVH_IONI_KOKOULIN": "1"},
    # THE FOUR CORRECTIONS ONE AT A TIME (2026-08-16), so that "they still
    # compose" is a measurement.  `pt3`/`pt40`/`real` are mu- models, which
    # makes two of these PREDICTED nulls rather than merely small ones:
    #   * `kokonly`  -- the Kokoulin term lives in the regime-2/3 delta channel
    #                   and there is no such channel without EXACTDELTA, so it
    #                   cannot apply and the export must equal `off`;
    #   * `spdonly`  -- the Tmax correction is on the scaled-proton-table
    #                   branch, which a muon never reaches, so it must equal
    #                   `off` for a muon (and NOT for the pion, which is what
    #                   speciesdedx.py measures).
    # `caonly` and `on` are the two that must be LIVE here.
    "caonly":  {"CVH_REF_CHARGEAWARE": "1"},
    "spdonly": {"CVH_REF_SPECIESDEDX": "1"},
    # THE UNPINNED ARM.  Every other arm here pins all four switches
    # explicitly, which is what keeps the archived comparisons valid -- and
    # which also means no other arm can tell you what the DEFAULT is.  `None`
    # means "leave the variable unset" (see deltaspec._clean_env), so this arm
    # runs with nothing set at all and its export is what a production job with
    # no environment gets.
    #
    # Since the four are DEFAULT-OFF (the 2026-08-16 flip was reverted) it must
    # equal `off` and must NOT equal `allon`.  That is the opposite of what it
    # was written to assert, and cmd_bitid spells the expectation out in the
    # row label so the polarity cannot silently rot again.
    "defaults": {k: None for k in ctr.CVH_DEFAULT_ON},
    # the explicit all-four-on arm the unpinned one is compared against
    "allon":   {k: "1" for k in ctr.CVH_DEFAULT_ON},
}

# archived references for the bit-identity control, per toy/real area
ARCHIVED = {
    "pt3":  os.path.join(SCRATCH, "tp", "pt3_K1_model.root"),
    "pt40": os.path.join(SCRATCH, "tp", "pt40_K1_model.root"),
    "real": os.path.join(SCRATCH, "ro", "real_model_pt3_radon.root"),
}
DSOUT = os.path.join(SCRATCH, "ds")


def mp(which, lab):
    return os.path.join(OUT, f"{which}_model_{lab}.root")


# ======================================================================
# export
# ======================================================================

def cmd_export(args):
    os.makedirs(OUT, exist_ok=True)
    for which in args.which:
        for lab in args.labels:
            out, log = mp(which, lab), mp(which, lab).replace(".root", ".log")
            if os.path.exists(out) and not args.force:
                print(f"[{which}/{lab}] exists, skipping")
                continue
            rc = ds.run_model(which, out, log, dict(ENVS[lab]))
            print(f"[{which}/{lab}] rc={rc} -> {out}")
            if rc:
                raise SystemExit(f"export failed, see {log}")


# ======================================================================
# bitid -- the same comparator as deltaspec, with its two guards kept
# ======================================================================

def cmd_bitid(args):
    print("=" * 100)
    print("BIT-IDENTITY OF THE EXPORT WITH THE KOKOULIN SWITCH OFF")
    print("=" * 100)
    print("Every branch of every TTree is hashed recursively (sha256 over the "
          "raw values,\ndtype included); the branch COUNT is printed on every "
          "line and a file with no TTree\nis a hard failure -- both guards are "
          "deltaspec's, and both caught real bugs.\n")
    pairs = []
    for which in args.which:
        # 1. the new code must not touch the regime-1 path at all
        pairs.append((ARCHIVED[which], mp(which, "off"),
                      f"{which} ARCHIVED vs fresh off   (regime 1, nothing set)"))
        # 2. nor the exact-delta path when the new switch is off
        pairs.append((os.path.join(DSOUT, f"{which}_model_on.root"),
                      mp(which, "on"),
                      f"{which} DELTASPEC on vs fresh on (regime 2, EXACTDELTA only)"))
        # 3. the switch alone, without exact delta, is a no-op by construction
        if args.kokonly:
            pairs.append((mp(which, "off"), mp(which, "kokonly"),
                          f"{which} off vs KOKOULIN-only  (must be a no-op)"))
        # 4. LIVE control
        pairs.append((mp(which, "on"), mp(which, "onk"),
                      f"{which} on vs on+KOKOULIN        (LIVE: expect Q, dQI, ioniurbanv)"))
        # 5. THE DEFAULT IS WHAT IT IS CLAIMED TO BE.
        #    `defaults` sets nothing at all; `allon` sets all four to 1, `off`
        #    sets all four to 0. This is the only check here that is not run
        #    against a pinned arm, and it is the only one that can catch a
        #    default that is not what the source says.
        #
        #    THE POLARITY OF THESE TWO ROWS IS THE TEST. They were written on
        #    2026-08-16 with the four DEFAULT-ON, i.e. `defaults` == `allon`
        #    and `defaults` != `off`. The flip was REVERTED (NOTES_CLOSURE_FINAL
        #    s1), so both expectations invert: an unpinned job is now the
        #    HISTORICAL state, and it must NOT be the all-four-on one. Getting
        #    this backwards is exactly the failure mode the rows exist to
        #    catch, which is why the expectation is spelled out in the label.
        if args.defaults:
            pairs.append((mp(which, "off"), mp(which, "defaults"),
                          f"{which} off vs UNPINNED         (DEFAULT-OFF: must be identical)"))
            #    ... and the switches must not be dead: turning them all on has
            #    to move the export.
            pairs.append((mp(which, "allon"), mp(which, "defaults"),
                          f"{which} all-four-ON vs UNPINNED (LIVE: must DIFFER)"))
        if args.compose:
            # each switch alone, against `off`.  Two are PREDICTED nulls on a
            # muon model and two are predicted live -- see ENVS.
            pairs.append((mp(which, "off"), mp(which, "on"),
                          f"{which} off vs EXACTDELTA alone  (LIVE)"))
            pairs.append((mp(which, "off"), mp(which, "kokonly"),
                          f"{which} off vs KOKOULIN alone    (null: no regime-2 channel)"))
            pairs.append((mp(which, "off"), mp(which, "caonly"),
                          f"{which} off vs CHARGEAWARE alone (LIVE: mu- reference moves)"))
            pairs.append((mp(which, "off"), mp(which, "spdonly"),
                          f"{which} off vs SPECIESDEDX alone (null: a muon has its own table)"))
            pairs.append((mp(which, "on"), mp(which, "allon"),
                          f"{which} EXACTDELTA vs all four   (LIVE: the other three add)"))
    for ref, new, lab in pairs:
        if not (os.path.exists(ref) and os.path.exists(new)):
            print(f"### {lab}\n    MISSING ({ref if not os.path.exists(ref) else new})\n")
            continue
        A, B = ds._branch_digests(ref), ds._branch_digests(new)
        ka, kb = set(A), set(B)
        same = sum(1 for k in ka & kb if A[k] == B[k])
        diff = sorted(k for k in ka & kb if A[k] != B[k])
        print(f"### {lab}")
        print(f"    {ref}\n    {new}")
        print(f"    branches: {len(ka)} vs {len(kb)}, common {len(ka & kb)}, "
              f"IDENTICAL {same}, different {len(diff)}")
        if ka ^ kb:
            print(f"    only on one side: {sorted(ka ^ kb)}")
        if diff:
            print(f"    DIFFERENT: {diff}")
        else:
            print("    BIT-IDENTICAL")
        print()


# ======================================================================
# gsig2 -- what the switch actually did to the variance, column by column
# ======================================================================

_COLS = ["regime", "gsig2", "a1", "e1", "a2", "e2", "a3", "e0r", "tmaxr",
         "scaling", "cs", "beta2", "etot"]


def _ioni(path):
    """per-leg (nstep, 11|13) ionization records of a model export.

    `cf_propagation_test.load_model` already resolves the stride from the
    per-leg step COUNT rather than from divisibility (11 and 13 both divide a
    multiple of 143), so it is used rather than re-derived here."""
    return [lg["ioni"] for lg in cpt.load_model(path)]


def cmd_gsig2(args):
    print("=" * 100)
    print("WHAT THE SWITCH DID TO THE RECORD, COLUMN BY COLUMN")
    print("=" * 100)
    print("`gsig2` IS the returned variance, so it must move; everything else "
          "in the record\nmust not.  Anything else moving means the correction "
          "leaked into the model itself.\n")
    for which in args.which:
        A, B = _ioni(mp(which, "on")), _ioni(mp(which, "onk"))
        a = np.concatenate([x for x in A if len(x)])
        b = np.concatenate([x for x in B if len(x)])
        if a.shape != b.shape:
            print(f"### {which}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        print(f"### {which}   {a.shape[0]} ionization steps, stride {a.shape[1]}")
        print(f"  {'column':<10}{'max |rel diff|':>16}{'bit-identical':>16}")
        for j in range(a.shape[1]):
            d = np.abs(b[:, j] - a[:, j]) / np.maximum(np.abs(a[:, j]), 1e-300)
            bit = np.array_equal(a[:, j], b[:, j])
            nm = _COLS[j] if j < len(_COLS) else f"col{j}"
            print(f"  {nm:<10}{d.max():>16.6e}{str(bit):>16}")
        m = a[:, 0] >= 2
        r = b[m, 1] / a[m, 1]
        print(f"  regime>=2 steps: {m.sum()}   gsig2 ratio  min {r.min():.6f}  "
              f"med {np.median(r):.6f}  max {r.max():.6f}  "
              f"mean {r.mean():.6f}")
        # the same ratio for the regime-1 steps, which must be exactly 1
        m1 = a[:, 0] == 1
        if m1.any():
            r1 = b[m1, 1] / a[m1, 1]
            print(f"  regime==1 steps: {m1.sum()}   gsig2 ratio  "
                  f"min {r1.min():.12f} max {r1.max():.12f}  (must be 1)")
        print()


# ======================================================================
# xcheck -- the C++ variance against an INDEPENDENT quadrature
# ======================================================================
#
# `gsig2` moving is not evidence that it moved by the right amount.  This
# rebuilds the exact-delta truncated variance from the record alone, in numpy,
# following the branch's own formulae, and adds the Kokoulin term with
# scipy.integrate.quad -- adaptive Gauss-Kronrod in T, i.e. a different
# quadrature in a different variable from the C++ log-Simpson.  The predicted
# ratio is then compared with the measured one.
#
# It also re-derives the UNCORRECTED variance from the record, which is a check
# on the record-to-variance bookkeeping itself (the `scaling` powers, the
# alpha-quantile) that has not been made anywhere else.

_NMAXCONT = 8.0
_ALPHA = 0.999
_MEL = 0.51099891  # CLHEP::electron_mass_c2, MeV
_MMU = 105.6583715


def _kok_excess(T, etot, mass):
    T = np.asarray(T, float)
    f = np.zeros_like(T)
    m = (T >= 0.1) & (T < etot - mass)
    if np.any(m):
        a1 = np.log(1. + 2. * T[m] / _MEL)
        a3 = np.log(4. * etot * (etot - T[m]) / (mass * mass))
        f[m] = a1 * (a3 - a1) / (2. * np.pi * 137.035999084)
    return f


def _predict(rec, mass=None):
    """(gsig2_tree, gsig2_kok) from ONE stride-13 regime-2/3 record row.

    Everything is in the record's own pre-`scaling` units, exactly as the C++
    branch works, and the two factors of `scaling` are applied at the end.
    The particle mass is taken from the record itself (m = E sqrt(1 - beta^2))
    rather than from a PDG constant, so a Geant4 mass-table difference cannot
    masquerade as a quadrature error."""
    from scipy.integrate import quad
    reg, _g, _a1, _e1, _a2, _e2, xi, t0, tmax, sc, _cs, b2, et = rec[:13]
    if mass is None:
        mass = et * np.sqrt(max(1. - b2, 0.))
    spin = (reg == 2)
    i0 = (1. / t0 - 1. / tmax) - b2 * np.log(tmax / t0) / tmax
    if spin:
        i0 += (tmax - t0) / (2. * et * et)
    n0 = xi * i0
    p3 = _NMAXCONT * n0 / (_NMAXCONT + n0) if n0 > _NMAXCONT else n0
    ta = min(1.0 / ((1. - _ALPHA) * p3 / xi + 1. / tmax), tmax)
    i2 = (ta - t0) - b2 * (ta * ta - t0 * t0) / (2. * tmax)
    if spin:
        i2 += (ta ** 3 - t0 ** 3) / (6. * et * et)
    lo = max(t0, 0.1 / sc)
    dk = 0.
    if ta > lo and et - mass > 1000.:
        def f(T):
            w = 1. - b2 * T / tmax + (T * T / (2. * et * et) if spin else 0.)
            return _kok_excess(np.array([sc * T]), et, mass)[0] * w
        dk = quad(f, lo, ta, limit=400)[0]
    return xi * i2 * sc * sc, xi * (i2 + dk) * sc * sc


def cmd_xcheck(args):
    print("=" * 100)
    print("THE C++ VARIANCE AGAINST AN INDEPENDENT QUADRATURE")
    print("=" * 100)
    print("The truncated exact-delta variance is rebuilt from the RECORD in "
          "numpy (which also\nchecks the record->variance bookkeeping itself), "
          "and the Kokoulin term is done with\nscipy's adaptive Gauss-Kronrod "
          "in T -- a different rule in a different variable from\nthe C++ "
          "log-Simpson.  Excitation channels are added from a1 e1 + a2 e2 the "
          "same way.\n")
    for which in args.which:
        A = np.concatenate([x for x in _ioni(mp(which, "on")) if len(x)])
        B = np.concatenate([x for x in _ioni(mp(which, "onk")) if len(x)])
        m = A[:, 0] >= 2
        rt, rp, dv = [], [], []
        for a, b in zip(A[m], B[m]):
            g0, g1 = _predict(a)
            rt.append(b[1] / a[1])
            rp.append(1. + (g1 - g0) / a[1])       # predicted gsig2 ratio
            dv.append((b[1] - a[1]) - (g1 - g0))   # absolute residual, MeV^2
        rt, rp, dv = np.array(rt), np.array(rp), np.array(dv)
        print(f"### {which}   {m.sum()} regime>=2 steps")
        print(f"  measured gsig2 ratio   min {rt.min():.8f}  "
              f"med {np.median(rt):.8f}  max {rt.max():.8f}")
        print(f"  predicted              min {rp.min():.8f}  "
              f"med {np.median(rp):.8f}  max {rp.max():.8f}")
        print(f"  max |measured - predicted| on the RATIO      "
              f"{np.abs(rt - rp).max():.3e}")
        print(f"  max |residual| / gsig2                       "
              f"{np.abs(dv / A[m, 1]).max():.3e}")
        print()


def cmd_nbin(args):
    """Quadrature convergence, measured on the export rather than asserted."""
    print("=" * 100)
    print("LOG-SIMPSON CONVERGENCE OF THE C++ KOKOULIN INTEGRAL")
    print("=" * 100)
    ref = None
    for nb in args.nbins:
        out = os.path.join(OUT, f"{args.which}_model_onk_nb{nb}.root")
        log = out.replace(".root", ".log")
        if not os.path.exists(out) or args.force:
            ev = dict(ENVS["onk"], CVH_IONI_KOKOULIN_NBIN=str(nb))
            rc = ds.run_model(args.which, out, log, ev)
            if rc:
                raise SystemExit(f"export failed, see {log}")
        v = np.concatenate([x for x in _ioni(out) if len(x)])[:, 1]
        if ref is None:
            ref = v
        print(f"  nbin {nb:>5}   max |gsig2/gsig2(nbin={args.nbins[0]}) - 1| "
              f"= {np.abs(v / ref - 1.).max():.3e}")


def cmd_qcmp(args):
    """The SIZE OF THE INPUT: what the two corrections do to the fit's own Q.

    Everything downstream is a response to this number, so it has to be on the
    table before any downstream response is called large or small.  `Q(0,0)` is
    the leg's transported ionization variance in curvilinear q/p units -- the
    diagonal the fit inverts."""
    print("=" * 100)
    print("THE INPUT: Q(0,0), THE FIT'S IONIZATION q/p NOISE, PER LEG")
    print("=" * 100)
    for which in args.which:
        legs = {lab: cpt.load_model(mp(which, lab)) for lab in args.labels}
        base = args.labels[0]
        n = len(legs[base])
        q0 = np.array([lg["Q"][0, 0] for lg in legs[base]])
        print(f"### {which}   {n} legs   Q(0,0) base [{q0.min():.3e}, "
              f"{q0.max():.3e}]  (GeV^-2)")
        for lab in args.labels[1:]:
            q = np.array([lg["Q"][0, 0] for lg in legs[lab]])
            r = q / q0
            print(f"  {lab:<6} / {base:<6}  ratio  min {r.min():.6f}  "
                  f"med {np.median(r):.6f}  max {r.max():.6f}")
            # the same for the TOTAL q/p variance the fit sees on that leg,
            # i.e. including MS, so the relative weight of the change is visible
        print()


# ======================================================================
# closure
# ======================================================================

MODEL_TCUT = {"pt3_cut1e4": 0.0, "pt40_cut001": 0.29638, "pt3_base": 17.8507,
              "pt40_base": 17.8507, "real": 0.0}
SIMTAG = {"real": "ARCHIVED"}
AREA = {"pt3_cut1e4": "pt3", "pt3_base": "pt3", "pt40_cut001": "pt40",
        "pt40_base": "pt40", "real": "real"}


def _closure(modelpath, tag, func, kok, tcut):
    """deltaspec._closure_rows with the offline Kokoulin term forced on/off.

    BOTH layers of the s_F cache are bypassed -- `fisher_norm.plane_scales`
    caches in-process AND on disk keyed by the model's CONTENT identity, which
    does not change when a module global changes, so without this the
    switched-ON run silently reuses the switched-OFF scale (samplergap's trap
    note; kept because it is still true)."""
    ctr.IONI_KOKOULIN = kok
    ctr.IONI_KOKOULIN_TCUT = tcut
    fn._SCALE_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        import tail_probe as tp
        import real_probe as rp
        stag = SIMTAG.get(tag, tag)
        if stag in tp.CONFIGS:
            sim = tp.sim_of(stag)
        else:
            sim = rp.sim_of(rp.ARCHIVED_SIM if stag == "ARCHIVED"
                            else rp.sim_glob(stag))
        (rows, errs, ks, ns, err), sc = ds._closure_rows(modelpath, sim, func)
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        fn._SCALE_CACHE.clear()
    return rows, err, ks, ns, sc


def cmd_closure(args):
    print("=" * 100)
    print("THE CLOSURE WITH BOTH CORRECTIONS APPLIED ON BOTH SIDES")
    print("=" * 100)
    print("`model on / kok 0` and `model on / kok 1` are NOTES_SAMPLERGAP's own "
          "two rows and\nmust reproduce it; `model onk / kok 1` is the same "
          "physics with the correction now\nALSO in the exported variance, "
          "i.e. the configuration a production would run.\n")
    for tag in args.configs:
        which = AREA[tag]
        tcut = args.tcut if args.tcut is not None else MODEL_TCUT.get(tag, 0.0)
        for func in args.funcs:
            print(f"### {tag}  {func}   Kokoulin from max(tcut, 100 keV), "
                  f"tcut = {tcut:g} MeV")
            print(f"  {'model':<8}{'kok':<6}"
                  + "".join(f"{u:>10.3g}" for u in fn.UCURVE) + f"{'rms':>10}")
            arms = [a if isinstance(a, tuple) else
                    (a.split(":")[0], float(a.split(":")[1])) for a in args.arms]
            for lab, kok in arms:
                path = mp(which, lab)
                if not os.path.exists(path):
                    print(f"  {lab:<8}{kok:<6} (missing {path})")
                    continue
                rows, err, ks, ns, sc = _closure(path, tag, func, float(kok), tcut)
                if args.kmax is not None:
                    sel = [i for i, k in enumerate(ks) if k <= args.kmax]
                    rows, ks = rows[sel], [ks[i] for i in sel]
                m = rows.mean(axis=0)
                print(f"  {lab:<8}{kok:<6}"
                      + "".join(f"{v:+10.5f}" for v in m)
                      + f"{np.sqrt(np.mean(m ** 2)):10.5f}"
                      + f"   [{len(ks)} pl, {int(np.median(ns))} ev]")
                print(f"  {'  s_F':<14}{np.mean(sc['sF']):.8e}   "
                      f"1/I {np.mean(sc['invI']):.6f}")
            print()


# ======================================================================
# cvh -- the real-track refit, four configurations
# ======================================================================

CVHENV = {
    "nominal": {},
    "exact":   {"CVH_IONI_EXACTDELTA": "1"},
    "kok":     {"CVH_IONI_EXACTDELTA": "1", "CVH_IONI_KOKOULIN": "1"},
    # THE NOISE FLOOR CONTROL.  Physically the same model as `kok`; only the
    # quadrature node count of the Kokoulin integral changes, which moves
    # gsig2 -- and therefore Q and dQI -- by a MEASURED 3.5e-10 relative
    # (`qvalid nbin`).  Anything the fit does in response to that is not
    # physics; it is the flat-direction reordering floor NOTES_CGFFIT
    # measured at gain ~1e12.  Without this row the size of the physics
    # response cannot be judged.
    "kok_nb384": {"CVH_IONI_EXACTDELTA": "1", "CVH_IONI_KOKOULIN": "1",
                  "CVH_IONI_KOKOULIN_NBIN": "384"},
    # THE CONTROL ON THE CONTROL.  Byte-identical configuration to `nominal`.
    # If two of these differ, the "noise floor" above is run-to-run
    # irreproducibility and not a property of the perturbation, and every
    # floor-relative statement collapses.  It has to be measured.
    "nominal2": {},
}


def run_cvh(script, inp, workdir, log, env_extra, nev, extra=""):
    """`runCvhSingleTrack.py` writes `globalcor_single_<stream>.root` into the
    CURRENT DIRECTORY and has no output= option, so each configuration gets its
    own working directory rather than a distinguishing filename.  cmsRun is
    launched from the release test/ directory's copy of the script but with cwd
    set to the work directory."""
    os.makedirs(workdir, exist_ok=True)
    # The CVH switches are ParameterSet parameters, not environment
    # variables (Geant4e b372e08). This runner bypasses
    # hadron_probe._run, so it needs the same translation or anything
    # it "sets" would be exported where nothing reads it.
    _swopts, env_extra = ctr.split_switches(env_extra)
    if _swopts:
        extra = f"{extra} {_swopts}"
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && cd {workdir} && "
           f"exec {SRCTEST}/cmsswlock.sh run cmsRun {SRCTEST}/{script} "
           f"input={inp} nEvents={nev} {extra}")
    with open(log, "w") as fh:
        p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                           stderr=subprocess.STDOUT, env=ds._clean_env(env_extra))
    return p.returncode


PREFIX = {"runCvhSingleTrack.py": "globalcor_single",
          "runCvhSingleTrackMC.py": "globalcor_singlemc"}


def cvh_out(tag, lab, script="runCvhSingleTrack.py"):
    return os.path.join(OUT, f"cvh_{tag}_{lab}",
                        f"{PREFIX.get(script, 'globalcor_single')}_0.root")


def cmd_cvh(args):
    os.makedirs(OUT, exist_ok=True)
    for lab in args.labels:
        wd = os.path.join(OUT, f"cvh_{args.tag}_{lab}")
        log = wd + ".log"
        if os.path.exists(cvh_out(args.tag, lab, args.script)) and not args.force:
            print(f"[{lab}] exists, skipping")
            continue
        rc = run_cvh(args.script, args.input, wd, log,
                     dict(CVHENV[lab]), args.nev, args.extra)
        print(f"[{lab}] rc={rc} -> {cvh_out(args.tag, lab, args.script)}")
        if rc:
            raise SystemExit(f"cvh run failed, see {log}")


# ======================================================================
# cvhcmp -- what the change did downstream of Q, on real tracks
# ======================================================================

def _trim(v, frac=0.95, centred=True):
    """rms of the smallest `frac` of |v|, i.e. with the chaotic tail removed."""
    v = np.asarray(v, float)
    if not len(v):
        return float("nan")
    w = np.sort(v ** 2)[:max(int(frac * len(v)), 1)]
    return float(np.sqrt(np.mean(w)))


_TRKKEYS = ["run", "lumi", "event", "trackPt", "trackEta", "trackPhi",
            "trackCharge"]
_FITKEYS = ["refParms", "refCov", "genParms", "niter", "edmval", "edmvalref",
            "chisqval", "ndof", "nValidHits", "nParms", "globalidxv",
            "jacrefv", "gradv", "hesspackedv", "trackQopErr"]


def _load_cvh(path, extra=()):
    import uproot
    f = uproot.open(path)
    t = f["tree"]
    have = set(t.keys())
    keys = [k for k in _TRKKEYS + _FITKEYS + list(extra) if k in have]
    a = t.arrays(keys, library="np")
    rt = f["runtree"].arrays(["iidx", "parmtype"], library="np") \
        if "runtree" in [k.split(";")[0] for k in f.keys()] else None
    return a, rt, keys


def _match(A, B):
    """index arrays aligning the two files on the INPUT track identity.

    The refit is what changes, so the key is built from quantities the refit
    cannot move: the event id and the seed track's own parameters."""
    def key(a):
        return list(zip(*[a[k] for k in _TRKKEYS]))
    ka, kb = key(A), key(B)
    ib = {k: i for i, k in enumerate(kb)}
    ia = [(i, ib[k]) for i, k in enumerate(ka) if k in ib]
    return np.array([x[0] for x in ia]), np.array([x[1] for x in ia])


def _fam(pt):
    """parmtype -> family label."""
    if pt <= 5:
        return "align"
    if pt in (7, 14):
        return "bfield"
    if pt in (15,):
        return "material"
    return f"pt{pt}"


def cmd_cvhcmp(args):
    print("=" * 100)
    print("DOWNSTREAM OF Q, ON REAL TRACKS")
    print("=" * 100)
    ref = args.labels[0]
    A, rtA, _ = _load_cvh(cvh_out(args.tag, ref, args.script))
    print(f"reference configuration: {ref}   {len(A['run'])} tracks\n")
    for lab in args.labels[1:]:
        B, rtB, _ = _load_cvh(cvh_out(args.tag, lab, args.script))
        i, j = _match(A, B)
        print(f"### {lab}  vs  {ref}      matched {len(i)} / "
              f"{len(A['run'])} vs {len(B['run'])}")

        # ---- fitted momenta
        qa = np.array([p[0] for p in A["refParms"]])[i]
        qb = np.array([p[0] for p in B["refParms"]])[j]
        pt = A["trackPt"][i]
        d = (qb - qa) / np.abs(qa)
        # WHY THREE SPREAD STATISTICS AND NOT ONE.  The per-track shift is a
        # MIXTURE: a coherent response to the model change on every track, plus
        # a rare, large kick on the few percent of tracks that the fit's flat
        # direction lets wander (NOTES_CGFFIT's 1e12 gain).  The plain rms is
        # therefore a TAIL statistic and is dominated by the kicks -- against a
        # null perturbation it reads 3.9e-7 while 88 % of tracks have moved by
        # EXACTLY zero.  The median and the fraction of exactly-unchanged
        # tracks are what separate the two, and getting this wrong understates
        # the signal/floor ratio by an order of magnitude.
        print(f"  fitted q/p   d(q/p)/|q/p|   mean {d.mean():+.4e} "
              f"+- {d.std() / np.sqrt(max(len(d), 1)):.2e}")
        print(f"    spread  median|.| {np.median(np.abs(d)):.4e}   "
              f"trimmed95 rms {_trim(d):.4e}   full rms {d.std():.4e}   "
              f"max|.| {np.abs(d).max():.3e}   exactly unchanged "
              f"{np.count_nonzero(d == 0.)}/{len(d)}")
        for lo, hi in ((0, 3), (3, 5), (5, 10), (10, 1e9)):
            m = (pt >= lo) & (pt < hi)
            if m.sum() > 5:
                print(f"    pT [{lo:g},{hi:g})  n {m.sum():5d}   "
                      f"mean {d[m].mean():+.4e}  rms {d[m].std():.4e}")

        # ---- reported covariance
        ca = np.array([c[0] for c in A["refCov"]])[i]
        cb = np.array([c[0] for c in B["refCov"]])[j]
        r = np.sqrt(cb / ca)
        print(f"  reported sigma(q/p) ratio   mean {r.mean():.6f}  "
              f"med {np.median(r):.6f}  min {r.min():.6f}  max {r.max():.6f}")
        # is the tightening EARNED?  pull of the shift against the claimed
        # error, and the claimed-vs-observed consistency of the SHIFT itself
        pull = (qb - qa) / np.sqrt(ca)
        print(f"  (q/p_new - q/p_old)/sigma_old   mean {pull.mean():+.4f}  "
              f"rms {pull.std():.4f}")

        # ---- is the change to the claimed resolution EARNED?  needs truth
        if args.gen and "genParms" in A:
            ga = np.array([p[0] for p in A["genParms"]])[i]
            gb = np.array([p[0] for p in B["genParms"]])[j]
            ok = (ga != 0.) & (gb != 0.) & (ga == gb)
            if ok.sum() > 20:
                for lab2, q, c, g in (("old", qa, ca, ga), ("new", qb, cb, gb)):
                    res = (q - g)[ok]
                    pl = (res / np.sqrt(c[ok]))
                    # trimmed spread: the q/p residual has a heavy dE/dx tail,
                    # so the RMS is a tail statistic; quote both
                    lo, hi = np.percentile(pl, [15.865, 84.135])
                    print(f"  [gen {lab2}]  residual rms {res.std():.4e}   "
                          f"pull rms {pl.std():.4f}   pull 68% half-width "
                          f"{0.5 * (hi - lo):.4f}   n {ok.sum()}")

        # ---- convergence
        for k, fmt in (("niter", "{:.4f}"), ("edmval", "{:.4e}"),
                       ("edmvalref", "{:.4e}"), ("chisqval", "{:.5f}")):
            if k not in A:
                continue
            va, vb = np.asarray(A[k], float)[i], np.asarray(B[k], float)[j]
            rel = (vb - va) / np.where(np.abs(va) > 0, np.abs(va), 1.)
            print(f"  {k:<10} mean {fmt.format(va.mean())} -> "
                  f"{fmt.format(vb.mean())}   "
                  f"changed on {np.count_nonzero(va != vb)} / {len(i)}   "
                  f"rel: mean {rel.mean():+.3e} rms {rel.std():.3e} "
                  f"max|.| {np.abs(rel).max():.3e}")
        if "ndof" in A:
            na = np.asarray(A["chisqval"], float)[i] / np.maximum(
                np.asarray(A["ndof"], float)[i], 1)
            nb = np.asarray(B["chisqval"], float)[j] / np.maximum(
                np.asarray(B["ndof"], float)[j], 1)
            print(f"  chi2/ndof  mean {na.mean():.4f} -> {nb.mean():.4f}")

        # ---- global-fit exports
        if "jacrefv" in A and "globalidxv" in A and rtA is not None:
            ptype = np.zeros(int(rtA["iidx"].max()) + 1, dtype=int)
            ptype[np.asarray(rtA["iidx"])] = np.asarray(rtA["parmtype"])
            acc = {}
            for ii, jj in zip(i, j):
                ga, gb = A["globalidxv"][ii], B["globalidxv"][jj]
                if len(ga) != len(gb) or np.any(ga != gb):
                    continue
                n = len(ga)
                ja = np.asarray(A["jacrefv"][ii], float).reshape(5, n)
                jb = np.asarray(B["jacrefv"][jj], float).reshape(5, n)
                fam = np.array([_fam(ptype[g]) for g in ga])
                for f in np.unique(fam):
                    m = fam == f
                    x, y = ja[0, m], jb[0, m]          # d(q/p)_ref / d a
                    nn = np.sqrt(np.sum(x ** 2))
                    acc.setdefault(f, []).append(
                        np.sqrt(np.sum((y - x) ** 2)) / nn if nn > 0 else 0.)
            print("  jacref row 0 (d q/p_ref / d a), by family "
                  "[PER-TRACK ||dJ||/||J||, same mixture as above]:")
            for f, v in sorted(acc.items()):
                v = np.asarray(v)
                print(f"    {f:<10} median {np.median(v):.4e}   "
                      f"trimmed95 {_trim(v, centred=False):.4e}   "
                      f"pooled {np.sqrt(np.sum(v ** 2) / len(v)):.4e}   "
                      f"exactly unchanged {np.count_nonzero(v == 0.)}/{len(v)}")

        # ---- the objects Millepede actually marginalises: grad and hess
        for k in ("gradv", "hesspackedv"):
            if k not in A or k not in B:
                continue
            v = []
            for ii, jj in zip(i, j):
                x = np.asarray(A[k][ii], float)
                y = np.asarray(B[k][jj], float)
                if x.shape != y.shape:
                    continue
                nn = np.sqrt(np.sum(x ** 2))
                v.append(np.sqrt(np.sum((y - x) ** 2)) / nn if nn > 0 else 0.)
            v = np.asarray(v)
            if len(v):
                print(f"  {k:<12} median {np.median(v):.4e}   "
                      f"trimmed95 {_trim(v, centred=False):.4e}   "
                      f"pooled {np.sqrt(np.sum(v ** 2) / len(v)):.4e}")
        print()


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("export")
    q.add_argument("which", nargs="*", default=["pt3", "pt40", "real"])
    q.add_argument("--labels", nargs="+", default=["off", "on", "onk"])
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_export)

    q = s.add_parser("bitid")
    q.add_argument("which", nargs="*", default=["pt3", "pt40", "real"])
    q.add_argument("--kokonly", action="store_true")
    q.add_argument("--defaults", action="store_true",
                   help="also compare the UNPINNED export against off "
                        "(must match: the four are DEFAULT-OFF) and against "
                        "all-four-on (must differ: they are not dead)")
    q.add_argument("--compose", action="store_true",
                   help="also compare each of the four switches ALONE against "
                        "off, and all four against EXACTDELTA alone")
    q.set_defaults(f=cmd_bitid)

    q = s.add_parser("gsig2")
    q.add_argument("which", nargs="*", default=["pt3", "pt40", "real"])
    q.set_defaults(f=cmd_gsig2)

    q = s.add_parser("xcheck")
    q.add_argument("which", nargs="*", default=["pt3", "pt40", "real"])
    q.set_defaults(f=cmd_xcheck)

    q = s.add_parser("nbin")
    q.add_argument("--which", default="pt3")
    q.add_argument("--nbins", nargs="+", type=int, default=[96, 48, 192, 384])
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_nbin)

    q = s.add_parser("qcmp")
    q.add_argument("which", nargs="*", default=["pt3", "pt40", "real"])
    q.add_argument("--labels", nargs="+", default=["off", "on", "onk"])
    q.set_defaults(f=cmd_qcmp)

    q = s.add_parser("closure")
    q.add_argument("--configs", nargs="+", default=["pt3_cut1e4", "pt40_cut001"])
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--arms", nargs="+", default=["on:0", "on:1", "onk:1"],
                   type=lambda x: (x.split(":")[0], float(x.split(":")[1])))
    q.add_argument("--tcut", type=float, default=None)
    q.add_argument("--kmax", type=int, default=None)
    q.set_defaults(f=cmd_closure)

    q = s.add_parser("cvh")
    q.add_argument("--tag", default="jpsi")
    q.add_argument("--script", default="runCvhSingleTrack.py")
    q.add_argument("--input", required=True)
    q.add_argument("--nev", type=int, default=200)
    q.add_argument("--labels", nargs="+", default=["nominal", "exact", "kok"])
    q.add_argument("--extra", default="")
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_cvh)

    q = s.add_parser("cvhcmp")
    q.add_argument("--tag", default="jpsi")
    q.add_argument("--script", default="runCvhSingleTrack.py")
    q.add_argument("--labels", nargs="+", default=["nominal", "exact", "kok"])
    q.add_argument("--gen", action="store_true",
                   help="MC only: pull of the fitted q/p against gen truth")
    q.set_defaults(f=cmd_cvhcmp)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
