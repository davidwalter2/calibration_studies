#!/usr/bin/env python3
"""How much closure does a magnetic-field error buy?

THE QUESTION
------------
After the basis fix (`hbasis.py`) the pT = 3 toy closes on locx, locy, dxdz and
dydz, leaving a +0.004 locx plateau and the qop items.  Is any of that the
field?

Rather than removing the field (B = 0 removes the MECHANISM too: no bending
means no bending/non-bending distinction, no sec(alpha) in H, and
cov(q/p, x_T) -- the one term H's qop row adds -- is identically zero), this
measures the RESPONSE: scale the field the MODEL is exported at, leave the SIM
at the real map, and read off how the closure moves.  Inverting that curve says
what field error would be required to produce the observed residual.

This is deliberately the half-switch the bending-plane handoff warns against.
Here the mismatch IS the measurement: a real field error is exactly a model
that disagrees with nature, and the closure's sensitivity to it is the number
we want.

The planes are held FIXED for the same reason -- a field error moves the
reference relative to fixed detector surfaces.  Regenerating them per arm would
absorb precisely the effect being measured.

WHY UNIFORM ON BOTH SIDES OF THE DERIVATIVE
-------------------------------------------
CMSSW has no scale knob on the tracker field (OAE takes a discrete `BValue`,
and `scalingVolumes` addresses yoke volumes).  So every arm uses
`UniformMagneticFieldESProducer`, including the reference arm at
B0 = 3.81459 T -- the PATH-AVERAGED real field along this toy's reference,
measured off the dumped grid.  The map -> uniform step is then common to all
arms and cancels in the difference; only the scale varies.  (That step is small
anyway: the real Bz runs 3.81143 - 3.82078 T along the path, and the resulting
mid-ladder reference displacement is ~23 um against a ~300 um sigma.)

WHAT TO EXPECT, AND WHY THE SIGN MATTERS
----------------------------------------
A field error is primarily a COHERENT BIAS on the reference, not a width error.
The closure statistic <e^{-u z^2}> is EVEN in z, so a mean offset mu (in sigma
units) can only push `data - model` NEGATIVE, by about -gauge * mu^2.  So the
response should be quadratic in the scale error and one-signed.  The surviving
locx residual is POSITIVE (+0.0042 toy, +0.0056 real), so if that holds, a
field error cannot be its cause at any magnitude -- and this scan is what turns
that argument into a measurement.

usage:
    python bscan.py setup                 # patch the arm drivers (idempotent)
    python bscan.py export                # cmsRun one model per field value
    python bscan.py closure               # the closure, sim FIXED
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hadron_probe as hp                                        # noqa: E402
import radoff_species as rs                                      # noqa: E402
import barkas_probe as bp                                        # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import hbasis as hb                                              # noqa: E402
import allcorr as ac                                             # noqa: E402
import deltaspec as ds                                           # noqa: E402

# Path-averaged Bz along this toy's reference, from the dumped CMSSW grid
# (MagneticField/Engine/test/field_polyfit3d_full.txt, integrated over the
# pt=3 eta=0.30 helix out to r = 106.8 cm).  The plane generator uses 3.8 flat,
# which is where the reference's 1.57 mrad outermost mis-tangency comes from.
B0 = 3.81459

# Relative scale errors.  1e-2 already moves the reference ~4 sigma, which
# collapses the statistic from the mean shift alone -- it is included as the
# large-signal anchor, not as a useful working point.
SCALES = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)

_MARK = "# --- bscan field override ---"


def _tag(eps):
    return "b0" if eps == 0 else ("bp%g" % eps).replace("-", "m").replace(
        ".", "").replace("+", "")


def model_path(eps):
    return rs.mp(13, True)[:-5] + "_" + _tag(eps) + ".root"


# ==========================================================================
# setup: the driver patch
# ==========================================================================

_PATCH = f'''{_MARK}
# TOY_BFIELD (Tesla) replaces the default map with a UNIFORM field.  Both the
# sim and the model read the UNLABELLED IdealMagneticFieldRecord product
# (SimG4Core's RunManagerMTWorker esConsumes with no label; runToyModel's
# fieldlabel defaults to ""), so swapping the ESProducer that fills that slot
# switches every consumer at once -- there is no half-switch to get wrong
# WITHIN a job.  Unset keeps the default map exactly.
if _os.environ.get("TOY_BFIELD"):
    process.load('MagneticField.Engine.uniformMagneticField_cfi')
    process.UniformMagneticFieldESProducer.ZFieldInTesla = cms.double(
        float(_os.environ["TOY_BFIELD"]))
    print('[toy] UNIFORM model field %g T'
          % process.UniformMagneticFieldESProducer.ZFieldInTesla.value())
else:
    process.load('Configuration.StandardSequences.MagneticField_cff')
{_MARK}'''


def cmd_setup(args):
    """Patch the per-arm driver COPY, not the shared source.

    `allcorr`/`hadron_probe` already rewrite `runToyModel.py` into
    `hp.testdir(g)`; this appends one more substitution to that copy, so the
    tree's own driver is untouched and a rerun of their setup wipes this
    cleanly.
    """
    for g in ("qm",):
        td = hp.testdir(g)
        p = os.path.join(td, "runToyModel.py")
        s = open(p).read()
        if _MARK in s:
            print(f"[{g}] already patched: {p}")
            continue
        if "import importlib, os as _os" not in s:
            raise SystemExit(
                f"{p} is not the patched arm copy (no `os as _os`). Run "
                f"`python allcorr.py setup` or `hadron_probe.py setup` first.")
        n = len(s)
        s = s.replace(
            "process.load('Configuration.StandardSequences.MagneticField_cff')",
            _PATCH, 1)
        if len(s) == n:
            raise SystemExit(f"MagneticField_cff load not found in {p}")
        open(p, "w").write(s)
        print(f"[{g}] patched {p}")


# ==========================================================================
# export
# ==========================================================================

def cmd_export(args):
    for eps in args.scales:
        out = model_path(eps)
        if os.path.exists(out) and not args.force:
            print(f"eps={eps:<8g} exists -> {os.path.basename(out)}")
            continue
        B = B0 * (1.0 + eps)
        log = out[:-5] + ".log"
        env = hp._env(13, "off", {"CVH_IONI_EXACTDELTA": "1",
                                  "CVH_IONI_KOKOULIN": "1",
                                  "TOY_BFIELD": repr(B)})
        rc = hp._run("qm", "runToyModel.py",
                     f"pt={hp.PT} eta={hp.ETA} phi={hp.PHI} partId=13 "
                     f"output={out}", log, env)
        print(f"eps={eps:<8g} B={B:.6f} T  rc={rc} -> {os.path.basename(out)}")
        if rc:
            raise SystemExit(f"export failed, see {log}")


# ==========================================================================
# closure
# ==========================================================================

def cmd_closure(args):
    import cf_propagation_test as cpt

    iu = list(ac.UCURVE).index(1.0)
    print("model field scanned, SIM FIXED at the real map.  seven corrections, "
          "radiation ON, u = 1, H basis, planes held fixed.")
    print(f"B0 = {B0} T (path-averaged real field along this reference)\n")
    rows = {}
    for func in args.funcs:
        print(f"{func}")
        print(f"  {'eps':>9} {'B [T]':>10} " +
              " ".join(f"{k:>7}" for k in range(14)) +
              f" {'ladder':>9} {'outer':>9}")
        for eps in args.scales:
            path = model_path(eps)
            if not os.path.exists(path):
                print(f"  {eps:9.1e}  MISSING {os.path.basename(path)}")
                continue
            # the arm's own field, in H as well as in the export
            oldB, oldmp = hb.BFIELD, rs.mp
            hb.BFIELD = hb._canonical().BFIELD = B0 * (1.0 + eps)
            rs.mp = lambda pdg, rad, _p=path: _p
            try:
                r = ac.cell(13, True, func, ms=True, useh=True)
            finally:
                hb.BFIELD = hb._canonical().BFIELD = oldB
                rs.mp = oldmp
            row = np.asarray(r["rows"])[:, iu]
            rows[(func, eps)] = row
            print(f"  {eps:9.1e} {B0*(1+eps):10.5f} " +
                  " ".join(f"{x:7.4f}" for x in row) +
                  f" {row.mean():9.4f} {row[-1]:9.4f}")
        # the response, referred to the eps = 0 arm
        base = rows.get((func, 0.0))
        if base is not None:
            print(f"  {'response vs eps=0 (outermost plane):':<45}")
            for eps in args.scales:
                if eps == 0.0 or (func, eps) not in rows:
                    continue
                d = rows[(func, eps)][-1] - base[-1]
                print(f"    eps={eps:8.1e}   d(closure) = {d:+.5f}   "
                      f"d/eps^2 = {d/eps**2:+.3g}")
        print()
    np.savez(os.path.join(hp.SCRATCH, "bscan.npz"),
             **{f"{f}_{_tag(e)}": v for (f, e), v in rows.items()})


def _scales(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("setup"); a.set_defaults(f=cmd_setup)
    for name, f in (("export", cmd_export), ("closure", cmd_closure)):
        q = sub.add_parser(name)
        # comma-separated, NOT nargs="+": argparse's negative-number matcher is
        # ^-\d+$|^-\d*\.\d+$, so "-1e-4" is read as an OPTION and the parser
        # dies on the whole scan. The negative arms are the point of the scan
        # (they are what shows the response is one-signed), so the CLI has to
        # take them.
        q.add_argument("--scales", type=_scales, default=list(SCALES),
                       help="comma-separated, e.g. -1e-3,-1e-4,0,1e-4")
        q.add_argument("--force", action="store_true")
        if name == "closure":
            q.add_argument("--funcs", nargs="+",
                           default=["locx", "dxdz", "qop"])
        q.set_defaults(f=f)
    args = p.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
