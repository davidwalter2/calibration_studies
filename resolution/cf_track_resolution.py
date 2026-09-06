"""Per-track resolution prediction from the CF product ("Level 2").

The linearized CVH fit maps the noise vector n to the momentum error via
delta(q/p) = w^T n with w = (C e_qop)^T F^T V^{-1}. The refit exports, per
resolution entry b (aligned with reseigidx), the variance contribution
v_b = w_b^T V_b w_b (resinfvarv) and their total resinfcov, which equals
refCov(0,0) EXACTLY (validated: ratio 1.0000 on every smoke track) -- every
noise dof feeding q/p carries an entry.

Independence of the blocks then gives the full non-Gaussian CF of the
standardized momentum error z = (qop_reco - qop_gen)/sigma, sigma^2 =
refCov(0,0):

  log phi_z(t) = -(Vg/sigma^2) t^2/2                        [hits, Gaussian]
      -- or, with --hitmode class, the MEASURED per-hit densities:
         sum_b log phi_c(b)( t sqrt(v_b)/sigma )              [hits, measured]
    + sum_MS  e^{k_ms}  sum_steps (chic2/chia2) G(t w_std sqrt(chia2); FF)
    + sum_ion e^{k_ion} sum_steps S_urban(t q w_std g_s ...)  [centered]
    + k_rad    sum_rad  sum_steps S_rad(t q w_std cs E ...)    [centered]

where per pooled block w_std = sqrt(v_pool / sigma_Q^2) / sigma is the
effective scalar weight (exact under per-collision azimuthal isotropy:
any linear functional of the two projected angles is again projected-
Moliere with |weight|), and sigma_Q^2 is the fit-assumed block variance
(sum thp2 for MS, sum gsig2 g^2 for ionization).

THE RADIATIVE BLOCK (bremsstrahlung + pair production), added 2026-09-03.
The offline CF had no radiative term at all while the in-fit CGF has had one
since 2026-08 (`CgfRadiativeChannel`, `cvhcgf::makeRadSpectrum`), and the
mismatch is one-sided: the reference trajectory SUBTRACTS the radiative mean
(`ComputeMuonDEDX` is built with ionOnly = false) while the fluctuation model
carried none, so the typical non-radiating muon had a mean removed that it
never lost.  The term is the compound Poisson of the two processes,

    S_rad(t) = sum_steps sum_{proc in brems,pair}
                 INT dv (dN/dv)_proc (e^{i a v E} - 1 - i a v E),
    a = t * q * w_std * cs ,

with (dN/dv)_proc the propagator's own tabulated Geant4 shape RENORMALIZED to
that process's own mean loss dedx_proc * step (so the mixture is right and the
centring subtracts exactly what the reference subtracted -- S'(0) = 0 by
construction).  The step records are the `radstepv`/`radstepspecv` export
(2026-09-03), one row per Geant4 step, tagged with the leg's PARMTYPE-11
global index -- the same index the ionization block is pooled by, because the
radiative loss enters the same q/p dof through the same transport.  So the
weight is the ionization block's weight, charge included: the fit's Q has no
radiative variance (deliberately -- for dsigma/dv ~ 1/v the second moment is
the catastrophic radiator, not the block), so there is nothing to recover a
radiative weight FROM, and nothing to recover: it is the same dof.

`--krad` scales it.  It is a LINEAR scale with default 1, unlike the log
scales `--khit/--kms/--kioni`, precisely so that `--krad 0` expresses "the
model as it was before this term existed" -- which is the control arm of
every closure below.  `--no-rad` is the EXTRACTION-side switch (for the
`nomsrad` sample, whose simulation has no brems/pair at all): it writes zero
arrays and the provenance key `rad_model = 0`.

THE IONIZATION WEIGHT CARRIES THE TRACK CHARGE q (fixed 2026-09-03,
Documents/Resolution/NOTES.md "2026-09-02/03 ... s3"). The exported
per-step factor `ioniurbanv[:,10] = us.cs = E/p^3` is POSITIVE for every
track; the physical map is delta(q/p) = q cs delta(E), and the in-fit CGF
block applies the `qsign` accordingly. This module did not until
2026-09-03, so every cached exponent was the mu+ exponent and the model
carried the mu+ skew for BOTH charges while a charge-symmetric sample
cancels it in the data. Only the ODD part is affected: Re S is even and
Im S odd in the weight (exactly -- see the comment at the call site), so
Sio_re and the whole even closure are unchanged and pre-fix caches differ
only by the sign of Sio_im on their mu- half. Caches written with the fix
carry the key `ioni_charge_signed`; consumers use its presence to decide
whether they still have to apply q themselves. VERIFIED 2026-09-02/03:
on a 50/50 mu+/mu- gun the DATA's odd moment cancels between the charges
while the unsigned MODEL's does not, which is what the earlier "model
over-predicts the skew 10x at low pT / zero mode shift" readings were.

THAT IONIZATION NORMALISATION USED TO BE AN IDENTITY ONLY FOR CgfQoPMode=0,
where the record's gsig2 IS the variance the fit put into dV_b. Under
CgfQoPMode>=1 (the cfi default since 2026-08-24) the fit substitutes the
block's Fisher weight while the record carries the untruncated second
cumulant, and the recovered weight came out ~20x too small (~400x in the
exponent). FIXED 2026-09-03 by exporting the substitution factor itself:
`ioniqscalev` carries, per leg, [sc, nstep] with

    sc = Q_ioni_applied(0,0) / dQ2_record(0,0)

(Geant4ePropagator::cgfQScale), and `ioni_sq2` multiplies each leg's step
sum by its own sc, which makes sqrt(v_b / sq2) exact for BOTH estimators.
sc is exactly 1.0 for CgfQoPMode=0 and for every two-track fit, and the
branch is auto-detected, so files written before the export -- and every
cache made from them -- are unchanged bit-for-bit.

`--ioni-norm raw` remains available and takes the ionization weight straight
from the exported per-dof influence weights, w_std = |w_b[0]| / sigma
(resinfv), which references neither gsig2 nor dV_b; it is independent of the
estimator but neglects the intra-leg transport (a one-sided ~2%/long-leg
deficit against `var`, measured 2026-09-02). Blocks are pooled by
global parameter index (steps are matched the same way as in cf_ms_exact /
cf_ioni_exact); pooling merges same-family crossings with a shared w --
exact for the common one-crossing case.

Closure ("--closure"): on gen-matched MC with the NOMINAL fit
(fitFromGenParms=False), compare per track
  data  e^{-u z_obs^2}          vs   model  E[e^{-u z^2}] (Weierstrass)
averaged in bins of predicted sigma and probe u, plus the averaged
predicted lineshape p(z) against the pull histogram.

usage:
  python cf_track_resolution.py --extract [--files GLOB] [--ntasks N]
  python cf_track_resolution.py --closure [--probes ...]
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
from cf_ms_exact import moliere_params, gshape, gshape_elec
import cf_ioni_exact
import hitres_classes
import cf_delta_ray
import cf_brems_exact
import pubhtml
import ratiopanel
import prodfiles

# Discrete delta-ray (knock-on) transverse recoil, see cf_delta_ray.  Stored as
# the NET change to the MS block: S_delta - carve*S_ms, so it is exactly zero
# when switched off and needs no second copy of Sms.  DELTA_TMAXCAP truncates
# the spectrum where the recoil stops being resolution and becomes a visible
# kink the selection removes (default 50 MeV ~ 30x the per-layer MS angle).
DELTA_ON = os.environ.get("CF_DELTA", "1") not in ("0", "", "false", "False")
DELTA_TCUT = float(os.environ.get("CF_DELTA_TCUT", "0.35e-3"))     # GeV
DELTA_TMAXCAP = float(os.environ.get("CF_DELTA_TMAXCAP", "0.05"))  # GeV

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

TG = np.linspace(0.0, 14.0, 448)  # t conjugate to standardized z
COVTOL = 5e-3                     # |resinfcov/refCov00 - 1| guard


# ------------------------------------------------------------------ SWITCHES
# The tri-state environment convention, IDENTICAL to `cvhcgf::envFlag` in
# TrackPropagation/Geant4e/src/CGFQoPBlock.cc:
#
#     unset                               -> `dflt`
#     "0"/"false"/"off"/"no"/"" (any case) -> False
#     anything else                        -> True
#
# It exists because on 2026-08-16 four corrections went DEFAULT-ON, and a
# presence-only reader (`os.environ.get(name)`) cannot express "off" once the
# default is on.  Attribution needs both directions: the whole point of the
# gated global-fit exercise is to run all four off against all four on.
#
# This lives in the DEEPEST module of the offline stack (nothing local is
# imported here), so every driver can reach it without a cycle.
def env_flag(name, dflt):
    v = os.environ.get(name)
    if v is None:
        return bool(dflt)
    return v.strip().lower() not in ("", "0", "false", "off", "no")


# The four energy-loss corrections.  They were flipped default-ON on
# 2026-08-16 (Documents/Resolution/NOTES_DEFAULTON.md) and REVERTED the same
# week (NOTES_CLOSURE_FINAL.md s1): the attribution gate is unchanged -- the
# global fit that consumes the exported Jacobians has not been run, and
# CVH_REF_CHARGEAWARE is charge-odd, i.e. degenerate with the calibration's
# `M`.  They are DEFAULT-OFF and closure studies enable them EXPLICITLY.
#
# The name is kept (rather than renamed to CVH_FOUR) because it is the list
# that matters, not the direction: every consumer wants "the four", and both
# overlays below are derived from it.
CVH_DEFAULT_ON = ("CVH_IONI_EXACTDELTA", "CVH_IONI_KOKOULIN",
                  "CVH_REF_CHARGEAWARE", "CVH_REF_SPECIESDEDX")
CVH_FOUR = CVH_DEFAULT_ON            # the direction-neutral alias

# The environment overlay that pins the historical state explicitly.  With the
# defaults back OFF this is a no-op on an otherwise-clean environment -- but it
# is KEPT and still applied in every cmsRun funnel, because it is what makes a
# control arm independent of the default: an arm that reads `{}` is at the
# mercy of whatever the ambient shell exports, and the whole point of these
# arms is that they reproduce published numbers.
SWITCHES_OFF = {n: "0" for n in CVH_DEFAULT_ON}

# The overlay a closure study uses to turn all four ON explicitly.  This is now
# the ONLY way they come on, which is deliberate: `SWITCHES_ON` in an arm dict
# is greppable and appears in the run log, where a default does not.
SWITCHES_ON = {n: "1" for n in CVH_DEFAULT_ON}


# ---------------------------------------------------------------------------
# env name -> cmsRun option.  THE SWITCHES ARE NO LONGER ENVIRONMENT VARIABLES:
# they are ParameterSet parameters on Geant4ePropagator (TrackPropagation/
# Geant4e/python/cvhSwitches.py), so that what ran is recoverable from the
# output file's provenance instead of from a shell that is gone.
#
# The dict vocabulary above is KEPT because ten modules build arms out of it
# and the names are the greppable record in every NOTES entry.  `hadron_probe.
# _run` is the single cmsRun funnel and translates these into `Name=value`
# command-line options there; nothing is exported.  A CVH_* name that is NOT in
# this table and NOT in `CVH_ENV_ONLY` is an error rather than being quietly
# exported, so a typo cannot silently become "the default".
CVH_OPTION = {
    "CVH_IONI_EXACTDELTA": "IoniExactDelta",
    "CVH_IONI_KOKOULIN": "IoniKokoulin",
    "CVH_REF_CHARGEAWARE": "ReferenceChargeAware",
    "CVH_REF_SPECIESDEDX": "ReferenceSpeciesDedx",
    "CVH_REF_HADRAD": "ReferenceHadronRadiative",
    "CVH_IONONLY": "ReferenceIonizationOnly",
    "CVH_IONI_URBAN2021": "IoniUrban2021",
    "CVH_REF_SPECIESDEDX_NBIN": "ReferenceSpeciesDedxNbin",
    "CVH_IONI_KOKOULIN_NBIN": "IoniKokoulinNbin",
    "CVH_IONI_KOKOULIN_CGFNBIN": "IoniKokoulinCgfNbin",
    "CVH_IONI_EXACTDELTA_T0": "IoniExactDeltaT0",
    "CVH_DEDX_SCALE": "DedxScale",
    "CVH_EM_HARMONISE": "EmHarmonise",
    "CVH_CGF_RADIATIVE": "CgfRadiativeChannel",
    "CVH_CGF_QOP": "CgfQoPMode",
    "CVH_CGF_QOP_REFRESH": "CgfQoPRefresh",
    "CVH_DUMP_EMPARAMS": "DumpEmParameters",
}

# CVH_DUMP_HADMODELS is gone entirely: it is now
# ProcessActivationWatcher's own `dumpHadronicModels` untracked parameter, set
# on the watcher PSet in the sim driver, not routed through here at all.
#
# Still environment, because their C++ readers have NOT been migrated yet:
# the LD_PRELOAD shim (CVH_SHIM_*, which is not a CMSSW module at all), and the
# knobs in G4TablesForExtrapolatorForCVH / G4ErrorPhysicsListForCVH /
# ProcessActivationWatcher / MaterialGroupModel / G4ErrorEnergyLossForCVH.
CVH_ENV_ONLY = (
    "CVH_SHIM_BARKAS_OFF", "CVH_SHIM_MOTT_OFF", "CVH_SHIM_BLOCH_OFF",
    "CVH_ELOSS_CYL_R", "CVH_ELOSS_CYL_Z", "CVH_ELOSS_CYL_EPS",
    "CVH_MATGROUP_PROBE", "CVH_MATGROUP_MEANONLY", "CVH_MATGROUP_EPS",
    "CVH_MS_SCALE", "CVH_MS_DISP_SCALE",
    "CVH_DEDX_DEBUG", "CVH_LOCAL_UPDATE",
    # CVH_CGF_QOP and CVH_CGF_QOP_REFRESH are NO LONGER environment variables:
    # the CGF weight is the production default now, so which estimator produced
    # a file has to be in its provenance. They are CgfQoPMode / CgfQoPRefresh
    # in CVH_OPTION above. What remains here is diagnostics only.
    "CVH_CGF_QOP_DEBUG", "CVH_CGF_QOP_GAUSSPSI",
    "CVH_CGF_QOP_LNCUT", "CVH_CGF_QOP_NPAD", "CVH_CGF_QOP_NT",
    "CVH_CGF_QOP_SIGSCALE", "CVH_CGF_QOP_SCALARONLY",
)


def split_switches(env):
    """(cmsRun option string, remaining env) for one arm's overlay dict."""
    opts, rest = [], {}
    for k, v in (env or {}).items():
        if k in CVH_OPTION:
            # the C++ takes 0/1 for the flags and a number for the rest; the
            # dicts already speak that vocabulary
            opts.append(f"{CVH_OPTION[k]}={v}")
        elif k.startswith("CVH_") and k not in CVH_ENV_ONLY:
            raise KeyError(
                f"{k} is not a known CVH switch. Add it to CVH_OPTION (if its "
                f"C++ reader takes a ParameterSet parameter) or to "
                f"CVH_ENV_ONLY (if it is still a getenv). Exporting an unknown "
                f"name silently would put the job on the default and look like "
                f"it had been set.")
        else:
            rest[k] = v
    return " ".join(opts), rest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_trackres_260802/task_*/globalcor_resclosure_*.root")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--max-tracks", type=int, default=100000)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf_trackres_cache.npz"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--closure", action="store_true")
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--khit", type=float, default=0.0,
                   help="log-scale applied to the hit (Gaussian) variance share")
    p.add_argument("--kms", type=float, default=0.0,
                   help="log-scale applied to the MS exponent (collision counts)")
    p.add_argument("--kioni", type=float, default=0.0,
                   help="log-scale applied to the ionization exponent")
    p.add_argument("--krad", type=float, default=1.0,
                   help="LINEAR scale on the radiative (brems+pair) exponent. "
                        "Linear, not a log scale, so that 0 is expressible: "
                        "--krad 0 is the pre-2026-09-03 model exactly")
    p.add_argument("--no-rad", action="store_true",
                   help="do not build the radiative term at extraction time "
                        "(writes zeros and rad_model=0). For the `nomsrad` "
                        "sample, whose SIM has no brems/pair")
    p.add_argument("--kdel", type=float, default=None,
                   help="log-scale on the discrete delta-ray recoil block "
                        "(net of the Moliere carve); omit to leave it off")
    p.add_argument("--ioni-norm", choices=["var", "raw"], default="var",
                   help="normalisation of the pooled ionization scalar weight. "
                        "'var' (default) = sqrt(v_pool/sq2)/sigma with "
                        "sq2 = sum_legs sc_leg * sum_steps gsig2*g^2, where "
                        "sc_leg is the exported ioniqscalev factor -- exact "
                        "for BOTH estimators since 2026-09-03 (sc == 1 for "
                        "CgfQoPMode=0; on files written before the export the "
                        "scale defaults to 1 and `var` is valid only for "
                        "mode 0, as before). 'raw' = |w_b[0]|/sigma from the "
                        "exported per-dof influence weights (resinfv), "
                        "independent of gsig2 and of the scale of dV_b but "
                        "neglecting the intra-leg transport (one-sided ~10% "
                        "low in the exponent, growing with leg length)")
    p.add_argument("--hitmode", choices=["gauss", "class"], default="gauss",
                   help="hit term: one Gaussian of the summed block variance "
                        "(the historical treatment, and the only block family "
                        "in the CF that was still Gaussian), or the MEASURED "
                        "per-class densities summed block by block")
    p.add_argument("--bank-subdir", default="hitres3")
    p.add_argument("--bank-tag", default="mugun_lowpt")
    p.add_argument("--bank-nfiles", type=int, default=25)
    p.add_argument("--nsigma-bins", type=int, default=4)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


_EULER = 0.5772156649015328606
# |a w| below which the Taylor series is used. The series converges for every
# argument; this bound keeps the alternating cancellation below one digit.
_DT_SER = 2.0
_DT_NSER = 80


def _ein_neg(s, out=None):
    """Ein(-s) = -sum_{k>=1} s^k/(k k!), for complex s. Entire, and finite at
    s = 0 (unlike E1, which is what makes it the right object here).

    Series for |s| <= _DT_SER; Ein(-s) = gamma + Log(-s) + E1(-s) beyond, where
    E1 comes from scipy (accurate to 1e-16 on the imaginary axis -- measured,
    2026-08-13 XX, so it is NOT the weak link).
    """
    s = np.asarray(s, dtype=np.complex128)
    res = np.empty(s.shape, dtype=np.complex128) if out is None else out
    sm = np.abs(s) <= _DT_SER
    if sm.any():
        ss = s[sm]
        term = np.ones_like(ss)
        acc = np.zeros_like(ss)
        for k in range(1, _DT_NSER):
            term = term * ss / k
            acc = acc - term / k
            if k > 3 and np.max(np.abs(term)) < 1e-19 * max(
                    float(np.max(np.abs(acc))), 1e-300):
                break
        res[sm] = acc
    bg = ~sm
    if bg.any():
        from scipy.special import exp1
        ss = s[bg]
        res[bg] = _EULER + np.log(-ss) + exp1(-ss)
    return res


def _delta_term_2d(a, w):
    """Vectorized <(e^{iaE} - 1 - iaE)/E^2> for the 1/E^2 spectrum on
    [1, w] (units of e0): a (ns, nt) real, w (ns,) scalar per step.
    Same math as cf_ioni_exact.delta_term, and the imaginary-argument twin of
    cgf_saddlepoint._delta_derivs (b -> ia).

    With x = i a w and y = i a, and using E1(-y) - E1(-x) - ln w
    = Ein(-y) - Ein(-x) exactly (the ln w and the two branch offsets cancel
    analytically, since x/y = w > 0):

        <...> * N = expm1(y) - expm1(x)/w + y (Ein(-y) - Ein(-x)),   N = 1 - 1/w
                  = sum_{k>=2} (x^k/w - y^k) / (k! (k-1))            [series]

    THE EXPANSION PARAMETER IS a*w, NOT a. The series of <e^{iaE}-1-iaE>
    requires a*E << 1 over the WHOLE support, and the support reaches E = w.
    For muons w = tmax/e0 ~ 1e8-1e10 (keV-scale e0 against a multi-GeV
    kinematic tmax), so a guard on |a| alone selects the series in a regime
    where it is wrong by orders of magnitude: at w = 1e9, a = 1e-6 the old
    |a|-guard gave Im = -8.3e-2 against the true -6.5e-6, and that spurious
    phase, stacked over ~350 steps, turned the model CF into a pure
    oscillation at small t. That guard is KEPT -- the switch below is on
    |a| * w.

    WHAT CHANGED 2026-08-13 (XX). Both branches were rewritten; neither was
    giving wrong VALUES (the argument is imaginary, |e^{iaw}| = 1, so nothing
    ever overflowed) but both lost precision, worst AT the old crossover:
      * the series kept only k = 2, 3, so it was truncated at (a w)^2/36
        -- 2.8e-6 at a w = 1e-2, 2.5e-5 at 3e-2. It now runs to convergence.
      * the closed form grouped e^{ia} - (1 - 1/w). fl(1 - 1/w) carries a
        rounding error of eps/2 ~ 5.5e-17 while the result is only O(a^2 w)
        = O(x^2/w), so the RELATIVE error grew linearly in w: 3e-13 at
        w = 20, 2.4e-10 at w = 1e4, 7.0e-5 at w = 1e9, 7.3e-4 at w = 1e10.
        expm1(y) - expm1(x)/w cancels the 1/w analytically instead.
    Measured worst error against mpmath at 60 dps is now 6e-15 over
    w = 20..1e10 and a w = 1e-6..1e2.
    """
    a = np.asarray(a, dtype=np.complex128)
    wb = np.broadcast_to(np.asarray(w, dtype=np.float64)[:, None], a.shape)
    y = 1j * a
    x = y * wb
    N = 1.0 - 1.0 / wb
    out = np.empty(a.shape, dtype=np.complex128)

    ser = np.abs(x) <= _DT_SER
    if ser.any():
        xs, ys, ws = x[ser], y[ser], wb[ser]
        tx = np.ones_like(xs)
        ty = np.ones_like(ys)
        acc = np.zeros_like(xs)
        for k in range(1, _DT_NSER):
            tx = tx * xs / k
            ty = ty * ys / k
            if k >= 2:
                acc = acc + (tx / ws - ty) / (k - 1.0)
                if np.max(np.abs(tx)) < 1e-19 * max(
                        float(np.max(np.abs(acc))), 1e-300):
                    break
        out[ser] = acc

    bg = ~ser
    if bg.any():
        xb, yb, ww = x[bg], y[bg], wb[bg]
        with np.errstate(all="ignore"):
            out[bg] = (np.expm1(yb) - np.expm1(xb) / ww
                       + yb * (_ein_neg(yb) - _ein_neg(xb)))
    return out / N


# ------------------------------------------------------- EXACT DELTA SPECTRUM
# The three integrals the exact spin-1/2 knock-on spectrum needs, in units of
# the channel's lower limit e0 (u = T/e0, alpha = a e0, W = tmax/e0):
#
#   J0 = int_1^W (e^{i alpha u} - 1 - i alpha u) / u^2 du     <- the 1/T^2 part
#   J1 = int_1^W (e^{i alpha u} - 1 - i alpha u) / u   du     <- the -b2 T/Tmax
#   J2 = int_1^W (e^{i alpha u} - 1 - i alpha u)       du     <- the spin term
#
# so that, for a step whose delta channel is
#     dN/dT = (xi/T^2) [1 - b2 T/tmax + T^2/(2 E^2)]  on [e0, tmax],
# the centred compound-Poisson log-CF at conjugate variable a is
#     S = (xi/e0) [ J0 - (b2 e0/tmax) J1 + (e0^2/(2 E^2)) J2 ].
#
# J0 is exactly the (unnormalized) integrand of `_delta_term_2d`, so the pure
# 1/E^2 branch is recovered by dropping the other two -- which is the check the
# regime-1 / regime-2 comparison uses.
#
# BRANCHING is on |alpha W| = |a * tmax|, NOT |alpha|, for the reason
# `_delta_term_2d` documents at length: the support reaches T = tmax and a
# guard on |a| alone selects the series in a regime where it is wrong by orders
# of magnitude.  Series:
#   J0 = sum_{k>=2} (x^k/W - y^k) / (k! (k-1))
#   J1 = sum_{k>=2} (x^k    - y^k) / (k!  k)
#   J2 = sum_{k>=2} (x^k W  - y^k) / (k! (k+1)),     x = i alpha W, y = i alpha
# Closed forms (E1 via the entire Ein, so nothing is evaluated near a branch
# cut and the ln W cancels analytically):
#   J0 = expm1(y) - expm1(x)/W + y (Ein(-y) - Ein(-x))
#   J1 = (Ein(-y) - Ein(-x)) - y (W - 1)
#   J2 = (e^x - e^y)/y - (W - 1) - y (W^2 - 1)/2

def _delta_terms_exact(a, w):
    """(J0, J1, J2) as above.  a: (ns, nt) real; w: (ns,) = tmax/e0."""
    a = np.asarray(a, dtype=np.complex128)
    wb = np.broadcast_to(np.asarray(w, dtype=np.float64)[:, None], a.shape)
    y = 1j * a
    x = y * wb
    J0 = np.empty(a.shape, dtype=np.complex128)
    J1 = np.empty(a.shape, dtype=np.complex128)
    J2 = np.empty(a.shape, dtype=np.complex128)

    ser = np.abs(x) <= _DT_SER
    if ser.any():
        xs, ys, ws = x[ser], y[ser], wb[ser]
        tx = np.ones_like(xs)
        ty = np.ones_like(ys)
        a0 = np.zeros_like(xs)
        a1_ = np.zeros_like(xs)
        a2_ = np.zeros_like(xs)
        for k in range(1, _DT_NSER):
            tx = tx * xs / k
            ty = ty * ys / k
            if k >= 2:
                a0 = a0 + (tx / ws - ty) / (k - 1.0)
                a1_ = a1_ + (tx - ty) / k
                a2_ = a2_ + (tx * ws - ty) / (k + 1.0)
                if np.max(np.abs(tx)) * float(np.max(ws)) < 1e-19 * max(
                        float(np.max(np.abs(a2_))), 1e-300):
                    break
        J0[ser], J1[ser], J2[ser] = a0, a1_, a2_

    bg = ~ser
    if bg.any():
        xb, yb, ww = x[bg], y[bg], wb[bg]
        with np.errstate(all="ignore"):
            ein = _ein_neg(yb) - _ein_neg(xb)
            J0[bg] = np.expm1(yb) - np.expm1(xb) / ww + yb * ein
            J1[bg] = ein - yb * (ww - 1.0)
            J2[bg] = (np.exp(xb) - np.exp(yb)) / yb - (ww - 1.0) \
                - yb * (ww * ww - 1.0) / 2.0
    return J0, J1, J2


def exact_delta_exponent(xi, e0, tmax, beta2, etot, a, spinhalf=True):
    """Centred log-CF of the exact delta channel, summed over steps.

    xi, e0, tmax, etot: (ns,) in MeV; beta2: (ns,); a: (ns, nt) conjugate
    variable in 1/MeV.  Returns (nt,) complex.

    NOTE `_delta_terms_exact` works in units of e0 -- its first argument is
    alpha = a * e0, exactly as `_delta_term_2d`'s is -- so the conversion is
    done HERE and not by the caller."""
    w = tmax / e0
    J0, J1, J2 = _delta_terms_exact(a * e0[:, None], w)
    pref = (xi / e0)[:, None]
    c1 = (beta2 * e0 / tmax)[:, None]
    S = pref * (J0 - c1 * J1)
    if spinhalf:
        # the spin term is present only for spin 1/2 (mu, p), exactly as in
        # G4BetheBlochModel's `0.5 == spin` branch -- pi and K do not have it
        S = S + pref * (e0 ** 2 / (2.0 * etot ** 2))[:, None] * J2
    return np.sum(S, axis=0)


# `gamma` from `tmax`, ASSUMING A MUON. Kept only as a cross-check of the
# exported beta2/etot columns for muon models -- it is NOT used to build the
# CF, because Tmax fixes gamma only once the particle mass is known, and
# inverting a 3 GeV KAON's Tmax = 41.1 MeV with the muon mass returns E = 698
# MeV instead of 3175 MeV and beta^2 = 0.9989 instead of 0.9761. beta^2
# multiplies the entire suppression term, so that is not a small error.
_MMU = 105.6583745
_MEL = 0.51099895


def gamma_from_tmax(tmax):
    r = _MEL / _MMU
    b = -2.0 * np.asarray(tmax, dtype=np.float64) * r
    c = -(2.0 * _MEL + np.asarray(tmax, dtype=np.float64) * (1.0 + r * r))
    return (-b + np.sqrt(b * b - 8.0 * _MEL * c)) / (4.0 * _MEL)


# ---------------------------------------------------------------- A3 GAUGE
# DIAGNOSTIC KNOB, default-inert (1.0), in exactly the style of the pre-existing
# KMS_SCALE / MS_NSUB module globals.
#
# The exported Urban record describes the block's ionization as two Poisson
# excitation channels (a1,e1),(a2,e2) plus `a3` collisions drawn from 1/E^2 on
# [e0, tmax = Tmax].  That third channel is the model's ENTIRE representation of
# hard delta rays.  Geant4's own explicit delta production can be counted
# directly in the simulation (tail_probe.py hardrate), and it is measurably not
# the same number.  `IONI_A3_SCALE` multiplies a3 so the size of that
# disagreement can be propagated through the closure's own transform and
# weights, and `IONI_EXC_SCALE` rescales the excitation channels so the block's
# MEAN loss can be held fixed while its hard-tail weight moves -- otherwise the
# gauge is a mean change as well as a shape change and the two cannot be told
# apart.  (The CF is centred, so the mean does not enter z directly; keeping it
# fixed is what makes the modified block a physically admissible alternative
# rather than a different amount of material.)
IONI_A3_SCALE = 1.0
IONI_EXC_SCALE = 1.0

# SYSTEMATIC KNOB, default-inert. Multiplies the regime-2/3 delta channel's
# Tmax -- the spectrum's upper limit AND the beta^2 T/Tmax suppression -- while
# holding beta^2 and E at their exported values. It answers exactly one
# question: how much would the closure move if the record's Tmax disagreed with
# the one Geant4 sampled from? (Measured disagreement: +2.3e-7 relative, from
# the propagator's hard-coded electron mass. See NOTES_DELTASPEC section 8.)
# It is NOT a tune and nothing is fitted to it.
IONI_TMAX_SCALE = 1.0


def a3_gauge_exc_scale(steps, f_a3):
    """The excitation scale that holds a pooled block's mean loss fixed when a3
    is multiplied by `f_a3`.  Returns 1.0 if there is no excitation to give."""
    reg, gam = steps[:, 0], steps[:, 9]
    a1, e1 = steps[:, 2], steps[:, 3] * gam
    a2, e2 = steps[:, 4], steps[:, 5] * gam
    a3, e0, tmx = steps[:, 6], steps[:, 7] * gam, steps[:, 8] * gam
    m = (reg != 0) & (a3 > 0) & (tmx > e0) & (e0 > 0)
    m1 = m & (reg != 2) & (reg != 3)
    m2 = m & ((reg == 2) | (reg == 3))
    C = np.zeros_like(a3)
    C[m1] = 1.0 / (1.0 / e0[m1] - 1.0 / tmx[m1])
    Ed = float(np.sum(a3[m1] * C[m1] * np.log(tmx[m1] / e0[m1])))
    if m2.any():
        # regime 2/3: column 6 is xi (already gam-scaled above); the channel's
        # mean is xi [ln(tmax/e0) - b2 (1-e0/tmax) (+ (tmax^2-e0^2)/(4E^2))]
        b2, et = steps[:, 11], steps[:, 12]
        t = (np.log(tmx[m2] / e0[m2]) - b2[m2] * (tmx[m2] - e0[m2]) / tmx[m2])
        half = reg[m2] == 2
        t = t + np.where(half, (tmx[m2] ** 2 - e0[m2] ** 2)
                         / (4.0 * np.maximum(et[m2], 1e-30) ** 2), 0.0)
        Ed += float(np.sum(a3[m2] * t))
    Ee = float(np.sum(np.where(reg != 0, a1 * e1 + a2 * e2, 0.0)))
    if Ee <= 0.0:
        return 1.0
    return max((Ee + Ed * (1.0 - f_a3)) / Ee, 0.0)


_QSC_WARNED = [False]


def ioni_sq2(steps, qsc=None):
    """The block variance the FIT used, sum_steps sc_leg * gsig2 * (cs*1e-3)^2.

    `steps` are the `ioniurbanv` rows pooled under one global parameter index
    and `qsc` the matching `ioniqscalev` rows RESHAPED TO (-1, 2), i.e. one
    [sc, nstep] pair per leg sharing that index, in drain order.

    WHY THE SCALE IS NEEDED (2026-09-03). `--ioni-norm var` recovers the
    block's scalar weight as sqrt(v_b / sq2), which is an identity only while
    the record's gsig2 IS the variance that went into dV_b. Under
    CgfQoPMode >= 1 the propagator substitutes the block's Fisher weight,
    dV_b -> sc * dV_b, while the record keeps the untruncated second cumulant
    (Geant4ePropagator::cgfQScale documents both sides), so `var` came out
    ~20x small in the weight and ~400x in the exponent. `sc` is exactly 1.0
    on every CgfQoPMode=0 file and on every two-track fit, so old caches and
    the legacy arm are unchanged bit-for-bit.

    WHY PER LEG AND NOT PER BLOCK. Several legs can share one global index
    (a track crossing the same module twice, ~1/3 of blocks) and each carries
    its own sc, so the pooled sum is sum_l sc_l * (step sum of leg l). The
    legs' rows are contiguous and in drain order, so the exported step counts
    split them exactly; if they do not add up (a file written by a mismatched
    build) this falls back to the mean sc and warns once, rather than
    silently mixing the two conventions.
    """
    # `steps[:,1] * gq * gq`, left to right, is the expression this replaced;
    # float multiplication is not associative, so keeping the order makes the
    # no-scale path BIT-identical to the pre-2026-09-03 caches.
    gq = steps[:, 10] * 1e-3
    w2 = steps[:, 1] * gq * gq
    if qsc is None or not len(qsc):
        return float(np.sum(w2))
    ns = qsc[:, 1].astype(np.int64)
    if ns.sum() != len(steps):
        if not _QSC_WARNED[0]:
            logger.warning(
                "ioniqscalev step counts (%d) do not match the pooled "
                "ioniurbanv rows (%d); falling back to the mean scale"
                % (int(ns.sum()), len(steps)))
            _QSC_WARNED[0] = True
        return float(np.sum(w2) * np.mean(qsc[:, 0]))
    off = np.concatenate(([0], np.cumsum(ns)))
    return float(sum(sc * w2[off[i]:off[i + 1]].sum()
                     for i, sc in enumerate(qsc[:, 0])))


def ioni_step_exponent(steps, wstd, tau):
    """Centered Urban log-CF exponent of one pooled ionization block,
    in standardized-z units: per step the raw qop-noise CF evaluated at
    wstd*tau (wstd = w/sigma includes the fit-unit conversion). steps:
    (n, 11) ioniurbanv records. Vectorized over steps. Returns complex
    array on tau."""
    if IONI_A3_SCALE != 1.0 or IONI_EXC_SCALE != 1.0:
        steps = steps.copy()
        steps[:, 6] *= IONI_A3_SCALE
        steps[:, 2] *= IONI_EXC_SCALE
        steps[:, 4] *= IONI_EXC_SCALE
    reg = steps[:, 0]
    gsig2 = steps[:, 1].astype(np.float64)
    g = steps[:, 10].astype(np.float64) * 1e-3  # qop per MeV
    gs = wstd * g                               # (ns,)
    S = np.zeros(len(tau), dtype=np.complex128)

    mg = reg == 0
    if mg.any():
        S += -0.5 * tau ** 2 * float(np.sum(gsig2[mg] * gs[mg] ** 2))

    mu = ~mg
    if mu.any():
        gam = steps[mu, 9]
        gsu = gs[mu]
        for ja, je in ((2, 3), (4, 5)):
            aj = steps[mu, ja]
            ej = steps[mu, je] * gam
            act = (aj > 0.) & (ej > 0.)
            if act.any():
                th = (gsu[act] * ej[act])[:, None] * tau[None, :]
                S += np.sum(aj[act][:, None] * (np.exp(1j * th) - 1. - 1j * th), axis=0)
        a3 = steps[mu, 6]
        e0 = steps[mu, 7] * gam
        tmax = steps[mu, 8] * gam
        # regime 2 (CVH_IONI_EXACTDELTA): the delta channel is the exact
        # spin-1/2 knock-on spectrum and column 6 carries its normalization xi
        # (an energy, so it takes the `scaling` factor) rather than a count.
        # Split on the regime rather than on a heuristic, so a regime-2 record
        # can never be read as a regime-1 one.
        regmu = reg[mu]
        ex = (regmu == 2) | (regmu == 3)
        act = (a3 > 0.) & (tmax > e0) & (e0 > 0.)
        if (act & ~ex).any():
            m = act & ~ex
            aarg = (gsu[m] * e0[m])[:, None] * tau[None, :]
            S += np.sum(a3[m][:, None] * _delta_term_2d(aarg, tmax[m] / e0[m]),
                        axis=0)
        if (act & ex).any():
            if steps.shape[1] < 13:
                raise ValueError(
                    "regime 2/3 record with stride 11: beta^2 and E are not in "
                    "the file. Re-export with CVH_IONI_EXACTDELTA (which writes "
                    "stride 13); they cannot be recovered from tmax without the "
                    "particle mass.")
            b2 = steps[mu, 11]
            et = steps[mu, 12]
            if IONI_TMAX_SCALE != 1.0:
                tmax = tmax * IONI_TMAX_SCALE
            for spin, half in ((2, True), (3, False)):
                m = act & (regmu == spin)
                if not m.any():
                    continue
                S += exact_delta_exponent(a3[m] * gam[m], e0[m], tmax[m],
                                          b2[m], et[m],
                                          gsu[m][:, None] * tau[None, :],
                                          spinhalf=half)
                if IONI_KOKOULIN != 0.0:
                    S += IONI_KOKOULIN * _kokoulin_exponent(
                        a3[m] * gam[m],
                        np.maximum(e0[m], max(IONI_KOKOULIN_TCUT, _KOK_TMIN)),
                        tmax[m], b2[m], et[m],
                        gsu[m][:, None] * tau[None, :],
                        nbin=IONI_KOKOULIN_NBIN, spinhalf=half)
    return S


# ---------------------------------------------------- KOKOULIN, default-inert
# Geant4's G4MuBetheBlochModel multiplies the knock-on cross section by
# R. Kokoulin's radiative correction  1 + (alpha/2pi) a1 (a3 - a1),
# a1 = ln(1 + 2T/m_e), a3 = ln(4 E (E-T)/M^2), above T = 100 keV for muons
# above 1 GeV.  It reaches +6 % at the hard end.  It is NOT part of the
# tree-level spectrum this module implements, and NOTES_DELTASPEC section 1.5
# recorded that omission as a known residual (it has no closed-form CF).
#
# `IONI_KOKOULIN` is a default-inert gauge in the style of IONI_A3_SCALE /
# IONI_TMAX_SCALE / MS_NSUB: 0.0 reproduces the published model bit-for-bit
# (the branch is not entered at all), 1.0 puts Geant4's own correction into
# the model.  `IONI_KOKOULIN_TCUT` is the e- production threshold of the
# simulation being compared against: the SIMULATION applies the correction
# only to the EXPLICIT secondaries, i.e. only above its production cut -- the
# Urban straggling below the cut has no radiative correction -- so the
# correction must start at max(tcut, 100 keV) and not at e0.
#
# THE SWITCH IS SHARED WITH THE C++ SIDE (2026-08-15, NOTES_QVALID).
# `CVH_IONI_KOKOULIN` is read by `cvhcgf::ioniKokoulinEnabled()` in
# TrackPropagation/Geant4e, which is the SINGLE C++ reader and is what puts the
# correction into the variance the track fit consumes (Q(0,0), and `gsig2` in
# the exported record).  Deriving this module's default from the SAME variable
# is the offline half of that coupling, in the pattern of
# `cvhcgf::referenceIsIonOnly()`: a job that exports with the correction on and
# then analyses in the same environment cannot end up half-corrected because
# somebody forgot a convention.
#
# Setting the module global directly still works and still wins (that is how
# `samplergap.py real --kok 0 1` runs both arms in one process); the
# environment only supplies the DEFAULT.
#
# DEFAULT OFF, mirroring `cvhcgf::ioniKokoulinEnabled()`.  It was ON for one
# week (2026-08-16, NOTES_DEFAULTON.md) and reverted; the SHARED-SWITCH
# property is what matters here and is unchanged.  `env_flag` is the SAME
# tri-state convention the C++ reader uses -- unset means the default, `=0`
# means off, `=1` means on -- so the two halves of the shared switch cannot
# disagree in either direction, at either default.
IONI_KOKOULIN = 1.0 if env_flag("CVH_IONI_KOKOULIN", True) else 0.0
IONI_KOKOULIN_TCUT = 0.0
IONI_KOKOULIN_NBIN = 96

# ------------------------------------------------------- the knob REGISTRY
# Every module-level global that changes what this module COMPUTES, in one
# place, so that a cache key can be built from the registry instead of from
# somebody's memory of the list.  That is not bookkeeping: `_PHI_CACHE`'s key
# carried IONI_A3_SCALE, IONI_EXC_SCALE and IONI_TMAX_SCALE but not
# IONI_KOKOULIN, which is why NOTES_RADOFF2 and NOTES_HADRONS both had to
# disable the cache by hand and why a switched-ON cell could silently reuse
# switched-OFF numbers.
#
# `_NOT_PHYSICS` is the explicit counterpart: names that look like knobs and
# are NOT, so the completeness audit (`barkas_probe.py guards`) can tell the
# two apart and fail on anything in neither list.
# ------------------------------------------------- the MS electron ceiling
# DIAGNOSTIC, DEFAULT OFF (0.0 -> `ms_step_exponent` is bit-identical to the
# production path).
#
# The Moliere chi_c^2 carries Z(Z+1): Z^2 for the nucleus and Z for the atomic
# electrons, with ONE angular range for both.  Geant4 does not.
# `G4WentzelOKandVIxSection::ComputeMaxElectronScattering` gives the electron
# term its OWN kinematic ceiling
#
#     1 - cos(theta_e,max) = min(cut, Tmax) * m_e / p^2
#
# and `ComputeTransportCrossSectionPerAtom` integrates the electron piece only
# to there (`costm = max(cosTMax, cosTetMaxElec)`) while the nuclear piece runs
# to cosTetMaxNuc.  Above `cut` the electron scattering is not lost -- it is
# delivered as EXPLICIT delta rays by muIoni/hIoni, whose recoil turns the
# primary -- but above the KINEMATIC Tmax it does not exist at all, while the
# model's Z(Z+1) keeps scattering off electrons all the way to the nuclear
# form-factor angle.
#
# theta(Tmax) is strongly species dependent at fixed momentum (8.5e-3 rad for a
# 3.1 GeV muon, 1.1e-3 rad for a proton), which is the one term in the whole
# enumeration that is.  Setting this to 1.0 imposes the ceiling and is the
# controlled test of that term.
#
# CAVEAT: `gshape`'s ymax is the SMOOTH dipole (1 + y^2/ymax^2)^-4, not a hard
# kinematic edge, so this sizes the LOG-RANGE removal (which dominates) and not
# the exact edge shape.  The Z^2 : Z split uses effZ, exact for the elementary
# toy material and approximate for a compound.
MS_ELEC_TMAX = 1.0
_ME_GEV = 0.51099895000e-3

# ------------------------------------- the SHAPE of the electron ceiling
# DIAGNOSTIC, DEFAULT 0.0 = the published `MS_ELEC_TMAX` behaviour, bit-for-bit
# (the `gshape` dipole at ym_e).  It only does anything when MS_ELEC_TMAX != 0.
#
# The caveat above -- "gshape's ymax is the SMOOTH dipole, not a hard kinematic
# edge" -- was corrected for ANALYTICALLY in NOTES_MSTERMS s8.3 with a factor
# 1.27-1.68, and that analytic correction is WRONG, because it compared the
# dipole against a SHARP CUT on the pure Wentzel law.  The exact kinematics of a
# heavy projectile on a free electron (cf_ms_exact, `_build_elec_tables`) give
#
#     theta^2(T) = (2 m_e T/p^2) (1 - T/Tmax)          [G4 keeps the first factor only]
#     theta_e,max = theta_G4/2                          EXACTLY, at T = Tmax/2
#     L_e = ln(t_G4/chi_a^2) - 2 - beta^2/2             [G4's f(x_e) gives -1, no beta term]
#
# so the correct electron transport log is a full unit of log BELOW the sharp-cut
# value s8.3 used, and the dipole is only 0.33 units above the truth, not 1.83.
#
#   0.0  `gshape` dipole at ym_e            -- the published path
#   0.5  the SAME dipole on the new, 4.4x finer quadrature -- the NUMERICS
#        control: 0.5 minus 0.0 is quadrature, not physics
#   1.0  hard edge at ym_e                  -- G4's own transport-XS range
#   2.0  exact kinematics, caustic kernel at ym_e/2, with the (1-beta^2 T/Tmax)
#        of the true spin-0 dsigma/dT       -- THE PHYSICS
#   3.0  the same with beta^2 = 0           -- isolates that factor
MS_ELEC_EDGE = 1.0

# ------------------------------------------ the MS kernel's own quadrature
# DIAGNOSTIC, DEFAULT OFF (0.0 -> `gshape` is called exactly as in production).
#
# `cf_ms_exact.gshape`'s table is built on `_Y2 = logspace(-4, 13, 360)` with
# `_DY2 = np.gradient(_Y2)` -- 21 nodes per decade, i.e. h = 0.109 in ln(y^2).
# NOTES_XXII s8.4 measured the resulting bias at **+2.03e-3 in |S|** and left it
# unfixed ("a resolution choice").  It has never been sized in closure units.
#
# Setting this to 1.0 evaluates the SAME dipole kernel on the 4.4x-finer grid
# built for the electron term (`cf_ms_exact._build_elec_tables`), which is
# validated against the closed-form transport log at 1e-6 where `gshape` is at
# 2e-3.  It is a PURE NUMERICS change -- identical physics, identical
# theta_FF -- and it is the model's own +0.2 % over-statement of every MS
# cumulant, which is exactly the currency the KMS_SCALE gauge is written in.
MS_FINE_G = 1.0

# ------------------------------------------------------------ the ymax snap
# DEFAULT 1.0 = the production behaviour: each step's form-factor ceiling
# ymax = theta_FF/chi_a is ROUNDED to the nearest of `cf_ms_exact._YMAXG`'s 13
# half-decade rows before `gshape` is called, so gshape's own interpolation
# never runs.  On the layered toy at pT = 3 the true ymax is 2.026e4 and the
# nearest row is 10^4.5 = 3.162e4, a factor 1.561 UP, and since G(tau -> 0) ~
# ln(ymax^2) that inflates the model's MS second moment by 5.2 %.  Setting this
# to 0.0 passes the step's own ymax and lets gshape interpolate, which is the
# controlled measurement of that approximation.
MS_SNAP_YMAX = 0.0

# ------------------------------------------- the WentzelVI INTERNAL split
# DIAGNOSTIC, DEFAULT OFF (0.0 -> `ms_step_exponent` is bit-identical to the
# production path: the branch is not entered at all).
#
# `G4WentzelVIModel::SampleScattering` does NOT sample the sub-threshold
# scattering from the single-scattering law.  It draws z ~ Exp(mean z0) with
# z0 = tPathLength/(2 lambdaeff), i.e. an EXACT Gaussian in the projected
# angle, and samples only the part above
#
#     1 - cos(theta_min) = 1.25 * tPathLength / lambda_transport
#
# explicitly.  (`useSecondMoment` is false AND
# `G4WentzelOKandVIxSection::ComputeSecondTransportMoment` returns 0.0
# unconditionally, so the Gamma(2,2) branch is dead code.)  The offline
# transform integrates ONE compound Poisson across that boundary.
#
# THE KEY STRUCTURAL FACT: lambdaeff is the RESTRICTED transport mean free
# path, so <1-cos>_Gaussian = t/lambdaeff is the exact first transport moment
# of the law it replaces -- and for an isotropic 2D deflection the first
# transport moment IS the projected variance, while a compound Poisson's
# second cumulant IS its transport moment.  The split therefore preserves the
# variance EXACTLY and changes only the fourth and higher cumulants.  With
# q = sqrt(chi_a^2) w tau the difference is the tail of the J0 series:
#
#     dS(q) = (chi_c^2/chi_a^2) sum_{k>=2} (-1)^(k+1) (q^2/4)^k M_k(U) / (k!)^2
#     M_k(U) = int_0^U u^k/(1+u)^2 du ,   U = 1.25 (chi_c,step^2/chi_a^2) L_g
#
# with L_g = [f(x_e) + Z f(x_N)]/(Z+1), G4's own transport log per unit
# chi_c^2 (15.0 for the toy; `wvisplit.py g4` measures it against G4's
# lambda).  dS < 0: the split makes the model LESS peaked, i.e. it moves the
# `locx` closure the WRONG way.
#
# `MS_WVI_NPERX` is what sets the SIMULATION's step length: the number of
# delta rays above the production cut per g/cm^2.  Nothing else limits a step
# inside a toy layer (msc defines 372 steps in 2e5 events; the range-based
# limits are metres), so a record of thickness x is crossed in 1 + Poisson(n)
# steps with n = MS_WVI_NPERX * x, and
#
#     sum_s x_s^2 / x^2  =  f(n) = 2/n - 2(1 - e^-n)/n^2
#
# exactly, for a Poisson process on [0, x] whose last interval runs to the
# boundary.  dS is QUADRATIC in the per-step chi_c^2, so f(n) -- not the mean
# step -- is the right reduction, and it differs from 1/n by a factor 2 at
# large n because the step lengths are exponential.  f -> 1 as x -> 0, so the
# exporter's zero-thickness filler records need no special case.
# MEASURED per species in `wvisplit.py steps` from the archived censuses
# (ionization secondaries, r < 107 cm gated): 9.49 /(g/cm^2) for the muon to
# 10.28 for the proton, i.e. 1/beta^2 as it must be.
# DEFAULT OFF, and NOT for the reason the other six are on.
#
# The 2026-08-18 flip put all seven harmonisations default-ON so that the model
# models the simulation.  This one cannot honour that: `wvi_split_exponent`
# represents dS by a J0 series truncated at _WVI_KMAX = 28, trusted only to
# q^2 U/4 = _WVI_ARGMAX = 60, and beyond that dS is CLAMPED TO ZERO -- legitimate
# only where the CF has already died, which `ms_step_exponent` asserts via
# _WVI_SMIN.  MEASURED on the real tracker (pT = 3, mu-, per plane):
#
#     plane  0   q^2U/4 =      0.3   clamp never fires
#     plane  5   q^2U/4 =   3266.9   S at clamp -6.093  (passes by 0.09)
#     plane  9   q^2U/4 =  18959.1   S at clamp -6.246  (passes by 0.25)
#     plane 14   q^2U/4 =  95365.6   S at clamp -5.209  GUARD FIRES
#     plane 18   q^2U/4 = 170352.6   S at clamp -5.086  GUARD FIRES
#
# i.e. the real geometry sits 3-4 ORDERS OF MAGNITUDE outside the series'
# validity region, the clamp is doing the work almost everywhere, and at the
# outer planes it is applied where |phi| ~ e^-5 -- the same size as the closure
# being measured.  Raising _WVI_ARGMAX/_WVI_KMAX cannot fix this: the terms go
# as q^{2k} and q2**kmax already overflows double at q2 ~ 5e10.
#
# This is a THIN-TARGET construction.  It was developed and gauged on the toy,
# whose dense 1 mm shells give small q^2 U; the real tracker's long air gaps and
# thin silicon are a different regime.  Making it universal needs a different
# representation of dS at large q^2 U (an asymptotic form, or direct quadrature
# of the J0 integral), not a bigger ceiling.
#
# Until then it stays opt-in: a default that hard-fails an entire geometry is
# worse than one that has to be asked for.  The other six harmonisations are
# unaffected and remain default-ON.
MS_WVI_SPLIT = 0.0
MS_WVI_NPERX = 9.494       # delta rays above the cut per g/cm^2
MS_WVI_LG = 15.00
_WVI_SSFACTOR = 1.25       # G4WentzelVIModel::SetSingleScatteringFactor(1.25)
_WVI_KMAX = 28             # terms of the J0 series; convergence asserted below
_WVI_ARGMAX = 60.0         # the largest q^2 U/4 the truncated series is trusted at
_WVI_MAXARG_SEEN = [0.0]   # diagnostic: the largest q^2 U/4 any call has used
# The MS exponent the ARGMAX clamp is required to sit below.  dS is
# negative-definite and monotonically steepening in q, so clamping it to zero
# can only UNDER-state the effect, and only where |phi| < e^{_WVI_SMIN}.
_WVI_SMIN = -6.0
# q^2/4 above which the series is clamped REGARDLESS of U: with kmax = 28,
# q2**kmax overflows double at q2 ~ 5e10, and for a record with a tiny U (the
# exporter's zero-thickness fillers) `q2 U <= _WVI_ARGMAX` does NOT bound q2.
# inf * 0 = nan, and the nan would then survive multiplication by the
# minNCollisions mask.  The real closure reaches q2 = 2.4e7 at most, so this is
# defensive; the `isfinite` assert below makes it non-silent either way.
_WVI_Q2MAX = 1.0e10
_WVI_SMAX_SEEN = [-np.inf]   # diagnostic: the largest MS exponent at the clamp
# `G4WentzelVIModel`'s own escape hatch, and it is not a detail: BOTH
# `ComputeGeomPathLength` and `ComputeTrueStepLength` contain
#     if(G4int(zPathLength*xtsec) < minNCollisions) { singleScatteringMode = true;
#                                                     lambdaeff = DBL_MAX; }
# with `minNCollisions = 10` and `xtsec` evaluated at cosThetaMin = 1, i.e. the
# TOTAL single-scattering rate.  A step with fewer than ten collisions gets NO
# Gaussian at all -- every scatter is explicit -- so the split does not act
# there.  chi_c^2/chi_a^2 IS that collision count (it reproduces the driver's
# `nssFull` to 2e-16, `wvisplit.py g4` V2), so the guard costs one comparison.
_WVI_MINCOLL = 10.0

PHYSICS_GLOBALS = ("IONI_A3_SCALE", "IONI_EXC_SCALE", "IONI_TMAX_SCALE",
                   "IONI_KOKOULIN", "IONI_KOKOULIN_TCUT", "IONI_KOKOULIN_NBIN",
                   "MS_ELEC_TMAX", "MS_ELEC_EDGE", "MS_FINE_G", "MS_SNAP_YMAX",
                   "MS_WVI_SPLIT", "MS_WVI_NPERX", "MS_WVI_LG")
_NOT_PHYSICS = ("COVTOL", "_ME_GEV",   # a tolerance and a physical constant
                # a G4 constant, a series length, and three numerical guards
                "_WVI_SSFACTOR", "_WVI_KMAX", "_WVI_ARGMAX", "_WVI_MAXARG_SEEN",
                "_WVI_SMIN", "_WVI_SMAX_SEEN", "_WVI_MINCOLL", "_WVI_Q2MAX")


def _wvi_moment(k, U):
    """M_k(U) = int_0^U u^k/(1+u)^2 du, in closed form (k >= 1).

    u^k/(1+u)^2 = sum_{j=0}^{k-2} C(k,j)... is avoidable: with v = 1+u,
    u^k = sum_j C(k,j) v^j (-1)^(k-j), so the integrand is a Laurent
    polynomial in v and every term integrates elementarily.  Written that way
    it is exact for any k and needs no quadrature."""
    from math import comb
    U = np.asarray(U, float)
    v = 1.0 + U
    hi = np.zeros_like(v)
    for j in range(k + 1):
        c = comb(k, j) * ((-1) ** (k - j))
        m = j - 2
        hi = hi + c * (np.log(v) if m == -1 else (v ** (m + 1) - 1.0) / (m + 1))
    if not np.any(U < 0.5):
        return hi
    # 1/(1+u)^2 = sum_j (-1)^j (j+1) u^j, |u| < 1.  The binomial form above
    # cancels catastrophically at small U (74 % error already at k = 4,
    # U = 1e-3), and the exporter's zero-thickness filler records land there.
    Us = np.minimum(U, 0.5)
    lo = np.zeros_like(v)
    for j in range(60):
        lo = lo + ((-1) ** j) * (j + 1) * Us ** (k + j + 1) / (k + j + 1)
    return np.where(U >= 0.5, hi, lo)


def wvi_split_exponent(q, U, kmax=None):
    """dS(q)/(chi_c^2/chi_a^2) for ONE model record: the difference between
    what WentzelVI samples and what the compound-Poisson transform has.

    The series is used rather than a quadrature because the closure never
    leaves the regime q^2 U << 1 (q sqrt(U) <= 0.4 on the layered toy at
    pT = 3, since |S| ~ 40 already at q ~ 4e-3), and the ratio of successive
    terms is ~ q^2 U/4.  Convergence is ASSERTED, not assumed."""
    kmax = _WVI_KMAX if kmax is None else kmax
    q2 = 0.25 * np.asarray(q, float) ** 2
    U = np.asarray(U, float)
    if U.ndim == 1 and q2.ndim == 2:
        U = U[:, None]
    from math import factorial
    # The moments depend on U ONLY.  Computing them on the (nrec, ntau)
    # broadcast instead of on the (nrec,) U vector costs a factor ntau ~ 400 and
    # makes a closure run take half an hour instead of two minutes.
    Mk = [_wvi_moment(k, U) / factorial(k) ** 2 for k in range(2, kmax + 1)]
    ok = (q2 * U <= _WVI_ARGMAX) & (q2 <= _WVI_Q2MAX)
    _WVI_MAXARG_SEEN[0] = max(_WVI_MAXARG_SEEN[0],
                              float(np.max(q2 * U)) if np.size(q2) else 0.0)
    q2s = np.where(ok, q2, 0.0)
    tot = np.zeros(np.broadcast(q2, U).shape)
    last = np.zeros_like(tot)
    p = q2s * q2s
    for k in range(2, kmax + 1):
        t = ((-1) ** (k + 1)) * p * Mk[k - 2]
        tot = tot + t
        last = t
        p = p * q2s
    m = ok & (np.abs(tot) > 0)
    if np.any(m):
        r = float(np.max(np.abs(last[m]) / np.abs(tot[m])))
        assert r < 1e-6, (
            f"wvi_split_exponent: the J0 series has not converged ({r:.2e} of "
            f"the total at k = {kmax}) -- q^2 U is out of the assumed regime")
    # Beyond q^2 U/4 = _WVI_ARGMAX the truncated series cancels and dS is set to
    # ZERO.  Two things make that safe, and both are checked rather than
    # assumed: (i) `ms_step_exponent` asserts that the MODEL's own MS exponent
    # is already below `_WVI_SMIN` everywhere the clamp fires, so |phi| there is
    # < e^{_WVI_SMIN}; (ii) moving `_WVI_ARGMAX` by a factor 2 must not move the
    # closure (`wvisplit.py gauge --argmax`).
    #
    # A closed-form asymptote was tried first and is WRONG: dropping the J0
    # integral as "oscillating away" gives -q^2/4 M_1(U) + M_0(U), which is
    # POSITIVE at the switch (the J0 integral is NOT small there -- it is
    # dominated by u < 1/q^2, where J0 ~ 1) and overflows exp().  dS is
    # negative-definite, so a positive value is a detectable error, and it was
    # detected this way.
    out = np.where(ok, tot, 0.0)
    assert np.all(np.isfinite(out)), (
        "wvi_split_exponent: non-finite dS -- the series overflowed inside the "
        "region it claims to be valid in")
    return out


def physics_state():
    """The module's physics knobs, as a hashable, printable tuple."""
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)

_ALPHA_PRIME = 1.0 / (2.0 * np.pi * 137.035999084)   # G4's alphaprime
_KOK_TMIN = 0.1          # G4MuBetheBlochModel::limitKinEnergy = 100 keV
_KOK_MUMIN = 1000.0      # G4MuBetheBlochModel::lowestKinEnergy = 1 GeV

# ---------------------------------------------------- THE MUON-ONLY GUARD
# Geant4 puts the Kokoulin factor in `G4MuBetheBlochModel` and NOWHERE else.
# Hadrons are ionized by `G4hIonisation`, whose high-energy model is
# `G4BetheBlochModel`, which has no such factor -- measured against Geant4's
# own `CrossSectionPerVolume` in NOTES_HADRONS s3.1 (the muon agrees with the
# Kokoulin-weighted integral to 3.8e-5 and every hadron with TREE LEVEL to
# 4e-5).  The C++ export has always known this:
#
#   G4UniversalFluctuationForExtrapolator.cc
#     kokoulinOn && ekin > kKokMuMin && std::abs(particle->GetPDGEncoding()) == 13
#
# This module did not, and the record carries no PDG column, so
# `CVH_IONI_KOKOULIN=1` in a hadron job silently applied a MUON-shaped factor
# (a3 = ln(4E(E-T)/m_mu^2)) to a kaon.  NOTES_HADRONS s3.4 measured the damage:
# it makes K- look 4x BETTER (0.00311 -> 0.00071) and pbar 3x better, because
# its u-shape mimics the charge-odd mean-loss bias -- i.e. the mistake is in
# the direction that looks like success.
#
# The mass is RECOVERABLE from the record, exactly, and does not need a new
# column: the regime-2/3 record carries `beta^2` and `E` (added by
# NOTES_DELTASPEC s9.2 precisely so that a kaon's Tmax is not inverted as a
# muon's), and
#
#       E * sqrt(1 - beta^2) = (gamma m) * (1/gamma) = m
#
# is an identity, not an approximation.  Measured on the exported records it
# returns the PDG masses to 1e-9 relative (`_mass_from_record` self-test in
# barkas_probe.py `guards`).  So the guard is applied HERE, on every existing
# file, rather than waiting for a record-layout change and a re-export of
# every model in the study.
#
# The tolerance is loose (1 MeV) on purpose: it has to separate the muon from
# the pion, its nearest neighbour, and 105.66 vs 139.57 MeV is a 34 MeV gap.
_KOK_MASS_TOL = 1.0      # MeV
_KOK_SUPPRESSED = {"n": 0, "masses": set()}


def _mass_from_record(beta2, etot):
    """The projectile mass implied by the regime-2/3 record, in MeV.

    m = E sqrt(1 - beta^2).  Exact for any particle; the record has both
    columns, and a record that does not (stride 11) never reaches this code --
    `ioni_step_exponent` raises on it."""
    return np.asarray(etot, float) * np.sqrt(
        np.clip(1.0 - np.asarray(beta2, float), 0.0, None))


def _is_muon_record(beta2, etot):
    """Elementwise: is this step a MUON's?  The offline half of the C++
    `std::abs(GetPDGEncoding()) == 13`."""
    return np.abs(_mass_from_record(beta2, etot) - _MMU) < _KOK_MASS_TOL


def kokoulin_factor(T, etot):
    """f_K(T), vectorized over T and over the per-step total energy."""
    T = np.asarray(T, dtype=np.float64)
    E = np.broadcast_to(np.asarray(etot, dtype=np.float64), T.shape)
    f = np.ones_like(T)
    m = (T > _KOK_TMIN) & (T < E - _MMU)
    if np.any(m):
        a1 = np.log(1.0 + 2.0 * T[m] / _MEL)
        a3 = np.log(4.0 * E[m] * (E[m] - T[m]) / (_MMU * _MMU))
        f[m] = 1.0 + _ALPHA_PRIME * a1 * (a3 - a1)
    return f


def _kokoulin_exponent(xi, t0, tmax, beta2, etot, a, nbin=96, spinhalf=True):
    """The Kokoulin CORRECTION's contribution to the delta channel's exponent.

    f_K - 1 is smooth and slowly varying in ln T; e^{i a T} is not, and at
    a T ~ 2e4 no practical quadrature in T resolves it.  So f_K - 1 is made
    piecewise constant on a log grid and the OSCILLATORY integral is done in
    CLOSED FORM on each bin with the same `exact_delta_exponent`.  The
    substitution beta^2 -> beta^2 hi/Tmax is what turns that routine's
    (upper limit == beta^2 denominator) convention into a genuine sub-range
    integral whose beta^2 term still carries the KINEMATIC Tmax; summing
    contiguous bins reproduces the full-range closed form to 7e-15.
    """
    xi = np.atleast_1d(np.asarray(xi, float))
    t0 = np.atleast_1d(np.broadcast_to(np.asarray(t0, float), xi.shape))
    tmax = np.atleast_1d(np.asarray(tmax, float))
    beta2 = np.atleast_1d(np.asarray(beta2, float))
    etot = np.atleast_1d(np.asarray(etot, float))
    a = np.atleast_2d(a)
    S = np.zeros(a.shape[1], dtype=np.complex128)
    lo0 = np.maximum(t0, _KOK_TMIN)
    ismu = _is_muon_record(beta2, etot)
    # `etot - _MMU > _KOK_MUMIN` is the C++ `ekin > kKokMuMin` and is only the
    # KINEMATIC half of the guard; `ismu` is the species half, which the C++
    # has had all along and this module did not.
    act = (lo0 < tmax) & (etot - _MMU > _KOK_MUMIN) & ismu
    nsup = int((~ismu).sum())
    if nsup:
        # Not an error -- Geant4 does exactly this -- but a job that ASKED for
        # the correction on a hadron has a configuration problem, so it is
        # said once per distinct mass rather than never.
        m0 = float(np.round(_mass_from_record(beta2[~ismu], etot[~ismu])[0], 3))
        if m0 not in _KOK_SUPPRESSED["masses"]:
            _KOK_SUPPRESSED["masses"].add(m0)
            msg = (f"IONI_KOKOULIN is on but {nsup} step(s) carry m = "
                   f"{m0:.3f} MeV, not the muon's {_MMU:.3f}: Geant4 applies "
                   f"the Kokoulin factor in G4MuBetheBlochModel ONLY, so the "
                   f"correction is SUPPRESSED for those steps (NOTES_HADRONS "
                   f"s3).")
            logger.warning(msg)
            # ...and to stderr as well, because a caller that has not set up
            # the wums root logger would otherwise see NOTHING, and silence
            # read as success is the exact failure this guard exists to stop.
            print("[cf_track_resolution] WARNING: " + msg, file=sys.stderr)
    _KOK_SUPPRESSED["n"] += nsup
    if not act.any():
        return S
    xi, lo0, tmx, b2, et, aa = (xi[act], lo0[act], tmax[act], beta2[act],
                                etot[act], a[act])
    fr = np.linspace(0.0, 1.0, nbin + 1)
    edges = lo0[:, None] * (tmx / lo0)[:, None] ** fr[None, :]
    for m in range(nbin):
        lo, hi = edges[:, m], edges[:, m + 1]
        kap = kokoulin_factor(np.sqrt(lo * hi), et) - 1.0
        g = kap != 0.0
        if not g.any():
            continue
        S += exact_delta_exponent(xi[g] * kap[g], lo[g], hi[g],
                                  b2[g] * hi[g] / tmx[g], et[g], aa[g],
                                  spinhalf=spinhalf)
    return S


def ms_step_exponent(steps, wstd, tau):
    """Moliere log-CF exponent of one pooled MS block in standardized-z
    units, FF-cut per step, vectorized over steps with the ymax bucketing
    of cf_ms_moliere._slot_S. steps: (n, 8) msmoliv records [effZ, effA,
    xg, pGeV, beta, thp2, dOverX0, stepGroup]."""
    ok = steps[:, 5] > 0.
    if not ok.any():
        return np.zeros(len(tau))
    # stride 10 (2026-08-08) carries the per-element sums in cols 8,9;
    # stride 8 files fall back to the effZ approximation inside
    # moliere_params.
    _st = steps[ok]
    if _st.shape[1] >= 10:
        prm = np.array([moliere_params(*s[:5], s[7], s[8]) for s in _st])
    else:
        prm = np.array([moliere_params(*s[:5]) for s in _st])  # (ns, 3)
    chic2, chia2, thff2 = prm[:, 0], prm[:, 1], prm[:, 2]
    act = (chic2 > 0.) & (chia2 > 0.)
    if not act.any():
        return np.zeros(len(tau))
    chic2, chia2, thff2 = chic2[act], chia2[act], thff2[act]
    args = np.sqrt(chia2)[:, None] * (wstd * tau)[None, :]
    ym = np.sqrt(thff2 / chia2)
    lg = np.log(np.clip(ym, 1e1, 1e7))
    if MS_SNAP_YMAX:
        rows = np.round((lg - np.log(1e1)) / (np.log(1e7 / 1e1) / 12)).astype(int)
    else:
        # no snap: group by the step's OWN ymax (to 1e-3 in ln) and let gshape
        # interpolate between the two bracketing rows
        rows = np.round(lg, 3)

    # Z^2 : Z split with the electron ceiling (MS_ELEC_TMAX, default off).
    # The mass is recovered from the record as m = p sqrt(1-beta^2)/beta, the
    # same trick `_kokoulin_exponent` uses, because msmoliv carries no PDG.
    fN = fE = None
    if MS_ELEC_TMAX:
        effZ = _st[act, 0]
        pg, bt = _st[act, 3], np.clip(_st[act, 4], 1e-9, 1. - 1e-15)
        gam = 1. / np.sqrt(1. - bt * bt)
        mgev = pg / (bt * gam)
        bg = bt * gam
        rat = _ME_GEV / mgev
        tmx = 2. * _ME_GEV * bg * bg / (1. + 2. * gam * rat + rat * rat)
        # 1 - cos = Tmax m_e / p^2 ;  theta^2 = 2 (1 - cos)
        te = 2. * tmx * _ME_GEV / (pg * pg)
        yme = np.minimum(np.sqrt(te / chia2), ym)
        bt2 = bt * bt
        fN = effZ / (effZ + 1.)
        fE = 1. / (effZ + 1.)

    # The WentzelVI internal split (MS_WVI_SPLIT, default off).  ADDITIVE in the
    # exponent and independent of the ymax snap and of the electron ceiling, so
    # the three can be combined without interference.
    wviS = None
    if MS_WVI_SPLIT:
        xg = np.clip(_st[act, 2], 0.0, None)
        Rr = chic2 / chia2                             # per MODEL record
        nd = np.clip(MS_WVI_NPERX * xg, 1e-12, None)   # delta rays in the record
        fq = 2.0 / nd - 2.0 * (-np.expm1(-nd)) / (nd * nd)   # sum x_s^2 / x^2
        Rs = Rr * np.clip(fq, 0.0, 1.0)                # per effective SIM step
        Umin = _WVI_SSFACTOR * Rs * MS_WVI_LG
        # G4's minNCollisions escape: a SIM step with fewer than ten collisions
        # is sampled entirely as single scatters, so the split does not act.
        # The mean sim step of a record of thickness xg carries Rr/(1 + nd)
        # collisions.
        onmsc = (Rr / (1.0 + nd)) >= _WVI_MINCOLL
        # Only the NUCLEAR share Z/(Z+1) of the model's chi_c^2 is Gaussianized
        # here.  G4's Gaussian covers the nucleus up to theta_min and the atomic
        # electrons only up to their own kinematic ceiling, which sits three
        # decades BELOW theta_min (1-cos = 4.93e-10 against 7.5e-8): between the
        # two the electron term simply does not scatter, and that is the SEPARATE
        # `MS_ELEC_TMAX` candidate.  Keeping the two disjoint is what lets them be
        # combined; without the Z/(Z+1) this term double-counts by 1/(Z+1) = 12 %
        # (measured in `wvisplit.py g4` V6 before the factor was applied).
        fNw = _st[act, 0] / (_st[act, 0] + 1.0) * onmsc
        wviS = np.sum((Rr * fNw)[:, None] * wvi_split_exponent(args, Umin), axis=0)
        wviAsym = np.any((0.25 * args ** 2 * Umin[:, None] > _WVI_ARGMAX)
                         & onmsc[:, None], axis=0)

    S = np.zeros(len(tau))
    for r in np.unique(rows):
        m = rows == r
        ymr = (float(np.exp(np.log(1e1) + r * (np.log(1e7 / 1e1) / 12)))
               if MS_SNAP_YMAX else float(np.exp(r)))
        gsh = (gshape_elec(args[m].ravel(), ymr ** 2, kind="dipole") if MS_FINE_G
               else gshape(args[m].ravel(), ymax=ymr)
               ).reshape(int(m.sum()), len(tau))
        if not MS_ELEC_TMAX:
            S += np.sum((chic2[m] / chia2[m])[:, None] * gsh, axis=0)
            continue
        S += np.sum((chic2[m] * fN[m] / chia2[m])[:, None] * gsh, axis=0)
        # the electron piece: its own ceiling, interpolated (NOT snapped --
        # there is no production path to mirror for a term that does not
        # exist there), grouped so the shape is evaluated once per distinct
        # (ceiling, beta^2).  MS_ELEC_EDGE selects the kernel; 0.0 is the
        # published dipole and is what the archived numbers were taken with.
        lye = np.round(np.log(np.clip(yme[m], 1e1, 1e7)), 3)
        lb2 = np.round(bt2[m], 6) if MS_ELEC_EDGE >= 2.0 else np.zeros_like(lye)
        for key in np.unique(np.stack([lye, lb2], axis=1), axis=0):
            v, b2 = float(key[0]), float(key[1])
            mm = (lye == v) & (lb2 == b2)
            idx = np.where(m)[0][mm]
            ymv = float(np.exp(v))
            if MS_ELEC_EDGE == 0.0:
                ge = gshape(args[idx].ravel(), ymax=ymv)
            elif MS_ELEC_EDGE == 0.5:
                ge = gshape_elec(args[idx].ravel(), ymv ** 2, kind="dipole")
            elif MS_ELEC_EDGE == 1.0:
                ge = gshape_elec(args[idx].ravel(), ymv ** 2, kind="hard")
            elif MS_ELEC_EDGE in (2.0, 3.0):
                # the TRUE ceiling is theta_G4/2, EXACTLY (see cf_ms_exact), so
                # in the y^2 variable the ceiling is ym_e^2/4
                ge = gshape_elec(args[idx].ravel(), 0.25 * ymv ** 2,
                                 kind="kine",
                                 beta2=(b2 if MS_ELEC_EDGE == 2.0 else 0.0))
            else:
                raise ValueError(f"MS_ELEC_EDGE = {MS_ELEC_EDGE} is not a kernel")
            ge = ge.reshape(len(idx), len(tau))
            S += np.sum((chic2[idx] * fE[idx] / chia2[idx])[:, None] * ge,
                        axis=0)
    if wviS is not None:
        # The asymptotic branch of `wvi_split_exponent` is only legitimate where
        # the CF is already zero.  CHECKED, not assumed: everywhere it was used,
        # the model's own MS exponent must be below -700, at which exp()
        # underflows to 0 in double and no downstream number can depend on it.
        if np.any(wviAsym):
            _WVI_SMAX_SEEN[0] = max(_WVI_SMAX_SEEN[0], float(np.max(S[wviAsym])))
            assert np.max(S[wviAsym]) < _WVI_SMIN, (
                "ms_step_exponent: the WentzelVI-split dS was clamped to zero "
                f"at tau where the MS exponent is only {np.max(S[wviAsym]):.4g} "
                "-- the CF there is NOT negligible")
        S = S + wviS
    return S


def extract(args):
    """Per selected track: z_obs, sigma, eta, and the three exponent
    components on TG (Gaussian variance share Vg_frac; MS real exponent;
    ionization complex exponent), stored separately so per-family k
    scalings can be applied at closure time."""
    files = prodfiles.resolve(args.files, args.ntasks, logger=logger.info)
    logger.info(f"{len(files)} files")
    zs, sigs, etas, phis, chgs, vgf = [], [], [], [], [], []
    # ragged per-block store: class index, v_b/refCov00, and the count per track
    hcls, hamp, hcnt = [], [], []
    # Track-quality and kinematics, stored so the closure can be scanned
    # AGAINST THE SELECTION rather than reported at one arbitrary cut. The
    # historical trimming-dependence (MS +0.007 at chi2/hit < 10 against
    # -0.073 at < 3) was never separable from model error without these
    # (2026-08-08).
    chi2n, nvhit, ptrk, ptgen, chisq, ndofs = [], [], [], [], [], []
    Sms_l, Sio_re_l, Sio_im_l, Sdel_l = [], [], [], []
    Srad_re_l, Srad_im_l = [], []
    nsel = ndropcov = ndropgen = 0
    pt = None
    _warned_noclass = False
    _warned_noqscale = [False]
    _warned_norad = [False]
    want_hitclass = False
    # provenance: 1 iff the radiative block was actually built. Set on the
    # first file and asserted on every later one, so a shard cannot silently
    # mix a production that has the export with one that does not.
    rad_model = None
    _RADB = ("radstepidx", "radstepv", "radstepspecv", "radvgrid")
    for fn in files:
        try:
            f = uproot.open(fn)
            if "tree" not in f:
                continue
            if pt is None:
                pt = f["runtree"]["parmtype"].array(library="np")
            t = f["tree"]
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        # The per-hit CLASS variables are present only on productions made
        # after the export was ungated (commits e81ef11 / 138169a); older
        # caches still work and simply fall back to the Gaussian hit term.
        _need = ["refParms", "refCov", "genParms", "resinfcov",
                 "resinfvarv", "reseigidx", "msmoliidx", "msmoliv",
                 "ioniurbanidx", "ioniurbanv",
                 "normalizedChi2", "nValidHits", "trackPt", "genPt",
                 "chisqval", "ndof"]
        if args.ioni_norm == "raw":
            if "resinfv" not in t.keys():
                raise ValueError(
                    f"--ioni-norm raw needs the per-dof influence weights, and "
                    f"{fn} has no `resinfv` branch")
            _need = _need + ["resinfv"]
        # The applied ionization-block scale (2026-09-03). AUTO-DETECTED:
        # productions from before the export simply have no such branch and
        # get scale 1.0, which is what they ran with (CgfQoPMode=0) or, for a
        # pre-export mode-1 file, all `--ioni-norm var` could ever do.
        want_qscale = ("ioniqscalev" in t.keys() and "ioniqscaleidx" in t.keys())
        if want_qscale:
            _need = _need + ["ioniqscaleidx", "ioniqscalev"]
        elif args.ioni_norm == "var" and not _warned_noqscale[0]:
            logger.warning("input has no `ioniqscalev`; the `var` ionization "
                           "normalisation assumes CgfQoPMode=0 (scale 1.0)")
            _warned_noqscale[0] = True
        # The RADIATIVE step export (2026-09-03). Absent on every production
        # made before it; `--no-rad` switches it off explicitly (the `nomsrad`
        # sample, whose simulation has no brems/pair, so a model term for it
        # would be a model of physics that is not in the data).
        want_rad = (not args.no_rad) and all(b in t.keys() for b in _RADB)
        if want_rad:
            _need = _need + list(_RADB)
        elif not args.no_rad and not _warned_norad[0]:
            logger.warning("input has no `radstepv`; the radiative CF term "
                           "will be absent (rad_model=0)")
            _warned_norad[0] = True
        if rad_model is None:
            rad_model = int(want_rad)
        elif rad_model != int(want_rad):
            raise ValueError(
                "the radiative export is present in some input files and not "
                "in others; a cache mixing the two would carry the term for "
                "part of the sample only")
        _cls = ["reshitidx", "hitDetId", "hitUProj", "clusterSizeX",
                "clusterChargeBin"]
        want_hitclass = all(b in t.keys() for b in _cls)
        if not want_hitclass and not _warned_noclass:
            logger.warning("input has no per-hit class variables; only the "
                           "Gaussian hit term will be available")
            _warned_noclass = True
        # --max-tracks could not shortcut anything before: t.arrays() reads
        # EVERY branch for EVERY entry up front (msmoliv/ioniurbanv are large
        # jagged arrays), so a 180 MB file cost minutes before the track loop
        # even started. Bound the read too, with headroom for the gen/cov drops.
        _stop = None
        if args.max_tracks:
            _stop = min(t.num_entries, 3 * int(args.max_tracks) + 100)
        a = t.arrays(_need + (_cls if want_hitclass else []), library="np",
                     entry_stop=_stop)
        for ic in range(len(a["resinfcov"])):
            qg = a["genParms"][ic][0]
            if qg == 0.:
                ndropgen += 1
                continue
            c00 = float(a["refCov"][ic][0])
            cov = float(a["resinfcov"][ic])
            if not (c00 > 0.) or abs(cov / c00 - 1.) > COVTOL:
                ndropcov += 1
                continue
            sig = np.sqrt(c00)
            # THE CHARGE OF THE TRACK, needed BEFORE the ionization exponent
            # (it is the sign of the ionization q/p map, see below) as well as
            # after it, where it is stored for the charge-odd/even split.
            chg = np.sign(qg)
            gi = np.asarray(a["reseigidx"][ic])
            vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
            # raw per-dof influence weights w_b (5 per entry, zero-padded,
            # aligned with reseigidx). Only --ioni-norm raw needs them.
            wraw = (np.asarray(a["resinfv"][ic], dtype=np.float64).reshape(-1, 5)
                    if args.ioni_norm == "raw" else None)
            fam = pt[gi]
            vgauss = vb[(fam == 8) | (fam == 9)].sum()
            # Per-BLOCK hit weights and classes, not just their sum. The sum
            # is enough only while the hit term is Gaussian; for a measured
            # per-class density log phi_hit = sum_b log phi_c(a_b t) is not a
            # function of sum_b v_b. It is also the spread of the a_b that
            # produces most of the hit term's non-Gaussianity at track level
            # (NOTES_HITRES section 13), so it must be kept.
            hcls_i, hamp_i = [], []
            if want_hitclass:
                hidx = np.asarray(a["reshitidx"][ic])
                hdet = np.asarray(a["hitDetId"][ic]).astype(np.uint32)
                hsd = (hdet >> 25) & 0x7
                hN = np.asarray(a["clusterSizeX"][ic])
                hU = np.asarray(a["hitUProj"][ic])
                hQ = np.asarray(a["clusterChargeBin"][ic])
                for j in np.where((fam == 8) | (fam == 9))[0]:
                    hh = hidx[j]
                    if hh < 0 or hh >= len(hsd) or vb[j] <= 0.:
                        continue
                    cl = hitres_classes.class_of(hsd[hh], hN[hh], hU[hh],
                                                 hQ[hh], fam[j] == 9)
                    hcls_i.append(hitres_classes.class_index(cl))
                    hamp_i.append(vb[j] / c00)

            uvm = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            uim = np.asarray(a["msmoliidx"][ic])
            uvm = uvm.reshape(-1, len(uvm) // max(len(uim), 1)) if len(uim) else uvm.reshape(0, 8)
            uvi = np.asarray(a["ioniurbanv"][ic], dtype=np.float64)
            uii = np.asarray(a["ioniurbanidx"][ic])
            uvi = uvi.reshape(-1, len(uvi) // max(len(uii), 1)) if len(uii) else uvi.reshape(0, 11)
            # per-leg [sc, nstep] pairs; absent -> scale 1.0 everywhere
            if want_qscale:
                qsi = np.asarray(a["ioniqscaleidx"][ic])
                qsv = np.asarray(a["ioniqscalev"][ic],
                                 dtype=np.float64).reshape(-1, 2)
            else:
                qsi, qsv = None, None
            # RADIATIVE step records, pooled by the SAME global index as the
            # ionization block. One row per Geant4 step (so 1:1 with msmoliv,
            # NOT with ioniurbanv -- the index VALUE is the join, never the row
            # position).
            if want_rad:
                ridx = np.asarray(a["radstepidx"][ic])
                rrec = np.asarray(a["radstepv"][ic], dtype=np.float64).reshape(
                    -1, cf_brems_exact.RADV_STRIDE)
                rspc = np.asarray(a["radstepspecv"][ic],
                                  dtype=np.float64).reshape(
                    -1, 2 * cf_brems_exact.NRADV)
                rvg = np.asarray(a["radvgrid"][ic], dtype=np.float64)
            else:
                ridx = rrec = rspc = rvg = None

            Sms = np.zeros(len(TG))
            Sdel = np.zeros(len(TG))
            Sio = np.zeros(len(TG), dtype=np.complex128)
            Srad = np.zeros(len(TG), dtype=np.complex128)
            ok = True
            for famcode, (uidx, uv) in ((10, (uim, uvm)), (11, (uii, uvi))):
                sel = fam == famcode
                for g in np.unique(gi[sel]):
                    vpool = vb[sel & (gi == g)].sum()
                    if vpool <= 0.:
                        continue
                    steps = uv[uidx == g]
                    if not len(steps):
                        ok = False
                        break
                    if famcode == 10:
                        sq2 = steps[:, 5].sum()
                        if sq2 <= 0.:
                            continue
                        wstd = np.sqrt(vpool / sq2) / sig
                        _sms = ms_step_exponent(steps, wstd, TG)
                        Sms += _sms
                        if DELTA_ON:
                            # carve, do NOT add: Moliere's Z(Z+1) already
                            # carries this electron scattering continuously.
                            _cf = cf_delta_ray.carve_factor(
                                steps, DELTA_TCUT, DELTA_TMAXCAP)
                            Sdel += (cf_delta_ray.delta_step_exponent(
                                steps, wstd, TG, DELTA_TCUT,
                                tmax_cap=DELTA_TMAXCAP) - _cf * _sms)
                    else:
                        if args.ioni_norm == "raw":
                            # THE gsig2-FREE NORMALISATION. The step's noise is
                            # injected into the block's q/p dof alone (errI has
                            # only (0,0)) and the fit's response to a unit
                            # residual in that dof is w_b[0] = resinfv[b][0], so
                            # to first order in the intra-leg transport
                            #     delta(q/p_ref) = w_b[0] * (q * cs * dE)
                            # per step -- no gsig2 and no dV_b anywhere. `var`
                            # below instead recovers |w_b| by dividing the
                            # fit's v_b = w_b^T dV_b w_b by the RECORD's
                            # variance sum, which is an identity only when the
                            # two are the same variance (CgfQoPMode=0). Under
                            # CgfQoPMode>=1 the fit substitutes the block's
                            # Fisher weight (dV_b -> (qcgf/dQ2(0,0)) dV_b) while
                            # the record's gsig2 is the UNTRUNCATED second
                            # cumulant, and `var` comes out ~300x too small in
                            # the weight (~500x in the exponent).
                            #
                            # Pooled blocks (2 legs sharing one global param,
                            # ~1/3 of them) cannot have their steps split
                            # between the entries, so the pooled scalar is the
                            # plain rms of the entries' |w_b[0]| -- the exact
                            # step-noise-weighted rms with an equal-share prior
                            # (a v_b-share weighting instead moves the pooled
                            # blocks by +0.2% in the median, +0.9% at 84%,
                            # measured on 3137 pooled blocks).
                            #
                            # NEGLECTED: the intra-leg transport mixing. The
                            # exact per-step weight is w_b^T A_s e_0 with A_s
                            # the transport from the injection point to the END
                            # of the leg (dQ2 = sum_s q_s (A_s e0)(A_s e0)^T);
                            # A_s is NOT exported (StepTransport is a
                            # cleanprop-only log), so this uses A_s = I. It is
                            # not suppressed by small weights -- the block's
                            # other dofs carry median 2x the q/p weight -- only
                            # by the smallness of A_s(k,0), so it grows with the
                            # leg length. Measured against `var` on the legacy
                            # arm, where `var` IS the exact step-weighted rms:
                            # raw/var = 0.98 for legs of <=10 steps, 0.90 in the
                            # median over all blocks (16-84%: 0.55-0.99), 0.84
                            # for 31-45-step legs (2026-09-02). It is a
                            # one-sided DEFICIT, so k_ioni absorbs the median
                            # and only the spread is left.
                            #
                            # CHARGE-BLIND, exactly like `var`: the sign of
                            # w_b[0] is dropped. The charge of the ionization
                            # q/p map is applied separately, below, so that
                            # both normalisations get it identically.
                            w0 = np.abs(wraw[sel & (gi == g), 0])
                            w0 = w0[w0 > 0.]
                            if not len(w0):
                                continue
                            wstd = np.sqrt(float(np.mean(w0 ** 2))) / sig
                        else:
                            # sq2 must be the variance the FIT used, i.e. the
                            # record sum TIMES the scale the CGF substitution
                            # applied to the block (1.0 on mode-0 files and on
                            # every file written before the export). See
                            # ioni_sq2 for why it is per leg, not per block.
                            sq2 = ioni_sq2(
                                steps,
                                qsv[qsi == g] if qsv is not None else None)
                            if sq2 <= 0.:
                                continue
                            wstd = np.sqrt(vpool / sq2) / sig
                        # THE CHARGE FACTOR (2026-09-02/03, NOTES.md s3).
                        # `ioniurbanv` column 10 is `us.cs = E/p^3`
                        # (Geant4ePropagator.cc:2360, "q/p per MeV"), POSITIVE
                        # for every track; the physical map is
                        #     delta(q/p) = q * cs * delta(E),
                        # so the exponent's step weight carries the charge.
                        # The in-fit CGF block applies it (`const double qsign
                        # = (charge >= 0. ? 1. : -1.); s.gs = qsign * wtr * cs
                        # * 1e-3;`, Geant4ePropagator.cc ~1524, "The CHARGE
                        # factor is not cosmetic"); the offline CF did not, so
                        # every cached exponent was the mu+ one and the model
                        # carried the mu+ skew for BOTH charges while a
                        # charge-symmetric sample cancels it in the data.
                        #
                        # Applied to the WEIGHT rather than to Im S because the
                        # weight is where the physics is. It is the same thing
                        # to the last bit: every channel enters as
                        # a(e^{i gs E t} - 1 - i gs E t) or an integral of that
                        # form, so w -> -w is t -> -t and phi(-t) = phi(t)*,
                        #     Re S(-w) = Re S(+w),  Im S(-w) = -Im S(+w),
                        # verified at 0.000e+00 in Urban regimes 0/1/2 with and
                        # without the Kokoulin term.  Hence Sio_re, and with it
                        # the EVEN closure (k_ms, k_hit, <e^{-u z^2}>), is
                        # bit-identical to the pre-fix caches and only Sio_im
                        # flips on the mu- half of the sample.
                        Sio += ioni_step_exponent(steps, chg * wstd, TG)
                        # THE RADIATIVE TERM, same block, same weight, same
                        # charge. `rad_exponent` takes cs from the record and
                        # the energies in GeV, so the weight passed here is
                        # the q/p -> z scalar itself (no 1e-3: the ionization
                        # records store MeV, the radiative ones GeV).
                        if want_rad:
                            rm = ridx == g
                            nrs = int(rm.sum())
                            if nrs:
                                Srad += cf_brems_exact.rad_exponent(
                                    TG, rrec[rm], rspc[rm], rvg,
                                    weights=np.full(nrs, chg * wstd))
                if not ok:
                    break
            if not ok:
                continue

            zs.append((a["refParms"][ic][0] - qg) / sig)
            sigs.append(sig)
            etas.append(-np.log(np.tan((np.pi / 2. - a["refParms"][ic][1]) / 2.)))
            # phi is stored so a SIM-vs-refit field mismatch can be TESTED
            # rather than assumed: the OAE tracker parametrization is
            # phi-symmetric, so any residual phi structure in <z> is the
            # sharpest handle on a field-model difference (2026-08-07).
            phis.append(a["refParms"][ic][2])
            # charge from sign(gen q/p): the discriminator between a
            # curvature-like (charge-ODD) and a material/eloss-like
            # (charge-EVEN) bias. Useless on the mu- only gun sample,
            # essential on the both-charge one (2026-08-07). It is ALSO the
            # sign applied to the ionization step weight above, so `charge`
            # and `Sio_im` are guaranteed consistent by construction.
            chgs.append(chg)
            vgf.append(vgauss / c00)
            hcls.extend(hcls_i); hamp.extend(hamp_i); hcnt.append(len(hcls_i))
            chi2n.append(float(a["normalizedChi2"][ic]))
            nvhit.append(float(a["nValidHits"][ic]))
            ptrk.append(float(a["trackPt"][ic]))
            ptgen.append(float(a["genPt"][ic]))
            # chisqval/nValidHits is the HISTORICAL trim variable (the
            # --max-chi2-per-hit of fit_global_grads). Stored raw so the exact
            # historical cut can be reproduced rather than approximated by
            # normalizedChi2 = chisqval/ndof.
            chisq.append(float(a["chisqval"][ic]))
            ndofs.append(float(a["ndof"][ic]))
            Sms_l.append(Sms.astype(np.float32))
            Sdel_l.append(Sdel.astype(np.float32))
            Sio_re_l.append(Sio.real.astype(np.float32))
            Sio_im_l.append(Sio.imag.astype(np.float32))
            Srad_re_l.append(Srad.real.astype(np.float32))
            Srad_im_l.append(Srad.imag.astype(np.float32))
            nsel += 1
            # --max-tracks used to break only at a FILE boundary, so with one
            # file per shard (extract_parallel) it did nothing at all and every
            # shard ground through its whole file. The per-track cost here is
            # ~2.4 s, dominated by the exact-delta ionization exponent, so that
            # is 2-3 h per shard rather than the intended cap.
            if nsel >= args.max_tracks:
                break
        logger.info(f"{fn.split('/')[-2]}: cumulative {nsel} tracks "
                    f"(drop gen {ndropgen}, cov {ndropcov})")
        if nsel >= args.max_tracks:
            break
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez_compressed(args.cache, z=np.array(zs), sigma=np.array(sigs),
                        eta=np.array(etas), phi=np.array(phis),
                        charge=np.array(chgs), vgf=np.array(vgf),
                        normchi2=np.array(chi2n), nvalidhits=np.array(nvhit),
                        trackpt=np.array(ptrk), genpt=np.array(ptgen),
                        chisqval=np.array(chisq), ndof=np.array(ndofs),
                        Sms=np.array(Sms_l), Sdel=np.array(Sdel_l), Sio_re=np.array(Sio_re_l),
                        Sio_im=np.array(Sio_im_l),
                        Srad_re=np.array(Srad_re_l), Srad_im=np.array(Srad_im_l),
                        tgrid=TG,
                        hitcls=np.array(hcls, dtype=np.int16),
                        hitamp2=np.array(hamp, dtype=np.float32),
                        hitcnt=np.array(hcnt, dtype=np.int32),
                        hitclsnames=np.array(hitres_classes.CLASSES),
                        # PROVENANCE FLAG, not a switch. Its presence says the
                        # cached `Sio_im` already carries the charge of the
                        # ionization q/p map (see the `chg * wstd` above);
                        # caches written before 2026-09-03 do not have it and
                        # consumers (cf_skew_closure.load) apply the factor
                        # themselves. Never read as a value, only as a key.
                        ioni_charge_signed=np.array(1),
                        # PROVENANCE VALUE (unlike the flag above, this one IS
                        # read): 1 = the radiative block was built from the
                        # `radstepv` export; 0 = it was not, either because
                        # the production predates the export or because
                        # --no-rad was given (the `nomsrad` sample). The
                        # Srad_* arrays are zero in the second case, so a
                        # consumer that ignores this key still gets the right
                        # model -- but a closure that does not KNOW which arm
                        # it is on cannot be interpreted.
                        rad_model=np.array(int(rad_model or 0)))
    logger.info(f"wrote {args.cache} ({nsel} tracks, rad_model="
                f"{int(rad_model or 0)})")


def hit_exponent(d, args, bank):
    """log phi_hit(t) per track from the MEASURED per-class densities.

    The CVH solve is linear least squares given the reference trajectory, so
    delta(q/p) = sum_b w_b n_b exactly and

        log phi_hit(t) = sum_b log phi_c(b)( t sqrt(v_b) / sigma )

    is exact too -- no linearisation. phi_c is the CF of the RAW per-hit pull,
    so it carries the class's variance ratio AND its shape in one object; a
    class missing from the bank falls back to a Gaussian of its own v_b, which
    is what the old term did for every class.

    Chunked over tracks: the intermediate is (nblocks x len(TG)) and the full
    sample would be ~10^9 numbers.
    """
    names = list(d["hitclsnames"])
    cnt = d["hitcnt"].astype(np.int64)
    off = np.concatenate(([0], np.cumsum(cnt)))
    cls = d["hitcls"].astype(np.int64)
    amp = np.sqrt(np.maximum(d["hitamp2"].astype(np.float64), 0.))
    n = len(cnt)
    S = np.zeros((n, len(TG)), dtype=np.complex128)
    have = np.array([names[i] in bank for i in range(len(names))])
    CHUNK = 2000
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        b0, b1 = off[lo], off[hi]
        if b1 == b0:
            continue
        cb, ab = cls[b0:b1], amp[b0:b1]
        trk = np.repeat(np.arange(lo, hi) - lo, cnt[lo:hi])
        arg = ab[:, None] * TG[None, :]
        val = np.zeros_like(arg, dtype=np.complex128)
        for ci in np.unique(cb):
            m = cb == ci
            if have[ci]:
                val[m] = hitres_classes.logphi(bank[names[ci]], arg[m])
            else:
                # unknown class -> Gaussian of the same variance
                val[m] = -0.5 * arg[m] ** 2
        acc = np.zeros((hi - lo, len(TG)), dtype=np.complex128)
        np.add.at(acc, trk, val)
        S[lo:hi] = acc
    return S


def model_phi(d, args, bank=None):
    """Complex phi_z(t) per track on TG with the per-family k applied.
    Total variance is renormalized so z stays standardized to the
    *scaled* model: sigma_model^2/sigma^2 = e^khit*vgf + scaled tails --
    handled by evaluating phi of the scaled physics and letting closure
    compare against z_obs built with the unscaled sigma."""
    if args.hitmode == "class":
        if bank is None:
            raise SystemExit("--hitmode class needs the measured CF bank")
        Shit = np.exp(args.khit) * hit_exponent(d, args, bank)
    else:
        vg = np.exp(args.khit) * d["vgf"][:, None]
        Shit = -0.5 * vg * TG[None, :] ** 2
    S = (Shit
         + np.exp(args.kms) * d["Sms"]
         + np.exp(args.kioni) * (d["Sio_re"] + 1j * d["Sio_im"]))
    # The radiative block. LINEAR scale, default 1; `krad = 0` is the model
    # exactly as it was before the term existed, which is the control arm of
    # every closure. Absent from pre-2026-09-03 caches (and zero in a
    # `--no-rad` one), so `getattr`/key guard both matter.
    kr = getattr(args, "krad", 1.0)
    if kr:
        keys0 = d.files if hasattr(d, "files") else d
        if "Srad_re" in keys0:
            S = S + kr * (d["Srad_re"] + 1j * d["Srad_im"])
    kd = getattr(args, "kdel", None)
    if kd is not None:
        keys = d.files if hasattr(d, "files") else d
        if "Sdel" in keys:
            S = S + np.exp(kd) * d["Sdel"]
    return np.exp(S)


def weier(phi, u):
    """E[e^{-u z^2}] per track from phi on TG (even weight, Re phi)."""
    w = np.exp(-TG ** 2 / (4. * u)) / np.sqrt(np.pi * u)
    return np.clip(np.trapezoid(phi.real * w[None, :], TG, axis=1), 0., 1.)


def closure(args, outdir):
    d = np.load(args.cache)
    n = len(d["z"])
    logger.info(f"{n} tracks from {args.cache}")
    bank = None
    if args.hitmode == "class":
        if "hitcnt" not in d:
            raise SystemExit(f"{args.cache} predates the per-block hit store; "
                             f"re-run --extract on a production with the class "
                             f"variables")
        logger.info(f"measured hit CF bank from {args.bank_subdir}_{args.bank_tag}:")
        bank, _ = hitres_classes.build_cf_bank(
            args.bank_subdir, args.bank_tag, args.bank_nfiles, logger=logger)
    phi = model_phi(d, args, bank)
    z = d["z"]

    # per-probe closure, inclusive and in predicted-sigma quartiles
    qs = np.quantile(d["sigma"], np.linspace(0., 1., args.nsigma_bins + 1))
    lines = []
    for u in args.probes:
        Em = weier(phi, u)
        Ed = np.exp(-u * z ** 2)
        dm, sm = Ed.mean() - Em.mean(), Ed.std() / np.sqrt(n)
        lines.append(f"u={u:5.2f}  <data>-<model> = {dm:+.5f} +- {sm:.5f}  "
                     f"(<model> = {Em.mean():.4f})")
    logger.info("inclusive closure:\n" + "\n".join("  " + s for s in lines))

    # FIGURE 1: pull histogram vs averaged predicted lineshape, + data/model
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(10., 7.6))
    zg = np.linspace(-8., 8., 161)
    # p(z) = 1/pi Int_0^inf Re[phi(t) e^{-itz}] dt, averaged over tracks.
    # Evaluated on the bin edges AND the bin centres: the edges draw the
    # curve, the Simpson combination of the three gives the bin integral
    # the ratio panel needs (the cusp at z = 0 is not linear across a bin).
    zc = 0.5 * (zg[1:] + zg[:-1])
    ph = phi.mean(axis=0)
    pofz = lambda zz: np.trapezoid((ph * np.exp(-1j * TG * zz)).real, TG) / np.pi
    pz = np.array([pofz(zz) for zz in zg])
    pzc = np.array([pofz(zz) for zz in zc])
    zcl = np.clip(z, zg[0], zg[-1])
    cnt, _ = np.histogram(zcl, bins=zg)
    ax.hist(zcl, bins=zg, density=True, histtype="step", color="black",
            label="pulls (gen-matched MC)")
    ax.plot(zg, pz, color="crimson", label="CF-product prediction")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2.)
    ax.set_ylabel("density")
    ax.legend()
    pbin = ratiopanel.bin_average(pz[:-1], pzc, pz[1:])
    _, rat, _, n_out = ratiopanel.draw_ratio(
        rax, zg, cnt, pbin, cnt.sum(),
        xlabel=r"$z = \Delta(q/p)/\sigma_{\mathrm{pred}}$")
    if n_out:
        logger.info(f"pull ratio: {n_out} bin(s) outside the clamped y range")
    name_p = f"trackres_pulls{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name_p, fig)
    output_tools.write_logfile(outdir, name_p, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    plt.close(fig)
    logger.info(f"wrote {outdir}/{name_p}")

    # FIGURE 2: data-model difference of the bounded statistic vs u, by sigma bin
    fig, ax = plt.subplots(figsize=(10., 7.))
    uu = np.geomspace(0.02, 4., 25)
    for ib in range(args.nsigma_bins):
        m = (d["sigma"] >= qs[ib]) & (d["sigma"] <= qs[ib + 1])
        dd = np.array([np.exp(-u * z[m] ** 2).mean() - weier(phi[m], u).mean()
                       for u in uu])
        ss = np.array([np.exp(-u * z[m] ** 2).std() / np.sqrt(m.sum()) for u in uu])
        ax.errorbar(uu, dd, ss, marker="o", markersize=4, linestyle="-",
                    label=rf"$\sigma \in [{1e2*qs[ib]:.2f}, {1e2*qs[ib+1]:.2f}]\times 10^{{-2}}$")
    ax.axhline(0., color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"probe $u$")
    ax.set_ylabel(r"$\langle e^{-uz^2}\rangle_{\mathrm{data}} - \langle e^{-uz^2}\rangle_{\mathrm{model}}$")
    ax.legend(fontsize="x-small")
    name = f"trackres_closure{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(outdir, f"{name}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    plt.close(fig)
    logger.info(f"wrote {outdir}/{name}")


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_trackres/")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    if args.extract:
        extract(args)
    if args.closure:
        closure(args, outdir)


if __name__ == "__main__":
    main()
