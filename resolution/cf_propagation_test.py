#!/usr/bin/env python3
"""Clean propagation test: predict the PDF of the propagated 5D state.

Josh Bendavid's suggestion (2026-08-04): take ONE particle with a FIXED
initial state, run the full Geant4 simulation many times, and try to predict
the distribution of the final helix state on a sensor surface -- with no hits,
no track fit, no FSR, no selection and no background in the way. It is the
ground-truth unit test the whole CF resolution programme has been missing:
Geant4 *is* the right answer by definition, so any disagreement is
unambiguously in the transport-fluctuation model.

Two inputs, both produced with identical geometry/field/conditions:
  * sim   -- Analysis/HitAnalyzer/test/runCleanPropSim.py    (N x the same muon)
  * model -- Analysis/HitAnalyzer/test/runCleanPropModel.py  (1 x deterministic)

What is compared, per crossed sensor plane k and per linear functional a of
the local 5D state (q/p, dx/dz, dy/dz, x, y):

    data:   z_e = a.(x_sim_{k,e} - x_ref_k) / sigma_pred_k
    model:  phi_k(t) = prod_s phi_{n_s}(w_{s,k} t / sigma_pred_k)

with the per-step weights w_{s,k} = (A_{s->k}^T a)_component taken from the
EXACT per-step cumulative transport Jacobians exported by the propagator,

    A_{s->k} = (prod_{m=j..k} F_m) . (Jacc^{(j)}_s)^{-1},   s in leg j

rather than one RMS-matched scalar per pooled block. In the track fit that
pooling is unavoidable; here it would be a confound, since the tails are the
whole point.

Comparison is done directly in transform space -- the empirical characteristic
function of the simulated sample is just mean(exp(i t z)), free to compute --
so the numerical inversion never enters the test and cannot be blamed for a
disagreement. Inversion is used only to draw the lineshape.

Note which physics each functional isolates: multiple scattering barely
changes |p|, so the q/p residual is essentially PURE ionization straggling --
the first real test of the muon Urban tail and its skew, which is invisible at
block level in the fit (hat values ~1e-6). Position/angle residuals are
Moliere-dominated.

Usage:
  # 1. target surfaces from the simulation (modal crossed-module sequence)
  python cf_propagation_test.py --targets --sim simstates.root --out targets.txt
  # 2. after running the model job:
  python cf_propagation_test.py --compare --sim simstates.root --model model.root
"""

import argparse
import datetime
import os
from collections import Counter

import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep

from wums import logging  # noqa: E402

from cf_track_resolution import ioni_step_exponent, ms_step_exponent
import cf_brems_exact

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

# Conjugate grid for the residual standardized by the propagator's own
# (alpha-truncated) sigma. That variable has a narrow core, |z| ~ 1, sitting
# under a delta-ray tail reaching |z| ~ 1e3, so its CF carries structure over
# three decades of t: the tail sets the initial decay near t ~ 1e-2 and the
# core sets the far behaviour near t ~ 10. A uniform grid resolves one or the
# other but not both, so the grid is logarithmic (the transforms below all
# integrate with trapezoid, which does not care).
TAU = np.concatenate([[0.0], np.geomspace(1e-3, 40.0, 1600)])

# linear functionals of the local 5D state (q/p, dx/dz, dy/dz, x, y)
# !! BASIS MISMATCH -- KNOWN, NOT YET FIXED (found 2026-08-06) !!
# The residuals below are LOCAL (DetUnit frame: q/p, dx/dz, dy/dz, x, y) but the
# exported Q/F/dQMS/dQI are CURVILINEAR (q/p, lambda, phi, yT, zT) and load_model
# reads them raw. Applying these a-vectors to a curvilinear covariance omits the
# curv->local Jacobian H.
#   qop  : unaffected (same variable in both frames).
#   locx : unaffected WHEN the shallow incidence lies in the local y-z plane
#          (true for the barrel geometries scanned so far) -- not general.
#   dxdz : WRONG by sec(theta_inc), because dx/dz = u_x/u_z carries 1/u_z.
#          Measured: rob68 * cos(theta_inc) collapses to 0.90-0.94 at eta =
#          0.30/1.00/1.60 while sec spans 1.05-2.61. Negligible centrally
#          (<=1.08), a factor 2.6 at eta = 1.6.
# The CVH FIT is not affected -- it applies curv2localJacobianAltelossD at every
# measurement surface. This is a defect of THIS TEST only.
# Fix: export H per leg from G4ePropagationExport.cc and use (H^T a) here.
KMS_SCALE = 1.0   # set from --kms in main()
MS_NSUB = 4       # sub-step quadrature of the MS kick; 1 = legacy point-like

FUNCTIONALS = {
    "qop": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    "dxdz": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    "locx": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
}
SIM_BRANCH = {"qop": "qop", "dxdz": "dxdz", "locx": "locx"}
REF_BRANCH = {"qop": "refqop", "dxdz": "refdxdz", "locx": "reflocx"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", required=True, help="runCleanPropSim.py output (or a glob)")
    p.add_argument("--model", default="", help="runCleanPropModel.py output")
    p.add_argument("--targets", action="store_true", help="write the target-surface list and exit")
    p.add_argument("--compare", action="store_true", help="run the data/model comparison")
    p.add_argument("--out", default="targets.txt", help="target list output (with --targets)")
    p.add_argument("--kms", type=float, default=0.0,
                   help="log-scale on the MS log-CF exponent, exactly as cf_track_resolution --kms. Scanning this here measures the SAME quantity as the track-level k_ms but with NO fit, NO hits and NO block pooling: exact per-step transport Jacobians. If the two agree, the discrepancy is in the Moliere FORMULA; if only the track-level one is non-zero, it is in the fit machinery (pooling / wstd / hit-MS split).")
    p.add_argument("--functionals", nargs="+", default=["qop", "locx"],
                   choices=sorted(FUNCTIONALS), help="which linear functionals to test")
    p.add_argument("--probes", nargs="+", type=float,
                   default=[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
                   help="Weierstrass probes u for the <exp(-u z^2)> comparison; "
                        "small u weights the delta-ray tail, u ~ 1 the core")
    p.add_argument("--layers", nargs="+", type=int, default=[],
                   help="restrict to these layer indices (default: all)")
    p.add_argument("--outpath", default="", help="plot output dir (default ~/public_html/cvh/<date>_cleanprop)")
    p.add_argument("--postfix", default="", help="suffix for output file names")
    p.add_argument("--label", default="",
                   help="campaign point label (e.g. 'pt40_eta0.30_pdg13'), stored "
                        "in the npz dump so a (pt, eta, species) scan can be "
                        "aggregated across runs")
    return p.parse_args()


# --------------------------------------------------------------------------
# simulation side
# --------------------------------------------------------------------------

def load_sim(path):
    """Per-event true local states, restricted to the modal module sequence.

    Events that miss a module (a large scatter near a module edge, or a decay
    in flight) have a different detid sequence; they are counted and dropped,
    and the drop fraction is reported -- it is the one selection effect this
    test cannot avoid, so it must be small and it must be quoted.
    """
    files = sorted(__import__("glob").glob(path)) if any(c in path for c in "*?[") else [path]
    keys = ["detid", "qop", "dxdz", "dydz", "locx", "locy", "locz", "pabs", "eloss", "globr", "nhit"]
    parts = {k: [] for k in keys}
    for fn in files:
        t = uproot.open(fn)["simstates/simstates"]
        a = t.arrays(keys, library="np")
        for k in keys:
            parts[k].append(a[k])
    arr = {k: np.concatenate(parts[k]) for k in keys}

    # Select on the joint (module, entry-face) pattern, not the module list
    # alone: a track that clips a module's SIDE gets a PSimHit on the right
    # detid but at an intermediate local z, i.e. it is not on the plane the
    # reference was propagated to. Both failure modes are geometric
    # (edge-grazing rays), both are quantified here, and the ray is chosen by
    # cleanprop/scan_ray.sh so that the drop fraction is negligible -- an
    # uncontrolled drop would be exactly the selection effect this test exists
    # to remove.
    seqs = [tuple(zip(d.tolist(), np.round(z, 4).tolist()))
            for d, z in zip(arr["detid"], arr["locz"])]
    counts = Counter(seqs)
    modal, nmodal = counts.most_common(1)[0]
    ntot = len(seqs)
    keep = np.array([s == modal for s in seqs])
    logger.info(f"sim: {ntot} events, {len(counts)} distinct (module, entry-face) patterns, "
                f"modal covers {nmodal} ({100.*nmodal/ntot:.4f}%), "
                f"dropped {ntot - nmodal}")

    out = {"detid": np.array([m[0] for m in modal], dtype=np.uint32),
           "locz_modal": np.array([m[1] for m in modal], dtype=np.float64),
           "nkept": int(keep.sum()), "ntot": ntot}
    for k in ("qop", "dxdz", "dydz", "locx", "locy", "locz", "pabs", "eloss", "globr"):
        out[k] = np.stack([np.asarray(v, dtype=np.float64) for v in arr[k][keep]])  # (nev, nlayer)
    return out


# --------------------------------------------------------------------------
# model side
# --------------------------------------------------------------------------

def _reshape_ms(v):
    """msmoliv is 8 doubles/step in legacy files, 10 since 2026-08-08."""
    v = np.asarray(v, dtype=np.float64)
    for w in (10, 8):
        if v.size % w == 0:
            return v.reshape(-1, w)
    raise ValueError(f'msmoliv size {v.size} matches neither stride')


def load_model(path):
    # TFileService puts the tree in a directory named after the module label
    f = uproot.open(path)
    tkey = next((k for k in f.keys() if k.split(";")[0].endswith("/legs")), None)
    assert tkey is not None, f"no .../legs tree in {path}; keys: {f.keys()}"
    t = f[tkey]
    keys = ["ileg", "detid", "ok", "zoff", "refqop", "refdxdz", "refdydz", "reflocx",
            "reflocy", "reflocz", "refglobr", "refp", "refpt", "F", "Q", "dQMS", "dQI",
            "msmoliv", "ioniurbanv", "stepjacc", "stepnms", "stepnioni"]
    have_rad = all(b in t.keys() for b in ("radv", "radspecv", "radvgrid"))
    if have_rad:
        keys += ["radv", "radspecv", "radvgrid"]
    a = t.arrays(keys, library="np")
    nlegs = len(a["ileg"])
    legs = []
    for k in range(nlegs):
        legs.append(dict(
            detid=int(a["detid"][k]), ok=bool(a["ok"][k]),
            refqop=float(a["refqop"][k]), refdxdz=float(a["refdxdz"][k]),
            refdydz=float(a["refdydz"][k]), reflocx=float(a["reflocx"][k]),
            reflocy=float(a["reflocy"][k]), refp=float(a["refp"][k]),
            refpt=float(a["refpt"][k]),
            F=np.asarray(a["F"][k], dtype=np.float64).reshape(5, 5),
            Q=np.asarray(a["Q"][k], dtype=np.float64).reshape(5, 5),
            dQMS=np.asarray(a["dQMS"][k], dtype=np.float64).reshape(5, 5),
            dQI=np.asarray(a["dQI"][k], dtype=np.float64).reshape(5, 5),
            # stride auto-detect: 8 (legacy) or 10 (with per-element sums)
            ms=_reshape_ms(a["msmoliv"][k]),
            ioni=np.asarray(a["ioniurbanv"][k], dtype=np.float64).reshape(-1, 11),
            jacc=np.asarray(a["stepjacc"][k], dtype=np.float64).reshape(-1, 5, 5),
            nms=np.asarray(a["stepnms"][k], dtype=np.int64),
            nioni=np.asarray(a["stepnioni"][k], dtype=np.int64),
            rad=(np.asarray(a["radv"][k], dtype=np.float64).reshape(
                -1, cf_brems_exact.RADV_STRIDE) if have_rad else None),
            radspec=(np.asarray(a["radspecv"][k], dtype=np.float64).reshape(
                -1, 2 * cf_brems_exact.NRADV) if have_rad else None),
            radvgrid=(np.asarray(a["radvgrid"][k], dtype=np.float64)
                      if have_rad else None),
        ))
    logger.info(f"model: {nlegs} legs, {sum(len(l['ms']) for l in legs)} MS steps, "
                f"{sum(len(l['ioni']) for l in legs)} ionization steps")
    return legs


def step_transports(legs, k):
    """Exact transport A_{s->k} of every step's noise up to the end of leg k.

    Returns (A_ms, A_ioni): lists over legs j<=k of (nstep_j, 5, 5) arrays
    aligned with that leg's msmoliv / ioniurbanv records.
    """
    # suffix products P_j = F_k F_{k-1} ... F_{j+1}
    suffix = [np.eye(5)]
    for m in range(k, 0, -1):
        suffix.append(suffix[-1] @ legs[m]["F"])
    suffix = suffix[::-1]  # suffix[j] = F_k...F_{j+1}, for j = 0..k

    A_ms, A_ioni, A_ms_start = [], [], []
    for j in range(k + 1):
        leg = legs[j]
        Pj = suffix[j] @ leg["F"]  # = F_k ... F_j
        jacc = leg["jacc"]
        if len(jacc) == 0:
            A_ms.append(np.zeros((0, 5, 5)))
            A_ioni.append(np.zeros((0, 5, 5)))
            A_ms_start.append(np.zeros((0, 5, 5)))
            continue
        # A for every step of this leg
        Astep = np.empty_like(jacc)
        for s in range(len(jacc)):
            Astep[s] = Pj @ np.linalg.inv(jacc[s])
        # map physics-log entries back to their step
        idx_ms = np.searchsorted(leg["nms"], np.arange(len(leg["ms"])) + 1, side="left")
        idx_io = np.searchsorted(leg["nioni"], np.arange(len(leg["ioni"])) + 1, side="left")
        idx_ms = np.clip(idx_ms, 0, len(jacc) - 1)
        idx_io = np.clip(idx_io, 0, len(jacc) - 1)
        A_ms.append(Astep[idx_ms])
        A_ioni.append(Astep[idx_io])
        # START-of-step transport: jacc[s-1] (identity before the first step).
        # A_{s->k} built from jacc[s] puts the kick at the END of the step, so
        # a step contributes only DD*D^2; the exact continuous result is
        # DD*(1/L) int_0^L (D+u)^2 du = DD*(D^2 + D*L + L^2/3). The missing
        # D*L + L^2/3 dominates for steps NEAR the target plane, where D->0.
        Astart = np.empty_like(jacc)
        for s in range(len(jacc)):
            Astart[s] = Pj @ np.linalg.inv(jacc[s - 1]) if s > 0 else Pj
        A_ms_start.append(Astart[idx_ms])
    return A_ms, A_ioni, A_ms_start


def model_variance(legs, k, avec):
    """Gaussian-limit variance of a.(x_k - x_ref) from the exported leg Q.

    This is the internal consistency check on the whole weight bookkeeping:
    it uses only the propagator's own per-leg noise matrices and leg
    Jacobians, i.e. exactly the quantity the CVH fit would use, and must
    agree with the curvature of the CF built from per-step weights.
    """
    suffix = [np.eye(5)]
    for m in range(k, 0, -1):
        suffix.append(suffix[-1] @ legs[m]["F"])
    suffix = suffix[::-1]
    var, varms, varioni = 0.0, 0.0, 0.0
    for j in range(k + 1):
        w = suffix[j].T @ avec
        var += float(w @ legs[j]["Q"] @ w)
        varms += float(w @ legs[j]["dQMS"] @ w)
        varioni += float(w @ legs[j]["dQI"] @ w)
    return var, varms, varioni


def model_phi(legs, k, avec, sigma, tau):
    """Model CF of the standardized residual z = a.(x - x_ref)/sigma."""
    A_ms, A_ioni, A_ms_start = step_transports(legs, k)
    S = np.zeros(len(tau), dtype=np.complex128)
    for j in range(k + 1):
        leg = legs[j]
        # --- ionization: the noise sits in the qop component only, so the
        # per-step weight is a scalar. Fold it into the step's own qop-per-MeV
        # column (cs) so the shared vectorized exponent can be called once for
        # the whole leg with unit weight -- exactly equivalent, much faster.
        if len(leg["ioni"]):
            # The exported cs = E/p^3 maps a step's energy loss to d(q/p) for a
            # POSITIVE charge: qop = q/p, so d(qop) = -q/p^2 dp = +q (E/p^3) dE.
            # The charge factor therefore has to be put back by hand, and it is
            # not cosmetic: it flips the sign of the ionization skew, i.e. of
            # the mean-vs-mode displacement that dominates this residual.
            q = np.sign(leg["refqop"]) or 1.0
            w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
            steps = leg["ioni"].copy()
            steps[:, 10] *= w
            S += ioni_step_exponent(steps, 1.0, tau)
        # --- radiative (brems + pair): same compound-Poisson structure and the
        # same qop-only weighting as ionization. Centred (the "-1-ix" in the
        # exponent), which is exactly right because the propagator's mean-loss
        # table is built with ionOnly=false and has ALREADY subtracted the
        # radiative mean -- so only the fluctuation is being added here, with
        # no double counting of the mean.
        if leg.get("rad") is not None and len(leg["rad"]):
            q = np.sign(leg["refqop"]) or 1.0
            w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
            # rad records are aligned with the MOLIERE log, ioni weights with
            # the ionization log; both are per-step, so reuse the ioni
            # transport when the counts match and fall back to the leg mean.
            wr = w if len(w) == len(leg["rad"]) else np.full(len(leg["rad"]), np.mean(w))
            S += cf_brems_exact.rad_exponent(tau, leg["rad"], leg["radspec"],
                                             leg["radvgrid"], weights=wr)
        # --- multiple scattering: an isotropic 2D kick in (lambda, phi).
        # Measuring the azimuthal angle as phi*cos(lambda) makes the two
        # projected angles iid with variance thp2, so by azimuthal isotropy
        # the CF depends only on the quadrature sum of the two weights.
        if len(leg["ms"]):
            wv = np.einsum("i,sij->sj", avec, A_ms[j])
            wv0 = np.einsum("i,sij->sj", avec, A_ms_start[j])
            coslam = leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0
            def _weff(v):
                return np.sqrt(v[:, 1] ** 2 + (v[:, 2] / max(coslam, 1e-3)) ** 2) / sigma
            weff, weff0 = _weff(wv), _weff(wv0)
            # SUB-STEP QUADRATURE (2026-08-08). Scattering happens continuously
            # THROUGH a step, not point-like at its end. Applying the kick at
            # the end gives a step only DD*D^2 of position variance; the exact
            # continuous result is DD*(1/L) int_0^L (D+u)^2 du =
            # DD*(D^2 + D*L + L^2/3). The omitted D*L + L^2/3 is 8.8% of the
            # position variance overall (verified against the propagator's own
            # dQMS, which reproduces exactly when the per-step thick-scatterer
            # S1/S3 terms are included) and DOMINATES for steps near the target
            # plane, where D -> 0 kills the angle term but not DD*L^2/3.
            # Splitting each step into NSUB equal sub-kicks, with the transport
            # weight interpolated linearly between the step start and end,
            # converges to the exact result; NSUB=4 leaves <1% of the term.
            for s in range(len(leg["ms"])):
                if weff[s] <= 0.0 and weff0[s] <= 0.0:
                    continue
                if MS_NSUB <= 1:
                    S += KMS_SCALE * ms_step_exponent(leg["ms"][s:s + 1], weff[s], tau)
                    continue
                rec = leg["ms"][s:s + 1].copy()
                rec[:, 2] /= MS_NSUB          # xg -> chi_c^2 scales with material
                for i in range(MS_NSUB):
                    f = (i + 0.5) / MS_NSUB   # 0 = step start, 1 = step end
                    w = weff0[s] + f * (weff[s] - weff0[s])
                    if w > 0.0:
                        S += KMS_SCALE * ms_step_exponent(rec, w, tau)
    return np.exp(S)


def weier_scalar(phi, u, tau):
    w = np.exp(-tau ** 2 / (4.0 * u)) / np.sqrt(np.pi * u)
    return float(np.clip(np.trapezoid(phi.real * w, tau), 0.0, 1.0))


def ecf(z, tau):
    """Empirical characteristic function of the simulated sample."""
    return np.exp(1j * np.outer(tau, z)).mean(axis=1)


# --------------------------------------------------------------------------

def write_targets(args):
    sim = load_sim(args.sim)
    # Build the whole list BEFORE opening the file. Writing incrementally meant
    # a mid-loop failure left a short but non-empty targets file, which the
    # campaign driver's `[[ -s ]]` check then accepted -- the model propagated
    # to the one surface that had been written and the comparison died with
    # "model has 1 legs but sim crossed 19 modules". Fail before touching disk.
    lines = ["# detid  localZ[cm]   (sensor entry face in the DetUnit frame)"]
    for i, did in enumerate(sim["detid"]):
        z = float(np.median(sim["locz"][:, i]))
        spread = float(np.std(sim["locz"][:, i]))
        # The modal pattern groups on round(locz, 4), so members of one group
        # may legitimately differ by up to 1e-4 cm; asserting 1e-6 here was
        # tighter than the grouping that produced the group and tripped on
        # perfectly good samples (pi- at pT=3 gave rms 1.7e-6). A REAL failure
        # -- the selection not isolating one entry face -- shows up at the
        # module half-thickness, ~1e-2 cm, so 1e-4 cm (1 um) is still strong.
        assert spread < 1e-4, (
            f"entry local z is not constant for detid {did} even after the "
            f"(module, entry-face) selection: rms {spread}")
        lines.append(f"{int(did)} {z:.6f}")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info(f"wrote {len(sim['detid'])} target surfaces to {args.out}")
    for i, did in enumerate(sim["detid"]):
        logger.info(f"  layer {i:2d}  detid {int(did):10d}  r = {np.median(sim['globr'][:, i]):7.3f} cm  "
                    f"localZ = {np.median(sim['locz'][:, i]):+.4f} cm")


def compare(args, outdir):
    sim = load_sim(args.sim)
    legs = load_model(args.model)
    assert len(legs) == len(sim["detid"]), \
        f"model has {len(legs)} legs but sim crossed {len(sim['detid'])} modules"
    for k, leg in enumerate(legs):
        assert leg["detid"] == sim["detid"][k], f"leg {k} detid mismatch"
        assert leg["ok"], f"leg {k} propagation failed"

    layers = args.layers if args.layers else list(range(len(legs)))
    tau = TAU
    rows = []
    results = {}

    for name in args.functionals:
        avec = FUNCTIONALS[name]
        for k in layers:
            d = sim[SIM_BRANCH[name]][:, k] - legs[k][REF_BRANCH[name]]
            var, varms, varioni = model_variance(legs, k, avec)
            if var <= 0:
                logger.warning(f"{name} layer {k}: non-positive model variance, skipping")
                continue
            # sigma is the propagator's OWN (alpha-truncated) width -- the
            # scale the Gaussian track fit works with. It is not the width of
            # the true distribution and is not meant to be: quoting the
            # residual in these units is what makes the size of the
            # non-Gaussian part legible.
            sigma = np.sqrt(var)
            z = d / sigma
            phi = model_phi(legs, k, avec, sigma, tau)
            q16, q50, q84 = np.percentile(z, [15.865, 50., 84.135])
            probes = {}
            for u in args.probes:
                fm = weier_scalar(phi, u, tau)
                fd = float(np.mean(np.exp(-u * z ** 2)))
                probes[u] = (fd, fm)
            rows.append(dict(func=name, layer=k, r=float(np.median(sim["globr"][:, k])),
                             sigma=sigma, sigma_ms=np.sqrt(max(varms, 0.)),
                             sigma_ioni=np.sqrt(max(varioni, 0.)),
                             mean=float(z.mean()), median=float(q50),
                             rob=float(0.5 * (q84 - q16)),
                             std=float(z.std()), probes=probes))
            results[(name, k)] = dict(z=z, phi=phi, sigma=sigma)

    report(rows, sim, outdir, args)
    dump_rows(rows, sim, outdir, args)
    make_plots(results, rows, sim, tau, outdir, args)
    return rows


def dump_rows(rows, sim, outdir, args):
    """Flat npz of the per-(functional, layer) scalars, for scan aggregation.

    The text summary is for reading; this is for scan_summary.py, which has to
    put many (pt, eta, species) points on one axis. Probe order is fixed by
    args.probes and stored alongside so the caller never has to guess it.
    """
    if not rows:
        return
    probes = np.asarray(args.probes, dtype=float)
    out = {
        "label": np.array(args.label),
        "probes": probes,
        "func": np.array([r["func"] for r in rows]),
        "layer": np.array([r["layer"] for r in rows], dtype=int),
        "nkept": np.array(sim["nkept"]), "ntot": np.array(sim["ntot"]),
    }
    for key in ("r", "sigma", "sigma_ms", "sigma_ioni",
                "mean", "median", "rob", "std"):
        out[key] = np.array([r[key] for r in rows], dtype=float)
    # (nrow, nprobe) data and model bounded averages
    out["fdata"] = np.array([[r["probes"][u][0] for u in args.probes] for r in rows])
    out["fmodel"] = np.array([[r["probes"][u][1] for u in args.probes] for r in rows])
    path = os.path.join(outdir, f"cleanprop_rows{args.postfix}.npz")
    np.savez(path, **out)
    logger.info(f"wrote {path}")


def report(rows, sim, outdir, args):
    lines = []
    lines.append(f"clean propagation test -- {sim['nkept']}/{sim['ntot']} events on the modal module sequence "
                 f"({100.*sim['nkept']/sim['ntot']:.3f}%)")
    lines.append("")
    lines.append("residual in units of the propagator's own truncated sigma "
                 "(mean/median/rob68/std of z)")
    hdr = (f"{'func':>5} {'lay':>3} {'r[cm]':>8} {'sigma':>11} {'ms/tot':>7} {'ioni/tot':>8} "
           f"{'mean':>9} {'med':>8} {'rob68':>8} {'std':>9}")
    lines.append(hdr)
    for r in rows:
        frac_ms = r["sigma_ms"] ** 2 / r["sigma"] ** 2
        frac_io = r["sigma_ioni"] ** 2 / r["sigma"] ** 2
        lines.append(f"{r['func']:>5} {r['layer']:>3} {r['r']:>8.2f} {r['sigma']:>11.4e} "
                     f"{frac_ms:>7.3f} {frac_io:>8.3f} "
                     f"{r['mean']:>9.3f} {r['median']:>8.3f} {r['rob']:>8.3f} {r['std']:>9.2f}")
    lines.append("")
    lines.append("<exp(-u z^2)>  data - model")
    lines.append(f"{'func':>5} {'lay':>3} " + " ".join(f"{'u=%.2g' % u:>12}" for u in args.probes))
    for r in rows:
        cells = []
        for u in args.probes:
            fd, fm = r["probes"][u]
            cells.append(f"{fd - fm:>+12.5f}")
        lines.append(f"{r['func']:>5} {r['layer']:>3} " + " ".join(cells))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(outdir, f"cleanprop_summary{args.postfix}.txt"), "w") as fh:
        fh.write(txt + "\n")


def make_plots(results, rows, sim, tau, outdir, args):
    """Three views per layer, in increasing distance from the raw comparison:
    the characteristic function itself (real and imaginary -- the imaginary
    part IS the skew, i.e. the mean-vs-mode physics), the bounded average
    <exp(-u z^2)> across probes, and the reconstructed lineshape."""
    funcs = sorted({r["func"] for r in rows})
    uscan = np.geomspace(1e-6, 10., 40)
    for name in funcs:
        ks = [r["layer"] for r in rows if r["func"] == name]
        if not ks:
            continue
        kshow = sorted({ks[0], ks[len(ks) // 2], ks[-1]})
        fig, axes = plt.subplots(3, len(kshow), figsize=(6.0 * len(kshow), 13.5),
                                 squeeze=False)
        for col, k in enumerate(kshow):
            res = results[(name, k)]
            z, phi = res["z"], res["phi"]
            e = ecf(z, tau)
            # --- row 0: the characteristic function, data vs model
            ax = axes[0][col]
            ax.plot(tau[1:], e.real[1:], "k-", lw=2.5, alpha=.55, label="sim  Re")
            ax.plot(tau[1:], phi.real[1:], "r--", lw=1.6, label="model  Re")
            ax.plot(tau[1:], e.imag[1:], "b-", lw=2.5, alpha=.55, label="sim  Im")
            ax.plot(tau[1:], phi.imag[1:], "g--", lw=1.6, label="model  Im")
            ax.set_xscale("log")
            ax.set_xlabel("t")
            ax.set_ylabel(r"$\varphi(t)$")
            ax.set_title(f"{name}, layer {k}  (r = {np.median(sim['globr'][:, k]):.1f} cm)")
            ax.axhline(0., color="0.7", lw=.8)
            ax.legend(fontsize=9)
            # --- row 1: bounded average across probes
            ax = axes[1][col]
            fd = np.array([np.mean(np.exp(-u * z ** 2)) for u in uscan])
            fm = np.array([weier_scalar(phi, u, tau) for u in uscan])
            ax.plot(uscan, fd, "k-", lw=2.5, alpha=.55, label="sim")
            ax.plot(uscan, fm, "r--", lw=1.6, label="model")
            ax.plot(uscan, fd - fm, "b-", lw=1.2, label="sim - model")
            ax.set_xscale("log")
            ax.set_xlabel("u")
            ax.set_ylabel(r"$\langle e^{-uz^2}\rangle$")
            ax.axhline(0., color="0.7", lw=.8)
            ax.legend(fontsize=9)
            # --- row 2: lineshape (inversion used only to draw, never to test)
            ax = axes[2][col]
            lim = float(np.percentile(np.abs(z), 99.5))
            zg = np.linspace(-lim, lim, 401)
            pz = np.array([np.trapezoid((phi * np.exp(-1j * tau * zz)).real, tau) / np.pi
                           for zz in zg])
            # Where the true density is essentially zero (energy loss is
            # one-sided, so there is no far positive tail) the inversion
            # oscillates about zero at the quadrature noise level; on a log
            # axis that reads as spurious structure. Mask it -- the test
            # itself never inverts, this panel is for display only.
            pz = np.where(pz > 1e-7 * np.nanmax(pz), pz, np.nan)
            ax.hist(z, bins=np.linspace(-lim, lim, 201), density=True, histtype="step",
                    color="k", lw=1.5, label="sim")
            ax.plot(zg, pz, "r--", lw=1.6, label="model")
            ax.set_yscale("log")
            ax.set_xlabel("z  (residual / truncated $\\sigma$)")
            ax.set_ylabel("density")
            ax.legend(fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"cleanprop_{name}{args.postfix}.{ext}"),
                        bbox_inches="tight")
        plt.close(fig)
    logger.info(f"plots written to {outdir}")


def main():
    args = parse_args()
    global KMS_SCALE
    KMS_SCALE = float(np.exp(args.kms))
    if args.targets:
        write_targets(args)
        return
    if not args.compare:
        raise SystemExit("nothing to do: pass --targets or --compare")
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_cleanprop/")
    os.makedirs(outdir, exist_ok=True)
    compare(args, outdir)


if __name__ == "__main__":
    logging.setup_logger(__file__, 3, False)
    main()
