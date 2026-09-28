#!/usr/bin/env python3
"""Ditrack clean-propagation closure on the real-material toy: a deterministic
two-body decay -- J/psi -> mu+ mu-, K_S -> pi+ pi-, Lambda -> p pi- -- with the
parent mass reconstructed from ONE plane (`--decay jpsi|ks|lam`).

WHAT IS MEASURED
----------------
On every plane k both legs' true states (sim) are compared with their
deterministic references (model), exactly as in the single-track closure
(realmat_closure.py).  Each leg's LOCAL deviation is carried back to the
vertex by the reference's own linear transport,

    dcurv_0 = P_k^{-1} H_k^{-1} dlocal_k ,      P_k = F_k ... F_0 ,

and the vertex mass moves by  dm = g . dcurv_0,  g = dm/d(q/p, lambda, phi)
of that leg at the vertex.  Per leg the mass is therefore one more linear
functional of the plane-k deviation:

    curvilinear  a_k = P_k^{-T} g            (what model_phi contracts)
    local        b_k = H_k^{-T} a_k          (what the sim residual is dotted into)

and the pair statistic is  z = (b_k^- . dlocal^- + b_k^+ . dlocal^+) / s_F.
The two legs are independent in Geant4 (different seeds; the per-plane
correlation is printed), so the model CF of the pair is the PRODUCT of the
legs' model CFs along a_k^-, a_k^+.  s_F = sigma sqrt(1/I) by exact FFT
inversion of that product (cgf_channels, the grid rules of fisher_norm).

Two statistics per plane and probe u (fisher_norm.UCURVE):

  even   <e^{-u z^2}>_sim - model          the single-track table's statistic;
                                            blind to a shift
  odd    <z e^{-u z^2}>_sim - model  ->  a LOCATION in MeV,
             dm(u) = s_F * odd(u) / D(u),   D(u) = E_model[(1 - 2 u z^2) e^{-u z^2}]
         the first-order response of the odd probe to a shift of the sim
         distribution; u -> 0 is the mean, u = 1 the core.  Model side
         from the CF: odd = int_0 Im phi (t/2u) w,  D = int_0 Re phi (t^2/2u) w,
         w = e^{-t^2/4u}/sqrt(pi u).

THE MODEL is the single-track model, which by default carries the exact hard
knock-on collision (cf_knockon): the joint law of energy loss and deflection
and the exact 1/p map.  With the linear, independent-channel model instead
(CF_KNOCKON_JOINT=0 CF_QOP_EXACT=0) this closure fails in shape (+0.0037),
core (+0.11 MeV) and mean (-0.12 MeV at the outer plane); outputs are tagged
by that state (`ditrack_closure{,_kj1_qx0,_kj0_qx1,_kj1_qx1}.json`).

The mass is linear in the back-propagated state here, on both sides.  The
second-order term of the mass map is printed separately (sim only: the exact
pair mass of the back-propagated momenta minus the linear one).

SIM-SIDE DIAGNOSTICS (per plane, in the JSON)
  mean_split   <b_c dlocal_c> per leg and local component, MeV
  jensen       the q/p slot's second-order part <b_0 (dqop - dqop_lin)>,
               dqop_lin = -q (p_sim - p_ref)/p_ref^2.  The model maps energy
               loss LINEARLY into q/p (dqop = q cs dE) and is centred, so its
               mean is zero; the sim's q/p is 1/p of a centred momentum and
               carries E[dqop]/qop = E[(dp/p)^2] (Jensen).  The bending that
               the excess curvature produces on the way out lands in dx/dz.
  shuffled     the even closure with each leg's dqop permuted across events:
               same marginals, energy loss and deflection decorrelated.  The
               model treats ionisation and scattering as independent, so a
               shift of the closure under the shuffle is the within-leg joint
               law (knock-on electrons: th = sqrt(2 m_e T)/p with the loss T).
               Clean near the vertex only: further out the transport couples
               q/p and angle (bending), which the model has and the shuffle
               destroys.

THE CONFIGURATION (DECAYS)
--------------------------
Each decay pairs an EXISTING leg -- realmat_closure.py's realmat_full sample
and model (pT 3, eta 0.30, phi 0.70, seeds 101-110) -- with a NEW leg at
phi 0.70 + DPHI (DPHI solved so that the pair has the parent's PDG mass) that
has its own plane set (gen_toy_realmat.helix_frames at the leg's pT, charge
and azimuth), the same tracker.xml, its own seeds and its own model export:
    jpsi  mu- + mu+ (pT 3,   seeds 201-210)
    ks    pi- + pi+ (pT 3,   seeds 301-310)
    lam   p   + pi- (pT 0.8, seeds 401-410)
The cylinders are azimuthally symmetric but the field is the real 3D map, so a
new leg is simulated and exported at its own azimuth rather than rotated from
the phi = 0.70 sample (`pairs` prints the difference for equal pT).
Arms (`--arm`, both legs): `off` (clean) or `elonly` (hadElastic on) with
`--model-tag _mat` and `closure --nucel` (the per-element elastic channel).

THE LOW-MOMENTUM LEG (Lambda's pi-, pT 0.8 GeV)
  * it LOOPS (helix diameter 1.40 m): ~70 % of events re-cross the outer
    shells inward; toy_loader keeps the FIRST (outgoing) crossing of a plane;
  * the propagator's momentum floor (PropagationPtotLimit, 1 GeV) refuses it:
    the export sets TOY_PLIMIT = 0.1;
  * `helix_frames`' fixed-radius planes miss its energy-losing reference by
    2 cm at the outer plane; `recentre` moves the planes onto the model's
    reference crossings (converges in one iteration; the sim is untouched);
  * H's energy-loss term (hbasis.H_ELOSS) is 0 for these layered geometries;
    at 1 it would over-state its local q/p width x2.1 at the outer planes.

THE EXTENDED SAMPLE (`--xstat`): 80 more jobs of 20 000 events per leg and
arm in sim_xstat/ (seeds `xstat_seeds`), 1.8 M per leg with the ten in sim/;
`sim --xstat` makes them for BOTH legs, every analysis under `--xstat` reads
sim/ + sim_xstat/ and writes to runs/<res>_xstat/.

SUBCOMMANDS (global options --decay, --arm, --model-tag, --xstat before the command)
    setup      private area of the new leg: geometry, planes, drivers (gun
               azimuth from TOY_PHI, plane module from TOY_PLANES_MOD)
    sim        the new leg, seeded split
    live       provenance, gun azimuth, switch census, acceptance
    export     the new leg's model
    recentre   put the new leg's planes on its reference crossings
    pairs      model/sim pair gate for both legs (+ rotation comparison)
    closure    per-plane even + odd closure of the parent mass -> JSON +
               figures (`--nucel`, `--only <leg>` for one leg's share)
    legclosure the five-direction single-track closure of one leg
    summary    per-plane shape and location figures from the JSON
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import realmat_closure as rc                                     # noqa: E402
from realmat_closure import cne, cpt, ctr, fn, gc, hb, hp, mx    # noqa: E402
import cgf_channels as cc                                        # noqa: E402

LOCAL = hb.LOCAL                    # (qop, dxdz, dydz, locx, locy)
REFB = {"qop": "refqop", "dxdz": "refdxdz", "dydz": "refdydz",
        "locx": "reflocx", "locy": "reflocy"}

# =========================================================================
# the decays
# =========================================================================
# Each decay pairs an EXISTING realmat_full leg (pT 3, eta 0.30, phi 0.70:
# realmat_closure.py's sample and model, `old`) with a NEW leg (`new`, pT
# `pt_new`, eta 0.30, phi 0.70 + DPHI) that has its own plane set, seeds, sim
# and model.  DPHI is solved so that the pair mass is the parent's PDG mass.
#   jpsi  mu- (old) + mu+ (new, pT 3)      DPHI 1.0819
#   ks    pi- (old) + pi+ (new, pT 3)      DPHI 0.1374
#   lam   p   (old) + pi- (new, pT 0.8)    DPHI 0.1283  -- equal momenta are
#         kinematically impossible (m >= 1335 MeV), the proton carries most of
#         the Lambda's momentum; pT_pi >= 0.61 GeV is needed to reach r = 107 cm
# Seeds of the new leg never overlap the old leg's (101-110): equal seeds would
# replay the same random stream in both legs.
DECAYS = {
    "jpsi": dict(tex=r"J/\psi", mass=3.0969, old=13, new=-13, pt_new=3.0,
                 geom="qpj", tag="jpsi", seed0=201,
                 res="realmat_ditrack_260926", fig="ditrack_cleanprop"),
    "ks": dict(tex=r"K^0_S", mass=0.497611, old=-211, new=211, pt_new=3.0,
               geom="ks_pip", tag="ks", seed0=301,
               res="realmat_ditrack_ks_260927", fig="ditrack_cleanprop_ks"),
    "lam": dict(tex=r"\Lambda", mass=1.115683, old=2212, new=-211, pt_new=0.8,
                geom="lam_pim", tag="lam", seed0=401,
                res="realmat_ditrack_lam_260927", fig="ditrack_cleanprop_lam"),
    "jpsiee": dict(tex=r"J/\psi \to ee", mass=3.0969, old=11, new=-11, pt_new=3.0,
                   geom="qpje", tag="jpsiee", seed0=501,
                   res="realmat_ditrack_jpsiee_260927", fig="ditrack_cleanprop_jpsiee"),
    # the Z: both legs are dedicated samples (`pt_old`), at the pT two
    # same-eta legs need for m_Z (>= m_Z/2 = 45.6 GeV; 48 GeV opens them by
    # 2.5 rad)
    "zmm": dict(tex=r"Z \to \mu\mu", mass=91.1876, old=13, new=-13, pt_new=48.0,
                pt_old=48.0, geom_old="zmm_m", geom="zmm_p", tag="zmm", seed0=601,
                res="realmat_ditrack_zmm_260927", fig="ditrack_cleanprop_zmm"),
    "zee": dict(tex=r"Z \to ee", mass=91.1876, old=11, new=-11, pt_new=48.0,
                pt_old=48.0, geom_old="zee_m", geom="zee_p", tag="zee", seed0=701,
                res="realmat_ditrack_zee_260927", fig="ditrack_cleanprop_zee"),
}
DECAY = None      # the configured decay (`configure`)
DEC = None
LEGS = {}         # leg key -> dict(pdg, q, pt, phi, geom, mass, new, seeds)
L0 = L1 = None    # the old and the new leg's key
DPHI = None
M_PARENT = None
RES = None
NUCEL = False     # the elastic channel in the prediction (`closure --nucel`)
XSTAT = False     # the extended sample: sim/ plus sim_xstat/ (`--xstat`)
XSTAT_JOBS = 80   # jobs of 20 000 events per leg and arm in sim_xstat/


def _mass(pdg):
    return hp.SPECIES[pdg]["mass"] * 1e-3


def _ptstr(pt):
    return f"{pt:g}".replace(".", "p")


def dphi_for_mass(m, pt1, m1, pt2, m2, eta=rc.ETA):
    """Azimuthal opening of two tracks at the same eta with pair mass m:
    m^2 = m1^2 + m2^2 + 2 (E1 E2 - pT1 pT2 cos dphi - pz1 pz2)."""
    e1 = math.hypot(pt1 * math.cosh(eta), m1)
    e2 = math.hypot(pt2 * math.cosh(eta), m2)
    pz = pt1 * pt2 * math.sinh(eta) ** 2
    c = (e1 * e2 - pz - 0.5 * (m * m - m1 * m1 - m2 * m2)) / (pt1 * pt2)
    if not -1.0 <= c <= 1.0:
        raise ValueError(f"no opening angle gives m = {m} (cos = {c})")
    return math.acos(c)


def configure(decay):
    global DECAY, DEC, LEGS, L0, L1, DPHI, M_PARENT, RES
    DECAY, DEC = decay, DECAYS[decay]
    old, new = DEC["old"], DEC["new"]
    own_old = "pt_old" in DEC          # the first leg a dedicated sample too
    pt_old = DEC.get("pt_old", rc.PT)
    DPHI = dphi_for_mass(DEC["mass"], pt_old, _mass(old), DEC["pt_new"], _mass(new))
    M_PARENT = DEC["mass"]
    L0, L1 = rc.lab(old), rc.lab(new)
    qo, qn = float(hp.SPECIES[old]["q"]), float(hp.SPECIES[new]["q"])
    LEGS = {L0: dict(pdg=old, q=qo, pt=pt_old, phi=rc.PHI,
                     geom=DEC["geom_old"] if own_old else rc.gkey(old),
                     mass=_mass(old), new=own_old,
                     seeds=list(range(DEC["seed0"] + 50, DEC["seed0"] + 60))),
            L1: dict(pdg=new, q=qn, pt=DEC["pt_new"], phi=rc.PHI + DPHI,
                     geom=DEC["geom"], mass=_mass(new), new=True,
                     seeds=list(range(DEC["seed0"], DEC["seed0"] + 10)))}
    for L in LEGS.values():
        if L["new"]:
            rc.GEOM[L["geom"]] = (f"realmat_full_{L['geom']}",
                                  f"toyPlanes_realmat_full_{L['geom']}_pt{_ptstr(L['pt'])}",
                                  L["q"])
    RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                       DEC["res"])


def variant():
    """File tag of the sim arm, the elastic channel (`_nojoint` when its joint
    law is off) and the knock-on state: realmat_closure's own (`_variant`), so
    the J/psi names are unchanged;
    `_hel<x>` when H's energy-loss term is on (hbasis.H_ELOSS != 0, a
    diagnostic: the closure geometries are layered)."""
    he = hb._canonical().H_ELOSS
    return rc._variant(NUCEL) + ("" if he == 0.0 else f"_hel{he:g}")


def sim_path(leg, seed):
    """A leg's sim file: the old leg under realmat_closure's name, the new legs
    under the decay's; in sim_xstat/ under `--xstat`."""
    L = LEGS[leg]
    base = (os.path.basename(rc.sim_path(L["pdg"], seed)) if not L["new"] else
            f"{leg}_{DEC['tag']}_pt{_ptstr(L['pt'])}_{rc.ARM}_s{seed}_sim.root")
    return os.path.join(rc.OUT, "sim_xstat" if XSTAT else "sim", base)


def sim_glob(leg):
    """The leg's sample: its ten seeds in sim/ (the old leg: realmat_closure's
    sample); under `--xstat` every seed of the leg in sim/ and sim_xstat/."""
    L = LEGS[leg]
    if XSTAT:
        stem = os.path.basename(sim_path(leg, 0))[:-len("0_sim.root")]
        return os.path.join(rc.OUT, "sim*", stem + "*_sim.root")
    if not L["new"]:
        return rc.sim_glob(L["pdg"])
    return os.path.join(rc.OUT, "sim", f"{leg}_{DEC['tag']}_pt{_ptstr(L['pt'])}_"
                                       f"{rc.ARM}_s{DEC['seed0'] // 100}??_sim.root")


def xstat_seeds(leg):
    """Seeds of a leg's sim_xstat/ jobs: 1000 + seed0 + i for the new leg,
    1100 + seed0 + i for the old one (distinct from each other and from every
    sample in sim/)."""
    first = 1000 + DEC["seed0"] + (0 if leg == L1 else 100)
    return list(range(first, first + XSTAT_JOBS))


def model_path(leg):
    """The old leg's model follows realmat_closure's MODEL_TAG (`_mat`: the
    export with the per-step material table); the new legs' exports carry the
    table in any case."""
    L = LEGS[leg]
    if not L["new"]:
        return rc.model_path(L["pdg"])
    return os.path.join(rc.OUT, "model",
                        f"model_{leg}_{DEC['tag']}_pt{_ptstr(L['pt'])}_all4.root")


def planes_path(leg):
    L = LEGS[leg]
    return rc.planes_path(L["geom"]) if L["new"] else rc.species_planes_path(L["pdg"])


def new_legs():
    """The legs with a dedicated sample and geometry (the second leg always,
    the first when the decay sets `pt_old`)."""
    return [lg for lg in (L0, L1) if LEGS[lg]["new"]]


configure("jpsi")


# =========================================================================
# kinematics
# =========================================================================

def p3(qop, lam, phi):
    p = 1.0 / abs(qop)
    return p * np.array([np.cos(lam) * np.cos(phi), np.cos(lam) * np.sin(phi),
                         np.sin(lam)])


def start_curv(leg):
    """(q/p, lambda, phi) of the leg's start state, from the gun settings."""
    L = LEGS[leg]
    return np.array([L["q"] / (L["pt"] * math.cosh(rc.ETA)),
                     math.atan(math.sinh(rc.ETA)), L["phi"]])


def pair_mass(c1, c2):
    """Pair mass from the old (c1) and new (c2) leg's (q/p, lambda, phi),
    arrays of shape (3,) or (3, n)."""
    def mom(c):
        p = 1.0 / np.abs(c[0])
        return p * np.array([np.cos(c[1]) * np.cos(c[2]),
                             np.cos(c[1]) * np.sin(c[2]), np.sin(c[1])])
    a, b = mom(np.asarray(c1)), mom(np.asarray(c2))
    ea = np.sqrt((a ** 2).sum(axis=0) + LEGS[L0]["mass"] ** 2)
    eb = np.sqrt((b ** 2).sum(axis=0) + LEGS[L1]["mass"] ** 2)
    return np.sqrt((ea + eb) ** 2 - ((a + b) ** 2).sum(axis=0))


def mass_gradient():
    """dm/d(q/p, lambda, phi) of each leg at the reference, analytic:
    dm = (E_other p_this/E_this - p_other) . dp_this / m."""
    c = {lg: start_curv(lg) for lg in LEGS}
    m0 = pair_mass(c[L0], c[L1])
    out = {}
    for lg, oth in ((L0, L1), (L1, L0)):
        qop, lam, phi = c[lg]
        q = np.sign(qop)
        p = 1.0 / abs(qop)
        n = np.array([np.cos(lam) * np.cos(phi), np.cos(lam) * np.sin(phi),
                      np.sin(lam)])
        dn_dlam = np.array([-np.sin(lam) * np.cos(phi), -np.sin(lam) * np.sin(phi),
                            np.cos(lam)])
        dn_dphi = np.array([-np.cos(lam) * np.sin(phi), np.cos(lam) * np.cos(phi),
                            0.0])
        pa, pb = p * n, p3(*c[oth])
        ea = math.sqrt(p * p + LEGS[lg]["mass"] ** 2)
        eb = math.sqrt(pb @ pb + LEGS[oth]["mass"] ** 2)
        v = (eb * pa / ea - pb) / m0
        out[lg] = np.array([v @ (-q * p * p * n), v @ (p * dn_dlam),
                            v @ (p * dn_dphi), 0.0, 0.0])
    return m0, out


def _check_gradient():
    """Finite-difference check of mass_gradient (central, relative 1e-6)."""
    m0, g = mass_gradient()
    c = {lg: start_curv(lg) for lg in LEGS}
    worst = 0.0
    for lg, oth in ((L0, L1), (L1, L0)):
        for i in range(3):
            h = 1e-6 * max(abs(c[lg][i]), 1e-3)
            cp, cm = c[lg].copy(), c[lg].copy()
            cp[i] += h
            cm[i] -= h
            if lg == L0:
                d = (pair_mass(cp, c[oth]) - pair_mass(cm, c[oth])) / (2 * h)
            else:
                d = (pair_mass(c[oth], cp) - pair_mass(c[oth], cm)) / (2 * h)
            worst = max(worst, abs(d - g[lg][i]) / max(abs(d), 1e-12))
    return m0, g, worst


# =========================================================================
# setup / sim / export / live / pairs  (the new leg)
# =========================================================================

def cmd_setup(args):
    m0, g, worst = _check_gradient()
    print(f"{DECAY}: " + " + ".join(
        f"{lg} (pT {LEGS[lg]['pt']:g}, phi {LEGS[lg]['phi']:.12f}"
        f"{', dedicated sample' if LEGS[lg]['new'] else ', single-track sample'})"
        for lg in (L0, L1)) + f"; DPHI = {DPHI:.12f} rad")
    print(f"pair mass of the two start states {1e3 * m0:.6f} MeV "
          f"(target {1e3 * M_PARENT:.3f}); analytic vs FD gradient max rel "
          f"diff {worst:.1e}")
    if abs(m0 - M_PARENT) > 1e-9 * max(1.0, M_PARENT) or worst > 1e-6:
        raise SystemExit("kinematics check failed")
    ref_xml = os.path.realpath(os.path.join(rc.SRCDATA, "realmat_full",
                                            "tracker.xml"))
    ref_pl = os.path.join(rc.SRCTEST, "toyPlanes_realmat_full_pt3.py")
    txt = open(ref_pl).read()
    head, body = txt.split("\n", 1)
    ns = {}
    exec(txt, ns)
    radii = list(ns["radii"])
    if rc._plane_body(radii, -1.0) != body:
        raise SystemExit("helix_frames(q=-1) does NOT reproduce the published "
                         "plane file")
    print("helix_frames(q=-1) reproduces the published plane file: IDENTICAL")
    import shutil
    for leg in new_legs():
        N = LEGS[leg]
        twoR = 2.0 * N["pt"] / (0.3 * rc.BFIELD) * 100.0
        if twoR <= max(radii):
            raise SystemExit(f"the {leg} leg curls up at r = {twoR:.1f} cm, before "
                             f"the outermost plane at {max(radii):.1f} cm")
        G = N["geom"]
        gdir, pmod, q = rc.GEOM[G]
        td = rc.testdir(G)
        dd = os.path.join(rc.geomdir(G), "Analysis", "HitAnalyzer", "data", gdir)
        os.makedirs(td, exist_ok=True)
        os.makedirs(dd, exist_ok=True)
        xo = os.path.join(dd, "tracker.xml")
        shutil.copyfile(ref_xml, xo)
        assert rc._md5(xo) == rc._md5(ref_xml)
        print(f"[{G}] {rc._md5(xo)}  {gdir}/tracker.xml  IDENTICAL to the reference")
        if not os.path.exists(rc.planes_path(G)) or args.force:
            out = (f"# generated by realmat_ditrack.py setup: gen_toy_realmat."
                   f"helix_frames(q={q:+.0f}, phi0={N['phi']!r}, pt={N['pt']!r}) on the "
                   f"radii of toyPlanes_realmat_full_pt3.py; do not hand edit\n"
                   + rc._plane_body(radii, q, phi0=N["phi"], pt=N["pt"]))
            open(rc.planes_path(G), "w").write(out)
        print(f"[{G}] {rc._md5(rc.planes_path(G))}  {pmod}.py")
        s, m = rc._driver_sources()
        s = hp._sub1(s, r"MinPhi=cms\.double\(0\.70\), MaxPhi=cms\.double\(0\.70\),",
                     'MinPhi=cms.double(float(os.environ["TOY_PHI"])),\n'
                     '        MaxPhi=cms.double(float(os.environ["TOY_PHI"])),',
                     "sim: gun azimuth")
        # the watcher radii come from the leg's own plane module (the stock
        # driver assumes `toyPlanes_<geometry>_pt3`; the radii are the same
        # cylinders for every leg, the module name is not)
        s = hp._sub1(s, r"_pmod = 'toyPlanes_%s_pt3' % opts\.toyGeom\.split\('/'\)\[-2\]",
                     "_pmod = os.environ.get('TOY_PLANES_MOD', 'toyPlanes_%s_pt3' "
                     "% opts.toyGeom.split('/')[-2])", "sim: watcher plane module")
        s += ("\nprint('[toy] gun phi = %.12f'\n"
              "      % process.generator.PGunParameters.MinPhi.value())\n")
        open(os.path.join(td, "runToyGeomCheck.py"), "w").write(s)
        # the propagator's momentum floor (PropagationPtotLimit, 1 GeV by
        # default: a guard against run-away legs in fits, not physics) refuses a
        # leg below it; the clean-propagation export takes it from TOY_PLIMIT
        m += ("\nprocess.Geant4ePropagator.PropagationPtotLimit = cms.double(\n"
              "    float(_os.environ.get('TOY_PLIMIT', '1.0')))\n"
              "print('[toy] PropagationPtotLimit = %g GeV'\n"
              "      % process.Geant4ePropagator.PropagationPtotLimit.value())\n")
        open(os.path.join(td, "runToyModel.py"), "w").write(m)
        print(f"[{G}] wrote runToyGeomCheck.py and runToyModel.py in {td}")


def _plane_offsets(leg):
    """(reference crossing of each flat plane in global coordinates, local
    offsets (x, y, z) of that crossing, reference direction) from the model."""
    import uproot
    from toy_loader import plane_frames
    ns = {}
    exec(open(planes_path(leg)).read(), ns)
    o, R = plane_frames(ns["origin"], ns["normal"], ns["uaxis"])
    f = uproot.open(model_path(leg))
    t = f[next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))]
    a = t.arrays(["reflocx", "reflocy", "reflocz", "refdxdz", "refdydz"],
                 library="np")
    loc = np.stack([a["reflocx"], a["reflocy"], a["reflocz"]], axis=1)
    X = o + np.einsum("kji,kj->ki", R, loc)
    dl = np.stack([a["refdxdz"], a["refdydz"], np.ones(len(o))], axis=1)
    d = np.einsum("kji,kj->ki", R, dl)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return ns, X, loc, d


def cmd_recentre(args):
    """Put every dedicated leg's planes ON its own reference trajectory
    (`realmat_closure.recentre_planes`): export, re-centre, repeat until the
    largest origin move is below --tol, then export once more.  The sim is
    untouched: it scores the same cylinders.  The single-track legs are
    re-centred by `realmat_closure.py recentre`."""
    for leg in new_legs():
        for it in range(args.iterations):
            cmd_export(argparse.Namespace(force=True, leg=leg))
            move = rc.recentre_planes(planes_path(leg), model_path(leg), leg)
            if move < args.tol:
                break
        cmd_export(argparse.Namespace(force=True, leg=leg))


def _env_new(leg, extra=None):
    N = LEGS[leg]
    sp = hp.SPECIES[N["pdg"]]
    e = dict(TOY_PLANES_MOD=rc.GEOM[N["geom"]][1], TOY_PDG=str(N["pdg"]),
             TOY_PNAME=sp["g4"], TOY_RMAX="107.0", TOY_CUT=repr(rc.CUT),
             TOY_INACT=",".join(hp.inact_of(N["pdg"], rc.ARM)),
             TOY_PHI=repr(N["phi"]))
    e.update(extra or {})
    return e


def _sim_one(a):
    leg, seed, nev, force = a
    N = LEGS[leg]
    out = sim_path(leg, seed)
    log = out[:-5] + ".log"
    if os.path.exists(out) and rc._complete(log, nev) and not force:
        return leg, seed, 0, log, "cached"
    census = {"TOY_CENSUS": out[:-9] + "_census.bin"}
    ee = _env_new(leg, census) if N["new"] else rc._env(N["pdg"], census)
    r = rc._run(N["geom"], "runToyGeomCheck.py",
                f"events={nev} pt={N['pt']!r} eta={rc.ETA} output={out} "
                f"seed={seed} toyGeom={rc.toygeom(N['geom'])}", log, ee)
    if r == 0 and not rc._complete(log, nev):
        r = 99
    return leg, seed, r, log, "ran"


def cmd_sim(args):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.dirname(sim_path(L1, 0)), exist_ok=True)
    legs = list(LEGS) if XSTAT else new_legs()
    seeds = {lg: xstat_seeds(lg) if XSTAT else LEGS[lg]["seeds"][:args.jobs] for lg in legs}
    jobs = [(lg, s, args.events, args.force) for lg in legs for s in seeds[lg]]
    for lg in legs:
        N = LEGS[lg]
        print(f"{lg}: {len(seeds[lg])} jobs x {args.events} events, pT {N['pt']:g}, "
              f"phi = {N['phi']:.6f}, arm {rc.ARM}"
              + (f", seeds {seeds[lg][0]}-{seeds[lg][-1]} -> sim_xstat/" if XSTAT else ""),
              flush=True)
    t0, bad = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for lg, seed, r, log, how in ex.map(_sim_one, jobs):
            print(f"  {lg} seed={seed} rc={r} {how:<6} {time.time() - t0:7.0f} s",
                  flush=True)
            if r:
                bad += 1
                print(f"    *** FAILED, see {log}", flush=True)
    if bad:
        raise SystemExit(f"{bad} jobs failed")


def cmd_export(args):
    os.makedirs(os.path.join(rc.OUT, "model"), exist_ok=True)
    legs = [args.leg] if getattr(args, "leg", None) else new_legs()
    for leg in legs:
        N = LEGS[leg]
        out = model_path(leg)
        log = out[:-5] + ".log"
        if os.path.exists(out) and not args.force:
            print(f"exists -> {out}")
            continue
        r = rc._run(N["geom"], "runToyModel.py",
                    f"pt={N['pt']!r} eta={rc.ETA} phi={N['phi']!r} partId={N['pdg']} "
                    f"output={out} toyGeom={rc.toygeom(N['geom'])}", log,
                    _env_new(leg, dict(rc.FOUR_ON, TOY_PLIMIT="0.1")))
        print(f"{leg}: rc={r} -> {out}", flush=True)
        if r:
            raise SystemExit(f"export failed, see {log}")
        t = open(log, errors="ignore").read()
        eff = re.search(r"\[cvh\] effective: (.*)", t)
        print(f"    {eff.group(0) if eff else '*** no [cvh] effective line ***'}")


_SIMC = {}


def load_sim(leg):
    key = (DECAY, leg, rc.ARM, XSTAT)
    if key not in _SIMC:
        from toy_loader import load_toy_sim
        ns = {}
        exec(open(planes_path(leg)).read(), ns)
        _SIMC[key] = load_toy_sim(sim_glob(leg), ns["origin"], ns["normal"],
                                  ns["uaxis"])
    return _SIMC[key]


def cmd_live(args):
    """The new leg's provenance (stepper, cut, particle, azimuth, planes) and
    process census: every process the arm switches off has ZERO primary steps
    (species-specific names zero in `all` as well), every process the arm
    restores relative to `off` fires on the primary."""
    for leg in new_legs():
        _live_one(leg)


def _live_one(leg):
    N = LEGS[leg]
    pdg = N["pdg"]
    logs = sorted(glob.glob(sim_glob(leg)[:-5] + ".log"))
    print(f"### {leg} ({DECAY} leg, arm {rc.ARM})  {len(logs)} logs")
    if not logs:
        raise SystemExit("no logs")
    tot, prov = {}, set()
    for lg in logs:
        t = open(lg, errors="ignore").read()
        for k, v in hp.steps_from_log(lg).items():
            a, b = tot.get(k, (0, 0))
            tot[k] = (a + v[0], b + v[1])
        st = "TIGHT" if "[toy] TIGHT stepper" in t else "NOT-TIGHT"
        cut = re.search(r"\[toy\] DefaultCutValue = (\S+) cm", t)
        pid = re.search(r"\[toy\] PartID = (-?\d+) \((\S+)\)", t)
        rad = re.search(r"\[toy\] watcher radii from (\S+) \((\d+)\)", t)
        phi = re.search(r"\[toy\] gun phi = (\S+)", t)
        prov.add((st, cut.group(1) if cut else "?", pid.group(1) if pid else "?",
                  rad.group(1) + "/" + rad.group(2) if rad else "?",
                  phi.group(1) if phi else "?"))
    for p in sorted(prov):
        print(f"    stepper={p[0]} cut={p[1]} cm PartID={p[2]} planes={p[3]} "
              f"gun phi={p[4]}")
    if len(prov) != 1:
        raise SystemExit("inconsistent provenance")
    st, cut, pid, pl, phi = next(iter(prov))
    if st != "TIGHT" or int(pid) != pdg or abs(float(phi) - N["phi"]) > 1e-9 \
            or rc.GEOM[N["geom"]][1] not in pl:
        raise SystemExit("wrong stepper, particle, azimuth or plane set")
    for nm in hp.inact_of(pdg, rc.ARM):
        got = tot.get(nm, (0, 0))
        bad = got[0] != 0 or (nm not in ("Decay", "hadElastic") and got[1] != 0)
        print(f"    census {nm:<24} primary {got[0]:>9} all {got[1]:>9}  "
              f"{'*** NONZERO ***' if bad else 'OK (primary zero)'}")
        if bad:
            raise SystemExit(f"{nm} is not switched off")
    off = set(hp.inact_of(pdg, "off"))
    for nm in sorted(off - set(hp.inact_of(pdg, rc.ARM))):
        got = tot.get(nm, (0, 0))
        print(f"    census {nm:<24} primary {got[0]:>9} all {got[1]:>9}  "
              f"{'OK (restored, fires on the primary)' if got[0] > 0 else '*** INERT ***'}")
        if got[0] == 0:
            raise SystemExit(f"{nm} is restored by arm {rc.ARM} but never fired")
    v = load_sim(leg)["valid"]
    print(f"    events {v.shape[0]}, reached every plane "
          f"{100 * v.all(axis=1).mean():.3f} %, outermost {100 * v[:, -1].mean():.3f} %")


def cmd_pairs(args):
    """realmat_closure's pair gate for both legs, then -- when the new leg has
    the old leg's pT -- the rotation comparison with the same species at
    phi = 0.70 (informational: the material is azimuthally symmetric, the
    field map is not exactly)."""
    for leg in (L0, L1):
        mp = model_path(leg)
        m = gc._model_meta(mp)
        files = sorted(glob.glob(sim_glob(leg)))
        s = gc._toy_sim_meta(files[0])
        sim = load_sim(leg)
        dE_m = 1e3 * float(m["refp"][0] - m["refp"][-1])
        nok = len(m["detid"]) == s["npl"]
        seqok = nok and bool(np.array_equal(np.asarray(m["detid"]), s["seq"]))
        dr = np.abs(np.asarray(m["refglobr"]) - s["r"]).max() if nok else np.nan
        ns = {}
        exec(open(planes_path(leg)).read(), ns)
        rpl = np.asarray(ns["radii"])
        dxs = np.array([np.nanmedian(sim["locx"][sim["valid"][:, k], k])
                        - m["reflocx"][k] for k in range(len(rpl))])
        ok = (nok and seqok and dr < 0.01 and np.abs(dxs).max() < 0.05
              and rc._dE_ok(dE_m, s))
        print(f"--- {leg}  model {os.path.basename(mp)}  sim {len(files)} files, "
              f"{sim['valid'].shape[0]} events")
        print(f"    (1) legs {len(m['detid'])} vs sim planes {s['npl']}  "
              f"(2) sequence identical {seqok}")
        print(f"    (3) max |refglobr - sim median r| = {dr:.5f} cm")
        print(f"    (4) dE model(mean) {dE_m:.3f} MeV  sim(mean, first file, tracks "
              f"on every plane) {s['dEmean']:.3f} +- {s['dEmean_err']:.3f} MeV  "
              f"(median {s['dE']:.3f})")
        print(f"    (5) max |median(sim locx) - reflocx| = "
              f"{1e4 * np.abs(dxs).max():.1f} um")
        print(f"    {'PASS' if ok else '*** FAIL ***'}")
        if not ok and not args.nofail:
            raise SystemExit("pair gate failed")
    N = LEGS[L1]
    if N["pt"] != rc.PT or LEGS[L0]["new"]:
        return
    a = cpt.load_model(rc.model_path(N["pdg"]))
    b = cpt.load_model(model_path(L1))
    rp = np.array([[l["refp"] for l in a], [l["refp"] for l in b]])
    print(f"rotation: refp(phi={N['phi']:.3f}) / refp(phi=0.70) - 1 per plane: "
          f"max {np.abs(rp[1] / rp[0] - 1).max():.2e}")
    va = [float(cpt.model_variance(a, k, cpt.FUNCTIONALS['qop'])[0])
          for k in range(len(a))]
    vb = [float(cpt.model_variance(b, k, cpt.FUNCTIONALS['qop'])[0])
          for k in range(len(b))]
    print(f"rotation: q/p model variance ratio - 1 per plane: max "
          f"{np.abs(np.array(vb) / np.array(va) - 1).max():.2e}")


# =========================================================================
# the closure
# =========================================================================

def weier_odd(phi, u, tau):
    w = (tau / (2.0 * u)) * np.exp(-tau ** 2 / (4.0 * u)) / np.sqrt(np.pi * u)
    return float(np.trapezoid(phi.imag * w, tau))


def weier_shift(phi, u, tau):
    """D(u) = E[(1 - 2 u z^2) e^{-u z^2}], the odd probe's response to a shift."""
    w = (tau ** 2 / (2.0 * u)) * np.exp(-tau ** 2 / (4.0 * u)) / np.sqrt(np.pi * u)
    return float(np.trapezoid(phi.real * w, tau))


def _setup_physics():
    """realmat_closure.cell's configuration, asserted: the six MS
    harmonisations, radiation on, Kokoulin for muons only (both legs must
    agree), the nuclear-elastic channel (recoil included) as `--nucel` says."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    assert (ctr.MS_ELEC_TMAX, ctr.MS_ELEC_EDGE, ctr.MS_SNAP_YMAX, ctr.MS_FINE_G,
            ctr.MS_WVI_SPLIT) == (1.0, 1.0, 0.0, 1.0, 0.0), "MS defaults moved"
    assert mx.MS_CHI0_G4 and mx.MS_FF_G4, "cf_ms_exact harmonisations off"
    kok = {bool(hp.kok_for(L["pdg"])) for L in LEGS.values()}
    assert len(kok) == 1, "the legs disagree on Kokoulin (one global switch)"
    cne.NUCEL_CHANNEL = bool(NUCEL)
    cne.NUCEL_RECOIL = True
    cpt.RAD_CHANNEL = True
    ctr.IONI_KOKOULIN = 1.0 if kok.pop() else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0


def leg_vectors(legs, g):
    """Per plane: curvilinear a_k = P_k^{-T} g and local b_k = H_k^{-T} a_k."""
    _, mass, bfield = hb.bound(legs)
    P = np.eye(5)
    out = []
    for k, leg in enumerate(legs):
        P = leg["F"] @ P
        a = np.linalg.solve(P.T, g)
        H, _ = hb.leg_H(legs, k, bfield=bfield, mass=mass)
        b = np.linalg.solve(H.T, a)
        out.append((a, b, P, H))
    return out


_CTX = None


def _plane_model(k):
    """Everything model-side for plane k: sigma, s_F, the product CF on the
    closure grid, and the product exponent on the density grid."""
    lm, lp, vm, vp, probes = _CTX
    am, ap = vm[k][0], vp[k][0]
    sig = math.sqrt(float(cpt.model_variance(lm, k, am)[0])
                    + float(cpt.model_variance(lp, k, ap)[0]))
    ch = ("ioni", "ms", "rad")
    top = min(cc.tau_reach(lm, k, am, sig, channels=ch),
              cc.tau_reach(lp, k, ap, sig, channels=ch))
    ts = np.concatenate([[0.0], np.geomspace(1e-4, 1.3 * top, 8000)])
    S = (cc.block_cf_exponent(lm, k, am, sig, ts, channels=ch)
         + cc.block_cf_exponent(lp, k, ap, sig, ts, channels=ch))
    z, p, dp = cc.invert_cf(S, ts, npad=32, nt=1 << 17, deriv=True)
    I, info = cc.fisher_exact(z, p, dp, floor=1e-8)
    sF = sig * math.sqrt(1.0 / I)
    tau = fn.closure_tau(float(np.max(probes)))
    phi = (cpt.model_phi(lm, k, am, sF, tau) * cpt.model_phi(lp, k, ap, sF, tau))
    even = np.array([cpt.weier_scalar(phi, u, tau) for u in probes])
    odd = np.array([weier_odd(phi, u, tau) for u in probes])
    D = np.array([weier_shift(phi, u, tau) for u in probes])
    # model mean of z from the slope of Im phi at the origin (smallest t)
    i1 = 1
    mean_z = float(phi[i1].imag / tau[i1])
    return dict(sigma=sig, sF=sF, invI=1.0 / I, mass_frac=info.get("mass_frac"),
                even=even, odd=odd, D=D, mean_z=mean_z, S=S, ts=ts)


def cmd_closure(args):
    _setup_physics()
    os.makedirs(RES, exist_ok=True)
    probes = np.asarray(fn.UCURVE)
    m0, g, worst = _check_gradient()
    assert abs(m0 - M_PARENT) < 1e-9 and worst < 1e-6
    legs = {}
    for leg, L in LEGS.items():
        path = model_path(leg)
        legs[leg] = cpt.load_model(path)
        if NUCEL and not all(cne._has_composition(x) for x in legs[leg]):
            raise SystemExit(f"{path} has no per-step material table: the "
                             f"elastic channel would fall back to one rounded "
                             f"element per step (use --model-tag _mat)")
        hb.bind(legs[leg], path, pdg=L["pdg"])
    # the elastic tables of every bucket, built in parallel and persisted,
    # before the serial warm in the parent
    cne.prewarm(list(legs.values()))
    for leg in LEGS:
        cne.warm(legs[leg])
    vec = {leg: leg_vectors(legs[leg], g[leg]) for leg in LEGS}
    if args.only:
        # ONE leg's share of the mass direction: the other leg's weights are
        # zero, so its CF factor is exactly 1 and it adds nothing to the sim
        if args.only not in LEGS:
            raise SystemExit(f"--only {args.only}: legs are {list(LEGS)}")
        for lg in LEGS:
            if lg != args.only:
                vec[lg] = [(0.0 * a, 0.0 * b, P, H) for a, b, P, H in vec[lg]]
    nl = min(len(legs[L0]), len(legs[L1]))
    global _CTX
    _CTX = (legs[L0], legs[L1], vec[L0], vec[L1], probes)
    t0 = time.time()
    ks = list(range(nl)) if args.planes is None else args.planes
    models = dict(zip(ks, fn.pmap(_plane_model, ks)))
    print(f"model: {len(ks)} planes in {time.time() - t0:.0f} s", flush=True)

    sims = {leg: load_sim(leg) for leg in LEGS}
    nev = min(s["valid"].shape[0] for s in sims.values())
    c0 = {leg: start_curv(leg) for leg in LEGS}
    iu1 = list(probes).index(1.0)
    rows, plots = [], []
    hdr = (f"{'k':>2} {'r[cm]':>7} {'n':>7} {'sigma':>7} {'s_F':>7} "
           f"{'even u=1':>16} {'dm(u=1) MeV':>17} {'dm(u=.01)':>15} "
           f"{'dm(u=.001)':>15} {'corr':>7} {'qop':>5} {'ang':>5} {'nl MeV':>8}")
    print(hdr)
    acc_e = np.zeros((len(probes), nev))
    acc_o = np.zeros((len(probes), nev))
    for k in ks:
        md = models[k]
        dm, good = {}, np.ones(nev, dtype=bool)
        dcurv0, dlocs = {}, {}
        for leg in LEGS:
            s = sims[leg]
            a, b, P, H = vec[leg][k]
            dloc = np.stack([s[c][:nev, k] - legs[leg][k][REFB[c]] for c in LOCAL],
                            axis=1)
            good &= s["valid"][:nev, k] & np.isfinite(dloc).all(axis=1)
            dlocs[leg] = dloc
            dm[leg] = dloc @ b
            dcurv0[leg] = np.linalg.solve(P, np.linalg.solve(H, np.nan_to_num(dloc).T))
            dm[leg + "_qop"] = dloc[:, 0] * b[0]
            dm[leg + "_ang"] = dloc[:, 1] * b[1] + dloc[:, 2] * b[2]
        dmt = dm[L0] + dm[L1]
        n = int(good.sum())
        # sim-side diagnostics of the LOCATION and of the within-leg joint law
        #   mean_split  per leg and local component, <b_c dlocal_c> in MeV
        #   jensen      the q/p slot's second-order part: <b_0 (dqop - dqop_lin)>,
        #               dqop_lin = -q (p_sim - p_ref)/p_ref^2 is the deviation the
        #               model's linear map (dqop = q cs dE) describes
        #   shuffle     even closure with each leg's dqop permuted across events:
        #               same marginals, energy loss and deflection decorrelated
        mean_split, jensen = {}, {}
        rng = np.random.default_rng(20260926)
        dsh = np.zeros(n)
        for leg in LEGS:
            b = vec[leg][k][1]
            dl = dlocs[leg][good]
            mean_split[leg] = (1e3 * b * dl.mean(axis=0)).tolist()
            qr = legs[leg][k]["refqop"]
            q = np.sign(qr)
            psim = 1.0 / np.abs(dl[:, 0] + qr)
            pref = 1.0 / abs(qr)
            dq_lin = -q * (psim - pref) / pref ** 2
            jensen[leg] = 1e3 * float(b[0] * np.mean(dl[:, 0] - dq_lin))
            dsh += dl[rng.permutation(n), 0] * b[0] + dl[:, 1:] @ b[1:]
        zsh = dsh / md["sF"]
        esh = np.exp(-probes[:, None] * zsh[None, :] ** 2).mean(axis=1) - md["even"]
        z = dmt[good] / md["sF"]
        e = np.exp(-probes[:, None] * z[None, :] ** 2)
        o = z[None, :] * e
        even = e.mean(axis=1) - md["even"]
        odd = o.mean(axis=1) - md["odd"]
        ee = e.std(axis=1) / math.sqrt(n)
        eo = o.std(axis=1) / math.sqrt(n)
        acc_e[:, good] += e * (nev / n)
        acc_o[:, good] += o * (nev / n)
        dmu = 1e3 * md["sF"] * odd / md["D"]            # MeV
        edmu = 1e3 * md["sF"] * eo / md["D"]
        corr = float(np.corrcoef(dm[L0][good], dm[L1][good])[0, 1])
        vt = float(np.var(dmt[good]))
        fq = float(np.var((dm[L0 + "_qop"] + dm[L1 + "_qop"])[good]) / vt)
        fa = float(np.var((dm[L0 + "_ang"] + dm[L1 + "_ang"])[good]) / vt)
        # second order of the mass map: exact pair mass of the back-propagated
        # momenta minus the linear prediction
        cm = c0[L0][:, None] + dcurv0[L0][:3][:, good]
        cp = c0[L1][:, None] + dcurv0[L1][:3][:, good]
        nlin = 1e3 * float(np.mean(pair_mass(cm, cp) - m0 - dmt[good]))
        r = float(legs[L0][k]["refglobr"])
        iu01 = list(probes).index(0.01)
        iu001 = list(probes).index(0.001)
        print(f"{k:>2} {r:7.2f} {n:>7} {1e3 * md['sigma']:7.2f} "
              f"{1e3 * md['sF']:7.2f} {even[iu1]:+8.4f}+-{ee[iu1]:.4f} "
              f"{dmu[iu1]:+8.4f}+-{edmu[iu1]:.4f} {dmu[iu01]:+7.3f}+-{edmu[iu01]:.3f} "
              f"{dmu[iu001]:+7.3f}+-{edmu[iu001]:.3f} {corr:+7.4f} {fq:5.2f} "
              f"{fa:5.2f} {nlin:+8.4f}", flush=True)
        print(f"   mean {1e3 * float(np.mean(dmt[good])):+.4f} MeV = "
              + " + ".join(f"{lg}[" + " ".join(f"{c}:{x:+.4f}" for c, x in
                                               zip(LOCAL, mean_split[lg])) + "]"
                           for lg in LEGS)
              + f";  jensen(q/p slot) {sum(jensen.values()):+.4f} MeV;  "
              f"shuffled even u=1 {esh[iu1]:+.4f}", flush=True)
        rows.append(dict(k=k, r=r, n=n, sigma=md["sigma"], sF=md["sF"],
                         invI=md["invI"], mass_frac=md["mass_frac"],
                         even=even.tolist(), even_err=ee.tolist(),
                         odd=odd.tolist(), odd_err=eo.tolist(),
                         D=md["D"].tolist(), dm_MeV=dmu.tolist(),
                         dm_err_MeV=edmu.tolist(),
                         model_even=md["even"].tolist(),
                         model_odd=md["odd"].tolist(),
                         model_mean_MeV=1e3 * md["sF"] * md["mean_z"],
                         sim_mean_MeV=1e3 * float(np.mean(dmt[good])),
                         sim_mean_err_MeV=1e3 * float(np.std(dmt[good]) / math.sqrt(n)),
                         corr=corr, share_qop=fq, share_ang=fa,
                         nonlinear_MeV=nlin, mean_split_MeV=mean_split,
                         jensen_qop_MeV=jensen, shuffled_even=esh.tolist(),
                         **{f"a_{lg}": vec[lg][k][0].tolist() for lg in LEGS},
                         **{f"b_{lg}": vec[lg][k][1].tolist() for lg in LEGS}))
        if k in args.plot:
            plots.append((k, r, dmt[good], md))
    nk = max(len(rows), 1)
    err_lad_e = (acc_e / nk).std(axis=1) / math.sqrt(nev)
    err_lad_o = (acc_o / nk).std(axis=1) / math.sqrt(nev)
    lad_e = np.mean([rw["even"] for rw in rows], axis=0)
    lad_o = np.mean([rw["odd"] for rw in rows], axis=0)
    print(f"ladder mean  even u=1 {lad_e[iu1]:+.4f} +- {err_lad_e[iu1]:.4f}   "
          f"odd u=1 {lad_o[iu1]:+.4f} +- {err_lad_o[iu1]:.4f}")
    print("mean of the vertex mass (MeV), sim vs model (u -> 0 limit):")
    for rw in rows:
        print(f"   k={rw['k']:>2}  sim {rw['sim_mean_MeV']:+.4f} +- "
              f"{rw['sim_mean_err_MeV']:.4f}   model {rw['model_mean_MeV']:+.4f}")
    import cf_knockon as ck
    out = dict(config=dict(decay=DECAY, knockon=dict(ck.physics_state()),
                           nucel=bool(NUCEL),
                           nucel_elements=bool(cne.NUCEL_ELEMENTS),
                           legs={lg: dict(pdg=L["pdg"], pt=L["pt"], eta=rc.ETA,
                                          phi=L["phi"], model=model_path(lg),
                                          sim=sim_glob(lg))
                                 for lg, L in LEGS.items()},
                           dphi=DPHI, m_parent=M_PARENT, arm=rc.ARM,
                           model_tag=rc.MODEL_TAG, nev=nev,
                           ucurve=list(probes)),
               ladder=dict(even=lad_e.tolist(), even_err=err_lad_e.tolist(),
                           odd=lad_o.tolist(), odd_err=err_lad_o.tolist()),
               planes=rows)
    full = args.planes is None
    tag = variant() + (f"_only{args.only}" if args.only else "")
    p = os.path.join(RES, (f"ditrack_closure{tag}.json" if full else
                           f"ditrack_closure{tag}_planes_"
                           + "_".join(map(str, ks)) + ".json"))
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")
    # figures only after the numbers are safe on disk
    for pl in plots:
        try:
            _plot(*pl)
        except Exception as exc:                          # noqa: BLE001
            print(f"*** figure for plane {pl[0]} failed: {exc!r}")


def cmd_legclosure(args):
    """The ordinary single-track closure (realmat_closure's five local
    directions, H basis, Fisher s_F, the full u curve) for ONE leg of the
    configured decay -- the diagnostic for a pair that does not close."""
    _setup_physics()
    leg = args.leg or L1
    L = LEGS[leg]
    path = model_path(leg)
    legs = cpt.load_model(path)
    hb.bind(legs, path, pdg=L["pdg"])
    cne.warm(legs)
    was = bool(hb._canonical().USE_H)
    hb.set_use_h(True)
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    os.makedirs(RES, exist_ok=True)
    try:
        sim = load_sim(leg)
        iu = list(fn.UCURVE).index(1.0)
        for func in args.funcs:
            t0 = time.time()
            sc = fn.plane_scales(legs, func, tag=path,
                                 channels=("ioni", "ms", "rad"))
            rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
            rows, errs = np.asarray(rows), np.asarray(errs)
            pull = rows / errs
            print(f"{leg} {func:<5} u=1 outer {rows[-1, iu]:+.4f}+-{errs[-1, iu]:.4f}  "
                  f"ladder {rows[:, iu].mean():+.4f}+-{err[iu]:.4f}  >3sig "
                  f"{int((np.abs(pull) > 3).sum())}/{pull.size} max "
                  f"{np.abs(pull).max():.1f}  ({time.time() - t0:.0f} s)", flush=True)
            print("      per plane " + " ".join(f"{x:+.4f}" for x in rows[:, iu]),
                  flush=True)
            out = dict(decay=DECAY, leg=leg, func=func, arm=rc.ARM,
                       nucel=bool(NUCEL), ucurve=list(fn.UCURVE),
                       rows=rows.tolist(), errs=errs.tolist(),
                       ks=np.asarray(ks).tolist(), err_ladder=np.asarray(err).tolist(),
                       sF=np.asarray(sc["sF"]).tolist())
            json.dump(out, open(os.path.join(RES, f"leg_{leg}_{func}{variant()}.json"),
                                "w"), indent=1)
    finally:
        hb.set_use_h(was)
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store


def _plot(k, r, dmt, md):
    """Vertex-mass lineshape from plane k: sim histogram vs the model density
    (FFT inversion of the product CF), with the ratio panel."""
    import matplotlib
    matplotlib.use("Agg")
    import pubhtml
    import ratiopanel as rp
    z, p = cc.invert_cf(md["S"], md["ts"], npad=32, nt=1 << 17, deriv=False)
    x = 1e3 * md["sigma"] * z                       # MeV
    dens = p / (1e3 * md["sigma"])
    v = 1e3 * dmt
    lo, hi = np.quantile(v, [2e-4, 1 - 2e-4])
    edges = np.linspace(lo, hi, 121)
    cnt, _ = np.histogram(v, edges)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(x))])
    mbin = np.diff(np.interp(edges, x, cdf)) / np.diff(edges)
    # FFT ringing can leave the far-tail bins of the model at or below zero;
    # they carry no model prediction to draw or to divide by
    mbin = np.where(mbin > 0, mbin, np.nan)
    fig, ax, rax = rp.make_ratio_fig()
    ctr_ = 0.5 * (edges[1:] + edges[:-1])
    w = np.diff(edges)
    ax.errorbar(ctr_, cnt / (len(v) * w), np.sqrt(cnt) / (len(v) * w), fmt="o",
                ms=3, color="black", label=f"Geant4, {len(v)} pairs")
    ax.plot(ctr_, mbin, color="tab:red", lw=1.5, label="model (CF product)")
    ax.set_yscale("log")
    ax.set_ylim(top=30 * float(np.nanmax(mbin)))
    ax.set_ylabel("density [1/MeV]")
    ax.legend(loc="upper left", fontsize="small", title_fontsize="small",
              title=f"${DEC['tex']}$ vertex mass from plane {k} (r = {r:.1f} cm)"
                    + (", elastic on" if rc.ARM == "elonly" else ""))
    rp.draw_ratio(rax, edges, cnt, mbin, len(v), ylabel="sim / model",
                  xlabel=f"$m - m_{{{DEC['tex']}}}$ [MeV]")
    d = pubhtml.figdir(DEC["fig"])
    os.makedirs(d, exist_ok=True)
    pubhtml.savefig(fig, os.path.join(d, f"vertexmass{variant()}_plane{k:02d}.pdf"))
    import matplotlib.pyplot as plt
    plt.close(fig)


SUMMARY_RUNS = {
    # J/psi: the linear, independent-channel model against the knock-on one
    "jpsi": [("linear model", "ditrack_closure.json", "0.55", "o"),
             ("knock-on joint law + exact 1/p", "ditrack_closure_kj1_qx1.json",
              "tab:red", "s")],
    # hadrons: the clean arm, and the elastic arm with the prediction's
    # elastic channel off and on
    "hadron": [("clean sim (no nuclear)", "ditrack_closure_kj1_qx1.json",
                "0.45", "o"),
               ("elastic sim, model without elastic",
                "ditrack_closure_elonly_nucel0_mat_kj1_qx1.json", "tab:blue", "^"),
               ("elastic sim, elastic channel, angle and recoil independent",
                "ditrack_closure_elonly_nucel1_nojoint_mat_kj1_qx1.json", "tab:red", "s"),
               ("elastic sim, elastic channel (joint angle-recoil law)",
                "ditrack_closure_elonly_nucel1_mat_kj1_qx1.json", "tab:green", "D")],
}


def cmd_summary(args):
    """Per-plane summary figures, one file per panel, from the JSON files
    `closure` wrote for the configured decay (SUMMARY_RUNS)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pubhtml
    runs = SUMMARY_RUNS["jpsi" if DECAY == "jpsi" else "hadron"]
    he = hb._canonical().H_ELOSS
    if he != 1.0:
        runs = [(lab, f.replace(".json", f"_hel{he:g}.json"), c, m)
                for lab, f, c, m in runs]
    runs = [(lab, json.load(open(os.path.join(RES, f))), c, m)
            for lab, f, c, m in runs if os.path.exists(os.path.join(RES, f))]
    u = list(runs[0][1]["config"]["ucurve"])
    iu, i3 = u.index(1.0), u.index(0.001)
    d = pubhtml.figdir(DEC["fig"])
    os.makedirs(d, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="gray", lw=1)
    for n, (lab, J, c, m) in enumerate(runs):
        P = J["planes"]
        r = np.array([x["r"] for x in P]) + 0.6 * n
        ax.errorbar(r, [x["even"][iu] for x in P], [x["even_err"][iu] for x in P],
                    fmt=m, color=c, label=lab)
    ax.set_xlabel("radius of the plane [cm]")
    ax.set_ylabel(r"$\langle e^{-uz^2}\rangle_{sim} - \langle e^{-uz^2}\rangle_{model}$, u = 1")
    ax.set_title(f"${DEC['tex']}$ vertex mass from one plane: shape",
                 fontsize="medium")
    if DECAY == "jpsi":
        ax.set_ylim(-0.002, 0.006)
    ax.legend(fontsize="small", loc="best")
    pubhtml.savefig(fig, os.path.join(d, "summary_shape_u1.pdf"))
    plt.close(fig)

    for tag, key, title, loc in (("core", iu, "core (u = 1)", "upper left"),
                                 ("mean", None, "mean", "lower left"),
                                 ("tail", i3, "u = 0.001", "lower left")):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(0, color="gray", lw=1)
        for n, (lab, J, c, m) in enumerate(runs):
            P = J["planes"]
            r = np.array([x["r"] for x in P]) + 0.6 * n
            if key is None:
                y = [x["sim_mean_MeV"] - x["model_mean_MeV"] for x in P]
                e = [x["sim_mean_err_MeV"] for x in P]
            else:
                y = [x["dm_MeV"][key] for x in P]
                e = [x["dm_err_MeV"][key] for x in P]
            ax.errorbar(r, y, e, fmt=m, color=c, label=lab)
        ax.set_xlabel("radius of the plane [cm]")
        ax.set_ylabel(r"sim $-$ model [MeV]")
        ax.set_title(f"${DEC['tex']}$ vertex mass from one plane: location, "
                     + title, fontsize="medium")
        ax.legend(fontsize="small", loc=loc if DECAY == "jpsi" else "best")
        pubhtml.savefig(fig, os.path.join(d, f"summary_location_{tag}_MeV.pdf"))
        plt.close(fig)
    print(f"-> {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decay", default="jpsi", choices=sorted(DECAYS))
    ap.add_argument("--arm", default="off", choices=sorted(hp.ARMS),
                    help="Geant4 process set of BOTH legs' sims (hadron_probe.ARMS); "
                         "`elonly` = hadElastic ON, inelastic and Decay OFF")
    ap.add_argument("--model-tag", default="",
                    help="the old leg's model-file suffix (`_mat`: the export with "
                         "the per-step material table the elastic channel needs)")
    ap.add_argument("--xstat", action="store_true",
                    help="the extended sample (sim/ plus sim_xstat/, 1.8 M per leg "
                         "and arm): `sim` makes the extension for both legs, the "
                         "analyses read it and write to runs/<res>_xstat/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--force", action="store_true",
                   help="rewrite the dedicated legs' planes (undoes re-centring)")
    s.set_defaults(f=cmd_setup)
    s = sub.add_parser("sim")
    s.add_argument("--events", type=int, default=rc.NEV)
    s.add_argument("--jobs", type=int, default=10)
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--force", action="store_true")
    s.set_defaults(f=cmd_sim)
    sub.add_parser("live").set_defaults(f=cmd_live)
    s = sub.add_parser("export")
    s.add_argument("--force", action="store_true")
    s.add_argument("--leg", default=None, help="one dedicated leg; default all")
    s.set_defaults(f=cmd_export)
    s = sub.add_parser("pairs")
    s.add_argument("--nofail", action="store_true")
    s.set_defaults(f=cmd_pairs)
    s = sub.add_parser("closure")
    s.add_argument("--planes", type=int, nargs="*", default=None)
    s.add_argument("--plot", type=int, nargs="*", default=[0, 9, 18])
    s.add_argument("--nucel", action="store_true",
                   help="nuclear-elastic channel (recoil, per-element targets) "
                        "ON in the prediction")
    s.add_argument("--only", default=None,
                   help="one leg's share of the mass direction (the other "
                        "leg's weights set to zero)")
    s.set_defaults(f=cmd_closure)
    sub.add_parser("summary").set_defaults(f=cmd_summary)
    s = sub.add_parser("recentre")
    s.add_argument("--iterations", type=int, default=4)
    s.add_argument("--tol", type=float, default=1e-4)
    s.set_defaults(f=cmd_recentre)
    s = sub.add_parser("legclosure")
    s.add_argument("--leg", default=None, help="leg key; default the new leg")
    s.add_argument("--funcs", nargs="+", default=list(rc.FUNCS))
    s.add_argument("--nucel", action="store_true")
    s.set_defaults(f=cmd_legclosure)
    a = ap.parse_args()
    global NUCEL, XSTAT, RES
    configure(a.decay)
    rc.ARM = a.arm
    rc.MODEL_TAG = a.model_tag
    NUCEL = bool(getattr(a, "nucel", False))
    XSTAT = a.xstat
    if XSTAT:
        RES += "_xstat"
    a.f(a)


if __name__ == "__main__":
    main()
