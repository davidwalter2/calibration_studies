#!/usr/bin/env python3
"""Ditrack clean-propagation closure on the real-material toy: a deterministic
J/psi -> mu+ mu- pair, the vertex mass reconstructed from ONE plane.

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

THE CONFIGURATION
-----------------
mu- : realmat_closure.py's realmat_full sample and model (pT 3, eta 0.30,
      phi 0.70, arm off, seeds 101-110, model_mum_pt3_all4.root) -- reused.
mu+ : pT 3, eta 0.30, phi 0.70 + DPHI, DPHI solved so that the pair mass is
      the J/psi mass.  Own plane set (gen_toy_realmat.helix_frames, q = +1,
      phi0 = 0.70 + DPHI), same tracker.xml, seeds 201-210 -- never the mu-
      seeds, which would replay the same random streams in both legs.
The cylinders are azimuthally symmetric but the field is the real 3D map, so
the mu+ leg is simulated and exported at its own azimuth rather than taken
from the phi = 0.70 mu+ sample by a rotation (`pairs` prints the difference).

SUBCOMMANDS
    setup    private area `qpj`: geometry, planes, drivers (gun azimuth from TOY_PHI)
    sim      the mu+ leg, seeded split
    live     provenance, gun azimuth, switch census, acceptance
    export   the mu+ model
    pairs    model/sim pair gate for both legs, plus the rotation comparison
    closure  per-plane even + odd closure of the vertex mass -> JSON + figures
    summary  per-plane shape and location figures from the JSON
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
import curv2local as c2l                                         # noqa: E402

M_JPSI = 3.0969                     # GeV (PDG: 3096.900 +- 0.006 MeV)
MMU = hp.SPECIES[13]["mass"] * 1e-3
LOCAL = hb.LOCAL                    # (qop, dxdz, dydz, locx, locy)
REFB = {"qop": "refqop", "dxdz": "refdxdz", "dydz": "refdydz",
        "locx": "reflocx", "locy": "reflocy"}


def dphi_for_mass(m, pt=rc.PT, eta=rc.ETA, mmu=MMU):
    """Azimuthal opening of two muons of equal pT and eta with pair mass m:
    m^2 = 2 mmu^2 + 2 (E^2 - pT^2 cos dphi - pz^2)."""
    e2 = (pt * math.cosh(eta)) ** 2 + mmu ** 2
    pz2 = (pt * math.sinh(eta)) ** 2
    return math.acos((e2 - pz2 - 0.5 * (m * m - 2.0 * mmu * mmu)) / (pt * pt))


DPHI = dphi_for_mass(M_JPSI)
PHI_P = rc.PHI + DPHI
GQPJ = "qpj"
rc.GEOM[GQPJ] = ("realmat_full_qpj", "toyPlanes_realmat_full_qpj_pt3", +1.0)
SEEDS_P = list(range(201, 211))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                   "realmat_ditrack_260926")


def variant():
    """File tag of the model's knock-on state (cf_knockon): '' with both
    switches off (the linear, independent-channel model), else
    `_kj{0,1}_qx{0,1}`."""
    import cf_knockon as ck
    if not ck.active():
        return ""
    return f"_kj{int(bool(ck.KNOCKON_JOINT))}_qx{int(bool(ck.QOP_EXACT))}"

# leg -> (pdg, charge, azimuth, geometry key)
LEGS = {"mum": (13, -1.0, rc.PHI, "qm"), "mup": (-13, +1.0, PHI_P, GQPJ)}


def sim_path(leg, seed):
    if leg == "mum":
        return rc.sim_path(13, seed)
    return os.path.join(rc.OUT, "sim", f"mup_jpsi_pt3_{rc.ARM}_s{seed}_sim.root")


def sim_glob(leg):
    if leg == "mum":
        return rc.sim_glob(13)
    return os.path.join(rc.OUT, "sim", f"mup_jpsi_pt3_{rc.ARM}_s2??_sim.root")


def model_path(leg):
    if leg == "mum":
        return rc.model_path(13)
    return os.path.join(rc.OUT, "model",
                        f"model_mup_jpsi_pt3_all4{rc.MODEL_TAG}.root")


def planes_path(leg):
    return rc.planes_path(LEGS[leg][3])


# =========================================================================
# kinematics
# =========================================================================

def p3(qop, lam, phi):
    p = 1.0 / abs(qop)
    return p * np.array([np.cos(lam) * np.cos(phi), np.cos(lam) * np.sin(phi),
                         np.sin(lam)])


def start_curv(leg):
    """(q/p, lambda, phi) of the leg's start state, from the gun settings."""
    _, q, phi, _ = LEGS[leg]
    return np.array([q / (rc.PT * math.cosh(rc.ETA)),
                     math.atan(math.sinh(rc.ETA)), phi])


def pair_mass(c1, c2):
    """Pair mass from two (q/p, lambda, phi) arrays of shape (3,) or (3, n)."""
    def mom(c):
        p = 1.0 / np.abs(c[0])
        return p * np.array([np.cos(c[1]) * np.cos(c[2]),
                             np.cos(c[1]) * np.sin(c[2]), np.sin(c[1])])
    a, b = mom(np.asarray(c1)), mom(np.asarray(c2))
    ea = np.sqrt((a ** 2).sum(axis=0) + MMU ** 2)
    eb = np.sqrt((b ** 2).sum(axis=0) + MMU ** 2)
    return np.sqrt((ea + eb) ** 2 - ((a + b) ** 2).sum(axis=0))


def mass_gradient():
    """dm/d(q/p, lambda, phi) of each leg at the reference, analytic:
    dm = (E_other p_this/E_this - p_other) . dp_this / m."""
    c = {lg: start_curv(lg) for lg in LEGS}
    m0 = pair_mass(c["mum"], c["mup"])
    out = {}
    for lg, oth in (("mum", "mup"), ("mup", "mum")):
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
        ea = math.sqrt(p * p + MMU ** 2)
        eb = math.sqrt(pb @ pb + MMU ** 2)
        v = (eb * pa / ea - pb) / m0
        out[lg] = np.array([v @ (-q * p * p * n), v @ (p * dn_dlam),
                            v @ (p * dn_dphi), 0.0, 0.0])
    return m0, out


def _check_gradient():
    """Finite-difference check of mass_gradient (central, relative 1e-6)."""
    m0, g = mass_gradient()
    c = {lg: start_curv(lg) for lg in LEGS}
    worst = 0.0
    for lg, oth in (("mum", "mup"), ("mup", "mum")):
        for i in range(3):
            h = 1e-6 * max(abs(c[lg][i]), 1e-3)
            cp, cm = c[lg].copy(), c[lg].copy()
            cp[i] += h
            cm[i] -= h
            if lg == "mum":
                d = (pair_mass(cp, c[oth]) - pair_mass(cm, c[oth])) / (2 * h)
            else:
                d = (pair_mass(c[oth], cp) - pair_mass(c[oth], cm)) / (2 * h)
            worst = max(worst, abs(d - g[lg][i]) / max(abs(d), 1e-12))
    return m0, g, worst


# =========================================================================
# setup / sim / export / live / pairs  (the mu+ leg)
# =========================================================================

def cmd_setup(args):
    m0, g, worst = _check_gradient()
    print(f"DPHI = {DPHI:.12f} rad  -> mu+ at phi = {PHI_P:.12f}")
    print(f"pair mass of the two start states {1e3 * m0:.6f} MeV "
          f"(target {1e3 * M_JPSI:.3f}); analytic vs FD gradient max rel "
          f"diff {worst:.1e}")
    if abs(m0 - M_JPSI) > 1e-9 or worst > 1e-6:
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
    gdir, pmod, q = rc.GEOM[GQPJ]
    td = rc.testdir(GQPJ)
    dd = os.path.join(rc.geomdir(GQPJ), "Analysis", "HitAnalyzer", "data", gdir)
    os.makedirs(td, exist_ok=True)
    os.makedirs(dd, exist_ok=True)
    xo = os.path.join(dd, "tracker.xml")
    import shutil
    shutil.copyfile(ref_xml, xo)
    assert rc._md5(xo) == rc._md5(ref_xml)
    print(f"[{GQPJ}] {rc._md5(xo)}  {gdir}/tracker.xml  IDENTICAL to the reference")
    out = (f"# generated by realmat_ditrack.py setup: gen_toy_realmat."
           f"helix_frames(q=+1, phi0={PHI_P!r}) on the radii of "
           f"toyPlanes_realmat_full_pt3.py; do not hand edit\n"
           + rc._plane_body(radii, +1.0, phi0=PHI_P))
    open(rc.planes_path(GQPJ), "w").write(out)
    print(f"[{GQPJ}] {rc._md5(rc.planes_path(GQPJ))}  {pmod}.py")
    s, m = rc._driver_sources()
    s = hp._sub1(s, r"MinPhi=cms\.double\(0\.70\), MaxPhi=cms\.double\(0\.70\),",
                 'MinPhi=cms.double(float(os.environ["TOY_PHI"])),\n'
                 '        MaxPhi=cms.double(float(os.environ["TOY_PHI"])),',
                 "sim: gun azimuth")
    s += ("\nprint('[toy] gun phi = %.12f'\n"
          "      % process.generator.PGunParameters.MinPhi.value())\n")
    open(os.path.join(td, "runToyGeomCheck.py"), "w").write(s)
    open(os.path.join(td, "runToyModel.py"), "w").write(m)
    print(f"[{GQPJ}] wrote runToyGeomCheck.py and runToyModel.py in {td}")
    # the new planes must be the phi = 0.70 mu+ planes rotated by DPHI
    a, b = {}, {}
    exec(open(rc.planes_path("qp")).read(), a)
    exec(open(rc.planes_path(GQPJ)).read(), b)
    c, s_ = math.cos(DPHI), math.sin(DPHI)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    for key in ("origin", "normal", "uaxis"):
        u = np.asarray(a[key]).reshape(-1, 3) @ R.T
        v = np.asarray(b[key]).reshape(-1, 3)
        print(f"rotation check {key:<6}: max |R(DPHI) qp - qpj| = "
              f"{np.abs(u - v).max():.2e}")


def _env_p(extra=None):
    sp = hp.SPECIES[-13]
    e = dict(TOY_PLANES_MOD=rc.GEOM[GQPJ][1], TOY_PDG="-13", TOY_PNAME=sp["g4"],
             TOY_RMAX="107.0", TOY_CUT=repr(rc.CUT),
             TOY_INACT=",".join(hp.inact_of(-13, rc.ARM)), TOY_PHI=repr(PHI_P))
    e.update(extra or {})
    return e


def _sim_one(a):
    seed, nev, force = a
    out = sim_path("mup", seed)
    log = out[:-5] + ".log"
    if os.path.exists(out) and rc._complete(log, nev) and not force:
        return seed, 0, log, "cached"
    ee = _env_p({"TOY_CENSUS": out[:-9] + "_census.bin"})
    r = rc._run(GQPJ, "runToyGeomCheck.py",
                f"events={nev} pt={rc.PT} eta={rc.ETA} output={out} seed={seed} "
                f"toyGeom={rc.toygeom(GQPJ)}", log, ee)
    if r == 0 and not rc._complete(log, nev):
        r = 99
    return seed, r, log, "ran"


def cmd_sim(args):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.join(rc.OUT, "sim"), exist_ok=True)
    jobs = [(s, args.events, args.force) for s in SEEDS_P[:args.jobs]]
    print(f"{len(jobs)} mu+ jobs x {args.events} events at phi = {PHI_P:.6f}",
          flush=True)
    t0, bad = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for seed, r, log, how in ex.map(_sim_one, jobs):
            print(f"  mu+ seed={seed} rc={r} {how:<6} {time.time() - t0:7.0f} s",
                  flush=True)
            if r:
                bad += 1
                print(f"    *** FAILED, see {log}", flush=True)
    if bad:
        raise SystemExit(f"{bad} jobs failed")


def cmd_export(args):
    os.makedirs(os.path.join(rc.OUT, "model"), exist_ok=True)
    out = model_path("mup")
    log = out[:-5] + ".log"
    if os.path.exists(out) and not args.force:
        print(f"exists -> {out}")
        return
    r = rc._run(GQPJ, "runToyModel.py",
                f"pt={rc.PT} eta={rc.ETA} phi={PHI_P!r} partId=-13 output={out} "
                f"toyGeom={rc.toygeom(GQPJ)}", log, _env_p(rc.FOUR_ON))
    print(f"rc={r} -> {out}", flush=True)
    if r:
        raise SystemExit(f"export failed, see {log}")
    t = open(log, errors="ignore").read()
    eff = re.search(r"\[cvh\] effective: (.*)", t)
    print(f"    {eff.group(0) if eff else '*** no [cvh] effective line ***'}")


_SIMC = {}


def load_sim(leg):
    if leg not in _SIMC:
        from toy_loader import load_toy_sim
        ns = {}
        exec(open(planes_path(leg)).read(), ns)
        _SIMC[leg] = load_toy_sim(sim_glob(leg), ns["origin"], ns["normal"],
                                  ns["uaxis"])
    return _SIMC[leg]


def cmd_live(args):
    logs = sorted(glob.glob(sim_glob("mup")[:-5] + ".log"))
    print(f"### mu+ (J/psi leg)  {len(logs)} logs")
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
    if st != "TIGHT" or int(pid) != -13 or abs(float(phi) - PHI_P) > 1e-9 \
            or rc.GEOM[GQPJ][1] not in pl:
        raise SystemExit("wrong stepper, particle, azimuth or plane set")
    for nm in hp.inact_of(-13, rc.ARM):
        got = tot.get(nm, (0, 0))
        bad = got[0] != 0 or (nm != "Decay" and got[1] != 0)
        print(f"    census {nm:<24} primary {got[0]:>9} all {got[1]:>9}  "
              f"{'*** NONZERO ***' if bad else 'OK (primary zero)'}")
        if bad:
            raise SystemExit(f"{nm} is not switched off")
    v = load_sim("mup")["valid"]
    print(f"    events {v.shape[0]}, reached every plane "
          f"{100 * v.all(axis=1).mean():.3f} %, outermost {100 * v[:, -1].mean():.3f} %")


def cmd_pairs(args):
    """realmat_closure's pair gate for both legs, then the rotation comparison
    of the new mu+ leg with the phi = 0.70 mu+ (informational)."""
    for leg in ("mum", "mup"):
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
              and 0.0 < dE_m - s["dE"] < 6.0)
        print(f"--- {leg}  model {os.path.basename(mp)}  sim {len(files)} files, "
              f"{sim['valid'].shape[0]} events")
        print(f"    (1) legs {len(m['detid'])} vs sim planes {s['npl']}  "
              f"(2) sequence identical {seqok}")
        print(f"    (3) max |refglobr - sim median r| = {dr:.5f} cm")
        print(f"    (4) dE model(mean) {dE_m:.3f} MeV  sim(median, first file) "
              f"{s['dE']:.3f} MeV  gap {dE_m - s['dE']:+.3f} MeV")
        print(f"    (5) max |median(sim locx) - reflocx| = "
              f"{1e4 * np.abs(dxs).max():.1f} um")
        print(f"    {'PASS' if ok else '*** FAIL ***'}")
        if not ok and not args.nofail:
            raise SystemExit("pair gate failed")
    # the same leg at phi = 0.70: material identical by construction, the
    # field map is the only difference
    a = cpt.load_model(rc.model_path(-13))
    b = cpt.load_model(model_path("mup"))
    rp = np.array([[l["refp"] for l in a], [l["refp"] for l in b]])
    print(f"rotation: refp(phi={PHI_P:.3f}) / refp(phi=0.70) - 1 per plane: "
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
    """realmat_closure.cell's configuration for muons, asserted."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    assert (ctr.MS_ELEC_TMAX, ctr.MS_ELEC_EDGE, ctr.MS_SNAP_YMAX, ctr.MS_FINE_G,
            ctr.MS_WVI_SPLIT) == (1.0, 1.0, 0.0, 1.0, 0.0), "MS defaults moved"
    assert mx.MS_CHI0_G4 and mx.MS_FF_G4, "cf_ms_exact harmonisations off"
    assert rc.ARM == "off" and rc.MODEL_TAG == ""
    cne.NUCEL_CHANNEL = False
    cpt.RAD_CHANNEL = True
    ctr.IONI_KOKOULIN = 1.0
    ctr.IONI_KOKOULIN_TCUT = 0.0


def leg_vectors(legs, g):
    """Per plane: curvilinear a_k = P_k^{-T} g and local b_k = H_k^{-T} a_k."""
    _, mass, bfield = hb.bound(legs)
    P = np.eye(5)
    out = []
    for k, leg in enumerate(legs):
        P = leg["F"] @ P
        a = np.linalg.solve(P.T, g)
        H, _ = c2l.leg_H(legs, k, bfield=bfield, mass=mass)
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
    assert abs(m0 - M_JPSI) < 1e-9 and worst < 1e-6
    legs = {}
    for leg, (pdg, _, _, _) in LEGS.items():
        path = model_path(leg)
        legs[leg] = cpt.load_model(path)
        hb.bind(legs[leg], path, pdg=pdg)
        cne.warm(legs[leg])
    vec = {leg: leg_vectors(legs[leg], g[leg]) for leg in LEGS}
    nl = min(len(legs["mum"]), len(legs["mup"]))
    global _CTX
    _CTX = (legs["mum"], legs["mup"], vec["mum"], vec["mup"], probes)
    t0 = time.time()
    ks = list(range(nl)) if args.planes is None else args.planes
    models = dict(zip(ks, fn.pmap(_plane_model, ks)))
    print(f"model: {len(ks)} planes in {time.time() - t0:.0f} s", flush=True)

    sims = {leg: load_sim(leg) for leg in LEGS}
    nev = min(s["valid"].shape[0] for s in sims.values())
    c0 = {leg: start_curv(leg) for leg in LEGS}
    iu1 = list(probes).index(1.0)
    rows = []
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
        dmt = dm["mum"] + dm["mup"]
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
        corr = float(np.corrcoef(dm["mum"][good], dm["mup"][good])[0, 1])
        vt = float(np.var(dmt[good]))
        fq = float(np.var((dm["mum_qop"] + dm["mup_qop"])[good]) / vt)
        fa = float(np.var((dm["mum_ang"] + dm["mup_ang"])[good]) / vt)
        # second order of the mass map: exact pair mass of the back-propagated
        # momenta minus the linear prediction
        cm = c0["mum"][:, None] + dcurv0["mum"][:3][:, good]
        cp = c0["mup"][:, None] + dcurv0["mup"][:3][:, good]
        nlin = 1e3 * float(np.mean(pair_mass(cm, cp) - m0 - dmt[good]))
        r = float(legs["mum"][k]["refglobr"])
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
                         a_mum=vec["mum"][k][0].tolist(),
                         a_mup=vec["mup"][k][0].tolist(),
                         b_mum=vec["mum"][k][1].tolist(),
                         b_mup=vec["mup"][k][1].tolist()))
        if k in args.plot:
            _plot(k, r, dmt[good], md)
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
    out = dict(config=dict(knockon=dict(ck.physics_state()),
                           pt=rc.PT, eta=rc.ETA, phi_mum=rc.PHI, phi_mup=PHI_P,
                           dphi=DPHI, m_jpsi=M_JPSI, arm=rc.ARM,
                           model_mum=model_path("mum"), model_mup=model_path("mup"),
                           sim_mum=sim_glob("mum"), sim_mup=sim_glob("mup"),
                           nev=nev, ucurve=list(probes)),
               ladder=dict(even=lad_e.tolist(), even_err=err_lad_e.tolist(),
                           odd=lad_o.tolist(), odd_err=err_lad_o.tolist()),
               planes=rows)
    full = args.planes is None
    p = os.path.join(RES, (f"ditrack_closure{variant()}.json" if full else
                           f"ditrack_closure{variant()}_planes_"
                           + "_".join(map(str, ks)) + ".json"))
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")


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
    fig, ax, rax = rp.make_ratio_fig()
    ctr_ = 0.5 * (edges[1:] + edges[:-1])
    w = np.diff(edges)
    ax.errorbar(ctr_, cnt / (len(v) * w), np.sqrt(cnt) / (len(v) * w), fmt="o",
                ms=3, color="black", label=f"Geant4, {len(v)} pairs")
    ax.plot(ctr_, mbin, color="tab:red", lw=1.5, label="model (CF product)")
    ax.set_yscale("log")
    ax.set_ylim(top=30 * float(np.max(mbin)))
    ax.set_ylabel("density [1/MeV]")
    ax.legend(loc="upper left", fontsize="small", title_fontsize="small",
              title=f"$J/\\psi$ vertex mass from plane {k} (r = {r:.1f} cm)")
    rp.draw_ratio(rax, edges, cnt, mbin, len(v), ylabel="sim / model",
                  xlabel=r"$m - m_{J/\psi}$ [MeV]")
    d = pubhtml.figdir("ditrack_cleanprop")
    os.makedirs(d, exist_ok=True)
    pubhtml.savefig(fig, os.path.join(d, f"vertexmass{variant()}_plane{k:02d}.pdf"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def cmd_summary(args):
    """Per-plane summary figures, one file per panel: the linear,
    independent-channel model (baseline) against the knock-on-corrected one
    (both cf_knockon switches on), from the JSON files `closure` wrote."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pubhtml
    runs = [("linear model", "ditrack_closure.json", "0.55", "o"),
            ("knock-on joint law + exact 1/p", "ditrack_closure_kj1_qx1.json",
             "tab:red", "s")]
    runs = [(lab, json.load(open(os.path.join(RES, f))), c, m)
            for lab, f, c, m in runs if os.path.exists(os.path.join(RES, f))]
    u = list(runs[0][1]["config"]["ucurve"])
    iu, i3 = u.index(1.0), u.index(0.001)
    d = pubhtml.figdir("ditrack_cleanprop")
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
    ax.set_title(r"$J/\psi$ vertex mass from one plane: shape", fontsize="medium")
    ax.set_ylim(-0.002, 0.006)
    ax.legend(fontsize="small", loc="upper right")
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
        ax.set_title(r"$J/\psi$ vertex mass from one plane: location, " + title,
                     fontsize="medium")
        ax.legend(fontsize="small", loc=loc)
        pubhtml.savefig(fig, os.path.join(d, f"summary_location_{tag}_MeV.pdf"))
        plt.close(fig)
    print(f"-> {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(f=cmd_setup)
    s = sub.add_parser("sim")
    s.add_argument("--events", type=int, default=rc.NEV)
    s.add_argument("--jobs", type=int, default=len(SEEDS_P))
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--force", action="store_true")
    s.set_defaults(f=cmd_sim)
    sub.add_parser("live").set_defaults(f=cmd_live)
    s = sub.add_parser("export")
    s.add_argument("--force", action="store_true")
    s.set_defaults(f=cmd_export)
    s = sub.add_parser("pairs")
    s.add_argument("--nofail", action="store_true")
    s.set_defaults(f=cmd_pairs)
    s = sub.add_parser("closure")
    s.add_argument("--planes", type=int, nargs="*", default=None)
    s.add_argument("--plot", type=int, nargs="*", default=[0, 9, 18])
    s.set_defaults(f=cmd_closure)
    sub.add_parser("summary").set_defaults(f=cmd_summary)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
