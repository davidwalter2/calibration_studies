#!/usr/bin/env python3
"""Path-averaged model of the real-tracker J/psi ditrack, WITH each path's shift.

THE QUESTION. On the real tracker the one-plane vertex mass of the J/psi
ditrack (realmat_ditrack.py --decay jpsi_real) closes on planes 0-7 and sits
at -0.57 / -0.70 MeV on planes 16-17, from the q/p slot of both legs.  The
model is built along ONE fixed reference; the simulated tracks spread over
millimetres and cross different material.  Is the location purely that
material sampling (which the CVH fit does not have: its reference follows the
hits)?

THE PATH ENSEMBLES (`design`, `export`, `--ens`). Per leg, reference paths
from perturbed start states, |p| fixed, each a runCleanPropModel export on the
leg's own target surfaces (the fixed reference's), so every path's local state
on every plane is defined relative to the fixed reference's.  Start offsets
are drawn in antithetic pairs (Sobol) from a Gaussian mixture q:

  rot  the start DIRECTION (dlambda, dphi) only, sigma_theta = 0.35, 0.9, 1.6,
       2.6 mrad (4 batches, 384 paths per leg).  A rotated ray spreads
       linearly in r, a scattered track like r^1.6: with the outer planes
       matched, the ray over-spreads the inner strip layers (x1.4 at 62 cm,
       x2.2 at 27 cm).
  cov  direction AND transverse start position (dx_T, dy_T), from
       s^2 C with s = 0.6, 1.0, 1.5 (192 paths per leg); C per transverse
       coordinate fitted (`covfit`) so that the fixed reference's linear
       transport reproduces the sim's sigma_68 on every fully covered plane
       beyond 20 cm (to +-7 % from 27 to 107 cm).  A start offset makes a
       kinked path, x = theta (r - r0), r0 ~ 15-25 cm; the pixel layers are
       then over-spread (0.5 mm against the sim's 3-55 um).

Either way the ensemble is REWEIGHTED PER PLANE,

    w_ij  propto  f_sim,j(dx_ij) / q_j(dx_ij),

with dx_ij the path's local (x, y) offset from the fixed reference on plane
j, f_sim,j the sim's own density of (locx, locy) - (reflocx, reflocy) there
(Gaussian KDE, bandwidth 0.1 sigma_68 per axis) and q_j the proposal's density
of that offset (the mixture pushed through the linear map fitted to the
paths): importance sampling of the target with the sim's marginal on plane j
and the proposal's conditional of the start given it.  The weighted ensemble
reproduces the sim's transverse distribution -- core and tails -- on plane j
(`validate`).  q/p on plane k accumulates the loss of every segment j <= k,
and a path weighted for plane k keeps the family's spread on the inner
segments; the MEAN is therefore also built segment by segment (`composite`:
each segment's increment with its own plane's weights), which has the sim's
marginal on every segment and ignores only the correlation between segments,
on which the mean does not depend.  MC errors take the antithetic pair as the
independent unit.

THE MODEL (`analyse`). Per path i, leg L and plane k, the path's shift in the
mass functional

    s_ik = b_k . (x_ref,i - x_ref,fixed)_k  -  g . dc0_i

(the q/p slot in the log variable cf_knockon.qop_dev; b_k the fixed
reference's local mass vector; g . dc0_i removes the injected start direction,
whose linear image b_k H_k P_k dc0 = g . dc0 is exactly the path's offset
from its own start and not a material effect -- the sim's tracks all start at
the fixed vertex state; the variant with it kept is printed too).  The
path-averaged pair CF on the closure grid, at the fixed reference's s_F:

    shift-only   f(t) = phi_fixed(t) prod_L sum_i w_i^L e^{i t s_i^L / s_F}
    full (--own-cf, selected planes)
                 f(t) = prod_L sum_i w_i^L phi_i^L(t) e^{i t s_i^L / s_F}

phi_i^L the model CF of the mass functional centred on path i (its own
material, its own transport a-vector).  Then cmd_closure's statistics: the
model's location relative to the fixed model, 1e3 s_F (odd_pa - odd_fix)/D,
against the sim's dm(u); and the sim's residual location against the
path-averaged model.  Per leg, the single-track q/p location: the sim's mean
qop_dev against sum_i w_ik qop_dev(path_i).

usage (the calibration_studies env; RES_NO_PHI_CACHE=1 is set here):
    python cleanprop/realtrk_path_average.py fixedcf
    python cleanprop/realtrk_path_average.py design --batches 4
    python cleanprop/realtrk_path_average.py --ens cov covfit
    python cleanprop/realtrk_path_average.py --ens cov design
    python cleanprop/realtrk_path_average.py [--ens cov] export --workers 24
    python cleanprop/realtrk_path_average.py [--ens cov] validate
    python cleanprop/realtrk_path_average.py [--ens cov] analyse --own-cf 16 17
    python cleanprop/realtrk_path_average.py plot
The paths' own CFs run in SPAWNED workers (a forked one inherits uproot's
reader locks and deadlocks) and are cached per (path, plane).
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("RES_NO_PHI_CACHE", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import realmat_ditrack as rd                                      # noqa: E402
from realmat_closure import cpt, ctr, fn, hb                      # noqa: E402
import realmat_closure as rc                                      # noqa: E402
import cf_knockon as ck                                           # noqa: E402

rd.configure("jpsi_real")
rc.ARM = "off"

OUT0 = os.path.join(rd.REAL_CEPH, "pathavg_jpsireal")
RES = os.path.join(os.path.dirname(HERE), "runs", "realtrk_pathavg_jpsi_260929")
# the ensemble (`--ens`): `rot` rotated rays from the vertex, `cov` rays whose
# start direction AND transverse start position are drawn from the Gaussian
# that matches the sim's cross-plane covariance of (locx, locy)
ENS = "rot"


def OUT():
    return OUT0 if ENS == "rot" else os.path.join(OUT0, ENS)


def RESE():
    return RES if ENS == "rot" else os.path.join(RES, ENS)


def RAYS():
    return os.path.join(OUT(), "rays.json")


# the proposal: sigma_theta [rad] and number of paths per component (even:
# antithetic pairs), per leg
COMPONENTS = ((0.35e-3, 12), (0.9e-3, 20), (1.6e-3, 28), (2.6e-3, 36))
# `cov`: scale of the fitted covariance and number of paths per component
COV_COMPONENTS = ((0.6, 24), (1.0, 96), (1.5, 72))
COV_RMIN = 20.0        # cm: the covariance fit uses the planes beyond
KDE_FRAC = 0.1          # KDE bandwidth / sigma_68 per axis
# planes where a leg's module is only partially covered by the ensemble (the
# one-plane mass there is a selection bias, realmat_ditrack's section)
PARTIAL = (8, 9, 10, 11, 15)


# =========================================================================
# the ensemble
# =========================================================================

def _start(leg):
    L = rd.LEGS[leg]
    lam0 = math.atan(math.sinh(rc.ETA))
    return lam0, L["phi"], L["pt"] * math.cosh(rc.ETA)


def cmd_design(args):
    """Path 0 (the fixed reference) and batch 0 of the proposal; `--batches B`
    appends batches 1..B-1 (same mixture, new Sobol seeds), so the union is
    still a sample of q and existing paths keep their numbers."""
    from scipy.stats import qmc, norm
    if ENS == "cov":
        return _design_cov(args)
    os.makedirs(OUT(), exist_ok=True)
    old = load_rays()[0] if os.path.exists(RAYS()) else {}
    rays = {}
    for li, leg in enumerate(rd.LEGS):
        lst = [dict(i=0, comp=-1, dlam=0.0, dphi=0.0)]
        for bt, (ci, (sg, n)) in [(bt, c) for bt in range(args.batches)
                                  for c in enumerate(COMPONENTS)]:
            s = qmc.Sobol(d=2, scramble=True,
                          seed=1000 * li + ci + 100 * bt).random(n // 2)
            z = norm.ppf(np.clip(s, 1e-12, 1 - 1e-12))
            for sgn in (1.0, -1.0):
                for zz in z:
                    lst.append(dict(i=len(lst), comp=ci, dlam=float(sgn * sg * zz[0]),
                                    dphi=float(sgn * sg * zz[1])))
        rays[leg] = lst
        if leg in old and old[leg] != lst[:len(old[leg])]:
            raise SystemExit(f"{leg}: the design would renumber existing paths")
        print(f"{leg}: {len(lst)} paths (path 0 = the fixed reference)")
    json.dump(dict(kind="rot", components=COMPONENTS, rays=rays), open(RAYS(), "w"),
              indent=1)
    print(f"-> {RAYS()}")


# the curvilinear start offset of a path: (q/p, lambda, phi, x_T, y_T)
CURV = ("dqop", "dlam", "dphi", "dxT", "dyT")


def cmd_covfit(args):
    """The start covariance C of (dlambda, dphi, dx_T, dy_T): per transverse
    coordinate a 2x2 block (dphi, dx_T -> locx; dlambda, dy_T -> locy; the
    sim's x-y correlation is <= 0.07), fitted so that the fixed reference's
    linear transport, sigma_k^2 = A_k C A_k^T with A_k the locx (locy) row of
    H_k P_k, reproduces the sim's per-plane sigma_68 in RELATIVE terms,
    sum_k ln^2(sigma_model / sigma_sim), on the fully covered planes with
    r > COV_RMIN (the strip tracker, where the location builds up; the
    pixel layers are then over-spread).  A line with a start offset is a kinked path, x = theta (r - r0):
    its spread grows from zero at r0, which follows the random walk's r^1.6
    far better than a ray from the vertex."""
    from scipy.optimize import minimize
    legs, vec, g, _ = _ctx_fixed()
    out = {}
    for leg in rd.LEGS:
        fx = _meta(rd.model_path(leg))
        s_ = rd.load_sim(leg)
        nk = len(fx)
        acc = s_["valid"].mean(axis=0)
        planes = [k for k in range(nk) if legs[leg][k]["refglobr"] > COV_RMIN and acc[k] > 0.99]
        HP = [vec[leg][k][3] @ vec[leg][k][2] for k in range(nk)]
        C = np.zeros((4, 4))
        rep = {}
        # (row of H P, curvilinear columns, start-offset slots in C, sim branch)
        for comp, row, cols, slots in (("x", 3, (2, 3), (1, 2)), ("y", 4, (1, 4), (0, 3))):
            ref = "reflocx" if comp == "x" else "reflocy"
            ss = np.array([_q68(s_["loc" + comp][s_["valid"][:, k], k] - fx[k][ref])
                           for k in planes])
            A = np.array([HP[k][row, list(cols)] for k in planes])

            def unpack(x):
                Lm = np.array([[x[0], 0.0], [x[1], x[2]]])
                return Lm @ Lm.T

            def obj(x):
                sm = np.sqrt(np.maximum(np.einsum("ki,ij,kj->k", A, unpack(x), A), 1e-30))
                return float((np.log(sm / ss) ** 2).sum())

            best = None
            for th0, x00 in ((1.5e-3, 0.0), (2e-3, -0.02), (2e-3, -0.04), (1e-3, 0.01)):
                r = minimize(obj, np.array([th0, x00, 1e-4]), method="Nelder-Mead",
                             options=dict(maxiter=40000, xatol=1e-9, fatol=1e-12))
                if best is None or r.fun < best.fun:
                    best = r
            Cb = unpack(best.x)
            for a, i in enumerate(slots):
                for b, j in enumerate(slots):
                    C[i, j] = Cb[a, b]
            sm = np.sqrt(np.einsum("ki,ij,kj->k", A, Cb, A))
            rep[comp] = dict(sim=ss.tolist(), model=sm.tolist(), objective=best.fun)
        sd = np.sqrt(np.diag(C))
        print(f"=== {leg}: {len(planes)} fully covered planes (r > {COV_RMIN:g} cm)")
        print("  sigma (dlam, dphi) = " + " ".join(f"{x * 1e3:.3f}" for x in sd[:2])
              + " mrad; (dx_T, dy_T) = " + " ".join(f"{x * 1e4:.1f}" for x in sd[2:])
              + f" um; corr(dphi, dx_T) {C[1, 2] / (sd[1] * sd[2]):+.4f}, "
              f"corr(dlam, dy_T) {C[0, 3] / (sd[0] * sd[3]):+.4f}")
        for i, k in enumerate(planes):
            print(f"  k={k:2d} r={legs[leg][k]['refglobr']:6.2f}  sigma68 x sim "
                  f"{1e4 * rep['x']['sim'][i]:6.0f} model {1e4 * rep['x']['model'][i]:6.0f}"
                  f" | y sim {1e4 * rep['y']['sim'][i]:6.0f} model {1e4 * rep['y']['model'][i]:6.0f}"
                  f" | rotated ray (outer-matched) {1e4 * rep['x']['sim'][-1] * legs[leg][k]['refglobr'] / legs[leg][planes[-1]]['refglobr']:6.0f}")
        out[leg] = dict(C=C.tolist(), planes=planes, fit=rep)
    os.makedirs(os.path.join(RES, "cov"), exist_ok=True)
    p = os.path.join(RES, "cov", "covfit.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")


def _design_cov(args):
    from scipy.stats import qmc, norm
    fit = json.load(open(os.path.join(RES, "cov", "covfit.json")))
    os.makedirs(OUT(), exist_ok=True)
    rays = {}
    for li, leg in enumerate(rd.LEGS):
        C = np.array(fit[leg]["C"])
        Lc = np.linalg.cholesky(C)
        lst = [dict(i=0, comp=-1, dlam=0.0, dphi=0.0, dxT=0.0, dyT=0.0)]
        for ci, (sc, n) in enumerate(COV_COMPONENTS):
            u = qmc.Sobol(d=4, scramble=True, seed=5000 + 1000 * li + ci).random(n // 2)
            z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
            for sgn in (1.0, -1.0):
                for zz in z:
                    d = sgn * sc * (Lc @ zz)
                    lst.append(dict(i=len(lst), comp=ci, dlam=float(d[0]), dphi=float(d[1]),
                                    dxT=float(d[2]), dyT=float(d[3])))
        rays[leg] = lst
        print(f"{leg}: {len(lst)} paths (path 0 = the fixed reference)")
    json.dump(dict(kind="cov", components=COV_COMPONENTS,
                   C={leg: fit[leg]["C"] for leg in rd.LEGS}, rays=rays),
              open(RAYS(), "w"), indent=1)
    print(f"-> {RAYS()}")


def load_rays():
    d = json.load(open(RAYS()))
    return d["rays"], d


def ray_path(leg, i):
    return os.path.join(OUT(), f"model_{leg}_path{i:03d}.root")


def component_covariances(leg, design):
    """[(pi_m, Sigma_m)] of the proposal mixture in the start-offset space
    (dlambda, dphi, dx_T, dy_T)."""
    comps = design["components"]
    ntot = sum(n for _, n in comps)
    if design.get("kind", "rot") == "rot":
        return [(n / ntot, np.diag([sg ** 2, sg ** 2, 0.0, 0.0])) for sg, n in comps]
    C = np.array(design["C"][leg])
    return [(n / ntot, sc ** 2 * C) for sc, n in comps]


def proposal_density(leg, D, design):
    """The mixture density q the paths were drawn from (path 0 excluded);
    D = (npath, 4) start offsets (dlambda, dphi, dx_T, dy_T)."""
    if design.get("kind", "rot") == "rot":
        comps = design["components"]
        ntot = sum(n for _, n in comps)
        q = 0.0
        for sg, n in comps:
            q = q + (n / ntot) * np.exp(-0.5 * (D[:, 0] ** 2 + D[:, 1] ** 2) / sg ** 2) / (
                2 * np.pi * sg ** 2)
        return q
    C = np.array(design["C"][leg])
    Ci = np.linalg.inv(C)
    q2 = np.einsum("ij,jk,ik->i", D, Ci, D)
    comps = design["components"]
    ntot = sum(n for _, n in comps)
    q = 0.0
    for sc, n in comps:
        q = q + (n / ntot) * np.exp(-0.5 * q2 / sc ** 2) / sc ** 4
    return q


def _export_one(job):
    """runCleanPropModel along the leg's targets from the perturbed start
    direction; realmat_ditrack._export_real's command, switches and
    environment (the CMSSW env only, local cwd, copied to ceph at the end)."""
    leg, ray, force = job
    L = rd.LEGS[leg]
    out = ray_path(leg, ray["i"])
    if os.path.exists(out) and not force:
        return leg, ray["i"], 0, "exists"
    lam0, phi0, p0 = _start(leg)
    if ray["i"] == 0:
        kin = f"pt={L['pt']:g} eta={rc.ETA:.2f} phi={L['phistr']}"
    else:
        lam = lam0 + ray["dlam"]
        kin = (f"pt={p0 * math.cos(lam)!r} eta={math.asinh(math.tan(lam))!r} "
               f"phi={phi0 + ray['dphi']!r}")
    # the transverse start offset, curvilinear U = z x T / |z x T| and
    # V = T x U at the unperturbed start direction, as a global position
    x0 = (ray.get("dxT", 0.0) * np.array([-math.sin(phi0), math.cos(phi0), 0.0])
          + ray.get("dyT", 0.0) * np.array([-math.sin(lam0) * math.cos(phi0),
                                            -math.sin(lam0) * math.sin(phi0),
                                            math.cos(lam0)]))
    penv = {f"CLEANPROP_{c}0": repr(float(v)) for c, v in zip("XYZ", x0)}
    swopts, _ = ctr.split_switches(dict(rc.FOUR_ON))
    work = tempfile.mkdtemp(prefix=f"pathavg_{ENS}_{leg}_{ray['i']:03d}_")
    tmp = os.path.join(work, os.path.basename(out))
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {rc.SRC} && eval $(scramv1 runtime -sh) && "
           f"cd {work} && exec cmsRun {HERE}/runCleanPropModelOffset.py "
           f"{kin} partId={L['pdg']} targets={rd.targets_path(leg)} output={tmp} "
           f"useIdealGeometry=True {swopts}")
    wlog = os.path.join(work, "export.log")
    t0 = time.time()
    with open(wlog, "w") as fh:
        fh.write(f"# {cmd}\n")
        fh.flush()
        r = subprocess.run(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT,
                           env=rc.ds._clean_env(penv)).returncode
    shutil.copyfile(wlog, out[:-5] + ".log")
    if r or not os.path.exists(tmp):
        return leg, ray["i"], 1, f"rc={r}"
    shutil.copyfile(tmp, out + ".part")
    os.replace(out + ".part", out)
    shutil.rmtree(work, ignore_errors=True)
    return leg, ray["i"], 0, f"{time.time() - t0:.0f} s"


def cmd_export(args):
    from multiprocessing.pool import ThreadPool
    rays, _ = load_rays()
    jobs = [(leg, r, args.force) for leg in rd.LEGS for r in rays[leg]
            if args.max is None or r["i"] < args.max]
    bad = 0
    with ThreadPool(args.workers) as pool:
        for leg, i, rcode, msg in pool.imap_unordered(_export_one, jobs):
            bad += rcode
            print(f"{leg} path {i:03d}: {'FAILED ' if rcode else ''}{msg}", flush=True)
    if bad:
        raise SystemExit(f"{bad} exports failed")


# =========================================================================
# loading: fixed reference, paths, offsets, weights
# =========================================================================

REFC = [rd.REFB[c] for c in rd.LOCAL]


def _local_offsets(pl, fx):
    """(nplane, 5) local offset of a path from the fixed reference, q/p in the
    map's variable."""
    out = np.zeros((len(fx), 5))
    for k in range(len(fx)):
        out[k, 0] = ck.qop_dev(pl[k]["refqop"], fx[k]["refqop"])
        for c in range(1, 5):
            out[k, c] = pl[k][REFC[c]] - fx[k][REFC[c]]
    return out


def _meta(path):
    import uproot
    f = uproot.open(path)
    tkey = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
    a = f[tkey].arrays(["detid", "ok", "refqop", "refdxdz", "refdydz", "reflocx",
                        "reflocy", "refp"], library="np")
    return [dict(detid=int(a["detid"][k]), ok=bool(a["ok"][k]),
                 **{c: float(a[c][k]) for c in ("refqop", "refdxdz", "refdydz",
                                                "reflocx", "reflocy", "refp")})
            for k in range(len(a["detid"]))]


def load_ensemble():
    """Per leg: the path offsets (npath, nplane, 5) relative to the fixed
    reference, the start perturbations, and the path-0 identity check."""
    rays, design = load_rays()
    ens = {}
    for leg in rd.LEGS:
        fx = _meta(rd.model_path(leg))
        offs, dl, dp, idx, dxy = [], [], [], [], []
        for r in rays[leg]:
            p = ray_path(leg, r["i"])
            if not os.path.exists(p):
                continue
            m = _meta(p)
            if len(m) != len(fx) or not all(x["ok"] for x in m) or \
                    [x["detid"] for x in m] != [x["detid"] for x in fx]:
                print(f"*** {leg} path {r['i']}: bad export (plane count/ok/detid)")
                continue
            o = _local_offsets(m, fx)
            if r["i"] == 0:
                ens.setdefault(leg, {})["identity"] = float(np.abs(o).max())
                continue
            offs.append(o)
            dl.append(r["dlam"])
            dp.append(r["dphi"])
            dxy.append((r.get("dxT", 0.0), r.get("dyT", 0.0)))
            idx.append(r["i"])
        dl, dp, dxy = np.array(dl), np.array(dp), np.array(dxy)
        # the paths start at the gun azimuth, the fixed reference at its
        # rounded spelling (phistr): the start offset g . dc0 uses the latter
        dphi0 = rd.LEGS[leg]["phi"] - float(rd.LEGS[leg]["phistr"])
        # the antithetic pairs (theta, -theta): the unit of the MC error
        D = np.column_stack([dl, dp, dxy])
        key = {tuple(np.round(d, 15)): j for j, d in enumerate(D)}
        pairs = sorted({tuple(sorted((j, key[tuple(np.round(-d, 15))])))
                        for j, d in enumerate(D) if tuple(np.round(-d, 15)) in key})
        ens.setdefault(leg, {}).update(
            offs=np.array(offs), dlam=dl, dphi=dp + dphi0, idx=np.array(idx),
            q=proposal_density(leg, D, design), fixed=fx, pairs=np.array(pairs),
            D=D, comp_cov=component_covariances(leg, design))
        if 2 * len(pairs) != len(idx):
            print(f"*** {leg}: {len(idx) - 2 * len(pairs)} paths without their partner")
    return ens


def sim_offsets(leg, fx):
    s = rd.load_sim(leg)
    nk = s["locx"].shape[1]
    return s, [(s["valid"][:, k],
                s["locx"][:, k] - fx[k]["reflocx"],
                s["locy"][:, k] - fx[k]["reflocy"]) for k in range(nk)]


def _q68(x):
    return 0.5 * (np.percentile(x, 84.1345) - np.percentile(x, 15.8655))


def kde_weights(leg, ens, sim_xy):
    """w[i, k] propto f_sim,k(dx_ik) / q_k(dx_ik), normalised per plane: the
    sim's density of the local (x, y) offset over the PROPOSAL's density of
    it on plane k, q_k = sum_m pi_m N(0, A_k Sigma_m A_k^T), A_k the linear map
    of the start offset onto plane k's (x, y) fitted to the paths themselves.
    This is importance sampling of the target that has the sim's marginal on
    plane k and the proposal's conditional of the start offset given it; for
    the rotated rays (two start variables, two offsets) it is f / q exactly."""
    E = ens[leg]
    npth, nk = E["offs"].shape[:2]
    W = np.zeros((npth, nk))
    for k in range(nk):
        Ak = np.linalg.lstsq(E["D"], E["offs"][:, k, 3:5], rcond=None)[0].T   # (2, 4)
        dxy = E["offs"][:, k, 3:5]
        qk = np.zeros(npth)
        for pi, Sg in E["comp_cov"]:
            Ck = Ak @ Sg @ Ak.T
            Ci = np.linalg.inv(Ck)
            qk += pi * np.exp(-0.5 * np.einsum("ij,jk,ik->i", dxy, Ci, dxy)) / (
                2 * np.pi * math.sqrt(np.linalg.det(Ck)))
        v, x, y = sim_xy[k]
        x, y = x[v], y[v]
        hx, hy = KDE_FRAC * _q68(x), KDE_FRAC * _q68(y)
        px, py = E["offs"][:, k, 3], E["offs"][:, k, 4]
        dens = np.zeros(npth)
        for c0 in range(0, len(x), 50000):
            xx, yy = x[c0:c0 + 50000], y[c0:c0 + 50000]
            dens += np.exp(-0.5 * (((px[:, None] - xx[None]) / hx) ** 2
                                   + ((py[:, None] - yy[None]) / hy) ** 2)).sum(axis=1)
        dens /= len(x) * 2 * np.pi * hx * hy
        w = dens / qk
        W[:, k] = w / w.sum()
    return W


def pair_draw(rng, pairs):
    """Path indices of a bootstrap sample of the antithetic pairs."""
    return pairs[rng.integers(0, len(pairs), len(pairs))].ravel()


def pair_mc_err(w, x, pairs):
    """MC error of the self-normalised weighted mean sum w x (w normalised),
    with the antithetic pair as the independent unit."""
    m = np.sum(w * x)
    return float(np.sqrt(np.sum(np.sum((w * (x - m))[pairs], axis=1) ** 2)))


def comp_mc_err(Wk, inc, pairs):
    """MC error of the segment composite sum_j sum_i W_ij inc_ij (each
    segment's weights normalised), the antithetic pair the independent unit."""
    dev = Wk * (inc - np.sum(Wk * inc, axis=0)[None, :])
    return float(np.sqrt(np.sum(np.sum(dev.sum(axis=1)[pairs], axis=1) ** 2)))


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def _wq68(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return 0.5 * (np.interp(0.841345, c, x[o]) - np.interp(0.158655, c, x[o]))


# =========================================================================
# validate: the weighted ensemble's transverse spread vs the sim's
# =========================================================================

def cmd_validate(args):
    ens = load_ensemble()
    out = {}
    for leg in rd.LEGS:
        E = ens[leg]
        print(f"=== {leg}: {len(E['idx'])} paths; path 0 vs fixed reference: "
              f"max |offset| = {E.get('identity', float('nan')):.3e}")
        s, sxy = sim_offsets(leg, E["fixed"])
        W = kde_weights(leg, ens, sxy)
        rows = []
        print(f"{'k':>2} {'r':>6} {'ESS':>5} | {'sim q68 x':>9} {'pa q68 x':>9} "
              f"{'sim rms x':>9} {'pa rms x':>9} | {'sim q68 y':>9} {'pa q68 y':>9} "
              f"{'sim rms y':>9} {'pa rms y':>9} | {'unw q68 x':>9}  [um]")
        for k in range(W.shape[1]):
            v, x, y = sxy[k]
            w = W[:, k]
            px, py = E["offs"][:, k, 3], E["offs"][:, k, 4]
            r = float(np.nanmedian(s["globr"][v, k]))
            mx_ = float(np.sum(w * px))
            my_ = float(np.sum(w * py))
            row = dict(k=k, r=r, ess=ess(w),
                       sim_q68x=_q68(x[v]), pa_q68x=_wq68(px, w),
                       sim_rmsx=float(np.std(x[v])), pa_rmsx=float(np.sqrt(np.sum(w * (px - mx_) ** 2))),
                       sim_q68y=_q68(y[v]), pa_q68y=_wq68(py, w),
                       sim_rmsy=float(np.std(y[v])), pa_rmsy=float(np.sqrt(np.sum(w * (py - my_) ** 2))),
                       unw_q68x=_q68(px), sim_meanx=float(np.mean(x[v])), pa_meanx=mx_,
                       sim_meany=float(np.mean(y[v])), pa_meany=my_)
            rows.append(row)
            print(f"{k:>2} {r:6.2f} {row['ess']:5.1f} | "
                  + " ".join(f"{1e4 * row[c]:9.0f}" for c in
                             ("sim_q68x", "pa_q68x", "sim_rmsx", "pa_rmsx"))
                  + " | " + " ".join(f"{1e4 * row[c]:9.0f}" for c in
                                     ("sim_q68y", "pa_q68y", "sim_rmsy", "pa_rmsy"))
                  + f" | {1e4 * row['unw_q68x']:9.0f}")
        # the rotated-ray approximation: paths weighted for the LAST plane
        # (the q/p location's plane) are over-spread on the inner planes
        wl = W[:, -2] if leg == rd.L0 else W[:, -1]
        spread = [dict(k=k, sim=_q68(sxy[k][1][sxy[k][0]]),
                       pa_lastw=_wq68(E["offs"][:, k, 3], wl)) for k in range(W.shape[1])]
        out[leg] = dict(rows=rows, identity=E.get("identity"), lastw_spread=spread)
    os.makedirs(RESE(), exist_ok=True)
    p = os.path.join(RESE(), "validate.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")


# =========================================================================
# analyse
# =========================================================================

def _g_dc0(g, E):
    return g[1] * E["dlam"] + g[2] * E["dphi"]


def _fixed_phi(k):
    """The fixed reference's pair CF on the closure grid and s_F (the
    closure's _plane_model), cached."""
    p = os.path.join(RES, "phifix", f"plane{k:02d}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return dict(sF=float(d["sF"]), tau=d["tau"], phi=d["phi"], phim=d["phim"],
                    phip=d["phip"], sigma=float(d["sigma"]))
    md = rd._plane_model(k)
    lm, lp, vm, vp, probes = rd._CTX
    tau = fn.closure_tau(float(np.max(probes)))
    phim = cpt.model_phi(lm, k, vm[k][0], md["sF"], tau)
    phip = cpt.model_phi(lp, k, vp[k][0], md["sF"], tau)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, sF=md["sF"], tau=tau, phi=phim * phip, phim=phim, phip=phip,
             sigma=md["sigma"])
    return dict(sF=md["sF"], tau=tau, phi=phim * phip, phim=phim, phip=phip,
                sigma=md["sigma"])


def _stats(phi, tau, probes):
    even = np.array([cpt.weier_scalar(phi, u, tau) for u in probes])
    odd = np.array([rd.weier_odd(phi, u, tau) for u in probes])
    D = np.array([rd.weier_shift(phi, u, tau) for u in probes])
    return even, odd, D, float(phi[1].imag / tau[1])


def _ctx_fixed():
    """realmat_ditrack's closure context (fixed models, vectors) for
    _plane_model."""
    rd._setup_physics()
    probes = np.asarray(fn.UCURVE)
    m0, g, worst = rd._check_gradient()
    assert abs(m0 - rd.M_PARENT) < 1e-9 and worst < 1e-6
    legs = {}
    for leg, L in rd.LEGS.items():
        path = rd.model_path(leg)
        legs[leg] = cpt.load_model(path)
        hb.bind(legs[leg], path, pdg=L["pdg"])
    vec = {leg: rd.leg_vectors(legs[leg], g[leg]) for leg in rd.LEGS}
    rd._CTX = (legs[rd.L0], legs[rd.L1], vec[rd.L0], vec[rd.L1], probes)
    return legs, vec, g, probes


def _own_cache(leg, i, k):
    return os.path.join(RESE(), "phiown", f"{leg}_path{i:03d}_plane{k:02d}.npy")


def _own_phi(job):
    """Path i's own model CF of the leg's mass functional on plane k (its own
    material and transport), at the fixed s_F; cached on disk.  Runs in a
    SPAWNED worker: a forked one inherits uproot's reader-thread locks from a
    parent that has read files and deadlocks on its first read."""
    global ENS
    leg, i, k, sF, tau, ENS = job
    out = _own_cache(leg, i, k)
    if os.path.exists(out):
        return out
    rd._setup_physics()
    _, g, _ = rd._check_gradient()
    p = ray_path(leg, i)
    legs = cpt.load_model(p)
    hb.bind(legs, p, pdg=rd.LEGS[leg]["pdg"])
    a = rd.leg_vectors(legs, g[leg])[k][0]
    phi = cpt.model_phi(legs, k, a, sF, tau)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + ".part.npy", phi)
    os.replace(out + ".part.npy", out)
    return out


def _run_own(jobs, nproc):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    todo = [j for j in jobs if not os.path.exists(_own_cache(*j[:3]))]
    print(f"own CFs: {len(jobs)} needed, {len(todo)} to compute", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(nproc, len(todo)),
                                 mp_context=mp.get_context("spawn")) as ex:
            for n, _ in enumerate(ex.map(_own_phi, todo, chunksize=1)):
                if (n + 1) % 20 == 0:
                    print(f"   {n + 1}/{len(todo)}", flush=True)
    return {tuple(j[:3]): np.load(_own_cache(*j[:3])) for j in jobs}


def cmd_fixedcf(args):
    """The fixed reference's pair CFs (the cache `analyse` reads)."""
    legs, _, _, _ = _ctx_fixed()
    nk = min(len(legs[rd.L0]), len(legs[rd.L1]))
    ks = list(range(nk)) if args.planes is None else args.planes
    t0 = time.time()
    fn.pmap(_fixed_phi, ks, nproc=min(len(ks), args.nproc))
    print(f"fixed CFs: {len(ks)} planes in {time.time() - t0:.0f} s")


def cmd_analyse(args):
    ens = load_ensemble()
    legs, vec, g, probes = _ctx_fixed()
    closure = json.load(open(os.path.join(rd.RES, "ditrack_closure_kj1_qx1.json")))
    crow = {r["k"]: r for r in closure["planes"]}
    nk = min(len(legs[rd.L0]), len(legs[rd.L1]))
    ks = list(range(nk)) if args.planes is None else args.planes
    W, S, Sfull, SQ, QOP = {}, {}, {}, {}, {}
    sims = {}
    for leg in rd.LEGS:
        E = ens[leg]
        s, sxy = sim_offsets(leg, E["fixed"])
        sims[leg] = s
        W[leg] = kde_weights(leg, ens, sxy)
        b = np.array([vec[leg][k][1] for k in range(len(E["fixed"]))])   # (nk, 5)
        full = np.einsum("ikc,kc->ik", E["offs"], b)                    # GeV
        S[leg] = full - _g_dc0(g[leg], E)[:, None]
        Sfull[leg] = full
        SQ[leg] = E["offs"][:, :, 0] * b[None, :, 0]                     # q/p slot
        QOP[leg] = E["offs"][:, :, 0]
    # the fixed pair CFs, plane-parallel
    t0 = time.time()
    fixed = dict(zip(ks, fn.pmap(_fixed_phi, ks, nproc=min(len(ks), args.nproc))))
    print(f"fixed CFs: {len(ks)} planes in {time.time() - t0:.0f} s", flush=True)
    iu = {u: list(probes).index(u) for u in (0.001, 0.01, 1.0)}

    # the path's own CFs on the selected planes (the full path average)
    own = {}
    if args.own_cf:
        jobs = []
        for k in args.own_cf:
            for leg in rd.LEGS:
                keep = np.arange(W[leg].shape[0])      # every path (pairs intact)
                for j in keep:
                    jobs.append((leg, int(ens[leg]["idx"][j]), k, fixed[k]["sF"],
                                 fixed[k]["tau"], ENS))
                own[(leg, k)] = keep
        t0 = time.time()
        ophi = _run_own(jobs, args.nproc)
        print(f"own CFs: {len(jobs)} in {time.time() - t0:.0f} s", flush=True)

    rows = []
    print(f"\n{'k':>2} {'r':>6} | {'sim mean':>9} {'fix mean':>9} {'pa mean':>9} "
          f"{'pa comp':>9} {'pa(full)':>9} | {'sim dm u=1':>10} {'pa dm u=1':>10} "
          f"{'res u=1':>8} {'res u=.01':>9}  [MeV]")
    for k in ks:
        F = fixed[k]
        sF, tau = F["sF"], F["tau"]
        e_fix, o_fix, D_fix, mz_fix = _stats(F["phi"], tau, probes)
        # shift-only path average: phase of each leg's weighted shifts
        C = np.ones(len(tau), dtype=np.complex128)
        for leg in rd.LEGS:
            w = W[leg][:, k]
            C *= (w[:, None] * np.exp(1j * np.outer(S[leg][:, k] / sF, tau))).sum(axis=0)
        f_pa = F["phi"] * C
        e_pa, o_pa, D_pa, mz_pa = _stats(f_pa, tau, probes)
        cr = crow[k]
        odd_sim = np.array(cr["odd"]) + np.array(cr["model_odd"])
        even_sim = np.array(cr["even"]) + np.array(cr["model_even"])
        # the recomputed fixed model must be the closure's
        chk = max(float(np.abs(e_fix - np.array(cr["model_even"])).max()),
                  float(np.abs(o_fix - np.array(cr["model_odd"])).max()))
        if chk > 1e-6:
            raise SystemExit(f"plane {k}: fixed CF differs from the closure's by {chk:.2e}")
        pa_loc = 1e3 * sF * (o_pa - o_fix) / D_fix       # model location vs fixed
        res_loc = 1e3 * sF * (odd_sim - o_pa) / D_pa     # sim vs path average
        # means (u -> 0): per leg weighted shift, and the segment composite
        mean_pa = sum(float(np.sum(W[leg][:, k] * S[leg][:, k])) for leg in rd.LEGS)
        mean_full = sum(float(np.sum(W[leg][:, k] * Sfull[leg][:, k])) for leg in rd.LEGS)
        comp, comp_start, perleg, comp_var = 0.0, 0.0, {}, 0.0
        for leg in rd.LEGS:
            inc = np.diff(np.concatenate([np.zeros((S[leg].shape[0], 1)),
                                          S[leg][:, :k + 1]], axis=1), axis=1)
            cl = float(np.sum(W[leg][:, :k + 1] * inc))
            wst = np.concatenate([W[leg][:, :1], W[leg][:, :k]], axis=1)
            comp += cl
            comp_var += comp_mc_err(W[leg][:, :k + 1], inc, ens[leg]["pairs"]) ** 2
            comp_start += float(np.sum(wst * inc))
            # single-track q/p location, sim vs path average (map variable,
            # relative units of q/p: dqop / |qop|)
            s = sims[leg]
            v = s["valid"][:, k]
            fq = legs[leg][k]["refqop"]
            dq = ck.qop_dev(s["qop"][v, k], fq) / abs(fq)
            qinc = np.diff(np.concatenate([np.zeros((QOP[leg].shape[0], 1)),
                                           QOP[leg][:, :k + 1]], axis=1), axis=1)
            perleg[leg] = dict(
                sim_qop_mean=float(np.mean(dq)),
                sim_qop_err=float(np.std(dq) / math.sqrt(v.sum())),
                pa_qop_mean=float(np.sum(W[leg][:, k] * QOP[leg][:, k])) / abs(fq),
                pa_qop_comp=float(np.sum(W[leg][:, :k + 1] * qinc)) / abs(fq),
                pa_qop_comp_err=comp_mc_err(W[leg][:, :k + 1], qinc,
                                            ens[leg]["pairs"]) / abs(fq),
                pa_qslot_MeV=1e3 * float(np.sum(W[leg][:, k] * SQ[leg][:, k])),
                sim_qslot_MeV=cr["mean_split_MeV"][leg][0],
                pa_mass_MeV=1e3 * float(np.sum(W[leg][:, k] * S[leg][:, k])),
                pa_mass_comp_MeV=1e3 * cl,
                ess=ess(W[leg][:, k]),
                path_spread_MeV=1e3 * float(np.sqrt(np.sum(W[leg][:, k] * (
                    S[leg][:, k] - np.sum(W[leg][:, k] * S[leg][:, k])) ** 2))))
        # bootstrap over the paths (resampled with their weights renormalised)
        rng = np.random.default_rng(20260929 + k)
        bs = np.zeros(args.nboot)
        bsq = {leg: np.zeros(args.nboot) for leg in rd.LEGS}
        for leg in rd.LEGS:
            w0, s0 = W[leg][:, k], S[leg][:, k]
            fq = abs(legs[leg][k]["refqop"])
            for ib in range(args.nboot):
                j = pair_draw(rng, ens[leg]["pairs"])
                bs[ib] += np.sum(w0[j] * s0[j]) / w0[j].sum()
                bsq[leg][ib] = np.sum(w0[j] * QOP[leg][j, k]) / w0[j].sum() / fq
        for leg in rd.LEGS:
            perleg[leg]["pa_qop_boot"] = float(np.std(bsq[leg]))
        # MC error of the path average (weighted, per leg, added)
        mc_err = math.sqrt(sum(pair_mc_err(W[leg][:, k], S[leg][:, k], ens[leg]["pairs"]) ** 2
                               for leg in rd.LEGS))
        for leg in rd.LEGS:
            perleg[leg]["pa_mass_mcerr_MeV"] = 1e3 * pair_mc_err(
                W[leg][:, k], S[leg][:, k], ens[leg]["pairs"])
            perleg[leg]["pa_qop_mcerr"] = pair_mc_err(
                W[leg][:, k], QOP[leg][:, k], ens[leg]["pairs"]) / abs(legs[leg][k]["refqop"])
        row = dict(k=k, r=cr["r"], sF=sF, n=cr["n"],
                   sim_mean_MeV=cr["sim_mean_MeV"], sim_mean_err_MeV=cr["sim_mean_err_MeV"],
                   fixed_mean_MeV=1e3 * sF * mz_fix,
                   pa_mean_MeV=1e3 * (sF * mz_fix + mean_pa),
                   pa_mean_comp_MeV=1e3 * (sF * mz_fix + comp),
                   pa_mean_comp_err_MeV=1e3 * math.sqrt(comp_var),
                   pa_mean_comp_start_MeV=1e3 * (sF * mz_fix + comp_start),
                   pa_mean_full_MeV=1e3 * (sF * mz_fix + mean_full),
                   pa_mean_mcerr_MeV=1e3 * mc_err,
                   pa_mean_boot_MeV=1e3 * float(np.std(bs)),
                   sim_dm_MeV=cr["dm_MeV"], sim_dm_err_MeV=cr["dm_err_MeV"],
                   pa_loc_MeV=pa_loc.tolist(), res_loc_MeV=res_loc.tolist(),
                   even_sim=even_sim.tolist(), even_fix=e_fix.tolist(),
                   even_pa=e_pa.tolist(), legs=perleg)
        if args.own_cf and k in args.own_cf:
            Cf = np.ones(len(tau), dtype=np.complex128)
            for leg in rd.LEGS:
                keep = own[(leg, k)]
                w = W[leg][keep, k]
                w = w / w.sum()
                acc = np.zeros(len(tau), dtype=np.complex128)
                for wj, j in zip(w, keep):
                    acc += wj * ophi[(leg, int(ens[leg]["idx"][j]), k)] * \
                        np.exp(1j * tau * S[leg][j, k] / sF)
                Cf *= acc
            e_o, o_o, D_o, mz_o = _stats(Cf, tau, probes)
            row.update(own_loc_MeV=(1e3 * sF * (o_o - o_fix) / D_fix).tolist(),
                       own_res_MeV=(1e3 * sF * (odd_sim - o_o) / D_o).tolist(),
                       own_mean_MeV=1e3 * sF * mz_o, even_own=e_o.tolist(),
                       own_npaths={leg: int(len(own[(leg, k)])) for leg in rd.LEGS})
        # bootstrap of the LOCATION statistics over the paths (shift-only,
        # and with the paths' own CFs where computed)
        rng = np.random.default_rng(7 + k)
        iul = [iu[0.01], iu[1.0]]
        bl, blo = [], []
        for _ in range(args.nboot_loc):
            Cb = np.ones(len(tau), dtype=np.complex128)
            Co = np.ones(len(tau), dtype=np.complex128) if "own_loc_MeV" in row else None
            for leg in rd.LEGS:
                j = pair_draw(rng, ens[leg]["pairs"])
                w = W[leg][j, k] / W[leg][j, k].sum()
                ph = np.exp(1j * np.outer(S[leg][j, k] / sF, tau))
                Cb *= (w[:, None] * ph).sum(axis=0)
                if Co is not None:
                    jj = j
                    ww = W[leg][jj, k] / W[leg][jj, k].sum()
                    acc = np.zeros(len(tau), dtype=np.complex128)
                    for wj, j1 in zip(ww, jj):
                        acc += wj * ophi[(leg, int(ens[leg]["idx"][j1]), k)] * \
                            np.exp(1j * tau * S[leg][j1, k] / sF)
                    Co *= acc
            ob = np.array([rd.weier_odd(F["phi"] * Cb, probes[i], tau) for i in iul])
            bl.append(1e3 * sF * (ob - o_fix[iul]) / D_fix[iul])
            if Co is not None:
                oo = np.array([rd.weier_odd(Co, probes[i], tau) for i in iul])
                blo.append(1e3 * sF * (oo - o_fix[iul]) / D_fix[iul])
        row["pa_loc_boot_MeV"] = dict(zip(("u0.01", "u1"), np.std(bl, axis=0).tolist()))
        if blo:
            row["own_loc_boot_MeV"] = dict(zip(("u0.01", "u1"), np.std(blo, axis=0).tolist()))
        rows.append(row)
        print(f"{k:>2} {cr['r']:6.2f} | {row['sim_mean_MeV']:+9.3f} {row['fixed_mean_MeV']:+9.3f} "
              f"{row['pa_mean_MeV']:+9.3f} {row['pa_mean_comp_MeV']:+9.3f} "
              f"{row['pa_mean_full_MeV']:+9.3f} | {cr['dm_MeV'][iu[1.0]]:+10.3f} "
              f"{pa_loc[iu[1.0]]:+10.3f} {res_loc[iu[1.0]]:+8.3f} {res_loc[iu[0.01]]:+9.3f}"
              f"   (MC {1e3 * mc_err:.3f}, comp MC {1e3 * math.sqrt(comp_var):.3f}, boot {1e3 * float(np.std(bs)):.3f}; ESS "
              + "/".join(f"{perleg[lg]['ess']:.0f}" for lg in rd.LEGS) + ")"
              + f"  loc u=1 boot {row['pa_loc_boot_MeV']['u1']:.3f}"
              + (f"  own: loc u=1 {row['own_loc_MeV'][iu[1.0]]:+.3f}+-{row['own_loc_boot_MeV']['u1']:.3f} "
                 f"res {row['own_res_MeV'][iu[1.0]]:+.3f}"
                 if "own_loc_MeV" in row else ""), flush=True)
    # the single-track q/p location on EVERY plane of each leg (mu- has 19)
    single = {}
    for leg in rd.LEGS:
        s = sims[leg]
        single[leg] = []
        for k in range(QOP[leg].shape[1]):
            v = s["valid"][:, k]
            fq = legs[leg][k]["refqop"]
            dq = ck.qop_dev(s["qop"][v, k], fq) / abs(fq)
            w = W[leg][:, k]
            qinc = np.diff(np.concatenate([np.zeros((QOP[leg].shape[0], 1)),
                                           QOP[leg][:, :k + 1]], axis=1), axis=1)
            single[leg].append(dict(
                k=k, r=float(legs[leg][k]["refglobr"]),
                sim=float(np.mean(dq)), sim_err=float(np.std(dq) / math.sqrt(v.sum())),
                sim_median=float(np.median(dq)),
                pa=float(np.sum(w * QOP[leg][:, k])) / abs(fq),
                pa_err=pair_mc_err(w, QOP[leg][:, k], ens[leg]["pairs"]) / abs(fq),
                pa_comp=float(np.sum(W[leg][:, :k + 1] * qinc)) / abs(fq),
                pa_comp_err=comp_mc_err(W[leg][:, :k + 1], qinc, ens[leg]["pairs"]) / abs(fq)))
    print("\nsingle-track q/p location, every plane, dqop/|qop| x 1e4: sim / path average")
    for leg in rd.LEGS:
        print(f"  {leg}: " + "  ".join(f"{x['k']}:{1e4 * x['sim']:+.2f}/{1e4 * x['pa']:+.2f}"
                                       f"({1e4 * x['pa_err']:.2f})"
                                       for x in single[leg]))
    print("\nsingle-track q/p location per leg, dqop/|qop| x 1e4 (sim vs path average):")
    for rw in rows:
        print(f"{rw['k']:>2} " + "   ".join(
            f"{lg}: sim {1e4 * d['sim_qop_mean']:+7.3f}+-{1e4 * d['sim_qop_err']:.3f} "
            f"pa {1e4 * d['pa_qop_mean']:+7.3f}+-{1e4 * d['pa_qop_boot']:.3f} "
            f"comp {1e4 * d['pa_qop_comp']:+7.3f}+-{1e4 * d['pa_qop_comp_err']:.3f}"
            for lg, d in rw["legs"].items()))
    os.makedirs(RESE(), exist_ok=True)
    # the transverse profile: every path's offsets and weights, per leg
    np.savez(os.path.join(RESE(), "paths.npz"),
             **{f"{leg}_{nm}": arr for leg in rd.LEGS for nm, arr in (
                 ("offs", ens[leg]["offs"]), ("w", W[leg]), ("S", S[leg]),
                 ("dlam", ens[leg]["dlam"]), ("dphi", ens[leg]["dphi"]),
                 ("refqop", np.array([x["refqop"] for x in ens[leg]["fixed"]])))})
    tag = "" if args.planes is None else "_planes_" + "_".join(map(str, ks))
    p = os.path.join(RESE(), f"pathavg{tag}.json")
    json.dump(dict(config=dict(components=COMPONENTS, kde_frac=KDE_FRAC,
                               npaths={lg: int(len(ens[lg]["idx"])) for lg in rd.LEGS},
                               ucurve=list(probes), own_cf=args.own_cf,
                               knockon=dict(ck.physics_state())),
                   planes=rows, single=single), open(p, "w"), indent=1)
    print(f"-> {p}")


# =========================================================================
# figures
# =========================================================================

def cmd_plot(args):
    """Both ensembles on the same axes (whichever have been analysed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    import pubhtml
    hep.style.use(hep.style.ROOT)
    outdir = pubhtml.figdir("realtrk_pathavg")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir)
    E = {}
    for e, lab, sty in (("rot", "rotated rays", dict(color="tab:red", marker="s")),
                        ("cov", "kinked (covariance-matched)", dict(color="tab:green", marker="D"))):
        d = os.path.join(RES if e == "rot" else os.path.join(RES, e))
        if os.path.exists(os.path.join(d, "pathavg.json")):
            E[e] = dict(d=json.load(open(os.path.join(d, "pathavg.json"))),
                        v=json.load(open(os.path.join(d, "validate.json"))),
                        z=np.load(os.path.join(d, "paths.npz")), lab=lab, sty=sty)
    ref = next(iter(E.values()))["d"]
    iu1 = ref["config"]["ucurve"].index(1.0)
    P = ref["planes"]
    r = np.array([p["r"] for p in P])
    good = np.array([p["k"] not in PARTIAL for p in P])

    def col(planes, key, sub=None):
        return np.array([(p[key] if sub is None else p[key][sub]) for p in planes])

    # (1) mean of the vertex mass
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(r[good], col(P, "sim_mean_MeV")[good], col(P, "sim_mean_err_MeV")[good],
                fmt="ko", label="sim")
    ax.plot(r[good], col(P, "fixed_mean_MeV")[good], "b^-", label="model, fixed reference")
    for i, (e, X) in enumerate(E.items()):
        Q = X["d"]["planes"]
        ax.errorbar(r[good] + 0.6 * (i + 1), col(Q, "pa_mean_MeV")[good],
                    col(Q, "pa_mean_mcerr_MeV")[good], ls="-", ms=6,
                    label=f"path-averaged, plane weights: {X['lab']}", **X["sty"])
        ax.errorbar(r[good] + 0.6 * (i + 1) + 0.3, col(Q, "pa_mean_comp_MeV")[good],
                    col(Q, "pa_mean_comp_err_MeV")[good], ls="--", ms=6, mfc="none",
                    label=f"path-averaged, segment composite: {X['lab']}", **X["sty"])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("plane radius [cm]")
    ax.set_ylabel(r"mean of the one-plane $J/\psi$ mass [MeV]")
    ax.legend(fontsize=12)
    pubhtml.savefig(fig, os.path.join(outdir, "mass_mean_vs_r.pdf"))
    plt.close(fig)
    # (2) core location u = 1
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(r[good], col(P, "sim_dm_MeV", iu1)[good], col(P, "sim_dm_err_MeV", iu1)[good],
                fmt="ko", label="sim vs fixed-reference model")
    for i, (e, X) in enumerate(E.items()):
        Q = X["d"]["planes"]
        ax.errorbar(r[good] + 0.6 * (i + 1), col(Q, "pa_loc_MeV", iu1)[good],
                    np.array([q["pa_loc_boot_MeV"]["u1"] for q in Q])[good], ls="-", ms=6,
                    label=f"shift-only: {X['lab']}", **X["sty"])
        own = [q for q in Q if "own_loc_MeV" in q]
        if own:
            ax.errorbar([q["r"] + 0.6 * (i + 1) for q in own],
                        [q["own_loc_MeV"][iu1] for q in own],
                        [q["own_loc_boot_MeV"]["u1"] for q in own], fmt="*", ms=16,
                        color=X["sty"]["color"], mfc="none",
                        label=f"own CFs: {X['lab']}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("plane radius [cm]")
    ax.set_ylabel(r"core location $\delta m(u=1)$ vs fixed model [MeV]")
    ax.legend(fontsize=14)
    pubhtml.savefig(fig, os.path.join(outdir, "mass_loc_u1_vs_r.pdf"))
    plt.close(fig)
    # (3) per-leg single-track q/p location, every plane of the leg
    for leg in rd.LEGS:
        fig, ax = plt.subplots(figsize=(10, 7))
        S0 = ref["single"][leg]
        ax.errorbar([x["r"] for x in S0], [1e4 * x["sim"] for x in S0],
                    [1e4 * x["sim_err"] for x in S0], fmt="ko", label="sim")
        for i, (e, X) in enumerate(E.items()):
            S1 = X["d"]["single"][leg]
            ax.errorbar([x["r"] + 0.6 * (i + 1) for x in S1], [1e4 * x["pa"] for x in S1],
                        [1e4 * x["pa_err"] for x in S1], ls="-", ms=6,
                        label=f"plane weights: {X['lab']}", **X["sty"])
            ax.errorbar([x["r"] + 0.6 * (i + 1) + 0.3 for x in S1],
                        [1e4 * x["pa_comp"] for x in S1],
                        [1e4 * x["pa_comp_err"] for x in S1], ls="--", ms=6, mfc="none",
                        label=f"segment composite: {X['lab']}", **X["sty"])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("plane radius [cm]")
        ax.set_ylabel(r"$\langle \delta(q/p) \rangle / |q/p|$ vs fixed reference [$10^{-4}$]")
        ax.set_title(f"{leg}: single-track q/p location", fontsize=18)
        ax.legend(fontsize=12)
        pubhtml.savefig(fig, os.path.join(outdir, f"qop_mean_{leg}.pdf"))
        plt.close(fig)
    # (4) transverse spread: the sim and each ensemble (weighted on the plane,
    # weighted for the outer mass plane, unweighted)
    for leg in rd.LEGS:
        fig, ax = plt.subplots(figsize=(10, 7))
        R0 = next(iter(E.values()))["v"][leg]["rows"]
        ax.plot([x["r"] for x in R0], [1e4 * x["sim_q68x"] for x in R0], "ko",
                label=r"sim $\sigma_{68}$(local x)")
        for e, X in E.items():
            R = X["v"][leg]["rows"]
            c = X["sty"]["color"]
            ax.plot([x["r"] for x in R], [1e4 * x["pa_q68x"] for x in R], "--", color=c,
                    marker=X["sty"]["marker"], label=f"{X['lab']}: plane weights")
            ax.plot([x["r"] for x in R], [1e4 * x["pa_lastw"] for x in X["v"][leg]["lastw_spread"]],
                    ":", color=c, label=f"{X['lab']}: outer-plane weights")
        ax.set_yscale("log")
        ax.set_xlabel("plane radius [cm]")
        ax.set_ylabel(r"$\sigma_{68}$ of local x [$\mu$m]")
        ax.set_title(leg, fontsize=18)
        ax.legend(fontsize=13)
        pubhtml.savefig(fig, os.path.join(outdir, f"spread_{leg}.pdf"))
        plt.close(fig)
    # (5) the transverse material profile: each path's q/p offset on the
    # outer mass plane against where it crosses it
    k = len(P) - 1
    for e, X in E.items():
        z = X["z"]
        for leg in rd.LEGS:
            o = z[f"{leg}_offs"]
            rel = 1e4 * o[:, k, 0] / np.abs(z[f"{leg}_refqop"][k])
            fig, ax = plt.subplots(figsize=(10, 8))
            vm = np.percentile(np.abs(rel), 98)
            sc = ax.scatter(1e4 * o[:, k, 3], 1e4 * o[:, k, 4], c=rel, cmap="coolwarm",
                            vmin=-vm, vmax=vm, s=10 + 3000 * z[f"{leg}_w"][:, k])
            fig.colorbar(sc, ax=ax, label=r"path $\delta(q/p)/|q/p|$ [$10^{-4}$]")
            ax.set_xlabel(r"path local x $-$ fixed reference [$\mu$m]")
            ax.set_ylabel(r"path local y $-$ fixed reference [$\mu$m]")
            ax.set_title(f"{leg}, plane {k} (r = {P[k]['r']:.1f} cm), {X['lab']}; "
                         "size = weight", fontsize=15)
            pubhtml.savefig(fig, os.path.join(outdir, f"profile_{e}_{leg}_plane{k}.pdf"))
            plt.close(fig)
    print(f"-> {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens", default="rot", choices=("rot", "cov"),
                    help="the path ensemble (see the module docstring)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("covfit").set_defaults(f=cmd_covfit)
    s = sub.add_parser("design")
    s.add_argument("--batches", type=int, default=1)
    s.set_defaults(f=cmd_design)
    s = sub.add_parser("export")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--force", action="store_true")
    s.add_argument("--max", type=int, default=None, help="paths i < max only")
    s.set_defaults(f=cmd_export)
    sub.add_parser("validate").set_defaults(f=cmd_validate)
    s = sub.add_parser("fixedcf")
    s.add_argument("--planes", type=int, nargs="*", default=None)
    s.add_argument("--nproc", type=int, default=18)
    s.set_defaults(f=cmd_fixedcf)
    s = sub.add_parser("analyse")
    s.add_argument("--planes", type=int, nargs="*", default=None)
    s.add_argument("--own-cf", type=int, nargs="*", default=[],
                   help="planes with the paths' own CFs (the full path average)")
    s.add_argument("--nproc", type=int, default=40)
    s.add_argument("--nboot", type=int, default=400)
    s.add_argument("--nboot-loc", type=int, default=100)
    s.set_defaults(f=cmd_analyse)
    sub.add_parser("plot").set_defaults(f=cmd_plot)
    a = ap.parse_args()
    global ENS
    ENS = a.ens
    a.f(a)


if __name__ == "__main__":
    main()
