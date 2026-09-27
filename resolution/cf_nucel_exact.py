"""Nuclear elastic (`hadElastic`) noise channel for the clean-propagation CF.

WHY THIS CHANNEL EXISTS
-----------------------
`cf_propagation_test.model_phi`'s other three channels -- ionization, multiple
scattering, radiative -- do not represent `hadElastic`, which the simulation
runs for every hadron.  The elastic/inelastic split (hadron_probe arms
`elonly`/`inelonly`) shows:

  * elastic is a PURE SHAPE effect: acceptance in `elonly` is 100.0000 % for
    pi/K/p and 99.9980 % for pbar, so there is no survivor selection to unfold;
  * elastic is essentially the WHOLE nuclear effect -- `inelonly` never exceeds
    1.5 sigma in locx despite removing 13-19 % of the tracks;
  * in locx it runs to 38-107 sigma, about 10x the residual the
    all-corrections closure is left with.  It is the dominant missing effect
    for hadrons.

THE STRUCTURE
-------------
Over the modelled path (12.45 g/cm^2 across the 14 legs) the expected number of
elastic collisions is only 0.026-0.100 depending on species, but the mean
deflection is 25-35 mrad -- three to four times a whole track's Moliere width.
So this is a RARE, LARGE-ANGLE TAIL, not a width: a few per cent of tracks take
exactly one big kick.  That is why it barely moves `qop` and wrecks `locx`.

A single elastic collision is an isotropic 2D kick of polar size theta.  Under
the same azimuthal-isotropy argument the MS channel uses -- measure the
azimuthal angle as phi*cos(lambda) so the two projected angles are iid, and the
CF then depends only on the quadrature sum of the two transport weights, which
is exactly `weff` -- the CF of one kick projected with weight w is

    E_psi[ exp(i * t * w * theta * cos psi) ]  =  J0(t * w * theta)

and the compound Poisson of mean N over a (sub-)step contributes

    S(t) = N * ( E_theta[ J0(t * w * theta) ] - 1 ).

There is no `- i*x` centring term here (unlike ionization and radiative): the
kick is isotropic, so its mean projection is exactly zero and nothing has been
pre-subtracted from the reference.

The kernel E_theta[J0(u*theta)] is a ONE-DIMENSIONAL function of u, so it is
tabulated once per species/energy bucket and interpolated -- the same device
`gshape` uses for Moliere.

WHAT IS TAKEN FROM GEANT4 AND WHAT IS NOT
-----------------------------------------
Everything.  The rate is `G4CrossSectionDataStore::GetCrossSection(dp, mat)` --
the very call `G4HadronicProcess` inverts for its mean free path -- evaluated
on the species-correct dataset that stock `G4HadronElasticPhysics` registers.
The kernel is the empirical distribution of the primary's deflection under
`ApplyYourself`, i.e. what the four models ACTUALLY draw, not a diffractive
parameterisation of them.  Both come from `nucel_g4driver`.  Nothing here is
fitted; see Documents/Resolution/NUCLEAR_ELASTIC.md for the parameter-free
check (shift/N_pred constant to 4 % across four species and four different G4
models).

THE MUON IS THE NULL.  It has no `hadElastic` at all, and the driver refuses to
run for it, so `species_kernel` refuses too rather than returning a zero that
could be mistaken for a computed result.
"""

import fcntl
import hashlib
import os
import subprocess

import numpy as np
from scipy.special import j0

# --------------------------------------------------------------------------
# the knob.  Default OFF, like every other correction in this study, pending
# the global fit.  Registered in PHYSICS_GLOBALS so it reaches both the
# _PHI_CACHE key and the sha256 scale-cache identity -- a physics switch that
# is not in both is silently ignored on a cache hit.
# --------------------------------------------------------------------------
NUCEL_CHANNEL = bool(int(os.environ.get("NUCEL_CHANNEL", "0")))

# The RECOIL term: an elastic collision also transfers energy to the nucleus,
# and the channel above injects the ANGLE only.  Measured: dE is a DETERMINISTIC
# function of the deflection, dE = (p*theta)^2 / (2*M_target), holding to 0.06 %
# with log-correlation 1.00000 -- it is the same momentum transfer seen in the
# conjugate variable, not an independent random quantity.  Size on the qop
# width: +0.01 % (pi-), +0.04 % (K-), +0.73 % (p), +3.26 % (pbar).
# This flag adds it as an INDEPENDENT compound Poisson in dE, which is the
# leading approximation (for the qop functional the angular part enters only
# through transport).  DIAGNOSTIC: it tests whether the recoil explains the
# proton's qop residual (0.00015 clean-arm -> 0.00070 once elastic is on, which
# the angle-only channel does not touch).  A correct treatment keeps the
# theta<->dE correlation; see Documents/Resolution/NUCLEAR_ELASTIC.md.
# DEFAULT ON: this is part of the nuclear-elastic physics, not a separate
# correction.  VERIFIED IN THE SIMULATION, not assumed:
#   * G4HadronElastic::ApplyYourself sets the primary's energy to
#     `nlv1.e() - m1`, i.e. it DOES remove the recoil;
#   * the sim's recoil is the two-body function of the deflection --
#     sim/(p*theta)^2/(2M) -> 1.013 once the recoil dominates the ionisation
#     pedestal;
#   * the sim loses energy the REFERENCE does not: ref - <pabs> at the
#     outermost plane is -0.0067 MeV in arm `off` and +0.0394 MeV in `elonly`,
#     a difference of 0.0461 MeV against the predicted N*<dE> = 0.043 (7 %).
# It makes the closure WORSE (Documents/Resolution/NUCLEAR_ELASTIC.md).  It is
# kept ON anyway: the
# physics is measured, and a correction that is right must not be dropped
# because it exposes a compensating error elsewhere -- that is how a lucky
# cancellation gets frozen in.  Set NUCEL_RECOIL=0 to A/B it.
NUCEL_RECOIL = bool(int(os.environ.get("NUCEL_RECOIL", "1")))

# PER-ELEMENT TARGETS.  The exported effZ/effA are mass averages: rounding
# them to one element replaces a compound by its dominant nucleus, which drops
# every minority target.  For elastic scattering that is not a small error when
# the minority is HYDROGEN: a proton target is ~3x wider in angle than carbon
# (median 65-103 mrad at pT = 3) and, at equal angle, takes a 12x larger recoil
# (dE = (p theta)^2 / 2M).  On the real-material toy the composites (carbon
# fibre, cables, connectors, Kapton) are 4-8 % hydrogen by mass -- 0.33 g/cm2,
# 2.7 % of the path -- and hydrogen alone reproduces the whole elastic-loss
# excess the rounded model misses (p-bar 0.295 vs 0.295 MeV per track).
# With this ON and an export that carries the material table (`msmatv` +
# the `materials` tree), each step's material is a rate-weighted mixture of
# its elements' kernels, for the angle AND the recoil.  Files without the table
# fall back to the rounded element, bit for bit.  Set NUCEL_ELEMENTS=0 to A/B.
NUCEL_ELEMENTS = bool(int(os.environ.get("NUCEL_ELEMENTS", "1")))

# THE JOINT LAW OF ONE ELASTIC COLLISION.  The angular and recoil families
# above are independent compound Poissons, but each collision's recoil is a
# function of its deflection (two-body kinematics: dE = (p theta)^2 / 2M on a
# heavy nucleus to 1e-3; on HYDROGEN at 0.8 GeV the small-angle form is 10-36 %
# off and a single collision turns a pion by ~37 deg and takes ~15 % of its
# momentum).  Per collision the exact CF is E[J0(b theta) e^{i a X(dE)}]; the
# model has E[J0(b theta)] + E[e^{i a dE}] - 1.  The difference,
#
#     dS = N sum_i W_i [ (J0(b th_i) - 1)(e^{i a X_i} - 1) + (e^{i a X_i} - e^{i a dE_i}) ]
#
# over the element-weighted (theta, dE) bins of the paired Geant4 samples, is
# added per MS row (b: the row's angular weight at the step midpoint, a: the
# recoil weight).  X = dE, or the exact 1/p map T_eff(dE) when
# cf_knockon.QOP_EXACT is on (the second bracket is then the recoil's own map
# correction; it vanishes for X = dE).  Needs NUCEL_RECOIL.  DEFAULT ON: the
# collision is one event (Lambda ditrack: even probes > 3 sigma 107 -> 71 of
# 171, the clean arm's level; K_S unchanged).
NUCEL_JOINT = bool(int(os.environ.get("NUCEL_JOINT", "1")))

PHYSICS_GLOBALS = ("NUCEL_CHANNEL", "NUCEL_RECOIL", "NUCEL_ELEMENTS",
                   "NUCEL_JOINT")
_NOT_PHYSICS = ("NUCEL_NSAMP", "NUCEL_SEED", "NUCEL_NU", "NUCEL_UMIN",
                "NUCEL_UMAX", "NUCEL_NBIN", "NUCEL_JBIN", "NUCEL_JSHARE")

# sampling/tabulation knobs -- resolution of the numerics, not physics
NUCEL_NSAMP = int(os.environ.get("NUCEL_NSAMP", "2000000"))
NUCEL_SEED = int(os.environ.get("NUCEL_SEED", "20260817"))
NUCEL_NU = 4000          # points in the log-u kernel table
NUCEL_NBIN = 100000      # log-theta bins used to quadrature the sampled kernel
NUCEL_UMIN = 1e-4        # below this J0 -> 1 to double precision for our thetas
NUCEL_UMAX = 1e9
NUCEL_JBIN = 200         # log-theta bins per element in the joint (theta, dE) table
NUCEL_JSHARE = 1e-5      # elements below this share of a material's rate are dropped
                         # from its joint table (trace elements; the families keep them)
_TABLE_VERSION = 1       # bump when a persisted table's construction changes


def physics_state():
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)


# --------------------------------------------------------------------------
# driver plumbing
# --------------------------------------------------------------------------

def _driver():
    """`NUCEL_DRIVER` names the wrapper `build_nucel.sh <dest>` writes; the
    default is the historical scratchpad location.  Read here rather than held
    as a module global: it selects where the sampler lives, not what it
    computes, so it is not a physics switch."""
    p = os.environ.get("NUCEL_DRIVER")
    if p:
        return p
    import hadron_probe as hp
    return os.path.join(hp.SCRATCH, "nucel_g4driver.sh")


def _cachedir():
    """`NUCEL_CACHEDIR` relocates the kernel cache.  Buckets are keyed on
    (species, Z, A, ekin, NSAMP, SEED) and are bit-reproducible, so the
    location never changes a result."""
    d = os.environ.get("NUCEL_CACHEDIR")
    if not d:
        import hadron_probe as hp
        d = os.path.join(hp.SCRATCH, "nucel")
    os.makedirs(d, exist_ok=True)
    return d


# The reference geometry the driver is called with.  The rate is returned as a
# MASS attenuation mu = Sigma/rho [cm^2/g], which is density independent, so
# these two numbers only have to be self-consistent -- they never enter a
# result.  (ToyLayerMat is rho = 0.1073 g/cm^3; using it keeps the driver's
# printed Sigma directly comparable with the toy.)
_REF_RHO = 0.1073         # g/cm^3
_REF_LEN = 100.0          # mm


def _bucket(pdg, Z, A, ekin):
    """Collapse the ~150 exported steps onto a few driver calls.  The elastic
    cross section and the angular kernel vary by well under a per cent over the
    0.8 % that the momentum drops across the whole track, so ekin is binned at
    1 % exactly as the radiative driver bins it."""
    return (int(pdg), round(float(Z), 4), round(float(A), 4),
            float(f"{float(ekin):.4g}"))


def _load_npz(npz):
    """Load a cached bucket, or None if it is absent or unreadable.

    Tolerating a corrupt file rather than raising is deliberate: see the lock
    comment in `_run_driver`.
    """
    if not os.path.exists(npz):
        return None
    try:
        d = np.load(npz)
        return float(d["mu"]), d["theta"]
    except Exception:
        return None


def _run_driver(pdg, Z, A, ekin):
    """Return (mu [cm^2/g], theta samples [rad]).  Cached on disk by bucket.

    CONCURRENCY.  `fisher_norm.plane_scales` evaluates the CF under a
    multiprocessing pool, so up to `nplane` workers reach this function for the
    SAME bucket within milliseconds of each other.  A plain savez plus a plain
    load kills the run with `BadZipFile: File is not a zip file` -- a worker
    reading a file another worker is still writing.  Two mechanisms, both
    needed:
      * an exclusive flock per bucket, so exactly one process runs the driver
        and the rest wait and then read the finished file;
      * an ATOMIC publish (write to a private temp name in the same directory,
        then os.replace), so a reader can never observe a partial file even if
        it somehow bypasses the lock.
    The driver's own scratch prefix carries the pid for the same reason.
    """
    key = _bucket(pdg, Z, A, ekin)
    tag = hashlib.sha256(repr((key, NUCEL_NSAMP, NUCEL_SEED)).encode()).hexdigest()[:16]
    npz = os.path.join(_cachedir(), f"nucel_{tag}.npz")
    hit = _load_npz(npz)
    if hit is not None:
        return hit

    lockpath = npz + ".lock"
    with open(lockpath, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        # re-check under the lock: the process we queued behind has very
        # probably just built it
        hit = _load_npz(npz)
        if hit is not None:
            return hit
        return _build_bucket(pdg, Z, A, ekin, tag, npz)


def _build_bucket(pdg, Z, A, ekin, tag, npz):
    prefix = os.path.join(_cachedir(), f"drv_{tag}_{os.getpid()}")
    # repr(np.float64(x)) is "np.float64(x)", which atof() reads as 0.0, so
    # every numeric argument goes through float() and "%.17g".
    cmd = [_driver(), "--pdg", "%d" % int(pdg),
           "--Z", "%.17g" % float(Z), "--A", "%.17g" % float(A),
           "--rho", "%.17g" % float(_REF_RHO),
           "--ekin", "%.17g" % float(ekin),
           "--len", "%.17g" % float(_REF_LEN),
           "--n", "%d" % int(NUCEL_NSAMP), "--seed", "%d" % int(NUCEL_SEED),
           "--out", prefix]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(prefix + ".rec"):
        raise SystemExit(f"nucel_g4driver failed for pdg={pdg} Z={Z} A={A} "
                         f"ekin={ekin}: rc={r.returncode}\n{r.stderr[-2000:]}")

    sigma_per_mm = None
    for line in open(prefix + ".rec"):
        p = line.split()
        if p and p[0] == "rate":
            sigma_per_mm = float(p[1])
            break
    if sigma_per_mm is None:
        raise SystemExit(f"no `rate` line in {prefix}.rec")
    # Sigma [1/mm] at _REF_RHO -> mass attenuation [cm^2/g]
    mu = sigma_per_mm * 10.0 / _REF_RHO

    theta = np.fromfile(prefix + ".theta.bin", dtype=np.float64)
    if theta.size == 0:
        raise SystemExit(f"{prefix}.theta.bin is empty -- the sampler drew "
                         f"nothing, which is never a physical result")
    # Atomic publish: same directory (so os.replace is a rename, not a copy),
    # private name, then a single atomic swap.
    # The ".npz" on the TEMP name is load-bearing: np.savez_compressed appends
    # ".npz" to any path that does not already end in it, so a temp named
    # "<x>.npz.tmp<pid>" is silently written to "<x>.npz.tmp<pid>.npz" and the
    # replace below then fails with FileNotFoundError on a path that the code
    # appears to have just created.
    tmp = f"{npz}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, mu=mu, theta=theta)
    os.replace(tmp, npz)
    return mu, theta


def warm(legs):
    """Pre-build every kernel these legs need, in the PARENT process.

    `plane_scales` and `closure_rows` both fan out over planes with a forking
    pool.  Warming here means the children inherit a populated `_KERNEL_CACHE`
    through the fork and never touch the driver, the lock or the disk at all --
    which is both much faster than N workers queueing on one flock and one
    fewer way for a half-written file to be observed.  Safe to call when the
    channel is off (it returns immediately) or on a muon (no kernel exists).
    """
    if not NUCEL_CHANNEL:
        return
    for leg in legs:
        if not len(np.asarray(leg["ms"])):
            continue
        pdg = pdg_from_leg(leg)
        if pdg is None:
            continue
        leg_rates(leg, pdg, mass_of(pdg), nsub=1)
        if NUCEL_RECOIL:
            # the recoil kernels too: a forked worker that builds one reads a
            # 16 MB sample file and runs a 4000 x 1e5 complex quadrature
            dE_step_kernels(leg, pdg)


# --------------------------------------------------------------------------
# persisted tabulations.  Tabulating one bucket's angular kernel is 4e8 J0
# evaluations (~20 s); a leg set needs ~200 buckets per species, so a closure
# spent ~70 min per species rebuilding tables that depend only on the bucket
# and the grid.  They are written next to the driver samples, keyed by the
# bucket's sample tag and every grid parameter, and published atomically.
# --------------------------------------------------------------------------

def _bucket_tag(pdg, Z, A, ekin):
    return hashlib.sha256(repr((_bucket(pdg, Z, A, ekin), NUCEL_NSAMP,
                                NUCEL_SEED)).encode()).hexdigest()[:16]


def _table_path(kind, pdg, Z, A, ekin, grid):
    gtag = hashlib.sha256(repr((kind, grid, _TABLE_VERSION)).encode()).hexdigest()[:8]
    return os.path.join(_cachedir(), f"{kind}_{_bucket_tag(pdg, Z, A, ekin)}_{gtag}.npz")


def _table_load(path, names):
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        return tuple(d[n] for n in names)
    except Exception:
        return None


def _table_store(path, **arrays):
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _paired_samples(pdg, Z, A, ekin):
    """(theta [rad], dE [MeV]) of the SAME collisions: the driver writes both
    per sample in one run; a prefix is used only when both files are there and
    have equal length.  Without a pair the driver is rerun (same seed, so the
    samples are the ones the cached kernels were built from)."""
    import glob as _g
    tag = _bucket_tag(pdg, Z, A, ekin)
    for th in sorted(_g.glob(os.path.join(_cachedir(), f"drv_{tag}_*.theta.bin"))):
        el = th[:-len(".theta.bin")] + ".eloss.bin"
        if os.path.exists(el) and os.path.getsize(el) == os.path.getsize(th):
            return np.fromfile(th, dtype=np.float64), np.fromfile(el, dtype=np.float64)
    npz = os.path.join(_cachedir(), f"nucel_{tag}.npz")
    with open(npz + ".lock", "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        prefix = os.path.join(_cachedir(), f"drv_{tag}_{os.getpid()}")
        cmd = [_driver(), "--pdg", "%d" % int(pdg), "--Z", "%.17g" % float(Z),
               "--A", "%.17g" % float(A), "--rho", "%.17g" % float(_REF_RHO),
               "--ekin", "%.17g" % float(ekin), "--len", "%.17g" % float(_REF_LEN),
               "--n", "%d" % int(NUCEL_NSAMP), "--seed", "%d" % int(NUCEL_SEED),
               "--out", prefix]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"nucel_g4driver failed for the paired samples of "
                             f"{(pdg, Z, A, ekin)}:\n{r.stderr[-2000:]}")
    return (np.fromfile(prefix + ".theta.bin", dtype=np.float64),
            np.fromfile(prefix + ".eloss.bin", dtype=np.float64))


_EJ_CACHE = {}


def element_joint(pdg, Z, A, ekin):
    """(theta, dE, w) of one bucket on NUCEL_JBIN log-theta bins: the bin's
    mean deflection, its mean recoil, and its share of ALL collisions (zero
    deflections carry no joint term and are left out of the bins, not of the
    normalisation).  Elastic two-body kinematics makes dE one-to-one in theta,
    so the per-bin mean is the exact relation up to the bin width."""
    key = _bucket(pdg, Z, A, ekin)
    hit = _EJ_CACHE.get(key)
    if hit is not None:
        return hit
    kf = _table_path("joint", pdg, Z, A, ekin, (NUCEL_JBIN,))
    got = _table_load(kf, ("theta", "dE", "w"))
    if got is None:
        th, dE = _paired_samples(pdg, Z, A, ekin)
        pos = th > 0.0
        t, e = th[pos], dE[pos]
        edges = np.geomspace(t.min(), t.max() * (1.0 + 1e-12), NUCEL_JBIN + 1)
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, NUCEL_JBIN - 1)
        cnt = np.bincount(idx, minlength=NUCEL_JBIN).astype(np.float64)
        st = np.bincount(idx, weights=t, minlength=NUCEL_JBIN)
        se = np.bincount(idx, weights=e, minlength=NUCEL_JBIN)
        m = cnt > 0
        got = (st[m] / cnt[m], se[m] / cnt[m], cnt[m] / float(th.size))
        _table_store(kf, theta=got[0], dE=got[1], w=got[2])
    _EJ_CACHE[key] = got
    return got


_MJ_CACHE = {}


def material_joint(pdg, comp, ekin):
    """(theta, dE, W) of a compound: its elements' joint bins, each weighted by
    the element's share w_i mu_i / sum_j w_j mu_j of the collisions (the
    mixture `material_kernel` uses); elements below NUCEL_JSHARE are dropped."""
    key = (int(pdg), comp, float(f"{float(ekin):.4g}"))
    hit = _MJ_CACHE.get(key)
    if hit is not None:
        return hit
    rates = []
    for z, a, w in comp:
        mu, _ = _run_driver(pdg, z, a, ekin)
        rates.append(w * mu)
    tot = float(sum(rates))
    if tot <= 0.0:
        _MJ_CACHE[key] = None
        return None
    th, de, ww = [], [], []
    for (z, a, w), r in zip(comp, rates):
        if r / tot < NUCEL_JSHARE:
            continue
        t, e, f = element_joint(pdg, z, a, ekin)
        th.append(t)
        de.append(e)
        ww.append(f * (r / tot))
    out = (np.concatenate(th), np.concatenate(de), np.concatenate(ww))
    _MJ_CACHE[key] = out
    return out


def joint_exponent(tau, nrate, jt, w, wq, E=None, p=None, chunk=2048):
    """The per-collision joint correction of one MS row (see NUCEL_JOINT):
    nrate * sum_i W_i [(J0(t w th_i) - 1)(e^{i t wq X_i} - 1) + (e^{i t wq X_i}
    - e^{i t wq dE_i})], with X = T_eff(dE) when cf_knockon.QOP_EXACT and the
    row's (E, p) [GeV] are given, else X = dE.  `wq` is the signed recoil weight
    [z per MeV] of the recoil family, `w` the row's angular weight."""
    tau = np.asarray(tau, dtype=np.float64)
    out = np.zeros(len(tau), dtype=np.complex128)
    if jt is None or nrate <= 0.0 or (w <= 0.0 and wq == 0.0):
        return out
    th, dE, W = jt
    X = dE
    exact = False
    if E is not None and p is not None:
        import cf_knockon as _ck
        if _ck.QOP_EXACT:
            X = _ck.t_eff(dE, 1e3 * E, 1e3 * p)
            exact = True
    for lo in range(0, len(tau), chunk):
        t = tau[lo:lo + chunk][:, None]
        jm1 = j0(t * (w * th)[None, :]) - 1.0
        yx = t * (wq * X)[None, :]
        term = jm1 * np.expm1(1j * yx)
        if exact:
            term = term + (np.exp(1j * yx) - np.exp(1j * t * (wq * dE)[None, :]))
        out[lo:lo + chunk] = nrate * (term @ W)
    return out


def rows_exponent(S, leg, tau, rid, wb, frac, wq_row, wb_mid):
    """THE nuclear-elastic exponent of one leg's scattering ROWS (`cf_rows`),
    accumulated INTO `S` in place:
      angular  per entry (row rid, angular weight wb, share frac of the row's
               collisions);
      recoil   per row at weight wq_row[s] [z per MeV of recoil energy: the
               q/p weight times the step's q/p per MeV, cs 1e-3]
               (NUCEL_RECOIL), the row's whole rate;
      joint    per row at (wb_mid[s], wq_row[s]) (NUCEL_JOINT).
    `leg` carries the rows (`ms`), their materials (`msmat`, `mattab`) and
    the species (`pdg_from_leg`)."""
    if not NUCEL_CHANNEL or not len(leg["ms"]):
        return S
    pdg = pdg_from_leg(leg)
    if pdg is None:
        return S
    r = leg_rates(leg, pdg, mass_of(pdg), nsub=1)
    if r is None:
        return S
    nrate, kidx, kernels = r
    for e in range(len(rid)):
        s_ = rid[e]
        if nrate[s_] <= 0.0 or wb[e] <= 0.0:
            continue
        ugrid, gtab = kernels[kidx[s_]]
        S += nucel_step_exponent(tau, nrate[s_] * frac[e], ugrid, gtab, wb[e])
    if not NUCEL_RECOIL:
        return S
    dkidx, dkern = dE_step_kernels(leg, pdg)
    for s_ in range(len(nrate)):
        if nrate[s_] <= 0.0:
            continue
        vg, gq = dkern[dkidx[s_]]
        S += nucel_qop_exponent(tau, nrate[s_], vg, gq, wq_row[s_])
    if NUCEL_JOINT and _has_composition(leg):
        ms = np.asarray(leg["ms"])
        m = mass_of(pdg)
        p0 = float(ms[0, 3])
        ekin = (np.sqrt(p0 * p0 + m * m) - m) * 1e3
        for s_ in range(len(nrate)):
            if nrate[s_] <= 0.0:
                continue
            comp = step_composition(leg, s_)
            jt = material_joint(pdg, comp, ekin) if comp else None
            ps = float(ms[s_, 3])
            S += joint_exponent(tau, nrate[s_], jt, wb_mid[s_], wq_row[s_],
                                np.sqrt(ps * ps + m * m), ps)
    return S


def leg_exponent(S, leg, A_ms, A_ms_start, A_ioni, avec, sigma, tau, nsub):
    """The nuclear-elastic exponent of one clean-propagation leg at plane k,
    INTO `S`: its scattering rows at the exact transport weights (`cf_rows`),
    NSUB sub-steps for the angular family.  The recoil takes the leg-mean
    ionisation weight (the ionisation and scattering logs are not parallel)
    and the joint term the step-midpoint angular weight.  The one place both
    CF builders call -- `cf_propagation_test.model_phi` (the closure) and
    `cgf_channels.block_cf_exponent` (the Fisher scale)."""
    if not NUCEL_CHANNEL or not len(leg["ms"]) or pdg_from_leg(leg) is None:
        return S
    import cf_rows
    rid, wq_s, wb, frac = cf_rows.transport_entries(leg, A_ms, A_ms_start, avec,
                                                    sigma, nsub, "length")
    _, wb_end = cf_rows.end_weights(leg, A_ms, avec, sigma)
    _, wb_start = cf_rows.end_weights(leg, A_ms_start, avec, sigma)
    wb_mid = 0.5 * (wb_end + wb_start)
    wq_eff = 0.0
    if NUCEL_RECOIL and len(leg["ioni"]):
        # the leg-mean ionisation weight per MeV: column 10 of the ionisation
        # record is q/p per GeV (the 1e-3: per MeV of recoil)
        wq_all, _ = cf_rows.end_weights(leg, A_ioni, avec, sigma)
        cs_all = np.asarray(leg["ioni"])[:, 10] * 1e-3
        wq_eff = float(np.mean(wq_all * cs_all)) \
            if len(wq_all) == len(cs_all) \
            else float(np.mean(wq_all) * np.mean(cs_all))
    wq_row = np.full(len(leg["ms"]), wq_eff)
    return rows_exponent(S, leg, tau, rid, wb, frac, wq_row, wb_mid)


def prewarm(legs_list, nproc=None):
    """Tabulate, in a forked pool, every bucket these legs need -- angular
    kernel, recoil kernel (NUCEL_RECOIL), joint table (NUCEL_JOINT) -- so the
    persisted tables are there before `warm` runs serially in the parent.
    Only buckets whose driver samples are cached are farmed out; a missing
    sample set is left to `warm`, which holds the per-bucket lock."""
    if not NUCEL_CHANNEL:
        return 0
    jobs = set()
    for legs in legs_list:
        for leg in legs:
            ms = np.asarray(leg["ms"])
            if not len(ms):
                continue
            pdg = pdg_from_leg(leg)
            if pdg is None:
                continue
            m = mass_of(pdg)
            p_ = float(ms[0, 3])
            ekin = (np.sqrt(p_ * p_ + m * m) - m) * 1e3
            if _has_composition(leg):
                for s in range(len(ms)):
                    for z, a, _ in step_composition(leg, s):
                        jobs.add((int(pdg), z, a, float(ekin)))
    jobs = sorted(j for j in jobs if os.path.exists(
        os.path.join(_cachedir(), f"nucel_{_bucket_tag(*j)}.npz")))
    if not jobs:
        return 0
    import multiprocessing as mp
    n = nproc or min(len(jobs), max(1, len(os.sched_getaffinity(0)) // 2))
    with mp.get_context("fork").Pool(processes=n) as pool:
        pool.map(_prewarm_one, jobs, chunksize=1)
    return len(jobs)


def _prewarm_one(job):
    pdg, z, a, ekin = job
    species_kernel(pdg, z, a, ekin)
    if NUCEL_RECOIL:
        dE_kernel(pdg, z, a, ekin)
    if NUCEL_JOINT:
        element_joint(pdg, z, a, ekin)


_KERNEL_CACHE = {}


def species_kernel(pdg, Z, A, ekin):
    """(mu [cm^2/g], ugrid, gtab) for one bucket.

    `gtab[i] = < J0(ugrid[i] * theta) >` over the sampled deflections, i.e. the
    projected single-collision CF.  Interpolated in log u by `_g_of`.
    """
    if abs(int(pdg)) == 13:
        raise SystemExit(
            "cf_nucel_exact: the muon has no hadElastic process; there is no "
            "nuclear elastic kernel for it.  This is the physics null -- the "
            "channel must not be applied to muons at all.")
    key = _bucket(pdg, Z, A, ekin)
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    kf = _table_path("kern", pdg, Z, A, ekin,
                     (NUCEL_NU, NUCEL_NBIN, NUCEL_UMIN, NUCEL_UMAX))
    got = _table_load(kf, ("mu", "u", "g"))
    if got is not None:
        out = (float(got[0]), got[1], got[2])
        _KERNEL_CACHE[key] = out
        return out
    mu, theta = _run_driver(pdg, Z, A, ekin)
    u = np.concatenate(([0.0], np.geomspace(NUCEL_UMIN, NUCEL_UMAX, NUCEL_NU)))
    # g(u) = <J0(u*theta)> over the SAMPLED deflections, evaluated as a
    # quadrature over the empirical distribution rather than as a direct mean
    # over 2e6 samples: the direct form is 4001 x 2e6 = 8e9 J0 evaluations per
    # species and dominates the whole closure.  Binning theta on a fine LOG
    # grid first makes it 4001 x NUCEL_NBIN and changes nothing that matters --
    # the residual phase error within a bin is u*theta*dln(theta), which is
    # ~1e-2 rad only where u*theta >~ 100, and there g is already consistent
    # with zero so the term has collapsed to the constant -nrate.
    pos = theta[theta > 0.0]
    nzero = theta.size - pos.size
    if pos.size == 0:
        raise SystemExit("cf_nucel_exact: every sampled deflection is zero -- "
                         "the model was not initialised (see the couple-table "
                         "trap in nucel_g4driver.cc)")
    edges = np.geomspace(pos.min(), pos.max() * (1.0 + 1e-12), NUCEL_NBIN + 1)
    cnt, _ = np.histogram(pos, bins=edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    wgt = cnt / float(theta.size)
    g = np.empty(len(u))
    CH = 256
    for i0 in range(0, len(u), CH):
        uu = u[i0:i0 + CH]
        g[i0:i0 + CH] = (j0(uu[:, None] * ctr[None, :]) * wgt[None, :]).sum(axis=1)
    # exactly-zero deflections contribute J0(0) = 1 at every u
    g += nzero / float(theta.size)
    out = (mu, u, g)
    _table_store(kf, mu=mu, u=u, g=g)
    _KERNEL_CACHE[key] = out
    return out


def dE_kernel(pdg, Z, A, ekin):
    """(vgrid, gqtab) with gqtab[i] = < exp(i*vgrid[i]*dE) > over the sampled
    recoils -- the single-collision CF of the recoil energy loss.

    Complex, unlike the angular kernel: the recoil is one-signed (the primary
    always LOSES energy), so this term carries skew, exactly as the ionization
    and radiative channels do.
    """
    key = _bucket(pdg, Z, A, ekin) + ("dE",)
    hit = _DEK_CACHE.get(key)
    if hit is not None:
        return hit
    kf = _table_path("dek", pdg, Z, A, ekin, (NUCEL_NU, NUCEL_NBIN))
    got = _table_load(kf, ("v", "g"))
    if got is not None:
        out = (got[0], got[1])
        _DEK_CACHE[key] = out
        return out
    _, theta = _run_driver(pdg, Z, A, ekin)
    prefix = None
    # the driver writes eloss alongside theta; recover it from the same bucket
    import glob as _g
    tag = hashlib.sha256(repr((_bucket(pdg, Z, A, ekin), NUCEL_NSAMP,
                               NUCEL_SEED)).encode()).hexdigest()[:16]
    cand = sorted(_g.glob(os.path.join(_cachedir(), f"drv_{tag}_*.eloss.bin")))
    if not cand:
        raise SystemExit(f"cf_nucel_exact: no eloss for bucket {key}; the "
                         f"driver scratch files were cleaned -- rerun it")
    dE = np.fromfile(cand[-1], dtype=np.float64)
    v = np.concatenate(([0.0], np.geomspace(1e-6, 1e4, NUCEL_NU)))
    pos = dE[dE > 0.0]
    nz = dE.size - pos.size
    edges = np.geomspace(pos.min(), pos.max() * (1 + 1e-12), NUCEL_NBIN + 1)
    cnt, _ = np.histogram(pos, bins=edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    wgt = cnt / float(dE.size)
    g = np.empty(len(v), dtype=np.complex128)
    CH = 256
    for i0 in range(0, len(v), CH):
        vv = v[i0:i0 + CH]
        g[i0:i0 + CH] = (np.exp(1j * vv[:, None] * ctr[None, :]) * wgt[None, :]).sum(axis=1)
    g += nz / float(dE.size)
    out = (v, g)
    _table_store(kf, v=v, g=g)
    _DEK_CACHE[key] = out
    return out


_DEK_CACHE = {}


def dE_kernel_for(leg, pdg):
    """dE kernel for the leg's DOMINANT target (the one carrying most xg).

    The recoil is a small correction to a small channel, so a per-leg dominant
    target is enough here; the ANGULAR kernel is per step, where it mattered.
    """
    ms = np.asarray(leg["ms"])
    xg = ms[:, 2].astype(np.float64)
    zs = np.rint(ms[:, 0].astype(np.float64)).astype(int)
    as_ = np.rint(ms[:, 1].astype(np.float64)).astype(int)
    best, bw = None, -1.0
    for z, a in set(zip(zs.tolist(), as_.tolist())):
        if z < 1 or a < 1:
            continue
        w = float(xg[(zs == z) & (as_ == a)].sum())
        if w > bw:
            best, bw = (z, a), w
    p_ = float(ms[0, 3])
    m = mass_of(pdg)
    ekin = (np.sqrt(p_ * p_ + m * m) - m) * 1e3
    return dE_kernel(pdg, float(best[0]), float(best[1]), ekin)


def nucel_qop_exponent(tau, nrate, vgrid, gqtab, wq):
    """Compound-Poisson exponent of the RECOIL energy loss for one (sub-)step.

        S(t) = nrate * ( < exp(i * t * wq * dE) > - 1 )

    `wq` is the qop weight the ionization channel uses (q * (a^T A)_qop / sigma)
    times the record's cs = E/p^3, i.e. it already converts MeV to d(q/p).
    NOT centred: unlike ionization and radiative, the propagator's mean-loss
    table never subtracted a nuclear-elastic recoil, so the mean belongs here.
    """
    if nrate <= 0.0 or wq == 0.0:
        return np.zeros(len(tau), dtype=np.complex128)
    x = np.abs(wq) * np.asarray(tau, dtype=np.float64)
    out = np.ones(len(x), dtype=np.complex128)
    m = x > vgrid[1]
    if m.any():
        xl = np.log(np.clip(x[m], vgrid[1], vgrid[-1]))
        lv = np.log(vgrid[1:])
        out[m] = (np.interp(xl, lv, gqtab[1:].real)
                  + 1j * np.sign(wq) * np.interp(xl, lv, gqtab[1:].imag))
    return nrate * (out - 1.0)


def _g_of(ugrid, gtab, x):
    """E_theta[J0(x*theta)] by log-u interpolation, clamped to the table.

    Below the table g -> 1 (no kick can be resolved); above it g -> the last
    tabulated value, which is already consistent with zero to MC precision.
    Interpolating in log u rather than u matters: the kernel falls over four
    decades of u and a linear grid would need ~10^6 points for the same
    accuracy near the knee.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    m = x > ugrid[1]
    if m.any():
        xl = np.log(np.clip(x[m], ugrid[1], ugrid[-1]))
        out[m] = np.interp(xl, np.log(ugrid[1:]), gtab[1:])
    return out


def nucel_step_exponent(tau, nrate, ugrid, gtab, w):
    """Compound-Poisson log-CF exponent of ONE (sub-)step, in standardized-z
    units.

        S(t) = nrate * ( E_theta[J0(t * w * theta)] - 1 )

    `nrate` is the expected number of elastic collisions in this (sub-)step and
    `w` the same `weff` the MS channel builds -- the quadrature sum of the two
    projected-angle transport weights divided by sigma.

    Real-valued: J0 is real and the kick is symmetric, so this channel adds no
    skew.  Returned complex only so it can be added to the complex exponent
    without an upcast at every call site.
    """
    if nrate <= 0.0 or w <= 0.0:
        return np.zeros(len(tau), dtype=np.complex128)
    g = _g_of(ugrid, gtab, np.abs(w) * np.asarray(tau, dtype=np.float64))
    # g - 1 is the small quantity; for w*tau*theta << 1 it is -(w*tau*theta)^2/4
    # and cancels to ~1e-16 if formed as (g) - (1).  np.interp already returns
    # g to full precision, and the subtraction is exact in IEEE for g near 1,
    # so no expm1-style rewrite is needed -- but the guard on the ARGUMENT
    # rather than on nrate is deliberate, as in the delta-ray term.
    return (nrate * (g - 1.0)).astype(np.complex128)


# masses in GeV; the ONLY species this channel is defined for (plus the muon,
# which is defined to have none).  Ordered so the lookup is unambiguous.
_SPECIES_MASS = ((0.13957039, 211), (0.493677, 321), (0.93827209, 2212))
_MMU = 0.1056583745
_MASS_TOL = 5e-3          # GeV; the three masses are 350 MeV apart


def pdg_from_leg(leg):
    """Recover the PDG code of the propagated particle from an exported leg.

    Mass comes from the MS record -- m = p sqrt(1-beta^2)/beta -- and the SIGN
    from sign(refqop).  Both are needed: the proton and the antiproton have the
    same mass but different elastic models (G4ChipsElasticModel vs
    G4AntiNuclElastic) and different cross-section datasets, so a mass-only
    lookup would silently give the antiproton the proton's physics.  That is
    the same class of bug as the Kokoulin species guard (NOTES_BARKAS s9.1).

    Returns None when the record is not one of the species this channel is
    defined for, so the caller can skip rather than guess.
    """
    ms = np.asarray(leg["ms"])
    if ms.size == 0:
        return None
    p, beta = float(ms[0, 3]), float(ms[0, 4])
    if not (0.0 < beta < 1.0) or p <= 0.0:
        return None
    m = p * np.sqrt(1.0 - beta * beta) / beta
    if abs(m - _MMU) < _MASS_TOL:
        return None                      # the muon: no hadElastic, by design
    q = np.sign(float(leg.get("refqop", 0.0))) or 1.0
    for m0, pdg in _SPECIES_MASS:
        if abs(m - m0) < _MASS_TOL:
            return int(pdg) * int(q)
    return None


def mass_of(pdg):
    """Rest mass in GeV for the species this channel supports."""
    for m0, p0 in _SPECIES_MASS:
        if abs(int(pdg)) == p0:
            return m0
    raise KeyError(f"cf_nucel_exact: no mass for pdg {pdg}")


def _has_composition(leg):
    return (NUCEL_ELEMENTS and leg.get("msmat") is not None
            and leg.get("mattab") is not None)


def step_composition(leg, s):
    """((Z, A, w), ...) of MS step s's material: Geant4's own elements, atomic
    masses [g/mole] and MASS fractions, rounded only for use as a cache key.
    Elements with no mass share are dropped."""
    m = leg["mattab"][int(leg["msmat"][s])]
    return tuple((round(float(z), 6), round(float(a), 6), round(float(w), 12))
                 for z, a, w in zip(m["Z"], m["A"], m["W"])
                 if z >= 1.0 and w > 0.0)


_MIX_CACHE = {}


def material_kernel(pdg, comp, ekin):
    """(mu, ugrid, gtab) of a compound: the rate-weighted mixture of its
    elements' kernels.  A collision in the compound picks element i with
    probability w_i mu_i / sum_j w_j mu_j, so the single-collision CF is that
    mixture and the per-gram rate is sum_i w_i mu_i.  Every element kernel is
    tabulated on the same u grid, so the mixture is exact on the grid."""
    key = (int(pdg), comp, float(f"{float(ekin):.4g}"))
    hit = _MIX_CACHE.get(key)
    if hit is not None:
        return hit
    tot, gmix, ug = 0.0, None, None
    for z, a, w in comp:
        mu, u, g = species_kernel(pdg, z, a, ekin)
        r = w * mu
        tot += r
        gmix = r * g if gmix is None else gmix + r * g
        ug = u
    out = (tot, ug, gmix / tot) if tot > 0.0 else None
    _MIX_CACHE[key] = out
    return out


_DEMIX_CACHE = {}


def material_dE_kernel(pdg, comp, ekin):
    """(vgrid, gqtab) of a compound's recoil: the same rate-weighted mixture
    as `material_kernel`, of the per-element recoil CFs (common v grid)."""
    key = (int(pdg), comp, float(f"{float(ekin):.4g}"))
    hit = _DEMIX_CACHE.get(key)
    if hit is not None:
        return hit
    tot, gmix, vg = 0.0, None, None
    for z, a, w in comp:
        mu, _ = _run_driver(pdg, z, a, ekin)
        v, g = dE_kernel(pdg, z, a, ekin)
        r = w * mu
        tot += r
        gmix = r * g if gmix is None else gmix + r * g
        vg = v
    out = (vg, gmix / tot) if tot > 0.0 else None
    _DEMIX_CACHE[key] = out
    return out


def dE_step_kernels(leg, pdg):
    """(kidx[nsteps], kernels) for the recoil, one kernel per MS step's
    material, aligned with `leg_rates`.  Without a material table every step
    gets the leg's dominant rounded target -- the original per-leg kernel, so
    that path is unchanged bit for bit."""
    ms = np.asarray(leg["ms"])
    if not _has_composition(leg):
        return np.zeros(len(ms), dtype=int), [dE_kernel_for(leg, pdg)]
    m = mass_of(pdg)
    p_ = float(ms[0, 3])
    ekin = (np.sqrt(p_ * p_ + m * m) - m) * 1e3
    kidx = np.zeros(len(ms), dtype=int)
    kernels, seen = [], {}
    for s in range(len(ms)):
        comp = step_composition(leg, s)
        if not comp:
            continue
        if comp not in seen:
            k = material_dE_kernel(pdg, comp, ekin)
            if k is None:
                continue
            seen[comp] = len(kernels)
            kernels.append(k)
        kidx[s] = seen[comp]
    if not kernels:
        kernels.append((np.array([0.0, 1.0]), np.array([1.0 + 0j, 1.0 + 0j])))
    return kidx, kernels


def leg_rates(leg, pdg, mass_gev, nsub=1):
    """Per-MS-step expected collision counts for one leg, with a PER-STEP target.

    Returns (nrate[nsteps], kidx[nsteps], kernels) where kernels[kidx[s]] is the
    (ugrid, gtab) pair for step s, or None when the leg has no material.

    THE TARGET VARIES WITHIN A LEG, so Z and A are read PER STEP and never
    from ms[0].  On the layered toy leg 0 contains effZ in {1, 4, 7.37, 8}, and
    even legs that start at Z=8 contain Z=1 steps.  Reading only the first step
    assigns 8.8 % of the modelled path (legs 0 and 2, whose first step happens
    to be Z=1) to a HYDROGEN target -- and pbar-hydrogen is both the largest
    per-nucleon elastic cross section of anything here and the one case
    G4AntiNuclElastic special-cases (theTargetDef == theProton), so it
    over-injects collisions with the wrong kernel, worst for the antiproton.

    With a material table on the leg (`msmat` + `mattab`, see
    `step_composition`) and NUCEL_ELEMENTS on, every step's material is the
    rate-weighted mixture of its elements (`material_kernel`), so minority
    nuclei -- above all hydrogen in composites -- are targets in their own
    right.  Without the table (older exports) Z and A are rounded to integers
    for the bucket because the driver builds a real G4Material and the
    exported effZ can be fractional (7.374 for a mixture); that rounding
    replaces a compound by its dominant element and DROPS its hydrogen, which
    the real-material closure showed fails the far tails at up to -24 sigma.
    """
    ms = np.asarray(leg["ms"])
    if ms.size == 0:
        return None
    xg = ms[:, 2].astype(np.float64)                 # g/cm^2 per step
    p = float(ms[0, 3])                              # GeV
    ekin = (np.sqrt(p * p + mass_gev ** 2) - mass_gev) * 1e3   # MeV

    nrate = np.zeros(len(xg))
    kidx = np.zeros(len(xg), dtype=int)
    kernels = []
    seen = {}
    if _has_composition(leg):
        # per-element targets from the exported material table
        for s in range(len(xg)):
            comp = step_composition(leg, s)
            mk = material_kernel(pdg, comp, ekin) if comp else None
            if mk is None:
                if not kernels:
                    kernels.append((np.array([0.0, 1.0]), np.array([1.0, 1.0])))
                continue
            if comp not in seen:
                seen[comp] = (len(kernels), mk[0])
                kernels.append((mk[1], mk[2]))
            i, mu = seen[comp]
            kidx[s] = i
            nrate[s] = mu * xg[s] / max(int(nsub), 1)
        if not kernels:
            return None
        return nrate, kidx, kernels
    zs = np.rint(ms[:, 0].astype(np.float64)).astype(int)
    as_ = np.rint(ms[:, 1].astype(np.float64)).astype(int)
    for s in range(len(xg)):
        key = (int(zs[s]), int(as_[s]))
        if key[0] < 1 or key[1] < 1:
            nrate[s] = 0.0
            kidx[s] = 0
            if not kernels:
                kernels.append((np.array([0.0, 1.0]), np.array([1.0, 1.0])))
            continue
        if key not in seen:
            mu, ug, gt = species_kernel(pdg, float(key[0]), float(key[1]), ekin)
            seen[key] = (len(kernels), mu)
            kernels.append((ug, gt))
        i, mu = seen[key]
        kidx[s] = i
        nrate[s] = mu * xg[s] / max(int(nsub), 1)
    if not kernels:
        return None
    return nrate, kidx, kernels
