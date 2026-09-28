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
tabulated and interpolated -- the same device `gshape` uses for Moliere.

WHAT IS TAKEN FROM GEANT4 AND WHAT IS NOT
-----------------------------------------
Everything.  The rate is `G4CrossSectionDataStore::GetCrossSection(dp, mat)` --
the very call `G4HadronicProcess` inverts for its mean free path -- evaluated
on the species-correct dataset that stock `G4HadronElasticPhysics` registers.
The kernel is the empirical distribution of the primary's deflection and
recoil under `ApplyYourself`, i.e. what the four models ACTUALLY draw, not a
diffractive parameterisation of them.  Both come from `nucel_g4driver`.
Nothing here is fitted; see Documents/Resolution/NUCLEAR_ELASTIC.md for the
parameter-free check (shift/N_pred constant to 4 % across four species and
four different G4 models).

THE TABLES.  The driver's samples are reduced once per (species, ELEMENT,
momentum node) by `ksclosure/nucel/nucel_tables.py` into the binary the
makers read, `TrackPropagation/Geant4e/data/cvhcf_nucel_v2.bin`
(`NUCEL_TABLE`): the rate on a fine momentum grid, the angular CF, the recoil
density and the joint (theta, dE) law, with compounds as rate-weighted
mixtures of their elements and every kernel at the step's own momentum
(the two nodes of its model segment that bracket it).  This module evaluates
them through `nucel_tables.Table.rows`, of which the in-maker
`cvhcf::nucelRows` is the port -- one model, one table, for the
clean-propagation closure and the fit.

THE MUON IS THE NULL.  It has no `hadElastic` at all and the tables carry no
muon, so `pdg_from_leg` returns None for it and the channel is skipped rather
than returning a zero that could be mistaken for a computed result.
"""

import fcntl
import hashlib
import os
import subprocess
import sys

import numpy as np

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
# With this ON each step's material is a rate-weighted mixture of its
# elements' kernels, for the angle, the recoil and the joint law: the material
# comes from the export's `msmatv` + `materials` tree, or, for records
# written without them, from the step's exported (effZ, effA, zzp1OverA)
# (`nucel_tables.Table.identify`, exact against the Geant4 table).
# NUCEL_ELEMENTS=0 replaces every step's material by its rounded element
# (Z = rint(effZ), A = rint(effA)) to A/B the minority targets.
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

# The shipped per-element tables (see THE TABLES above).  A physics switch: a
# different table is a different kernel.
NUCEL_TABLE = os.environ.get(
    "NUCEL_TABLE", "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/"
                   "TrackPropagation/Geant4e/data/cvhcf_nucel_v2.bin")

PHYSICS_GLOBALS = ("NUCEL_CHANNEL", "NUCEL_RECOIL", "NUCEL_ELEMENTS",
                   "NUCEL_JOINT", "NUCEL_TABLE")
_NOT_PHYSICS = ("NUCEL_NSAMP", "NUCEL_SEED")

# the driver's sampling knobs (`_run_driver`, the diagnostics' direct runs)
NUCEL_NSAMP = int(os.environ.get("NUCEL_NSAMP", "2000000"))
NUCEL_SEED = int(os.environ.get("NUCEL_SEED", "20260817"))


def physics_state():
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)


# --------------------------------------------------------------------------
# direct driver runs: the diagnostics' own samples at one (species, target,
# energy) -- the recoil census, the per-element tail models.  The channel
# itself reads only the tables.
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
    """`NUCEL_CACHEDIR` relocates the sample cache.  Buckets are keyed on
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
    """The cache key of one driver run: ekin binned at 1 %, as the radiative
    driver bins it (the elastic cross section and kernels vary by well under a
    per cent over it)."""
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

    CONCURRENCY.  A diagnostic under a multiprocessing pool reaches this
    function from several workers for the SAME bucket within milliseconds of
    each other.  A plain savez plus a plain load kills the run with
    `BadZipFile: File is not a zip file` -- a worker reading a file another
    worker is still writing.  Two mechanisms, both needed:
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


# --------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------
_TABLE = [None]


def table():
    """The shipped tables (`NUCEL_TABLE`) as a `nucel_tables.Table`, loaded once
    per path; its material mixtures are built on first use and kept."""
    t = _TABLE[0]
    if t is None or t.path != NUCEL_TABLE:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ksclosure", "nucel")
        if d not in sys.path:
            sys.path.insert(0, d)
        import nucel_tables
        t = nucel_tables.Table(NUCEL_TABLE)
        _TABLE[0] = t
    return t


_FILEMATS = {}
_ROUNDED = {}


def _filemats(mattab):
    """The leg's material table ({index: dict(name, Z, A, W)}) as the list
    `Table.register` takes; one list per table object, so the registration is
    cached across the legs of a file."""
    hit = _FILEMATS.get(id(mattab))
    if hit is None or hit[0] is not mattab:
        hit = (mattab, [dict(index=int(i), name=m["name"], Z=m["Z"], A=m["A"], W=m["W"])
                        for i, m in sorted(mattab.items())])
        _FILEMATS[id(mattab)] = hit
    return hit[1]


def leg_materials(T, leg):
    """Table material row of every MS row of the leg: through `msmat` and the
    leg's material table when it carries them, else identified from the
    row's exported (effZ, effA, zzp1OverA) (columns 0, 1, 7).  -1: not
    resolved.  Under NUCEL_ELEMENTS=0 every row's material is its rounded
    element instead."""
    ms = np.asarray(leg["ms"])
    if not NUCEL_ELEMENTS:
        out = np.full(len(ms), -1, dtype=np.int64)
        for s_ in range(len(ms)):
            z, a = int(np.rint(ms[s_, 0])), int(np.rint(ms[s_, 1]))
            if z < 1 or a < 1:
                continue
            key = (id(T), z, a)
            if key not in _ROUNDED:
                _ROUNDED[key] = T._add_material(dict(index=-(10 ** 6) - len(_ROUNDED),
                                                     name=f"rounded:Z{z}A{a}",
                                                     Z=[float(z)], A=[float(a)], W=[1.0]))
            out[s_] = _ROUNDED[key]
        return out
    if leg.get("msmat") is not None and leg.get("mattab") is not None:
        rowmap = T.register(_filemats(leg["mattab"]))
        return np.array([rowmap.get(int(i), -1) for i in leg["msmat"]], dtype=np.int64)
    r = T.identify(ms[:, 0], ms[:, 1], ms[:, 7])
    return np.where(r >= 0, r, -1)


def warm(legs):
    """Load the tables and build every material mixture these legs need, in
    the PARENT process: `plane_scales` and `closure_rows` fan out over planes
    with a forking pool, and the children then inherit the mixtures instead
    of each rebuilding them.  Safe to call when the channel is off (it
    returns immediately) or on a muon (nothing to build)."""
    if not NUCEL_CHANNEL:
        return 0
    T = table()
    n = 0
    for leg in legs:
        if not len(np.asarray(leg["ms"])):
            continue
        pdg = pdg_from_leg(leg)
        if pdg is None:
            continue
        isp = T.species_index(pdg)
        for im in np.unique(leg_materials(T, leg)):
            if im >= 0:
                T.mixture(isp, int(im))
                n += 1
    return n


def prewarm(legs_list, nproc=None):
    """`warm` over several leg sets (the ditrack's two legs); the mixtures are
    cheap enough to build serially, so `nproc` is not used."""
    return sum(warm(legs) for legs in legs_list)


def rows_exponent(S, leg, tau, rid, wb, frac, wq_row, wb_mid):
    """THE nuclear-elastic exponent of one leg's scattering ROWS (`cf_rows`),
    accumulated INTO `S` in place -- `nucel_tables.Table.rows` on the leg's
    rows, species and materials:
      angular  per entry (row rid, angular weight wb, share frac of the row's
               collisions);
      recoil   per row at weight wq_row[s] [z per MeV of recoil energy: the
               q/p weight times the step's q/p per MeV, cs 1e-3]
               (NUCEL_RECOIL);
      joint    per row at (wb_mid[s], wq_row[s]) (NUCEL_JOINT, with the
               recoil), X = T_eff(dE) under cf_knockon.QOP_EXACT.
    `leg` carries the rows (`ms`), their materials (`msmat`, `mattab`) and
    the species (`pdg_from_leg`)."""
    if not NUCEL_CHANNEL or not len(leg["ms"]):
        return S
    pdg = pdg_from_leg(leg)
    if pdg is None:
        return S
    import cf_knockon
    T = table()
    _, ang, rec, jnt = T.rows(tau, np.asarray(leg["ms"]), leg_materials(T, leg), pdg,
                              mass_of(pdg), rid, wb, frac, wq_row, wb_mid,
                              recoil=NUCEL_RECOIL, joint=NUCEL_JOINT,
                              exact=bool(cf_knockon.QOP_EXACT))
    S += ang
    S += rec
    S += jnt
    return S


def leg_entries(leg, A_ms, A_ms_start, A_ioni, avec, sigma, nsub):
    """The entries of one clean-propagation leg at plane k: its scattering rows
    at the exact transport weights (`cf_rows.transport_entries`, NSUB
    sub-steps, for the angular family), the recoil weight per row -- the
    leg-mean ionisation weight per MeV (the ionisation and scattering logs are
    not parallel) -- and the step-midpoint angular weight of the joint term.
    (rid, wb, frac, wq_row, wb_mid)."""
    import cf_rows
    rid, _, wb, frac = cf_rows.transport_entries(leg, A_ms, A_ms_start, avec,
                                                 sigma, nsub, "length")
    _, wb_end = cf_rows.end_weights(leg, A_ms, avec, sigma)
    _, wb_start = cf_rows.end_weights(leg, A_ms_start, avec, sigma)
    wb_mid = 0.5 * (wb_end + wb_start)
    wq_eff = 0.0
    if NUCEL_RECOIL and len(leg["ioni"]):
        # column 10 of the ionisation record is q/p per GeV (the 1e-3: per
        # MeV of recoil)
        wq_all, _ = cf_rows.end_weights(leg, A_ioni, avec, sigma)
        cs_all = np.asarray(leg["ioni"])[:, 10] * 1e-3
        wq_eff = float(np.mean(wq_all * cs_all)) \
            if len(wq_all) == len(cs_all) \
            else float(np.mean(wq_all) * np.mean(cs_all))
    return rid, wb, frac, np.full(len(leg["ms"]), wq_eff), wb_mid


def leg_exponent(S, leg, A_ms, A_ms_start, A_ioni, avec, sigma, tau, nsub):
    """The nuclear-elastic exponent of one clean-propagation leg at plane k,
    INTO `S`, at the entries of `leg_entries`.  The one place both CF builders
    call -- `cf_propagation_test.model_phi` (the closure) and
    `cgf_channels.block_cf_exponent` (the Fisher scale)."""
    if not NUCEL_CHANNEL or not len(leg["ms"]) or pdg_from_leg(leg) is None:
        return S
    import cf_rows
    cx = cf_rows._cxx()
    fn = cx.rows_exponent if cx is not None else rows_exponent
    return fn(S, leg, tau, *leg_entries(leg, A_ms, A_ms_start, A_ioni, avec, sigma, nsub))


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
