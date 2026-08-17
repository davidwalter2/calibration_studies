"""Nuclear elastic (`hadElastic`) noise channel for the clean-propagation CF.

WHY THIS CHANNEL EXISTS
-----------------------
`cf_propagation_test.model_phi` carried three channels -- ionization, multiple
scattering, radiative -- and no nuclear elastic term, while the simulation runs
`hadElastic` for every hadron.  The elastic/inelastic split (hadron_probe arms
`elonly`/`inelonly`, 2026-08-17) showed:

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
fitted; see NOTES for the parameter-free check (shift/N_pred constant to 4 %
across four species and four different G4 models).

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
# _PHI_CACHE key and the sha256 scale-cache identity (trap #8).
# --------------------------------------------------------------------------
NUCEL_CHANNEL = bool(int(os.environ.get("NUCEL_CHANNEL", "0")))

PHYSICS_GLOBALS = ("NUCEL_CHANNEL",)
_NOT_PHYSICS = ("NUCEL_NSAMP", "NUCEL_SEED", "NUCEL_NU", "NUCEL_UMIN",
                "NUCEL_UMAX", "NUCEL_NBIN")

# sampling/tabulation knobs -- resolution of the numerics, not physics
NUCEL_NSAMP = int(os.environ.get("NUCEL_NSAMP", "2000000"))
NUCEL_SEED = int(os.environ.get("NUCEL_SEED", "20260817"))
NUCEL_NU = 4000          # points in the log-u kernel table
NUCEL_NBIN = 100000      # log-theta bins used to quadrature the sampled kernel
NUCEL_UMIN = 1e-4        # below this J0 -> 1 to double precision for our thetas
NUCEL_UMAX = 1e9


def physics_state():
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)


# --------------------------------------------------------------------------
# driver plumbing
# --------------------------------------------------------------------------

def _driver():
    import hadron_probe as hp
    return os.path.join(hp.SCRATCH, "nucel_g4driver.sh")


def _cachedir():
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
    SAME bucket within milliseconds of each other.  The first version wrote the
    .npz with a plain savez and read it with a plain load, and the run died
    with `BadZipFile: File is not a zip file` -- a worker had read a file
    another worker was still writing.  Two fixes, both needed:
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
    # trap #11: repr(np.float64(x)) is "np.float64(x)", which atof() reads as
    # 0.0.  Every numeric argument goes through float() and "%.17g".
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
    _KERNEL_CACHE[key] = out
    return out


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
    # rather than on nrate is deliberate (the delta-ray term's lesson).
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


def leg_rates(leg, pdg, mass_gev, nsub=1):
    """Per-MS-step expected collision counts for one leg, with a PER-STEP target.

    Returns (nrate[nsteps], kidx[nsteps], kernels) where kernels[kidx[s]] is the
    (ugrid, gtab) pair for step s, or None when the leg has no material.

    THE TARGET VARIES WITHIN A LEG.  The first version read Z, A from ms[0] and
    applied them to the whole leg.  That is wrong: on the layered toy leg 0
    contains effZ in {1, 4, 7.37, 8} and even legs that start at Z=8 contain
    Z=1 steps.  Reading only the first step assigned 8.8 % of the modelled path
    (legs 0 and 2, whose first step happens to be Z=1) to a HYDROGEN target --
    and pbar-hydrogen is both the largest per-nucleon elastic cross section of
    anything here and the one case G4AntiNuclElastic special-cases
    (theTargetDef == theProton), so it over-injected collisions with the wrong
    kernel, worst for the antiproton.

    Z and A are rounded to integers for the bucket because the driver builds a
    real G4Material and the exported effZ can be fractional (7.374 for a
    mixture); the rounding is an approximation of a mixture by its dominant
    element, which is what an "effective Z" already is.
    """
    ms = np.asarray(leg["ms"])
    if ms.size == 0:
        return None
    xg = ms[:, 2].astype(np.float64)                 # g/cm^2 per step
    p = float(ms[0, 3])                              # GeV
    ekin = (np.sqrt(p * p + mass_gev ** 2) - mass_gev) * 1e3   # MeV

    zs = np.rint(ms[:, 0].astype(np.float64)).astype(int)
    as_ = np.rint(ms[:, 1].astype(np.float64)).astype(int)
    nrate = np.zeros(len(xg))
    kidx = np.zeros(len(xg), dtype=int)
    kernels = []
    seen = {}
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
