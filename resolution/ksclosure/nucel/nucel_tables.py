#!/usr/bin/env python3
"""Nuclear-elastic kernel tables: one bucket per (species, ELEMENT, momentum node).

For every bucket the driver `nucel_g4driver` (the species-correct Geant4
elastic model and cross-section dataset, exactly as `cf_nucel_exact` uses it)
is run at the element's own Z and atomic mass A [g/mole] -- hydrogen at
1.00794, never a rounded mixture -- and reduced to

  g(u) - 1   on UG (log-uniform, 0.1 .. 1e7 /rad, 30 per decade):
             E_theta[J0(u theta)] - 1, the projected single-collision CF of the
             deflection
  dE density on DEE (log bins, 1e-3 .. 1e5 MeV, 32 per decade): per bin the
             partial moments (P_b, D1_b, D2_b) = E[(1, dE - c_b, (dE - c_b)^2)
             1{dE in b}] about the bin centre c_b, plus (P, E[dE], E[dE^2]) of
             the part below 1e-3 MeV
  mom        <theta>, <theta^2>, <dE>, <dE^2>
  joint      the (theta, dE) law of the SAME collisions: JBIN log-theta bins,
             each bin's mean deflection, mean recoil and share of all the
             collisions (zero deflections carry no joint term and are left out
             of the bins, not of the normalisation) -- elastic two-body
             kinematics makes dE one-to-one in theta, so the per-bin mean is
             the exact relation up to the bin width
and the rate mu = Sigma/rho [cm^2/g] on its own fine momentum grid PRATE.
Species: pi+, pi-, K+, K-, p, pbar.  Elements: every element of the job's
Geant4 material table (the `materials` tree the clean-propagation export and
the makers write), plus the elements the 2016 geometry XML references beyond
it (`EXTRA_ELEMENT_A`).  The material list carried with the tables is that
job table plus every material of the geometry XML resolved to Geant4's
element mass fractions (`xml_materials`).

THE SINGLE-COLLISION CFs.
  angular   g(u) at u in [UG_0, UG_last] by Catmull-Rom interpolation in ln u
            of the tabulated g - 1 (uniform in ln u; the end intervals use the
            linearly extrapolated neighbour); below UG_0 the second-order
            expansion 1 - u^2 <theta^2>/4; above UG_last the last value.
  recoil    h(v) = E[exp(i v dE)] from the density: within each bin the
            recoil is taken uniform over the interval with the bin's own mean
            mu_b = c_b + D1_b/P_b and variance s2_b = D2_b/P_b - (D1_b/P_b)^2, so
               h(v) = P_low + i v M1_low - v^2 M2_low / 2
                      + sum_b P_b exp(i v mu_b) sinc(v sqrt(3 s2_b))
            (v of either sign, sinc(x) = sin(x)/x).  The recoil CF oscillates
            in v wherever the recoil spectrum has structure on the scale 1/v
            (hydrogen: a kinematic end point), so it is NOT tabulated on a v
            grid -- interpolating it between grid nodes is wrong at the
            per-cent level -- and a bin-uniform density is not enough either
            (4e-2 at 16 bins per decade); matching each bin's first two
            moments brings the representation to the sampled CF's own noise
            (max 1.9e-3 at 32 per decade, measured on 48 random buckets).
            The channel takes h - 1 in the cancellation-free, exactly
            normalised form of `Table.hm1_node`.
  joint     one collision's CF is E[J0(b theta) e^{i a X(dE)}], the two
            families carry E[J0(b theta)] + E[e^{i a dE}] - 1; the difference
               sum_i W_i [(J0(b th_i) - 1)(e^{i a X_i} - 1) + (e^{i a X_i} - e^{i a dE_i})]
            over the joint bins (X = dE, or the exact 1/p map T_eff(dE) under
            cf_knockon.QOP_EXACT) is the angle-recoil joint term
            (cf_nucel_exact.NUCEL_JOINT).

COMPOUNDS.  A step's material is a compound.  A collision in it picks element
i with probability w_i mu_i / sum_j w_j mu_j (w = mass fractions), so

  mu_mat = sum_i w_i mu_i,    K_mat = sum_i w_i mu_i K_i / mu_mat

for every kernel quantity K (g - 1, the dE partial moments, the moments) --
exact, all being linear in the underlying distribution (which is why the dE
bins carry partial moments rather than means and variances).  The collision shares use the rates at the kernel's own node.
The joint bins of a compound are its elements' bins, each weighted by the
element's collision share at the node; elements below JSHARE of the node's
collisions are left out of them (trace elements; the families keep them).
An element whose atomic mass differs from the tabulated one takes
mu_i * A_tab / A (the per-atom cross section is the same).

MOMENTUM.  Kernels are tabulated on per-species nodes: the geometric grid
PGRID = geomspace(0.18, 48, 74) GeV plus a node on each side of every model
EDGE (`edges`): G4ElasticHadrNucleusHE (pi+-) switches its sampling table at
fixed kinetic energies, where its kernel jumps by up to 40 % in p*theta, and
the antiproton changes model at 100 MeV kinetic energy.  A step at momentum p
in segment s (the number of edges below p) takes
  * the kernels of the two nodes of segment s that bracket p, linearly in
    ln p: K(p) = (1-f) K_j + f K_{j+1}; the angular argument of each node is
    rescaled to fixed momentum transfer, g_j(u p_j / p); the recoil, which is
    t/2M at fixed momentum transfer, is not rescaled.  Outside the segment's
    node range: the end node alone (rescaled).
  * the rate log-log between the bracketing nodes of PRATE (no edges: the
    cross sections are continuous), constant beyond the ends.
  * the joint bins of the same two nodes, the deflection rescaled to fixed
    momentum transfer (theta p_j / p), the recoil not.
The residual interpolation error is measured against direct driver runs off
the grid by `nucel_table_checks.py rescale`.

MATERIAL LOOKUP.  A step's material is found by its index (`msmatv` + the
file's material tree) or, for step records written without one, by the
exported (effZ, effA, zzp1OverA) of the step (`Table.identify`), exact
against the Geant4 table: effZ = sum w Z, effA = sum w A, zzp1OverA = sum w
Z(Z+1)/A.

THE SHIPPED TABLE is the binary `TrackPropagation/Geant4e/data/cvhcf_nucel_v2.bin`
written by `make_cvhcf_nucel_tables.py` next to it from `assemble`'s npz;
`Table` reads that binary, so the offline references -- the fit-level
`ks_nucel_cf.step_family` and the clean-propagation `cf_nucel_exact` -- and
the in-maker port evaluate identical numbers.

usage:
  python3 nucel_tables.py elements --mattab <root with a materials tree> --out elements.json
  python3 nucel_tables.py edges    --out edges.json
  python3 nucel_tables.py build    --elements elements.json --edges edges.json --cache <dir> [--task i --ntask n]
                                   (kernel buckets and the per-element rate grids)
  python3 nucel_tables.py assemble --elements elements.json --edges edges.json --cache <dir> --out table.npz
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
from scipy.special import j0

DRIVER = os.environ.get(
    'NUCEL_DRIVER',
    '/ceph/submit/data/user/d/david_w/ZMass/cvh/nucel_trackfit_260926/tables/bin/nucel_g4driver.sh')
SPECIES = (211, -211, 321, -321, 2212, -2212)
# Geant4 11.2's own masses (what the driver converts kinetic energy with) [MeV]
G4MASS = {211: 139.5701, 321: 493.677, 2212: 938.272013}
REF_RHO = 0.1073        # g/cm^3 -- only self-consistency matters (mu is rho-free)
REF_LEN = 100.0         # mm
NSAMP = int(os.environ.get('NUCEL_NSAMP', '500000'))
SEED = 20260926
PGRID = np.geomspace(0.18, 48.0, 74)          # GeV/c, kernel base nodes, ratio 1.0795
PRATE = np.geomspace(0.18, 48.0, 561)         # GeV/c, rate nodes, ratio 1.0100
EDGE_DELTA = 1e-5                             # edge nodes at p_edge (1 -+ EDGE_DELTA)
UG = np.geomspace(1e-1, 1e7, 241)             # 1/rad, 30 nodes per decade
DE_LOW = 1e-3                                 # MeV
DEE = np.geomspace(DE_LOW, 1e5, 257)          # MeV, 256 bins, 32 per decade
VCHK = np.geomspace(1e-3, 1e3, 61)            # 1/MeV, exact sample CF of dE (checks)
UGRID = np.concatenate(([0.0], np.geomspace(1e-1, 1e7, 700)))   # 1/rad (checks)
NBIN = 50000
JBIN = 200          # joint (theta, dE) bins per bucket (log theta)
JSHARE = 1e-5       # elements below this share of a material's collisions are
                    # left out of its joint bins


def mass_mev(pdg):
    return G4MASS[abs(int(pdg))]


def ekin_mev(pdg, p):
    m = mass_mev(pdg)
    return np.sqrt((np.asarray(p, dtype=np.float64) * 1e3) ** 2 + m * m) - m


def species_nodes(pdg, edges):
    """(node momenta, segment of each node) of a species: PGRID plus a node on
    each side of every edge inside the grid."""
    e = sorted(float(x) for x in edges.get(str(int(pdg)), []))
    extra = [x * f for x in e if PGRID[0] < x < PGRID[-1]
             for f in (1.0 - EDGE_DELTA, 1.0 + EDGE_DELTA)]
    p = np.unique(np.concatenate([PGRID, extra]))
    seg = np.searchsorted(np.asarray(e), p, side='right').astype(np.int32)
    return p, seg


def bucket_seed(pdg, Z, inode):
    return SEED + 1000000 * SPECIES.index(int(pdg)) + 1000 * int(round(Z)) + int(inode)


def bucket_path(cache, pdg, Z, inode):
    return os.path.join(cache, f'b_{int(pdg)}_{int(round(Z)):03d}_{int(inode):03d}.npz')


def rate_path(cache, pdg, Z):
    return os.path.join(cache, f'r_{int(pdg)}_{int(round(Z)):03d}.npz')


# ---------------------------------------------------------------------------
# reduction of the sampled collisions
# ---------------------------------------------------------------------------

def _j0_cf(x, grid):
    """E[J0(u x)] on `grid` as a quadrature over NBIN log bins of the positive
    samples (exact zeros contribute 1).  The phase error of a bin,
    u x dln(x)/2, matters only where the CF has already decayed to noise."""
    pos = x[x > 0.0]
    nz = x.size - pos.size
    if pos.size == 0:
        return np.ones(len(grid))
    edges = np.geomspace(pos.min(), pos.max() * (1 + 1e-12), NBIN + 1)
    cnt, _ = np.histogram(pos, bins=edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    wgt = cnt / float(x.size)
    keep = wgt > 0
    ctr, wgt = ctr[keep], wgt[keep]
    out = np.empty(len(grid))
    for i0 in range(0, len(grid), 32):
        gg = grid[i0:i0 + 32]
        out[i0:i0 + 32] = (j0(gg[:, None] * ctr[None, :]) * wgt[None, :]).sum(1)
    return out + nz / float(x.size)


def joint_bins(th, de):
    """(theta, dE, w) of one bucket's collisions on JBIN log-theta bins: the
    bin's mean deflection, its mean recoil and its share of ALL collisions
    (zero deflections are left out of the bins, not of the normalisation)."""
    pos = th > 0.0
    t, e = th[pos], de[pos]
    edges = np.geomspace(t.min(), t.max() * (1.0 + 1e-12), JBIN + 1)
    idx = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, JBIN - 1)
    cnt = np.bincount(idx, minlength=JBIN).astype(np.float64)
    st = np.bincount(idx, weights=t, minlength=JBIN)
    se = np.bincount(idx, weights=e, minlength=JBIN)
    m = cnt > 0
    return st[m] / cnt[m], se[m] / cnt[m], cnt[m] / float(th.size)


def _exp_cf_exact(x, grid):
    """E[exp(i v x)] on `grid` by direct summation over the samples."""
    out = np.empty(len(grid), dtype=np.complex128)
    for k, v in enumerate(grid):
        out[k] = np.exp(1j * v * x).mean()
    return out


def de_density(de):
    """(P, M1, M2)[256] partial moments per DEE bin and (P, M1, M2) below
    DE_LOW of recoil samples de [MeV]."""
    n = float(de.size)
    if np.any(de >= DEE[-1]):
        raise RuntimeError(f'recoil above {DEE[-1]} MeV: extend DEE')
    low = de < DE_LOW
    x = de[~low]
    b = np.searchsorted(DEE, x, side='right') - 1
    nb = len(DEE) - 1
    pm = np.stack([np.bincount(b, minlength=nb) / n,
                   np.bincount(b, weights=x, minlength=nb) / n,
                   np.bincount(b, weights=x * x, minlength=nb) / n])
    return pm, np.array([low.sum() / n, de[low].sum() / n, (de[low] ** 2).sum() / n])


DEC = 0.5 * (DEE[:-1] + DEE[1:])              # MeV, the bin centres the moments refer to


def centre_moments(pm, dee=DEE):
    """(P, D1, D2) = E[(1, dE - c_b, (dE - c_b)^2) 1{dE in b}] from the raw
    partial moments (P, M1, M2)[..., 3, nbin], c_b the bin centre.  Still
    linear in the distribution, and free of the cancellation that forming a
    variance from raw moments suffers in float32."""
    c = 0.5 * (dee[:-1] + dee[1:])
    P, M1, M2 = pm[..., 0, :], pm[..., 1, :], pm[..., 2, :]
    return np.stack([P, M1 - c * P, M2 - 2.0 * c * M1 + c * c * P], axis=-2)


def h_from_density(v, pdc, low, dee=DEE):
    """h(v) = E[exp(i v dE)] from the centred partial moments pdc = (P, D1, D2)
    [3, nbin] (`centre_moments`) and low = (P, M1, M2) below DE_LOW; v any
    sign, array.  Each bin is uniform over mu_b -+ sqrt(3 s2_b), its own mean
    mu_b = c_b + D1/P and variance s2_b = D2/P - (D1/P)^2."""
    v = np.asarray(v, dtype=np.float64)
    out = low[0] + 1j * v * low[1] - 0.5 * v * v * low[2]
    P, D1, D2 = pdc
    nz = P > 0.0
    c = 0.5 * (dee[:-1] + dee[1:])[nz]
    P, D1, D2 = P[nz], D1[nz], D2[nz]
    d = D1 / P
    hw = np.sqrt(3.0 * np.maximum(D2 / P - d * d, 0.0))
    vv = v[..., None]
    return out + (P * np.exp(1j * vv * (c + d)) * np.sinc(vv * hw / np.pi)).sum(-1)


def _read_rec(fn):
    r = {}
    for line in open(fn):
        w = line.split()
        if not w or w[0].startswith('#'):
            continue
        r.setdefault(w[0], []).append(w[1:])
    return {k: (v[0] if len(v) == 1 else v) for k, v in r.items()}


def run_driver(pdg, p, out_npz, Z=None, A=None, comp=None, nsamp=NSAMP, seed=SEED,
               keep_raw=None):
    """Run the driver at momentum p [GeV] on one element (Z, A) or on a
    compound comp = [(Z, A, w), ...] and write the reduced bucket to out_npz
    (atomic publish).  keep_raw: directory to copy the raw sample files into."""
    ek = float(ekin_mev(pdg, p))
    tmpd = tempfile.mkdtemp(prefix='nucel_')
    prefix = os.path.join(tmpd, 'drv')
    cmd = [DRIVER, '--pdg', str(int(pdg)), '--rho', '%.17g' % REF_RHO,
           '--ekin', '%.17g' % ek, '--len', '%.17g' % REF_LEN,
           '--n', str(int(nsamp)), '--seed', str(int(seed)), '--out', prefix]
    if comp is None:
        cmd += ['--Z', '%.17g' % float(Z), '--A', '%.17g' % float(A)]
    else:
        cmd += ['--mat', ','.join('%.17g:%.17g:%.17g' % (float(z), float(a), float(w))
                                  for z, a, w in comp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'driver failed {cmd}: {r.stderr[-1500:]}')
        rec = _read_rec(prefix + '.rec')
        th = np.fromfile(prefix + '.theta.bin', dtype=np.float64)
        de = np.fromfile(prefix + '.eloss.bin', dtype=np.float64)
        zs = (np.fromfile(prefix + '.z.bin', dtype=np.int32)
              if os.path.exists(prefix + '.z.bin') else None)
        if keep_raw:
            os.makedirs(keep_raw, exist_ok=True)
            base = os.path.join(keep_raw, os.path.basename(out_npz)[:-4])
            for ext in ('.rec', '.theta.bin', '.eloss.bin', '.z.bin'):
                if os.path.exists(prefix + ext):
                    shutil.copy(prefix + ext, base + ext)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    if th.size == 0 or not np.any(th > 0):
        raise RuntimeError(f'empty kernel for {cmd}')
    sig = float(rec['rate'][0])            # 1/mm at REF_RHO
    mu = sig * 10.0 / REF_RHO              # cm^2/g
    pm, low = de_density(de)
    jth, jde, jw = joint_bins(th, de)
    extra = {}
    if zs is not None:
        uz, cz = np.unique(zs, return_counts=True)
        extra = dict(zhit=uz, zhitfrac=cz / float(zs.size))
    tmp = out_npz + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, mu=mu, gm1=_j0_cf(th, UG) - 1.0, g_full=_j0_cf(th, UGRID),
             pde=pm, low=low, hchk=_exp_cf_exact(de, VCHK), jth=jth, jde=jde, jw=jw,
             p=float(p), ekin=ek, plab=float(rec['input'][10]) * 1e-3,
             mass=float(rec['input'][8]),
             Z=np.nan if Z is None else float(Z), A=np.nan if A is None else float(A),
             comp=np.zeros((0, 3)) if comp is None else np.asarray(comp, dtype=np.float64),
             mom=np.array([th.mean(), (th ** 2).mean(), de.mean(), (de ** 2).mean()]),
             q50=np.quantile(th, 0.5), q99=np.quantile(th, 0.99),
             nsamp=int(th.size), seed=int(seed), model=' '.join(rec['model']),
             xs=' '.join(rec['xs']), **extra)
    os.replace(tmp, out_npz)
    return out_npz


def run_rates(pdg, Z, A, out_npz):
    """mu(p) [cm^2/g] on PRATE from one driver process (`--ekinlist`)."""
    tmpd = tempfile.mkdtemp(prefix='nucel_')
    try:
        ek = ekin_mev(pdg, PRATE)
        r = subprocess.run([DRIVER, '--pdg', str(int(pdg)), '--Z', '%.17g' % Z, '--A', '%.17g' % A,
                            '--rho', '%.17g' % REF_RHO, '--ekin', '%.17g' % ek[0],
                            '--ekinlist', ','.join('%.17g' % x for x in ek),
                            '--out', tmpd + '/r'], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'rate driver failed {pdg} {Z}: {r.stderr[-1500:]}')
        rows = [l.split() for l in open(tmpd + '/r.rec') if l.startswith('rateat')]
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    sig = np.array([float(x[2]) for x in rows])
    if len(sig) != len(PRATE):
        raise RuntimeError(f'{len(sig)} rates for {len(PRATE)} momenta')
    tmp = out_npz + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, mu=sig * 10.0 / REF_RHO, p=PRATE, Z=float(Z), A=float(A))
    os.replace(tmp, out_npz)
    return out_npz




# ---------------------------------------------------------------------------
# the material table
# ---------------------------------------------------------------------------

def read_materials(fn):
    """The Geant4 material table as the exports write it (`materials` tree:
    index, name, density, elemZ, elemA [g/mole], elemW [mass fraction])."""
    import uproot
    f = uproot.open(fn)
    key = next(k for k in f.keys(recursive=True)
               if k.split(';')[0].split('/')[-1] == 'materials')
    a = f[key].arrays(library='np')
    mats = []
    for i in range(len(a['index'])):
        mats.append(dict(index=int(a['index'][i]), name=str(a['name'][i]),
                         density=float(a['density'][i]),
                         Z=[float(x) for x in a['elemZ'][i]],
                         A=[float(x) for x in a['elemA'][i]],
                         W=[float(x) for x in a['elemW'][i]]))
    return mats


# The natural-isotope atomic mass Geant4 gives an element [g/mole], for the
# elements the 2016 geometry references that the job table used for the element
# list does not contain.
EXTRA_ELEMENT_A = {27: 58.933194}
GEOMETRY_CFI = ('/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_15_0_19/src/'
                'Geometry/CMSCommonData/python/cmsExtendedGeometry2016XML_cfi.py')


def xml_materials(cfi, elemA):
    """Every material DEFINED in the DDD XML files of a geometry cfi, resolved
    to Geant4's element mass fractions (compounds are mixtures by weight,
    recursively), with the element atomic masses Geant4 uses (`elemA`, Z ->
    g/mole).  Materials with an element outside `elemA` are skipped.

    Needed for step records written without a material index: the job table
    of the clean-propagation export (a toy tracker inside the real detector)
    does not contain the real tracker's compounds, and the maker's records are
    identified against these by their (effZ, effA, zzp1OverA)."""
    import re
    src = open(cfi).read()
    base = cfi.split('/Geometry/')[0] + '/'
    files = [base + f for f in re.findall(r"'([A-Za-z]+/[A-Za-z0-9_]+/data/[^']+\.xml)'", src)]
    elem, comp = {}, {}
    for f in files:
        if not os.path.exists(f):
            continue
        t = open(f, errors='ignore').read()
        ns = os.path.basename(f)[:-4]
        for e in re.finditer(r'<ElementaryMaterial\s+name="([^"]+)"[^>]*?atomicNumber="([\d.]+)"', t):
            elem.setdefault(f'{ns}:{e.group(1)}', float(e.group(2)))
        for c in re.finditer(r'<CompositeMaterial\s+name="([^"]+)"(.*?)</CompositeMaterial>', t, re.S):
            fr = re.findall(r'fraction="([\d.eE+-]+)"\s*>\s*<rMaterial\s+name="([^"]+)"', c.group(2))
            comp.setdefault(f'{ns}:{c.group(1)}',
                            [(float(a), b if ':' in b else f'{ns}:{b}') for a, b in fr])

    def resolve(name, w, out):
        if name in elem:
            z = int(round(elem[name]))
            out[z] = out.get(z, 0.0) + w
            return
        tot = sum(f for f, _ in comp[name])
        for f, sub in comp[name]:
            resolve(sub, w * f / tot, out)

    mats = []
    for n in sorted(set(comp) | set(elem)):
        out = {}
        try:
            resolve(n, 1.0, out)
        except (KeyError, RecursionError, ZeroDivisionError):
            continue
        zs = sorted(z for z in out if z >= 1 and out[z] > 0.0)
        if not zs or any(z not in elemA for z in zs):
            continue
        mats.append(dict(index=-1 - len(mats), name='xml:' + n, density=float('nan'),
                         Z=[float(z) for z in zs], A=[elemA[z] for z in zs],
                         W=[out[z] for z in zs]))
    return mats


def elements(args):
    mats = read_materials(args.mattab)
    el = {}
    for m in mats:
        for z, a in zip(m['Z'], m['A']):
            if z >= 1.0:
                if int(round(z)) in el and el[int(round(z))] != a:
                    raise SystemExit(f'Z={z} appears with two atomic masses '
                                     f'({el[int(round(z))]}, {a}): bucket by (Z, A) instead')
                el[int(round(z))] = a
    for z, a in EXTRA_ELEMENT_A.items():
        el.setdefault(z, a)
    xm = xml_materials(args.geometry, el)
    out = dict(source=os.path.abspath(args.mattab), geometry=args.geometry,
               elements=[[z, el[z]] for z in sorted(el)], materials=mats + xm)
    json.dump(out, open(args.out, 'w'), indent=1)
    print(f'{len(mats)} job-table materials + {len(xm)} resolved from the geometry XML, '
          f'{len(el)} elements -> {args.out}')


# ---------------------------------------------------------------------------
# kernel discontinuities of the Geant4 models
# ---------------------------------------------------------------------------

def _pq50(pdg, Z, A, p, n=40000, seed=5):
    """p * median(theta) [GeV rad] from a short driver run."""
    tmpd = tempfile.mkdtemp(prefix='nucel_')
    try:
        subprocess.run([DRIVER, '--pdg', str(int(pdg)), '--Z', '%.17g' % Z, '--A', '%.17g' % A,
                        '--rho', '0.1', '--ekin', '%.17g' % ekin_mev(pdg, p), '--len', '100',
                        '--n', str(n), '--seed', str(seed), '--out', tmpd + '/x'],
                       capture_output=True, check=True)
        th = np.fromfile(tmpd + '/x.theta.bin', dtype=np.float64)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    return float(np.median(th) * p)


def _bisect_edge(a):
    """Locate one jump of p*theta_50 inside [lo, hi] to relative 1e-6."""
    pdg, Z, A, lo, hi = a
    vlo, vhi = _pq50(pdg, Z, A, lo), _pq50(pdg, Z, A, hi)
    while hi / lo - 1.0 > 1e-6:
        mid = np.sqrt(lo * hi)
        vm = _pq50(pdg, Z, A, mid)
        if abs(np.log(vm / vlo)) < abs(np.log(vm / vhi)):
            lo, vlo = mid, vm
        else:
            hi, vhi = mid, vm
    return float(np.sqrt(lo * hi)), vlo, vhi


def edges(args):
    """Momenta at which a species' elastic MODEL switches its sampling table.

    G4ElasticHadrNucleusHE (pi+-) samples from per-energy tables and changes
    table at fixed kinetic energies: its kernel jumps there by up to 40 % in
    p*theta while it is continuous in between.  The antiproton switches from
    G4HadronElastic to G4AntiNuclElastic at 100 MeV kinetic energy.  Neither
    kernel may be interpolated across such an edge.  Found by a scan in p of
    p*theta_50 (jumps > 4 % between neighbours 0.9 % apart; carbon and lead,
    whose edges coincide) and bisection to relative 1e-6."""
    P = np.geomspace(PGRID[0], PGRID[-1], 600)
    out = {}
    with Pool(args.jobs) as pool:
        for pdg in SPECIES:
            v = np.array(pool.map(_pq50_star, [(pdg, 6.0, 12.01073638, p) for p in P]))
            j = np.where(np.abs(np.diff(v) / v[:-1]) > 0.04)[0]
            res = pool.map(_bisect_edge, [(pdg, 6.0, 12.01073638, P[i], P[i + 1]) for i in j])
            out[str(pdg)] = [r[0] for r in res]
            print(pdg, ' '.join(f'{r[0]:.7g}({r[1]:.4f}->{r[2]:.4f})' for r in res), flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)


def _pq50_star(a):
    return _pq50(*a)




# ---------------------------------------------------------------------------
# build / assemble
# ---------------------------------------------------------------------------

def _build_task(task):
    kind = task[0]
    if kind == 'rate':
        _, pdg, Z, A, cache = task
        fn = rate_path(cache, pdg, Z)
        return fn if os.path.exists(fn) else run_rates(pdg, Z, A, fn)
    _, pdg, Z, A, inode, p, cache = task
    fn = bucket_path(cache, pdg, Z, inode)
    if os.path.exists(fn):
        return fn
    return run_driver(pdg, p, fn, Z=Z, A=A, seed=bucket_seed(pdg, Z, inode))


def all_tasks(els, edges, cache):
    tasks = [('rate', pdg, float(z), float(a), cache) for pdg in SPECIES for (z, a) in els]
    for pdg in SPECIES:
        nodes, _ = species_nodes(pdg, edges)
        tasks += [('kern', pdg, float(z), float(a), i, float(p), cache)
                  for (z, a) in els for i, p in enumerate(nodes)]
    return tasks


def _done(t):
    if t[0] == 'rate':
        return os.path.exists(rate_path(t[4], t[1], t[2]))
    return os.path.exists(bucket_path(t[6], t[1], t[2], t[4]))


def build(args):
    els = json.load(open(args.elements))['elements']
    edges = json.load(open(args.edges))
    os.makedirs(args.cache, exist_ok=True)
    tasks = all_tasks(els, edges, args.cache)
    mine = tasks[args.task::args.ntask]
    todo = [t for t in mine if not _done(t)]
    print(f'{len(tasks)} tasks in total; slice {args.task}/{args.ntask}: '
          f'{len(mine)}, {len(todo)} to run', flush=True)
    with Pool(args.jobs) as pool:
        for k, _ in enumerate(pool.imap_unordered(_build_task, todo, chunksize=1)):
            if (k + 1) % 20 == 0:
                print(f'  {k+1}/{len(todo)}', flush=True)
    print('done', flush=True)


def assemble(args):
    """Collect the buckets into one float64 npz: per species `{pdg}_nodes`,
    `{pdg}_seg`, `{pdg}_edges` and, over (element, node), `{pdg}_mom` (4),
    `{pdg}_low` (3), `{pdg}_gm1` (UG), `{pdg}_pde` (3, DEE bins), the joint
    bins `{pdg}_jth`, `_jde`, `_jw` (flat, element-major) with their counts
    `{pdg}_jcnt` (element, node), plus the
    diagnostics `{pdg}_gfull` (UGRID), `{pdg}_hchk` (VCHK), `{pdg}_munode`;
    `logmu` (species, element, PRATE)."""
    js = json.load(open(args.elements))
    els = js['elements']
    edges = json.load(open(args.edges))
    out = dict(species=np.array(SPECIES), elemZ=np.array([e[0] for e in els], dtype=np.int32),
               elemA=np.array([e[1] for e in els]), ug=UG, dee=DEE, prate=PRATE, vchk=VCHK,
               ugrid=UGRID, nsamp=NSAMP, seed=SEED, edge_delta=EDGE_DELTA)
    logmu = np.zeros((len(SPECIES), len(els), len(PRATE)))
    models = set()
    for s, pdg in enumerate(SPECIES):
        nodes, seg = species_nodes(pdg, edges)
        out[f'{pdg}_nodes'], out[f'{pdg}_seg'] = nodes, seg
        out[f'{pdg}_edges'] = np.array(sorted(float(x) for x in edges.get(str(pdg), [])))
        sh = (len(els), len(nodes))
        arr = {k: np.zeros(sh + (n,)) for k, n in
               (('mom', 4), ('low', 3), ('gm1', len(UG)), ('gfull', len(UGRID)))}
        arr['pde'] = np.zeros(sh + (3, len(DEE) - 1))
        arr['hchk'] = np.zeros(sh + (len(VCHK),), dtype=np.complex128)
        arr['munode'] = np.zeros(sh)
        jcnt = np.zeros(sh, dtype=np.int32)
        jth, jde, jw = [], [], []
        for ie, (z, a) in enumerate(els):
            r = np.load(rate_path(args.cache, pdg, z))
            if abs(float(r['A']) - a) > 1e-9:
                raise SystemExit(f'rate file A mismatch {pdg} {z}')
            logmu[s, ie] = np.log(r['mu'])
            for i, p in enumerate(nodes):
                fn = bucket_path(args.cache, pdg, z, i)
                d = np.load(fn)
                if (abs(float(d['A']) - a) > 1e-9 or int(d['nsamp']) != NSAMP
                        or abs(float(d['p']) / p - 1.0) > 1e-12):
                    raise SystemExit(f'{fn}: A/nsamp/p do not match')
                arr['mom'][ie, i] = d['mom']
                arr['low'][ie, i] = d['low']
                arr['gm1'][ie, i] = d['gm1']
                arr['pde'][ie, i] = d['pde']
                arr['gfull'][ie, i] = d['g_full']
                arr['hchk'][ie, i] = d['hchk']
                arr['munode'][ie, i] = d['mu']
                jcnt[ie, i] = len(d['jw'])
                jth.append(d['jth'])
                jde.append(d['jde'])
                jw.append(d['jw'])
                models.add((int(pdg), str(d['model']), str(d['xs'])))
        for k, v in arr.items():
            out[f'{pdg}_{k}'] = v
        out[f'{pdg}_jcnt'] = jcnt
        out[f'{pdg}_jth'] = np.concatenate(jth)
        out[f'{pdg}_jde'] = np.concatenate(jde)
        out[f'{pdg}_jw'] = np.concatenate(jw)
    out['logmu'] = logmu
    out['models'] = np.array([f'{a} {b} {c}' for a, b, c in sorted(models)])
    out['mattab_source'] = np.array([js['source']])
    np.savez(args.out, **out)
    print(f'wrote {args.out}')
    for m in sorted(models):
        print(' ', m)


# ---------------------------------------------------------------------------
# the shipped binary (layout: TrackPropagation/Geant4e/data/make_cvhcf_nucel_tables.py)
# ---------------------------------------------------------------------------

BIN_MAGIC = b'CVHNUCEL'
BIN_VERSION = 2
BIN_TAIL = b'CVHNUEND'
BIN_DEFAULT = ('/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/TrackPropagation/'
               'Geant4e/data/cvhcf_nucel_v2.bin')


def read_bin(fn):
    """The binary as a dict of float64 / int arrays; `rec[s]` is the
    (element, node, 7 + nU + 3 nD) record block of species s:
    [mom(4), low(3), g - 1 (nU), P (nD), D1 (nD), D2 (nD)], the dE bins
    expanded from their stored nonzero range; `joint[s][e][n]` the
    (theta, dE, w) joint bins of that record."""
    b = open(fn, 'rb').read()
    o = [0]

    def take(dt, n):
        a = np.frombuffer(b, dtype=dt, count=n, offset=o[0])
        o[0] += a.nbytes
        return a

    if b[:8] != BIN_MAGIC:
        raise ValueError(f'{fn}: not a cvhcf nucel table')
    o[0] = 8
    ver, ns, ne, nu, nde, npr = take('<i4', 6)
    if ver != BIN_VERSION:
        raise ValueError(f'{fn}: version {ver}')
    t = dict(version=int(ver), ug=take('<f8', nu).copy(), dee=take('<f8', nde + 1).copy(),
             prate=take('<f8', npr).copy(), species=take('<i4', ns).copy(),
             elemZ=take('<i4', ne).copy(), elemA=take('<f8', ne).copy())
    t['edges'], t['nodes'], t['seg'] = [], [], []
    for s in range(ns):
        nedge, nnode = take('<i4', 2)
        t['edges'].append(take('<f8', nedge).copy())
        t['nodes'].append(take('<f8', nnode).copy())
        t['seg'].append(take('<i4', nnode).copy())
    t['logmu'] = take('<f8', ns * ne * npr).reshape(ns, ne, npr).copy()
    nfix = 7 + nu
    fixed = [take('<f4', ne * len(t['nodes'][s]) * nfix).reshape(ne, len(t['nodes'][s]), nfix)
             for s in range(ns)]
    nrec = sum(ne * len(t['nodes'][s]) for s in range(ns))
    idx = take('<i4', 2 * nrec).reshape(nrec, 2)
    data = take('<f4', 3 * int(idx[:, 1].sum()))
    t['rec'] = []
    k = 0
    off = 0
    for s in range(ns):
        nn = len(t['nodes'][s])
        r = np.zeros((ne, nn, nfix + 3 * nde))
        r[..., :nfix] = fixed[s]
        for e in range(ne):
            for n in range(nn):
                b0, nb = idx[k]
                blk = data[off:off + 3 * nb].reshape(3, nb)
                for q in range(3):
                    r[e, n, nfix + q * nde + b0:nfix + q * nde + b0 + nb] = blk[q]
                off += 3 * nb
                k += 1
        t['rec'].append(r)
    jn = take('<i4', nrec)
    t['joint'] = []
    k = 0
    for s in range(ns):
        js = []
        for e in range(ne):
            je = []
            for n in range(len(t['nodes'][s])):
                m = int(jn[k])
                blk = take('<f4', 3 * m).astype(np.float64).reshape(3, m)
                je.append((blk[0].copy(), blk[1].copy(), blk[2].copy()))
                k += 1
            js.append(je)
        t['joint'].append(js)
    if b[o[0]:o[0] + 8] != BIN_TAIL or o[0] + 8 != len(b):
        raise ValueError(f'{fn}: bad trailer / size')
    return t


# ---------------------------------------------------------------------------
# the table as a lookup
# ---------------------------------------------------------------------------

def catmull_rom(y, t):
    """Catmull-Rom interpolation of the node values y[..., n] at fractional
    node positions t (0 .. n-1, array); the end intervals use the linearly
    extrapolated outer neighbour."""
    n = y.shape[-1]
    i = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
    s = t - i
    p1 = np.take_along_axis(y, i, -1) if y.ndim > 1 else y[i]
    p2 = np.take_along_axis(y, i + 1, -1) if y.ndim > 1 else y[i + 1]
    ym = y[..., 0:1] * 2 - y[..., 1:2]
    yp = y[..., -1:] * 2 - y[..., -2:-1]
    ye = np.concatenate([ym, y, yp], axis=-1)
    p0 = np.take_along_axis(ye, i, -1) if y.ndim > 1 else ye[i]
    p3 = np.take_along_axis(ye, i + 3, -1) if y.ndim > 1 else ye[i + 3]
    return 0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s
                  + (3 * p1 - p0 - 3 * p2 + p3) * s * s * s)


# sin(y)/y - 1 = sum_{k>=1} SINCM1_C[k-1] y^{2k}, (-1)^k / (2k+1)!
SINCM1_C = (-0.16666666666666666, 0.008333333333333333, -0.0001984126984126984,
            2.7557319223985893e-06, -2.505210838544172e-08, 1.6059043836821613e-10,
            -7.647163731819816e-13, 2.8114572543455206e-15, -8.22063524662433e-18)


def sincm1(y):
    """sin(y)/y - 1 without the cancellation near 0: the Taylor series
    (Horner in y^2, complete to double precision) below |y| = 1, the direct
    form above."""
    y = np.asarray(y, dtype=np.float64)
    y2 = y * y
    p = np.full(y.shape, SINCM1_C[-1])
    for c in SINCM1_C[-2::-1]:
        p = p * y2 + c
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.abs(y) < 1.0, p * y2, np.sin(y) / y - 1.0)


class Table:
    """The shipped per-(species, element, node) tables and their compound
    mixtures, evaluated exactly as the in-maker port evaluates them.

    `materials`: the material list (dicts with index, name, Z, A, W) the steps
    refer to -- `elements.json`'s list (job table + geometry XML) by default.
    `mixture(isp, imat)` builds a material's arrays once; `rows` is the
    channel's row function (the clean-propagation `cf_nucel_exact.rows_exponent`
    and the fit-level `ks_nucel_cf.step_family` both evaluate through it);
    `nodes(isp, p)`, `g`, `h`, `mom` and `rate` evaluate the kernels for
    arrays of steps (diagnostics)."""

    def __init__(self, fn=BIN_DEFAULT, materials=None):
        t = read_bin(fn)
        self.t = t
        self.path = fn
        self.species = [int(x) for x in t['species']]
        self.ug = t['ug']
        self.lug0 = np.log(t['ug'][0])
        self.dlug = (np.log(t['ug'][-1]) - self.lug0) / (len(t['ug']) - 1)
        self.dee = t['dee']
        self.lpr = np.log(t['prate'])
        self.nu, self.nde = len(t['ug']), len(t['dee']) - 1
        self.eidx = {int(z): i for i, z in enumerate(t['elemZ'])}
        self.elemA = {int(z): float(a) for z, a in zip(t['elemZ'], t['elemA'])}
        if materials is None:
            materials = json.load(open(os.path.join(
                '/ceph/submit/data/user/d/david_w/ZMass/cvh/nucel_trackfit_260926/tables',
                'elements_full.json')))['materials']
        self.mats = []
        self.trip = np.zeros((0, 3))
        self._mix = {}
        self._reg = {}
        for m in materials:
            self._add_material(m)
        self.byindex = {m['index']: i for i, m in enumerate(self.mats)}
        # per species: node index range [a, b] of every segment
        self.segrange = []
        for s in range(len(self.species)):
            seg = t['seg'][s]
            self.segrange.append({int(k): (int(np.where(seg == k)[0][0]), int(np.where(seg == k)[0][-1]))
                                  for k in np.unique(seg)})

    def _add_material(self, m):
        Z, A, W = (np.asarray(m[k], dtype=np.float64) for k in ('Z', 'A', 'W'))
        self.mats.append(dict(index=int(m['index']), name=m['name'], Z=Z, A=A, W=W))
        self.trip = np.vstack([self.trip, [float((W * Z).sum()), float((W * A).sum()),
                                           float((W * Z * (Z + 1) / A).sum())]])
        return len(self.mats) - 1

    def register(self, filemats):
        """{file material index: table row} for a material list read from a
        file (`read_materials`): a material whose name and composition match a
        row maps to it, any other is appended.  Cached per list."""
        key = id(filemats)
        hit = self._reg.get(key)
        if hit is not None:
            return hit[1]
        rowmap = {}
        for m in filemats:
            Z, W = np.asarray(m['Z'], dtype=np.float64), np.asarray(m['W'], dtype=np.float64)
            row = -1
            for i, t in enumerate(self.mats):
                if (t['name'] == m['name'] and len(t['Z']) == len(Z) and np.array_equal(t['Z'], Z)
                        and np.allclose(t['W'], W, rtol=1e-12, atol=0.0)):
                    row = i
                    break
            if row < 0:
                row = self._add_material(m)
            rowmap[int(m['index'])] = row
        self._reg[key] = (filemats, rowmap)
        return rowmap

    def species_index(self, pdg):
        return self.species.index(int(pdg))

    def composition(self, imat):
        """(element indices, mass fractions, atomic masses) of material row
        `imat`; elements with no mass share are dropped.  None if an element
        is not tabulated."""
        m = self.mats[imat]
        keep = (m['Z'] >= 1.0) & (m['W'] > 0.0)
        zs = [int(round(z)) for z in m['Z'][keep]]
        if any(z not in self.eidx for z in zs):
            return None
        return (np.array([self.eidx[z] for z in zs], dtype=int), m['W'][keep], m['A'][keep])

    def _loglog(self, lmu, p):
        lp = np.log(np.asarray(p, dtype=np.float64))
        j = np.clip(np.searchsorted(self.lpr, lp) - 1, 0, len(self.lpr) - 2)
        f = np.clip((lp - self.lpr[j]) / (self.lpr[j + 1] - self.lpr[j]), 0.0, 1.0)
        return (1 - f) * lmu[..., j] + f * lmu[..., j + 1]

    def mixture(self, isp, imat):
        """dict(nodes, seg, lmu[PRATE], rec[node, 7+nU+3nD], share[comp, node],
        joint[node] = (theta, dE, W)) of material row `imat` for species index
        `isp`, or None."""
        key = (isp, imat)
        if key in self._mix:
            return self._mix[key]
        c = self.composition(imat)
        if c is None or not len(c[0]):
            self._mix[key] = None
            return None
        ei, w, a = c
        t = self.t
        scale = w * t['elemA'][ei] / a                      # w_i A_tab / A_i
        mu_pr = scale[:, None] * np.exp(t['logmu'][isp, ei])  # (comp, PRATE)
        nodes = t['nodes'][isp]
        r = scale[:, None] * np.exp(self._loglog(t['logmu'][isp, ei], nodes))   # (comp, node)
        share = r / r.sum(0)[None, :]
        rec = np.einsum('cn,cnk->nk', share, t['rec'][isp][ei])
        joint = []
        for n in range(len(nodes)):
            th, de, ww = [], [], []
            for c, e in enumerate(ei):
                if share[c, n] < JSHARE:
                    continue
                jt, jd, jw = t['joint'][isp][e][n]
                th.append(jt)
                de.append(jd)
                ww.append(jw * share[c, n])
            joint.append((np.concatenate(th), np.concatenate(de), np.concatenate(ww)))
        out = dict(nodes=nodes, seg=t['seg'][isp], lmu=np.log(mu_pr.sum(0)), rec=rec, share=share,
                   joint=joint)
        self._mix[key] = out
        return out

    def rate(self, mx, p):
        """mu_mat(p) [cm^2/g]."""
        return np.exp(self._loglog(mx['lmu'], p))

    def nodes(self, isp, p):
        """(j0, j1, f) per momentum: the two nodes of p's segment that bracket
        p and the ln-p weight of j1; outside the segment's nodes, the end
        node twice with f = 0."""
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        P = self.t['nodes'][isp]
        seg = np.searchsorted(self.t['edges'][isp], p, side='right')
        j0 = np.zeros(len(p), dtype=int)
        j1 = np.zeros(len(p), dtype=int)
        f = np.zeros(len(p))
        for k, (a, b) in self.segrange[isp].items():
            m = seg == k
            if not m.any():
                continue
            pp = p[m]
            j = np.clip(np.searchsorted(P[a:b + 1], pp) - 1 + a, a, b)
            jj = np.minimum(j + 1, b)
            with np.errstate(divide='ignore', invalid='ignore'):
                ff = np.where(jj > j, np.log(pp / P[j]) / np.log(P[jj] / P[j]), 0.0)
            lo = pp <= P[a]
            hi = pp >= P[b]
            j = np.where(lo, a, np.where(hi, b, j))
            jj = np.where(lo, a, np.where(hi, b, jj))
            ff = np.where(lo | hi, 0.0, np.clip(ff, 0.0, 1.0))
            j0[m], j1[m], f[m] = j, jj, ff
        return j0, j1, f

    def g_node(self, row, u):
        """g(u) of one node record (mixture or element), u >= 0 array."""
        u = np.asarray(u, dtype=np.float64)
        gm1 = row[7:7 + self.nu]
        out = np.empty(u.shape)
        lo = u < self.ug[0]
        hi = u > self.ug[-1]
        mid = ~(lo | hi)
        out[lo] = 1.0 - 0.25 * u[lo] ** 2 * row[1]
        out[hi] = 1.0 + gm1[-1]
        if mid.any():
            t = (np.log(u[mid]) - self.lug0) / self.dlug
            out[mid] = 1.0 + catmull_rom(gm1, t)
        return out

    def h_node(self, row, v):
        """h(v) of one node record, v any sign, array."""
        return h_from_density(v, row[7 + self.nu:].reshape(3, self.nde), row[4:7], self.dee)

    def g(self, mx, j0, j1, f, p, x):
        """g at argument x (per step, any trailing shape) for steps at momenta
        p with node weights (j0, j1, f): the fixed-momentum-transfer rescaled
        node kernels, mixed linearly in ln p."""
        P = mx['nodes']
        x = np.asarray(x, dtype=np.float64)
        ex = (Ellipsis,) + (None,) * (x.ndim - np.ndim(p))
        s0 = (P[j0] / p)[ex]
        s1 = (P[j1] / p)[ex]
        out = np.zeros(x.shape)
        for jj, ss, ww in ((j0, s0, (1.0 - f)[ex]), (j1, s1, f[ex])):
            for k in np.unique(jj):
                m = jj == k
                out[m] += ww[m] * self.g_node(mx['rec'][k], x[m] * ss[m])
        return out

    def h(self, mx, j0, j1, f, x):
        """h at argument x (any sign), linear in ln p between the nodes."""
        x = np.asarray(x, dtype=np.float64)
        ex = (Ellipsis,) + (None,) * (x.ndim - np.ndim(f))
        out = np.zeros(x.shape, dtype=np.complex128)
        for jj, ww in ((j0, (1.0 - f)[ex]), (j1, f[ex])):
            for k in np.unique(jj):
                m = jj == k
                out[m] += ww[m] * self.h_node(mx['rec'][k], x[m])
        return out

    def gm1_node(self, row, u):
        """g(u) - 1 of one node record, u >= 0 array (`g_node` minus one,
        formed without the cancellation of 1 + (g - 1) - 1)."""
        u = np.asarray(u, dtype=np.float64)
        gm1 = row[7:7 + self.nu]
        out = np.empty(u.shape)
        lo = u < self.ug[0]
        hi = u > self.ug[-1]
        mid = ~(lo | hi)
        out[lo] = -0.25 * u[lo] * u[lo] * row[1]
        out[hi] = gm1[-1]
        if mid.any():
            out[mid] = catmull_rom(gm1, (np.log(u[mid]) - self.lug0) / self.dlug)
        return out

    def recoil_bins(self, row):
        """(P_b, mean_b, half-width_b) of one node record's populated recoil
        bins: each bin uniform over its own mean -+ sqrt(3) x its own std."""
        P, D1, D2 = row[7 + self.nu:].reshape(3, self.nde)
        nz = P > 0.0
        c = (0.5 * (self.dee[:-1] + self.dee[1:]))[nz]
        P, D1, D2 = P[nz], D1[nz], D2[nz]
        d = D1 / P
        return P, c + d, np.sqrt(3.0 * np.maximum(D2 / P - d * d, 0.0))

    def hm1_node(self, row, v, chunk=4096):
        """h(v) - 1 of one node record, v any sign, 1-d array (`h_node` minus
        one), formed without cancellation: the recoil below the first bin by
        its partial moments, each bin uniform over its own mean -+ sqrt(3) x
        its own std (`recoil_bins`),
            h - 1 = i v M1_low - v^2 M2_low / 2
                    + sum_b P_b [e^{i v mu_b} sinc(v a_b) - 1],
        e^{ix} s - 1 = -2 sin^2(x/2) s + (s - 1) + i sin(x) s,  s - 1 by
        `sincm1`.  P_low + sum_b P_b = 1 is used to drop the constant, so the
        CF is normalised exactly (the stored float32 P's sum to 1 only to
        their precision)."""
        v = np.asarray(v, dtype=np.float64)
        P, mu, hw = self.recoil_bins(row)
        out = -0.5 * v * v * row[6] + 1j * (v * row[5])
        for lo in range(0, len(v), chunk):
            vv = v[lo:lo + chunk, None]
            x = vv * mu[None, :]
            y = vv * hw[None, :]
            with np.errstate(invalid='ignore', divide='ignore'):
                sc = np.where(y != 0.0, np.sin(y) / y, 1.0)
            h = np.sin(0.5 * x)
            re = (-2.0 * h * h) * sc + sincm1(y)
            im = np.sin(x) * sc
            out[lo:lo + chunk] += (P * re).sum(-1) + 1j * (P * im).sum(-1)
        return out

    def joint_node(self, mx, k, sc, b, a, X, exact, chunk=2048):
        """One node's joint term at the arguments b = |w_ang| tau >= 0 and
        a = w_rec tau (1-d, same length): over the node's joint bins
        (theta_i, dE_i, W_i), the deflections rescaled by `sc` = p_node / p,
            sum_i W_i [(J0(b theta_i sc) - 1)(e^{i a X_i} - 1) + (e^{i a X_i} - e^{i a dE_i})]
        with X the bins' q/p-equivalent loss: T_eff(dE) under the exact map
        (`exact`), where the second bracket is the recoil's own map
        correction, else dE, where it vanishes and is not formed (`rows`
        forms X)."""
        th, de, W = mx['joint'][k]
        thp = th * sc
        out = np.zeros(len(a), dtype=np.complex128)
        for lo in range(0, len(a), chunk):
            bb = b[lo:lo + chunk, None]
            aa = a[lo:lo + chunk, None]
            jm1 = j0(bb * thp[None, :]) - 1.0
            yx = aa * X[None, :]
            # e^{iy} - 1 = (-2 s^2, 2 s c) at the half angle
            hs, hc = np.sin(0.5 * yx), np.cos(0.5 * yx)
            re = jm1 * (-2.0 * hs * hs)
            im = jm1 * (2.0 * hs * hc)
            if exact:
                # e^{iyx} - e^{iyd} by the sum-to-product forms
                yd = aa * de[None, :]
                yp, ym = 0.5 * (yx + yd), np.sin(0.5 * (yx - yd))
                re = re + (-2.0 * np.sin(yp) * ym)
                im = im + (2.0 * np.cos(yp) * ym)
            out[lo:lo + chunk] = (re @ W) + 1j * (im @ W)
        return out

    def rows(self, tau, M, imat, pdg, mass, rid, wb, frac, wq_row, wb_mid,
             recoil=True, joint=True, exact=False):
        """THE ROW FUNCTION of the nuclear-elastic channel, in `cf_rows`'
        entry convention (`cvhcf::nucelRows` is its C++ port).  Over the MS
        rows M (xg column 2 [g/cm^2], p column 3 [GeV]) with table material
        rows `imat`, species `pdg` (0: none) and masses `mass` [GeV]:

          angular  per entry (row rid, angular weight wb, share frac of the
                   row's collisions):  N_s frac (g_s(|wb| tau) - 1)
          recoil   per row at the signed weight wq_row [z per MeV of recoil
                   energy] (`recoil`):  N_s (h_s(wq tau) - 1)
          joint    per row at (|wb_mid|, wq_row) (`joint`, with `recoil`):
                   N_s x `joint_node` over the row's two nodes

        N_s = mu_{m(s)}(p_s) xg_s; every kernel at the row's own momentum: the
        two nodes of its model segment that bracket p_s, mixed linearly in
        ln p, the deflections rescaled to fixed momentum transfer.  X =
        T_eff(dE) at the row's (E, p) under `exact` (cf_knockon.t_eff), else
        dE.  The exact map ends where the knock-on channel's does
        (cf_knockon.law_top): a recoil that would leave the primary with less
        than PMIN_FRAC of its momentum -- the upper node's recoils near the
        kinematic end point can exceed the row's kinetic energy, since the
        recoil is carried at fixed momentum transfer -- keeps the linear map
        (X = dE).  A zero weight carries no term.  The recoil of rows that share a
        mixture, a node and a weight is evaluated once for all of them.

        Returns (N, ang, rec, jnt): the expected collisions and the three
        exponents summed over the rows, (nt,) real / complex / complex.  Rows
        with a species whose material does not resolve raise."""
        import cf_knockon as ck
        tau = np.asarray(tau, dtype=np.float64)
        nt = len(tau)
        ang = np.zeros(nt)
        rec = np.zeros(nt, dtype=np.complex128)
        jnt = np.zeros(nt, dtype=np.complex128)
        M = np.asarray(M)
        n = len(M)
        if not n:
            return 0.0, ang, rec, jnt
        p = M[:, 3].astype(np.float64)
        xg = M[:, 2].astype(np.float64)
        pdg = np.broadcast_to(np.asarray(pdg, dtype=np.int64), (n,))
        mass = np.broadcast_to(np.asarray(mass, dtype=np.float64), (n,))
        imat = np.broadcast_to(np.asarray(imat, dtype=np.int64), (n,))
        wq = np.broadcast_to(np.asarray(wq_row, dtype=np.float64), (n,))
        wbm = np.broadcast_to(np.asarray(wb_mid, dtype=np.float64), (n,))
        ok = (pdg != 0) & (xg > 0.0) & (p > 0.0)
        if np.any(ok & (imat < 0)):
            raise ValueError('a hadron row has no resolved material')
        N = np.zeros(n)
        mx, nd, isp = [None] * n, [None] * n, [None] * n
        for s in np.flatnonzero(ok):
            isp[s] = self.species_index(pdg[s])
            m = self.mixture(isp[s], int(imat[s]))
            if m is None:
                raise ValueError(f'material row {imat[s]} has an element the table lacks')
            mx[s] = m
            N[s] = float(self.rate(m, p[s])) * xg[s]
            j0_, j1_, f_ = self.nodes(isp[s], p[s])
            nd[s] = ((int(j0_[0]), 1.0 - float(f_[0])), (int(j1_[0]), float(f_[0])))
        live = ok & (N > 0.0)
        for e in range(len(rid)):
            s = int(rid[e])
            if not live[s] or wb[e] == 0.0:
                continue
            x = abs(float(wb[e])) * tau
            for k, ww in nd[s]:
                if ww == 0.0:
                    continue
                ang += (N[s] * float(frac[e]) * ww) * self.gm1_node(
                    mx[s]['rec'][k], x * (mx[s]['nodes'][k] / p[s]))
        if recoil:
            grp = {}
            for s in np.flatnonzero(live):
                if wq[s] == 0.0:
                    continue
                for k, ww in nd[s]:
                    if ww == 0.0:
                        continue
                    key = (id(mx[s]), k, float(wq[s]))
                    if key not in grp:
                        grp[key] = [mx[s], k, float(wq[s]), 0.0]
                    grp[key][3] += N[s] * ww
            for m, k, w, c in grp.values():
                rec += c * self.hm1_node(m['rec'][k], w * tau)
            if joint:
                for s in np.flatnonzero(live):
                    if wq[s] == 0.0:
                        continue
                    E = np.sqrt(p[s] * p[s] + mass[s] * mass[s])
                    b = abs(float(wbm[s])) * tau
                    a = float(wq[s]) * tau
                    Em, pm = 1e3 * E, 1e3 * p[s]
                    tcap = Em - np.sqrt(Em * Em - pm * pm + (ck.PMIN_FRAC * pm) ** 2)
                    for k, ww in nd[s]:
                        if ww == 0.0:
                            continue
                        de = mx[s]['joint'][k][1]
                        X = (np.where(de < tcap, ck.t_eff(np.minimum(de, tcap), Em, pm), de)
                             if exact else de)
                        jnt += (N[s] * ww) * self.joint_node(
                            mx[s], k, mx[s]['nodes'][k] / p[s], b, a, X, exact)
        return float(N.sum()), ang, rec, jnt

    def mom(self, mx, j0, j1, f, p):
        """<theta> and <theta^2> (rescaled to p), <dE>, <dE^2> per step."""
        P = mx['nodes']
        r0, r1 = mx['rec'][j0, :4], mx['rec'][j1, :4]
        s0, s1 = (P[j0] / p), (P[j1] / p)
        sc0 = np.stack([s0, s0 ** 2, np.ones_like(s0), np.ones_like(s0)], -1)
        sc1 = np.stack([s1, s1 ** 2, np.ones_like(s1), np.ones_like(s1)], -1)
        return (1 - f)[:, None] * r0 * sc0 + f[:, None] * r1 * sc1

    def identify(self, effZ, effA, zzp1, rtol=2e-6):
        """Material row for step records without an index, by the exported
        (effZ, effA, zzp1OverA) triple (float32 in the maker's records).
        -1 where nothing matches within rtol, -2 where materials of DIFFERENT
        composition match (ambiguous)."""
        q = np.stack([np.asarray(effZ, dtype=np.float64), np.asarray(effA, dtype=np.float64),
                      np.asarray(zzp1, dtype=np.float64)], axis=-1)
        uq, inv = np.unique(q, axis=0, return_inverse=True)
        res = np.full(len(uq), -1, dtype=int)
        for k, row in enumerate(uq):
            d = np.abs(self.trip - row[None, :]) / np.maximum(np.abs(row[None, :]), 1e-30)
            hit = np.where(d.max(1) <= rtol)[0]
            if not len(hit):
                continue
            res[k] = hit[0] if all(self._same_comp(hit[0], i) for i in hit[1:]) else -2
        return res[inv.ravel()]

    def _same_comp(self, i, j, wtol=1e-5):
        """Same elements and mass fractions within wtol (the job table's and
        the XML's resolution of one material differ at the 1e-9 level)."""
        a, b = self.mats[i], self.mats[j]
        ka, kb = a['W'] > 0.0, b['W'] > 0.0
        za, zb = a['Z'][ka], b['Z'][kb]
        if len(za) != len(zb):
            return False
        oa, ob = np.argsort(za), np.argsort(zb)
        return (np.array_equal(za[oa], zb[ob])
                and np.max(np.abs(a['W'][ka][oa] - b['W'][kb][ob])) <= wtol)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    s0 = sp.add_parser('elements')
    s0.add_argument('--mattab', required=True)
    s0.add_argument('--out', required=True)
    s0.add_argument('--geometry', default=GEOMETRY_CFI,
                    help='geometry cfi whose DDD XML materials are resolved as well')
    s1 = sp.add_parser('edges')
    s1.add_argument('--out', required=True)
    s1.add_argument('--jobs', type=int, default=48)
    s2 = sp.add_parser('build')
    s2.add_argument('--elements', required=True)
    s2.add_argument('--edges', required=True)
    s2.add_argument('--cache', required=True)
    s2.add_argument('--task', type=int, default=0)
    s2.add_argument('--ntask', type=int, default=1)
    s2.add_argument('--jobs', type=int, default=8)
    s3 = sp.add_parser('assemble')
    s3.add_argument('--elements', required=True)
    s3.add_argument('--edges', required=True)
    s3.add_argument('--cache', required=True)
    s3.add_argument('--out', required=True)
    args = ap.parse_args()
    {'elements': elements, 'edges': edges, 'build': build, 'assemble': assemble}[args.cmd](args)


if __name__ == '__main__':
    main()
