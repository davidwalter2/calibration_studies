#!/usr/bin/env python3
"""The nuclear-elastic CF family on the K_S -> pi+ pi- MASS functional.

WHAT IS ADDED
-------------
The in-maker mass CF (`cfmass_*`) composes hit / MS / ionisation / radiative
(+ delta) exponents on the 64-point tau grid.  A pion also takes `hadElastic`
collisions -- rare (N ~ 0.05 per track), large (25-35 mrad at 3 GeV, 0.2 rad at
0.2 GeV) isotropic kicks -- which no family carries.  This module builds that
family OFFLINE from the raw step records of the step-record re-production and
writes it into a pairs cache beside the maker's own exponents:

  angular part (the channel of `cf_nucel_exact`, on the mass functional):
      S_ang(tau) = sum_b sum_{s in b} N_s * ( g_s(w_b tau) - 1 )
  recoil sub-channel (`NUCEL_RECOIL`, as in `cf_nucel_exact`):
      S_rec(tau) = sum_s N_s * ( h_s(wq_s tau) - 1 )

* the BLOCKS b and their weights are the maker's own MS blocks: parmtype 10,
  pooled by global index, `w_b = sqrt(v_b / sum_s thp2_s) / sigma_m` with v_b
  the fit's influence variance (`resinfvarv`) -- exactly the weight the maker
  gives the Moliere exponent of that block (`cvhcf::trackExponents`,
  reproduced by `cvhcf_validate.py`).  A collision is an isotropic 2D kick at
  a point inside the step, like a Moliere kick, so it rides on the same
  weight.  That weight is checked here, not assumed: `--gate-ms N` rebuilds
  `cfmass_ms` from the same records and weights and compares it with what the
  maker wrote.
* N_s = mu_mat(species, p_s) * xg_s, PER STEP, with one target PER ELEMENT
  of the step's material: mu_mat = sum_i w_i mu_i and the single-collision
  kernels are the rate-weighted mixtures of the element kernels
  (`nucel_tables.Table.mixture`), so the hydrogen of the composites is a
  target in its own right.  The material is the record's `msmatv` index
  resolved through the file's `materials` tree when the file carries them,
  and otherwise identified from the record's exported (effZ, effA, zzp1OverA)
  (`Table.identify`, exact against the Geant4 material table).  The step's
  own momentum; the species from the leg: pi+ on the positive leg, pi- on the
  negative one.  The records are drained leg by leg, so a leg boundary is the
  one place the momentum jumps; each segment is given to the leg whose fitted
  momentum is nearest.
* g, h, mu: the Geant4 per-element tables of `nucel_tables.py` (the
  species-correct elastic model and cross-section dataset through
  `nucel_g4driver`), read from the shipped binary `cvhcf_nucel_v2.bin` and
  evaluated by `nucel_tables.Table` exactly as the in-maker port does: the
  two momentum nodes of the step's model segment that bracket p_step, each
  with its angular argument rescaled by p_node/p_step (fixed momentum
  transfer), mixed linearly in ln p; the recoil CF from the tabulated dE
  density; the rate log-log on its own fine grid.
* wq_s: the recoil moves q/p like an ionisation loss, so its weight is the
  maker's ionisation weight of the SAME leg at the SAME step,
  `w_io(block) * cs_s * 1e-3 * IONI_SGN`: the radiative records are one per
  Geant4 step, parallel to the MS records, and carry the leg's ionisation
  block index and the step's cs (`maker_step_weights`, the rule of the
  in-maker family).  Records without them fall back to the leg's ionisation
  weight interpolated along the leg in p.
* The joint law of one collision (`cf_nucel_exact.NUCEL_JOINT`, `step_family`):
  the recoil is a function of the deflection, so the families' independent
  product is corrected per row by the joint term over the tables' (theta, dE)
  bins (`nucel_tables.Table.rows`).
* The species: the maker's `mspdgv` per record when the file carries it.
* `step_family` is the reference of the in-maker family (`cvhcf` NucelExponents,
  exported as `<cf>_nuc_*` under `exportCfNucel`) that `cvhcf_validate.py`
  holds the port to; the pairs cache below (`candidate`) carries the same
  angular, recoil and joint parts, with its survival diagnostics.
* NOT centred: the Geant4e reference carries no nuclear-elastic mean (neither
  the kick, which is isotropic, nor the recoil loss, which is one-signed).

The channel is GATED exactly as the offline one: nothing is added unless
`cf_nucel_exact.NUCEL_CHANNEL` (env NUCEL_CHANNEL=1); the recoil only under
`cf_nucel_exact.NUCEL_RECOIL`.  Both parts are written separately so the
recoil-on and recoil-off variants are two views of one cache.

COLUMNS ADDED to the ks_pairs cache (rows identical to `ks_pairs.py`'s):
  Snuc_ang (n, 64)       the angular exponent
  Snuc_rec_re/_im        the recoil exponent
  Snuc_jnt_re/_im        the joint (angle-recoil) exponent (`cf_nucel_exact.NUCEL_JOINT`)
  nuc_N, nuc_Np, nuc_Nm  expected collisions (both legs / + / -)
  nuc_v                  second cumulant of the angular part, in z^2 units
  nuc_vrec, nuc_mrec     second cumulant and MEAN of the recoil part (z units)
  nuc_xgunk              fraction of the candidate's g/cm^2 whose material is not
                         resolved (no or an ambiguous match)
  Snuc_first, nuc_Nfirst the angular exponent / collisions of each leg's FIRST MS
                         block (decay vertex to the first measured module)
  nuc_Nbig(first)        collisions whose mean kick alone is > 3 sigma_m
  Snuc_surv, nuc_Nsurv   (--survival) the SURVIVAL-WEIGHTED angular exponent and
                         collisions: each kick weighted by its survival on the
                         reconstructed candidate, S(q) of its chi2 significance in
                         its block and the first block's migration factor
                         (survival/survival_model.py); the recoil is scaled by
                         nuc_Nsurv / nuc_N downstream (make_variants.py)

usage:
  NUCEL_CHANNEL=1 python3 ks_nucel_cf.py --prod <steprec prod> [--table cvhcf_nucel_v2.bin] \
      --cache out.npz [--jobs 32] [--gate-ms 200]
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.dirname(HERE)
RES = os.path.dirname(KS)
sys.path.insert(0, RES)
sys.path.insert(0, KS)
sys.path.insert(0, os.path.join(HERE, 'survival'))
os.environ.setdefault("CVH_IONI_KOKOULIN", "0")

import cf_nucel_exact                                   # noqa: E402
import nucel_tables                                     # noqa: E402
import cf_track_resolution as ctr                        # noqa: E402
import ks_pairs                                          # noqa: E402
import prodfiles                                         # noqa: E402
from cf_mass_likelihood import IONI_SGN                  # noqa: E402

M_PI = 0.13957039
_T = {}
# the survival-weighted variant: {'model': survival_model.json dict, 'theta': path}
# (set by --survival; off by default)
SURVIVAL = None


def _theta_table():
    if 'TT' not in _T:
        sys.path.insert(0, os.path.join(HERE, 'survival'))
        import theta_tables
        _T['TT'] = theta_tables.ThetaTable(SURVIVAL['theta'], _T['T'])
    return _T['TT']


def load_table(fn):
    """The shipped binary with the default material list (the job table of the
    clean-propagation export plus the 2016 geometry XML)."""
    return nucel_tables.Table(fn)


def step_materials(T, M, msmat=None, filemats=None):
    """Table material row of every MS record: through `msmatv` and the file's
    material tree when both are given, else from the exported (effZ, effA,
    zzp1OverA) (columns 0, 1, 7).  -1: unresolved or ambiguous."""
    if msmat is not None and filemats is not None:
        rowmap = T.register(filemats)
        return np.array([rowmap.get(int(i), -1) for i in msmat], dtype=int)
    r = T.identify(M[:, 0], M[:, 1], M[:, 7])
    return np.where(r >= 0, r, -1)


def _p_from_cs(cs):
    """|p| [GeV] from the record's cs = E/p^3 [GeV^-2] (pion mass), by Newton
    in ln p on f = ln(E) - 3 ln p - ln cs, which is monotonic."""
    lc = np.log(np.maximum(cs, 1e-300))
    lp = -lc / 3.0                       # the massless start
    for _ in range(40):
        p = np.exp(lp)
        e2 = p * p + M_PI * M_PI
        f = 0.5 * np.log(e2) - 3.0 * lp - lc
        df = p * p / e2 - 3.0
        step = f / df
        lp = lp - step
        if np.max(np.abs(step)) < 1e-13:
            break
    return np.exp(lp)


def _segments(p):
    """Leg segments of a drain-ordered record list: a new segment wherever the
    momentum INCREASES (a leg only loses momentum) or jumps by > 5 %."""
    if len(p) < 2:
        return [np.arange(len(p))]
    dl = np.diff(np.log(p))
    cut = np.where((dl > 1e-9) | (np.abs(dl) > 0.05))[0] + 1
    return np.split(np.arange(len(p)), cut)


def _leg_sign(p_first, pp, pm):
    return +1 if abs(np.log(p_first / pp)) <= abs(np.log(p_first / pm)) else -1


def block_weights(gi, fam, vb, famcode, idx, sq2_of, sig, sign=1.0):
    """{global index: weight} of the pooled blocks of one family, exactly as
    `cvhcf::trackExponents` forms them: v_b pooled over the resolution
    entries of the index, sq2 over the step rows of the index,
    w = sign * sqrt(vpool / sq2) / sigma.  Blocks with vpool <= 0, no rows or
    sq2 <= 0 get no weight."""
    out = {}
    sel = fam == famcode
    for g in np.unique(gi[sel]):
        vpool = vb[sel & (gi == g)].astype(np.float64).sum()
        if vpool <= 0.:
            continue
        rows = np.where(idx == g)[0]
        if not len(rows):
            continue
        sq2 = sq2_of(g, rows)
        if sq2 <= 0.:
            continue
        out[int(g)] = sign * np.sqrt(vpool / sq2) / sig
    return out


def maker_step_weights(rec, sig, ioni_sign):
    """Per MS row: the angular weight (its MS block's) and the signed recoil
    weight -- the ionisation weight of the SAME leg at the SAME step,
    w_io(block) * cs * 1e-3 -- as the in-maker family pairs them: the
    radiative rows are one per Geant4 step, parallel to the MS rows, and carry
    the leg's ionisation block index (`radstepidx`) and the step's cs = E/p^3
    (column 10).  rec: gi, fam, vb, M, midx, I, iidx, qsi, qsv, ridx, R."""
    M, midx, I, iidx = rec['M'], rec['midx'], rec['I'], rec['iidx']
    gi, fam, vb = rec['gi'], rec['fam'], rec['vb']
    wms = block_weights(gi, fam, vb, 10, midx,
                        lambda g, r: M[r, 5].astype(np.float64).sum(), sig)

    def io_sq2(g, r):
        qs = rec['qsv'][rec['qsi'] == g] if rec.get('qsv') is not None else None
        return ctr.ioni_sq2(I[r].astype(np.float64), None if qs is None else qs.astype(np.float64))
    wio = block_weights(gi, fam, vb, 11, iidx, io_sq2, sig, ioni_sign)
    R, ridx = rec['R'], rec['ridx']
    if len(R) != len(M) or (len(M) and not np.array_equal(R[:, 4], M[:, 3])):
        raise ValueError('radiative rows are not parallel to the MS rows')
    wang = np.array([wms.get(int(g), 0.0) for g in midx], dtype=np.float64)
    wrec = np.array([wio.get(int(g), 0.0) for g in ridx], dtype=np.float64) \
        * R[:, 10].astype(np.float64) * 1e-3
    return wang, wrec


def step_family(T, M, imat, pdg, wang, wrec, tau, groups=None):
    """The nuclear-elastic family of MS rows M with table material rows
    `imat`, species `pdg` (0: none), angular weights `wang` and signed recoil
    weights `wrec` [z per MeV]: `nucel_tables.Table.rows` with one entry per
    row at its block's angular weight, the recoil at its own weight and the
    joint term at both, under cf_nucel_exact's NUCEL_RECOIL / NUCEL_JOINT and
    cf_knockon.QOP_EXACT.  Returns (S_ang, S_rec, S_jnt, N), or with `groups`
    (the rows' material group) also {group: (S_ang, S_rec, S_jnt, N)} over the
    groups with collisions, the flat family then the sum of the groups in
    ascending order.  The reference of the in-maker family (`cvhcf`
    nucelFunctional).  Rows whose material does not resolve raise, as the
    maker does."""
    import cf_knockon
    M = np.asarray(M)
    pdg = np.asarray(pdg)
    imat = np.asarray(imat)
    wang = np.asarray(wang, dtype=np.float64)
    wrec = np.asarray(wrec, dtype=np.float64)
    mass = np.array([cf_nucel_exact.mass_of(int(x)) if x != 0 else 0.0 for x in pdg])
    kw = dict(recoil=bool(cf_nucel_exact.NUCEL_RECOIL), joint=bool(cf_nucel_exact.NUCEL_JOINT),
              exact=bool(cf_knockon.QOP_EXACT))

    def run(r):
        N, a, rc, j = T.rows(tau, M[r].astype(np.float64), imat[r], pdg[r], mass[r],
                             np.arange(len(r)), wang[r], np.ones(len(r)), wrec[r], wang[r], **kw)
        return a, rc, j, N

    if groups is None:
        return run(np.arange(len(M)))
    nt = len(tau)
    flat = [np.zeros(nt), np.zeros(nt, dtype=np.complex128), np.zeros(nt, dtype=np.complex128), 0.0]
    grp = {}
    groups = np.asarray(groups)
    for g in np.unique(groups[pdg != 0]):
        r = np.flatnonzero((pdg != 0) & (groups == g))
        a, rc, j, N = run(r)
        if N == 0.0:
            continue
        grp[int(g)] = (a, rc, j, N)
        flat[0] = flat[0] + a
        flat[1] = flat[1] + rc
        flat[2] = flat[2] + j
        flat[3] += N
    return tuple(flat), grp


def candidate(rec, tau, T, want_ms=False):
    """(Snuc_ang, Srec, Sjnt, diag, Sms_rebuilt or None) for one candidate:
    the angular, recoil and joint (angle-recoil, `Table.rows`) exponents.

    rec: dict of this candidate's arrays (gi, fam, vb, M, midx, I, iidx, qsi,
    qsv, sigma, pp, pm)."""
    nt = len(tau)
    sig = rec['sigma']
    gi, fam, vb = rec['gi'], rec['fam'], rec['vb']
    M, midx = rec['M'], rec['midx']
    Sang = np.zeros(nt)
    Srec = np.zeros(nt, dtype=np.complex128)
    Sjnt = np.zeros(nt, dtype=np.complex128)
    Sms = np.zeros(nt) if want_ms else None
    diag = dict(N=0.0, Np=0.0, Nm=0.0, v=0.0, vrec=0.0, mrec=0.0, xgunk=0.0,
                Nfirst=0.0, Nbig=0.0, Nbigfirst=0.0, Sfirst=np.zeros(nt),
                Nsurv=0.0, Ssurv=np.zeros(nt), Nan=0.0, San1=np.zeros(nt), San2=np.zeros(nt),
                Nfirstmig=0.0, Sfirstmig=np.zeros(nt), msi_legs=[])
    if not len(M):
        return Sang, Srec, Sjnt, diag, Sms

    # ---- leg / species of every MS record --------------------------------
    p = M[:, 3].astype(np.float64)
    sgn = np.zeros(len(p), dtype=int)
    # the leg's FIRST MS block: the material from the decay vertex up to and
    # including the first measured module -- a kick there is invisible to the
    # pattern recognition (no hit before it to see a kink against)
    first = np.zeros(len(p), dtype=bool)
    for seg in _segments(p):
        sgn[seg] = _leg_sign(p[seg[0]], rec['pp'], rec['pm'])
        first[seg[midx[seg] == midx[seg[0]]]] = True
    # ---- MS block weights: exactly the maker's -------------------------------
    w = np.zeros(len(p))
    sel10 = fam == 10
    for g in np.unique(gi[sel10]):
        vpool = vb[sel10 & (gi == g)].astype(np.float64).sum()
        if vpool <= 0.:
            continue
        rows = np.where(midx == g)[0]
        if not len(rows):
            continue
        sq2 = M[rows, 5].astype(np.float64).sum()
        if sq2 <= 0.:
            continue
        wb = np.sqrt(vpool / sq2) / sig
        w[rows] = wb
        if want_ms:
            Sms += ctr.ms_step_exponent(M[rows].astype(np.float64), wb, tau)

    # ---- per-step rate: per-element targets of the step's material ---------
    xg = M[:, 2].astype(np.float64)
    imat = step_materials(T, M, rec.get('msmat'), rec.get('filemats'))
    unk = (imat < 0) & (M[:, 0] >= 1.0)
    diag['xgunk'] = float(xg[unk].sum() / max(xg.sum(), 1e-300))
    if rec.get('mspdg') is not None:
        # the species the maker propagated each row as (`mspdgv`)
        pdg = np.asarray(rec['mspdg'])
        sgn = np.where(pdg > 0, 1, -1)
        isp = np.array([T.species_index(int(x)) for x in pdg], dtype=int)
    else:
        isp = np.where(sgn > 0, T.species_index(211), T.species_index(-211))
    ok = (imat >= 0) & (xg > 0.)
    # node weights: the two nodes of the step's model segment bracketing p
    j0 = np.zeros(len(p), dtype=int)
    j1 = np.zeros(len(p), dtype=int)
    f = np.zeros(len(p))
    for sp_ in np.unique(isp):
        m = isp == sp_
        j0[m], j1[m], f[m] = T.nodes(sp_, p[m])
    N = np.zeros(len(p))
    mom = np.zeros((len(p), 4))                # <theta>, <theta^2> at p_s, <dE>, <dE^2>
    groups = []
    for sp_, im_ in sorted(set(zip(isp[ok].tolist(), imat[ok].tolist()))):
        mx = T.mixture(sp_, im_)
        r = np.where(ok & (isp == sp_) & (imat == im_))[0]
        if mx is None:
            diag['xgunk'] += float(xg[r].sum() / max(xg.sum(), 1e-300))
            continue
        N[r] = T.rate(mx, p[r]) * xg[r]
        mom[r] = T.mom(mx, j0[r], j1[r], f[r], p[r])
        groups.append((mx, r))
    diag['N'] = float(N.sum())
    diag['Np'] = float(N[sgn > 0].sum())
    diag['Nm'] = float(N[sgn < 0].sum())
    diag['v'] = float(np.sum(N * w ** 2 * mom[:, 1] / 2.0))
    diag['Nfirst'] = float(N[first].sum())
    # collisions whose MEAN kick alone is a > 3 sigma_m mass shift
    big = w * mom[:, 0] > 3.0
    diag['Nbig'] = float(N[big].sum())
    diag['Nbigfirst'] = float(N[big & first].sum())

    # ---- the first block's migration factor (SURVIVAL) ---------------------
    # silicon layers (runs of pure-silicon steps) crossed in the leg's first
    # block before its target module
    # (the missing-inner-module count); the family of that block is scaled by
    # MIG[count] (survival/survival_model.py)
    migf = np.ones(len(p))
    issi = np.array([imat[i] >= 0 and len(T.mats[imat[i]]['Z']) == 1 and int(T.mats[imat[i]]['Z'][0]) == 14
                     for i in range(len(p))])
    if SURVIVAL:
        MIG = np.asarray(SURVIVAL['model'].get('MIG', [1.0]), dtype=np.float64)
        for seg in _segments(p):
            fr = seg[midx[seg] == midx[seg[0]]]
            si = issi[fr]
            runs = int(np.sum(si & np.concatenate(([True], ~si[:-1])))) if len(si) else 0
            migf[fr] = MIG[min(max(runs - 1, 0), len(MIG) - 1)]
            diag['msi_legs'].append(max(runs - 1, 0))

    # ---- the angular exponent, per (species, material) ----------------------
    for mx, r in groups:
        r = r[(w[r] > 0.) & (N[r] > 0.)]
        if not len(r):
            continue
        g = T.g(mx, j0[r], j1[r], f[r], p[r], w[r][:, None] * tau[None, :])
        Sang += (N[r][:, None] * (g - 1.0)).sum(axis=0)
        diag['Sfirst'] += ((N[r] * first[r])[:, None] * (g - 1.0)).sum(axis=0)
        diag['Sfirstmig'] += ((N[r] * first[r] * migf[r])[:, None] * (g - 1.0)).sum(axis=0)

    # ---- the survival-weighted family (SURVIVAL, survival/survival_model.py) --
    diag['Nfirstmig'] = float((N * first * migf).sum())
    # the q-survival forms need the block eigenvalues (`reseigv`), which only
    # the single-track maker exports
    if SURVIVAL and rec.get('lam') is not None and rec['lam'].shape[1] > 0:
        import survival_model as smod
        lam = rec['lam']
        mig = migf
        _, cb = smod.block_quantities(midx, M.astype(np.float64), gi, fam, vb.astype(np.float64), lam, sig)
        _, Ss, _, _, Ns = smod.steps_family(T, _theta_table(), p, xg, imat, isp, w, cb, first, mig,
                                            tau, SURVIVAL['model'])
        diag['Ssurv'] = Ss
        diag['Nsurv'] = Ns
        # the anisotropic form (survival on the kick's resolved projection):
        # response along the best-resolved direction (an1) and independent of
        # it (an2) -- the two limits the eigenvalue-only exports allow
        Sf = smod.sfun_factory(SURVIVAL['model'], max(float(rec.get('ndof', 1.0)), 1.0))
        MIGA = np.asarray(SURVIVAL['model'].get('MIG_ANISO', [1.0]), dtype=np.float64)
        for g in np.unique(midx):
            r = np.where((midx == g) & (N > 0.) & (w > 0.))[0]
            ent = np.where((fam == 10) & (gi == g))[0]
            if not len(r) or not len(ent):
                continue
            sq2 = M[midx == g, 5].astype(np.float64).sum()
            if sq2 <= 0.:
                continue
            th, Nb = smod.block_theta_quantiles(_theta_table(), isp[r], imat[r], p[r], N[r])
            if th is None:
                continue
            i1, i2 = lam[ent[0], 0] / sq2, lam[ent[0], 1] / sq2
            a1, n1 = smod.aniso_block_exponent(th, Nb, w[r[0]], i1, i2, tau, Sf)
            a2, _ = smod.aniso_block_exponent_iso(th, Nb, w[r[0]], i1, i2, tau, Sf)
            fm = 1.0
            if first[r[0]]:
                si = issi[midx == g]
                runs = int(np.sum(si & np.concatenate(([True], ~si[:-1])))) if len(si) else 0
                fm = MIGA[min(max(runs - 1, 0), len(MIGA) - 1)]
            diag['San1'] += fm * a1
            diag['San2'] += fm * a2
            diag['Nan'] += fm * n1

    # ---- the recoil: the ionisation weight of the same leg at the same p ----
    if cf_nucel_exact.NUCEL_RECOIL and len(rec['I']) and rec.get('R') is not None:
        # the in-maker pairing: the radiative rows are parallel to the MS rows
        # and name each step's leg ionisation block and its cs
        wq = maker_step_weights(rec, sig, IONI_SGN)[1]
    elif cf_nucel_exact.NUCEL_RECOIL and len(rec['I']):
        # records without the radiative rows: the leg's ionisation weight
        # interpolated in p along the leg
        I, iidx = rec['I'], rec['iidx']
        wio = np.zeros(len(I))
        sel11 = fam == 11
        for g in np.unique(gi[sel11]):
            vpool = vb[sel11 & (gi == g)].astype(np.float64).sum()
            if vpool <= 0.:
                continue
            rows = np.where(iidx == g)[0]
            if not len(rows):
                continue
            qs = rec['qsv'][rec['qsi'] == g] if rec['qsv'] is not None else None
            sq2 = ctr.ioni_sq2(I[rows].astype(np.float64),
                               None if qs is None else qs.astype(np.float64))
            if sq2 <= 0.:
                continue
            wio[rows] = IONI_SGN * np.sqrt(vpool / sq2) / sig
        cs = I[:, 10].astype(np.float64)
        wq_i = wio * cs * 1e-3                    # z per MeV of recoil loss
        pio = _p_from_cs(cs)
        sgi = np.zeros(len(pio), dtype=int)
        for seg in _segments(pio):
            sgi[seg] = _leg_sign(pio[seg[0]], rec['pp'], rec['pm'])
        wq = np.zeros(len(p))
        for s in (+1, -1):
            mi = (sgi == s) & (wq_i != 0.)
            mm = sgn == s
            if mi.any() and mm.any():
                o = np.argsort(pio[mi])
                wq[mm] = np.interp(p[mm], pio[mi][o], wq_i[mi][o])
    else:
        return Sang, Srec, Sjnt, diag, Sms
    # the recoil and the joint term of every row with a tabulated mixture, at
    # (the block's angular weight, the step's recoil weight)
    diag['vrec'] = float(np.sum(N * wq ** 2 * mom[:, 3]))
    diag['mrec'] = float(np.sum(N * wq * mom[:, 2]))
    rr = np.concatenate([r for _, r in groups]) if groups else np.zeros(0, dtype=int)
    if len(rr):
        import cf_knockon
        pdg = np.array([T.species[i] for i in isp[rr]], dtype=int)
        mass = np.array([cf_nucel_exact.mass_of(x) for x in pdg])
        none = np.zeros(0)
        _, _, Srec, Sjnt = T.rows(tau, M[rr].astype(np.float64), imat[rr], pdg, mass,
                                  np.zeros(0, dtype=int), none, none, wq[rr], w[rr],
                                  recoil=True, joint=bool(cf_nucel_exact.NUCEL_JOINT),
                                  exact=bool(cf_knockon.QOP_EXACT))
    return Sang, Srec, Sjnt, diag, Sms


_BR = ['reseigidx', 'resinfvarv', 'msmoliv', 'msmoliidx', 'ioniurbanv', 'ioniurbanidx',
       'Jpsi_sigmamass', 'Muplus_pt', 'Muminus_pt', 'Muplus_eta', 'Muminus_eta',
       'cfmass_ms', 'run', 'lumi', 'event']


def process_task(arg):
    """One (task, truth) chunk: the standard ks_pairs columns + the family."""
    cidx, fs, tn, tabfn, mopts, gate_ms = arg
    if 'T' not in _T:
        _T['T'] = load_table(tabfn)
    T = _T['T']
    t = ks_pairs.load_truth([tn], quiet=True)
    cols = {}
    state = {'tgrid': None, 'tag': '', 'aux': mopts['aux'], 'auxi': mopts['auxi']}
    margs = argparse.Namespace(**mopts['match'])
    extra = {k: [] for k in ('Snuc_ang', 'Snuc_rec_re', 'Snuc_rec_im', 'Snuc_jnt_re',
                             'Snuc_jnt_im', 'nuc_N', 'nuc_Np',
                             'nuc_Nm', 'nuc_v', 'nuc_vrec', 'nuc_mrec', 'nuc_xgunk',
                             'nuc_Nfirst', 'nuc_Nbig', 'nuc_Nbigfirst', 'Snuc_first',
                             'nuc_Nsurv', 'Snuc_surv', 'nuc_Nan', 'Snuc_an1', 'Snuc_an2',
                             'nuc_Nfirstmig', 'Snuc_firstmig', 'nuc_msimax', 'zchk')}
    gate = []
    for fn in fs:
        ks_pairs.read_one(fn, t, margs, cols, state)
        f = uproot.open(fn)
        if 'tree' not in f:
            continue
        tr = f['tree']
        pt = f['runtree']['parmtype'].array(library='np')
        have = set(tr.keys())
        br = _BR + [b for b in ('ioniqscaleidx', 'ioniqscalev', 'cfmass_ok', 'Jpsi_mass',
                                'Jpsi_x', 'Jpsi_y', 'Jpsi_z', 'Muplus_phi', 'Muminus_phi')
                    if b in have]
        br += [b for b in ('radstepidx', 'radstepv', 'mspdgv', 'reseigv', 'ndof') if b in have]
        br += prodfiles.stride_keys(have, ('msmoliv', 'ioniurbanv', 'radstepv'))
        # the per-step material index and the job's material table, when the
        # file carries them; otherwise materials are identified from the records
        filemats = None
        if 'msmatv' in have:
            br.append('msmatv')
            filemats = nucel_tables.read_materials(fn)
        a = tr.arrays(br, library='np')
        if not len(a['Jpsi_mass']):
            continue
        tau = np.asarray(state['tgrid'], dtype=np.float64)
        # the SAME selection read_one applied (same inputs, same function)
        tab = {k: np.asarray(a[k], dtype=np.float64) for k in
               ('Muplus_eta', 'Muminus_eta', 'Muplus_pt', 'Muminus_pt',
                'Muplus_phi', 'Muminus_phi', 'Jpsi_x', 'Jpsi_y', 'Jpsi_z')}
        for k in ('run', 'lumi', 'event'):
            tab[k] = np.asarray(a[k], dtype=np.int64)
        jrow, _, _ = ks_pairs.match(tab, t, margs)
        sig = np.asarray(a['Jpsi_sigmamass'], dtype=np.float64)
        okf = np.asarray(a['cfmass_ok']).astype(bool)
        good = okf & np.isfinite(sig) & (sig > 0.) & (jrow >= 0)
        idx = np.where(good)[0]
        for ic in idx:
            gi = np.asarray(a['reseigidx'][ic])
            M = prodfiles.reshape_records(a['msmoliv'][ic], nrec=len(a['msmoliidx'][ic]),
                                          stride=prodfiles.entry_stride(a, 'msmoliv', ic),
                                          branch='msmoliv')
            I = prodfiles.reshape_records(a['ioniurbanv'][ic], nrec=len(a['ioniurbanidx'][ic]),
                                          stride=prodfiles.entry_stride(a, 'ioniurbanv', ic),
                                          branch='ioniurbanv')
            rec = dict(gi=gi, fam=pt[gi], vb=np.asarray(a['resinfvarv'][ic], dtype=np.float32),
                       lam=(np.asarray(a['reseigv'][ic], dtype=np.float64).reshape(len(gi), -1)
                            if 'reseigv' in a else None),
                       ndof=(float(a['ndof'][ic]) if 'ndof' in a else 1.0),
                       M=M, midx=np.asarray(a['msmoliidx'][ic]), I=I,
                       iidx=np.asarray(a['ioniurbanidx'][ic]),
                       qsi=(np.asarray(a['ioniqscaleidx'][ic]) if 'ioniqscaleidx' in a else None),
                       qsv=(np.asarray(a['ioniqscalev'][ic], dtype=np.float32).reshape(-1, 2)
                            if 'ioniqscalev' in a else None),
                       msmat=(np.asarray(a['msmatv'][ic]) if filemats is not None else None),
                       mspdg=(np.asarray(a['mspdgv'][ic]) if 'mspdgv' in a else None),
                       ridx=(np.asarray(a['radstepidx'][ic]) if 'radstepidx' in a else None),
                       R=(prodfiles.reshape_records(a['radstepv'][ic], nrec=len(a['radstepidx'][ic]),
                                                    stride=prodfiles.entry_stride(a, 'radstepv', ic),
                                                    branch='radstepv')
                          if 'radstepidx' in a else None),
                       filemats=filemats,
                       sigma=float(sig[ic]),
                       pp=float(a['Muplus_pt'][ic] * np.cosh(a['Muplus_eta'][ic])),
                       pm=float(a['Muminus_pt'][ic] * np.cosh(a['Muminus_eta'][ic])))
            want = gate_ms > 0 and len(gate) < gate_ms
            if cf_nucel_exact.NUCEL_CHANNEL:
                Sa, Sr, Sj, dg, Sms = candidate(rec, tau, T, want_ms=want)
            else:
                Sa, Sr = np.zeros(len(tau)), np.zeros(len(tau), dtype=np.complex128)
                Sj = np.zeros(len(tau), dtype=np.complex128)
                dg = dict(N=0., Np=0., Nm=0., v=0., vrec=0., mrec=0., xgunk=0.,
                          Nfirst=0., Nbig=0., Nbigfirst=0., Sfirst=np.zeros(len(tau)),
                          Nsurv=0., Ssurv=np.zeros(len(tau)), Nan=0., San1=np.zeros(len(tau)),
                          San2=np.zeros(len(tau)), Nfirstmig=0., Sfirstmig=np.zeros(len(tau)), msi_legs=[])
                Sms = None
            if Sms is not None:
                ref = np.asarray(a['cfmass_ms'][ic], dtype=np.float64)
                gate.append((float(np.max(np.abs(Sms - ref))), float(np.max(np.abs(ref)))))
            extra['Snuc_ang'].append(Sa.astype(np.float32))
            extra['Snuc_rec_re'].append(Sr.real.astype(np.float32))
            extra['Snuc_rec_im'].append(Sr.imag.astype(np.float32))
            extra['Snuc_jnt_re'].append(Sj.real.astype(np.float32))
            extra['Snuc_jnt_im'].append(Sj.imag.astype(np.float32))
            for k in ('N', 'Np', 'Nm', 'v', 'vrec', 'mrec', 'xgunk', 'Nfirst', 'Nbig',
                      'Nbigfirst', 'Nsurv', 'Nan', 'Nfirstmig'):
                extra[f'nuc_{k}'].append(dg[k])
            extra['Snuc_first'].append(dg['Sfirst'].astype(np.float32))
            extra['Snuc_surv'].append(dg['Ssurv'].astype(np.float32))
            extra['Snuc_an1'].append(dg['San1'].astype(np.float32))
            extra['Snuc_an2'].append(dg['San2'].astype(np.float32))
            extra['Snuc_firstmig'].append(dg['Sfirstmig'].astype(np.float32))
            extra['nuc_msimax'].append(max(dg['msi_legs']) if dg['msi_legs'] else -1)
            extra['zchk'].append(float(a['Jpsi_mass'][ic]))
    return cidx, cols, extra, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', required=True)
    ap.add_argument('--table', default=nucel_tables.BIN_DEFAULT)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--jobs', type=int, default=32)
    ap.add_argument('--maxtasks', type=int, default=0)
    ap.add_argument('--slice', type=int, default=0, help='this slice of the chunks (merge with --merge)')
    ap.add_argument('--nslice', type=int, default=1)
    ap.add_argument('--merge', nargs='+', default=None,
                    help='merge slice caches (in chunk order) into --cache and exit')
    ap.add_argument('--survival', default='',
                    help='survival_model.json: also write the survival-weighted family '
                         '(Snuc_surv, nuc_Nsurv; survival/survival_model.py)')
    ap.add_argument('--theta-table', default='/ceph/submit/data/user/d/david_w/ZMass/cvh/'
                    'nucel_survival_260926/theta_tables/theta_table.npz')
    ap.add_argument('--gate-ms', type=int, default=0,
                    help='rebuild cfmass_ms from the records for the first N '
                         'candidates of each chunk and compare with the maker')
    args = ap.parse_args()
    if args.merge:
        parts = [dict(np.load(fn, allow_pickle=True)) for fn in args.merge]
        rowk = [k for k, v in parts[0].items() if np.ndim(v) >= 1 and len(v) == len(parts[0]['z'])]
        ch = np.concatenate([p['chunk'] for p in parts])
        o = np.argsort(ch, kind='stable')
        out = {k: np.concatenate([p[k] for p in parts])[o] for k in rowk}
        for k, v in parts[0].items():
            if k not in out:
                out[k] = v
        np.savez_compressed(args.cache, **out)
        print(f'merged {len(parts)} slices -> {args.cache}: {len(out["z"])} rows')
        return
    global SURVIVAL
    if args.survival:
        import json
        SURVIVAL = dict(model=json.load(open(args.survival)), theta=args.theta_table)
        print(f"survival-weighted family: model {SURVIVAL['model'].get('tag')} ({args.survival})")
    print(f'NUCEL_CHANNEL={int(cf_nucel_exact.NUCEL_CHANNEL)} '
          f'NUCEL_RECOIL={int(cf_nucel_exact.NUCEL_RECOIL)}')
    tdir = os.path.join(args.prod, 'truth')
    pairs = []
    for td in sorted(glob.glob(os.path.join(args.prod, 'task_[0-9]*'))):
        idx = os.path.basename(td).split('_')[-1]
        if not os.path.exists(os.path.join(td, '.complete')):
            continue
        tn = os.path.join(tdir, f'truth_{idx}.npz')
        fs = sorted(glob.glob(os.path.join(td, 'globalcor_ks_*.root')))
        if fs and os.path.exists(tn):
            pairs.append((idx, fs, tn))
    if args.maxtasks:
        pairs = pairs[:args.maxtasks]
    if args.nslice > 1:
        pairs = pairs[args.slice::args.nslice]
    print(f'{len(pairs)} complete chunks')
    aux = dict(ks_pairs.cf_inmaker._MASS_AUX)
    aux.pop('mpre', None)
    aux.pop('w', None)
    aux['munc'] = 'Jpsi_mass_unc'
    aux['vtxres'] = 'Jpsi_vtxres'
    aux['covmassvtx'] = 'Jpsi_covmassvtx'
    mopts = dict(aux=aux, auxi=dict(ks_pairs.cf_inmaker._MASS_AUX_INT),
                 match=dict(max_dvtx=2.0, max_dlam=0.25, max_dphi=0.25, max_dp=0.50,
                            keep_unmatched=False))
    tasks = [(c, fs, tn, os.path.abspath(args.table), mopts, args.gate_ms)
             for c, fs, tn in pairs]
    res = {}
    gate = []
    with Pool(args.jobs) as pool:
        for k, (c, cols, extra, g) in enumerate(pool.imap_unordered(process_task, tasks)):
            res[c] = (cols, extra)
            gate += g
            if (k + 1) % 50 == 0:
                print(f'  {k+1}/{len(tasks)}', flush=True)
    out = {}
    tgrid = None
    for c in sorted(res):
        cols, extra = res[c]
        if not cols:
            continue
        nstd = sum(len(x) for x in cols['z'])
        nex = len(extra['zchk'])
        if nstd != nex:
            raise SystemExit(f'chunk {c}: {nstd} standard rows vs {nex} nucel rows')
        mr = np.concatenate(cols['ks_mreco'])
        if not np.array_equal(mr, np.asarray(extra['zchk'])):
            raise SystemExit(f'chunk {c}: row alignment broken')
        for k, v in cols.items():
            out.setdefault(k, []).append(np.concatenate(v))
        out.setdefault('chunk', []).append(np.full(nstd, int(c)))
        for k, v in extra.items():
            if k == 'zchk':
                continue
            out.setdefault(k, []).append(np.asarray(v))
    out = {k: np.concatenate(v) for k, v in out.items()}
    # tau grid and model tag from one file
    f0 = glob.glob(os.path.join(args.prod, 'task_*', 'globalcor_ks_*.root'))[0]
    tg, tag = ks_pairs.cf_inmaker._runtree_grid(uproot.open(f0))
    out['tgrid'] = np.asarray(tg, dtype=np.float64)
    out['cf_source'] = np.array(['inmaker-ks+nucel-offline'])
    out['cf_model'] = np.array([tag])
    out['ioni_sign_fixed'] = np.array([1])
    out['nucel_channel'] = np.array([int(cf_nucel_exact.NUCEL_CHANNEL)])
    out['nucel_recoil'] = np.array([int(cf_nucel_exact.NUCEL_RECOIL)])
    out['nucel_table'] = np.array([os.path.abspath(args.table)])
    out['nucel_survival'] = np.array([json.dumps(SURVIVAL['model']) if SURVIVAL else ''])
    np.savez_compressed(args.cache, **out)
    print(f'wrote {args.cache}: {len(out["z"])} rows')
    if gate:
        g = np.array(gate)
        print(f'MS WEIGHT GATE ({len(g)} candidates): max|Sms(records) - cfmass_ms| = '
              f'{g[:, 0].max():.3e} (max|S| {g[:, 1].max():.3g}; float32 floor '
              f'{6e-8 * g[:, 1].max():.1e})')
    if cf_nucel_exact.NUCEL_CHANNEL:
        print(f'N per candidate: mean {out["nuc_N"].mean():.4f} median {np.median(out["nuc_N"]):.4f}; '
              f'v_nuc median {np.median(out["nuc_v"]):.4e} mean {out["nuc_v"].mean():.4e}; '
              f'xg without a table bucket: max {out["nuc_xgunk"].max():.2e}')


if __name__ == '__main__':
    main()
