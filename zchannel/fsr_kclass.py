#!/usr/bin/env python3
"""Resolution classes: the selection-conditional FSR kernel under conditioning.

The Z likelihood conditions every candidate on its own mass resolution -- the
v-form conditions on ``k = sigma_m / m^p``.  The selected FSR content depends on
``k``: ``<u>`` runs 16.1e-3 to 12.5e-3 across the ``k`` quintiles of the reco MC
(`ptres.py sigcond`), and it does so identically under a cut on the true
``p_T``, so it is a **kinematic** correlation -- a large-``sigma`` candidate is a
forward or soft one, which has a different acceptance and therefore a different
selected radiation spectrum.  A population-level ``K_sel(u|m)`` inside a
``k``-conditioned likelihood is therefore misspecified per candidate.

The fix needs no new axis on the boson-kinematics interface.  The mass
resolution of a pair is, to the accuracy of the track fit itself,

    sigma_m / m = 1/2 sqrt( (sigma_{pT,+}/pT_+)^2 + (sigma_{pT,-}/pT_-)^2 )

-- ``ln m = 1/2 [ln pT_+ + ln pT_- + ln(2(cosh dEta - cos dPhi))]`` and the
angular term carries no measurable resolution -- so ``k`` is a function of the
two legs' ``(p_T, eta)``, which are exactly the axes of the ``h4`` table the
provider already hands over.  A **resolution class** is therefore a partition of
that table:

    k_pred(pT_+, eta_+, pT_-, eta_-) = 1/2 sqrt(s(pT_+,eta_+)^2 + s(pT_-,eta_-)^2)

with ``s`` the measured ``sigma_pT/pT`` map (`ptres.Resolution.sigma`), and the
class is a bin of ``k_pred``.  ``K_sel(u|m, class)`` and ``A(m|class)`` follow
from the existing chain with the table restricted to the class, so SCETLib +
DYTurbo supply exactly what they supply already.

Subcommands
-----------
``classes``  the class edges, from the ``k_pred`` quantiles of a gen record
``check``    the reco MC: is ``k_pred``-class conditioning enough for ``k``?
``build``    the per-class ``h`` tables, kernels and acceptances
``figs``     the figures

usage:
  python3 fsr_kclass.py classes --gen data/genmerged_full.npz \
      --res data/ptres_dyv2.npz --nclass 5 --pt-cuts 25 10 -o data/kcl5.json
  python3 fsr_kclass.py check --aux ../fullscale/runs/auxgen_dyv2.npz \
      --pairs ../fullscale/runs/zpairs_dyv2_full.npz --res data/ptres_dyv2.npz
  python3 fsr_kclass.py build --gen data/genmerged_full.npz \
      --classes data/kcl5.json --run data/photos/gen_mcMix.npz
"""
import argparse
import json
import os
import time

import numpy as np

#: the |eta| fiducial cut, everywhere
ETA_CUT = 2.4


# --------------------------------------------------------------------------
# k_pred: the pair mass resolution from the two legs' (pT, eta)
# --------------------------------------------------------------------------
def kpred(resol, pt1, eta1, pt2, eta2):
    """``sigma_m/m`` predicted from the ``sigma_pT/pT(pT, eta)`` map.

    ``m^2 = 2 pT_+ pT_- (cosh dEta - cos dPhi)``, so

        d ln m = 1/2 (d ln pT_+ + d ln pT_-) + 1/2 d ln(2(cosh dEta - cos dPhi))

    and with the two legs' momentum errors independent and the angular term
    resolution-free,

        sigma_m/m = 1/2 sqrt( s_+^2 + s_-^2 ) ,  s_q = sigma_pT/pT(pT_q, eta_q).

    Measured against the two-track fit's own ``sigma_m`` on the reco MC
    (`check`): using the fit's per-leg ``sigma_rel`` for ``s_q`` the identity
    holds to 0.3 %, so the opening angle and the leg-leg correlation of the
    covariance are negligible and what is left is the map itself.
    """
    s1 = np.asarray(resol.sigma(pt1, eta1), float)
    s2 = np.asarray(resol.sigma(pt2, eta2), float)
    return 0.5 * np.sqrt(s1 * s1 + s2 * s2)


def smeared_pt(g, resol, seed, post=True):
    """The per-muon smeared ``p_T`` of `fit_gen.fiducial`, same draw.

    Reproduces the seeded, fixed-order draw of `fit_gen.fiducial` so that the
    class assignment sees exactly the smeared momenta the selection did.
    """
    sfx = "" if post else "_pre"
    rng = np.random.default_rng(seed)
    out = []
    for i in ("1", "2"):
        pt = np.asarray(g[f"pt{i}{sfx}"], float)
        eta = np.asarray(g[f"eta{i}{sfx}"], float)
        out.append(pt * (1.0 + resol.sample(pt, eta, rng)))
    return out


def gen_kpred(g, resol, kin="post", smear=None, seed=None):
    """``k_pred`` of every event of a gen record.

    ``kin`` is ``post`` (the observed, status-1 muons -- what the likelihood
    can assign a real candidate by) or ``pre`` (the Born muons, for which the
    class is an exact partition of the ``h`` table's own cells).  ``smear``
    replaces the post-FSR ``p_T`` by the reconstructed one of the acceptance
    toy, with `fit_gen.fiducial`'s own draw.
    """
    sfx = "" if kin == "post" else "_pre"
    eta1 = np.asarray(g[f"eta1{sfx}"], float)
    eta2 = np.asarray(g[f"eta2{sfx}"], float)
    if smear is not None:
        pt1, pt2 = smeared_pt(g, smear, seed, post=(kin == "post"))
    else:
        pt1 = np.asarray(g[f"pt1{sfx}"], float)
        pt2 = np.asarray(g[f"pt2{sfx}"], float)
    return kpred(resol, pt1, eta1, pt2, eta2)


def assign(kp, edges):
    """Class index of every ``k_pred``: ``edges`` are the interior boundaries."""
    return np.clip(np.searchsorted(np.asarray(edges, float), kp, "right"),
                   0, len(edges))


def load_classes(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# the class-restricted h table
# --------------------------------------------------------------------------
def class_htable(m, pt1, eta1, pt2, eta2, w, bands, keep, **kw):
    """`fsr_perleg.build_htable` of a subpopulation, normalised to the whole.

    The table's contract is that a band sums to ``P(|eta| < cut and both pT
    above the grid floor | m)`` -- normalised to the band's **total** weight,
    not to the subpopulation's.  Building it on the class subsample normalises
    to the class, so the result is rescaled by the class fraction of each band;
    then ``sum_cells h_C = P(class C and eta cut | m)`` and everything
    downstream reads ``A_C(m) = P(pass and class C | m)`` without knowing that
    a class is involved.

    ``m_bar`` and ``sumw`` are kept at the **population** values, so every
    class's ``A(m)`` is tabulated on the same mass grid.
    """
    import fsr_perleg as FP

    keep = np.asarray(keep, bool)
    ht = FP.build_htable(m[keep], pt1[keep], eta1[keep], pt2[keep], eta2[keep],
                         w[keep], bands, **kw)
    m = np.asarray(m, float)
    w = np.asarray(w, float)
    tot = np.zeros(len(bands))
    mbar = np.zeros(len(bands))
    nev = np.zeros(len(bands), np.int64)
    for k, (lo, hi) in enumerate(bands):
        s = (m >= lo) & (m < hi)
        if not s.any():
            continue
        tot[k] = w[s].sum()
        mbar[k] = float(np.sum(w[s] * m[s]) / tot[k]) if tot[k] else 0.0
        nev[k] = int(s.sum())
    frac = np.where(tot > 0, ht["sumw"] / np.where(tot > 0, tot, 1.0), 0.0)
    ht["h"] = ht["h"] * frac[:, None, None]
    ht["m_bar"] = mbar
    ht["sumw"] = tot
    ht["nev_class"] = ht["nev"].copy()
    ht["nev"] = nev
    ht["frac"] = frac
    return ht


# --------------------------------------------------------------------------
# the class as part of the PASS REGION
# --------------------------------------------------------------------------
# A resolution class is a region in the **post-FSR** muon kinematics, exactly
# like the two ``p_T`` thresholds are.  It is therefore NOT a partition of the
# ``h`` table, whose axes are the pre-FSR ``(b, eta)``: a candidate lands in a
# low-``k_pred`` class *because* it radiated, and a table restricted to the
# class would make that a property of its Born kinematics.  The class belongs
# where the cut belongs -- in ``G(u_+, u_-)``.
#
# With ``s(p_T, eta)`` monotonically increasing in ``p_T`` (all three terms of
# the 4-parameter form are), ``k_pred^2 = (t_+ + t_-)/4``, ``t = s^2``, is
# increasing in both post-FSR momenta at fixed ``eta``, so
#
#     U(T) = { t_a(v_+) + t_c(v_-) >= T }
#
# is an upper-right set whose boundary ``v_- = g(v_+)`` is decreasing, i.e. a
# **staircase**, and a decreasing staircase is a signed sum of quadrants:
#
#     P(U u_k Q_k) = sum_k S(x_k, y_k) - sum_k S(x_{k+1}, y_k) ,
#     Q_k = {v_+ > x_k, v_- > y_k} ,  y_k = g(x_k) decreasing
#
# (the pairwise intersections are nested, so inclusion-exclusion truncates).
# A class is ``U(T_lo) \ U(T_hi)``, and each quadrant is intersected with the
# lepton thresholds by raising its shifts, so everything stays a quadrant and
# the whole region is read off the SAME survival function of the same table.
def _sigma2(resol, iband, v, pt_ref):
    """``(sigma_pT/pT)^2`` of an eta band at ``p_T = pt_ref e^v``."""
    import ptres
    return ptres.sig_model(pt_ref * np.exp(np.asarray(v, float)),
                           resol.coef[iband]) ** 2


def _staircase(resol, a, c, T, pt_ref, knots):
    """``[(x_k, y_k)]`` of the upper set ``t_a(v_+) + t_c(v_-) >= T``.

    ``y_k`` is clipped to the knot range: below it the leg is unconstrained
    (the quadrant is then the plain threshold), above it the knot contributes
    nothing and is dropped.
    """
    lo, hi = float(knots[0]), float(knots[-1])
    need = T - _sigma2(resol, a, knots, pt_ref)
    tc = _sigma2(resol, c, knots, pt_ref)
    out = []
    for x, nd in zip(knots, need):
        if nd > tc[-1]:                     # no v_- in range can reach T
            continue
        if nd <= tc[0]:                     # any v_- does: leg unconstrained
            out.append((float(x), lo))
            break                           # and so does every larger x
        out.append((float(x), float(np.interp(nd, tc, knots))))
    return out


def _quadrants(stair):
    """The signed quadrant list of a decreasing staircase."""
    if not stair:
        return []
    t = [(x, y, +1.0) for x, y in stair]
    t += [(stair[i + 1][0], stair[i][1], -1.0) for i in range(len(stair) - 1)]
    return t


class ClassPassRegion:
    """``G(u_+, u_-|m)`` of ``pass AND resolution class``, off the ``h4`` table.

    Duck-types `fsr_perleg.PassRegion` -- `build_corr_kernel` only ever calls
    ``grid(k)`` -- so the class costs the kernel chain nothing but a longer
    term list.  ``iclass = None`` reproduces the plain pass region and is the
    closure test of the construction.
    """

    def __init__(self, ht, h4, resol, cuts, kedges, iclass, nknot=90,
                 v_split=2.0, cuts_resol=None):
        import fsr_perleg as FP

        self.b_edges = np.asarray(ht["b_edges"], float)
        self.pt_ref = float(ht["pt_ref"])
        c = np.atleast_1d(np.asarray(cuts, float))
        self.cuts = ((float(c[0]), float(c[0])) if c.size == 1
                     else (float(np.max(c)), float(np.min(c))))
        self.shifts = FP.cut_shifts(self.cuts, self.pt_ref)
        self.terms = FP.rect_terms(*self.shifts)
        self.resol = cuts_resol                    # the CUT's resolution (None)
        self.h4 = h4
        self.H4 = h4["h4"]
        self.h4_b = np.asarray(h4["b_edges"], float)
        self.h4_eta = np.asarray(h4["eta_edges"], float)
        self.iclass = iclass
        self.kedges = list(kedges)
        ne = len(self.h4_eta) - 1
        lo = min(self.shifts[1], float(self.h4_b[0]))
        hi = float(self.h4_b[-1])
        knots = np.concatenate([np.linspace(lo, v_split, nknot),
                                np.linspace(v_split, hi, nknot // 4 + 1)[1:]])
        self.knots = knots
        self.ac_terms = [[self._terms_ac(resol, a, c_) for c_ in range(ne)]
                         for a in range(ne)]
        n = sum(len(x) for row in self.ac_terms for x in row)
        self.n_terms = n

    def _terms_ac(self, resol, a, c):
        """The signed quadrant list of ``pass AND class`` for one eta pair."""
        if self.iclass is None:
            return list(self.terms)
        e = [0.0] + list(self.kedges) + [np.inf]
        quads = []
        for sgn, K in ((+1.0, e[self.iclass]), (-1.0, e[self.iclass + 1])):
            if not np.isfinite(K):
                continue
            if K <= 0.0:
                quads.append((self.knots[0], self.knots[0], sgn))
                continue
            st = _staircase(resol, a, c, 4.0 * K * K, self.pt_ref, self.knots)
            quads += [(x, y, sgn * s) for x, y, s in _quadrants(st)]
        # intersect every quadrant with the lepton thresholds: raising a
        # quadrant's shifts keeps it a quadrant.  Many then collapse onto the
        # same pair of shifts -- every knot below the threshold clamps to it --
        # so they are summed and the cancelling ones dropped.
        agg = {}
        for x, y, s in quads:
            for sp, sm, sg in self.terms:
                key = (round(max(x, sp), 9), round(max(y, sm), 9))
                agg[key] = agg.get(key, 0.0) + s * sg
        return [(x, y, s) for (x, y), s in agg.items() if abs(s) > 1e-12]

    def grid(self, k):
        import fsr_perleg as FP

        H = np.asarray(self.H4[k], float)
        ne = len(self.h4_eta) - 1
        n = len(self.b_edges)
        G = np.zeros((n, n))
        for a in range(ne):
            for c in range(ne):
                M = H[a, :, c, :]
                if not M.any():
                    continue
                g = FP.survival(M, self.h4_b)
                for sp, sm, sgn in self.ac_terms[a][c]:
                    R = FP._interp_rows(g, self.h4_b, self.b_edges + sm)
                    G += sgn * FP._interp_cols(R, self.h4_b,
                                               self.b_edges + sp)
        return G

    def label(self):
        return (f"pT > {self.cuts[0]:g}/{self.cuts[1]:g}, class "
                f"{self.iclass} of {len(self.kedges)+1}")


def save_htable(path, ht, meta):
    np.savez_compressed(path, h=ht["h"].astype(np.float32),
                        b_edges=ht["b_edges"], bands=ht["bands"],
                        m_bar=ht["m_bar"], sumw=ht["sumw"], nev=ht["nev"],
                        provenance=np.array([json.dumps(meta)]))


# --------------------------------------------------------------------------
# the class edges
# --------------------------------------------------------------------------
def _load_gen(path, wclip=100.0):
    d = np.load(path)
    g = {k: d[k] for k in d.files if d[k].ndim == 1}
    w = np.asarray(g["weight"], np.float64)
    aw = np.abs(w)
    w = np.clip(w, -wclip * np.median(aw), wclip * np.median(aw))
    g["w"] = w * (len(w) / w.sum())
    return g


def _fiducial(g, cuts, eta_max=ETA_CUT, smear=None, seed=None):
    import fit_gen as FG
    return FG.fiducial(g, cuts, eta_max, post=True, smear=smear,
                       seed=(FG.SMEAR_SEED if seed is None else seed))


def _wquantile(x, w, qs):
    """Weighted quantiles, safe against the negative MiNNLO weights.

    4.8 % of the weights are negative, so the raw cumulative is not monotone
    and `np.interp` would read a non-monotone ``xp`` -- silently, and with
    grossly wrong edges.  The cumulative is therefore monotonised; the
    correction is local jitter of order one weight.
    """
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.maximum.accumulate(np.cumsum(w))
    return np.interp(np.asarray(qs) * c[-1], c, x)


def _classes_cmd(args):
    import ptres

    R = ptres.Resolution(args.res, mode=args.res_mode)
    g = _load_gen(args.gen, args.wclip)
    sm = R if args.smear else None
    kp = gen_kpred(g, R, kin=args.kin, smear=sm, seed=args.smear_seed)
    cuts = (args.pt_cuts[0], args.pt_cuts[-1])
    sel = _fiducial(g, cuts, args.eta_cut, smear=sm, seed=args.smear_seed)
    q = np.linspace(0.0, 1.0, args.nclass + 1)[1:-1]
    edges = [float(f"{v:.6g}") for v in _wquantile(kp[sel], g["w"][sel], q)]
    cl = assign(kp, edges)
    out = dict(kind="kpred", nclass=args.nclass, edges=edges,
               res=os.path.abspath(args.res), res_mode=args.res_mode,
               kin=args.kin, smear=bool(args.smear),
               smear_seed=args.smear_seed, quantiles=q.tolist(),
               pt_cuts=list(cuts), eta_cut=args.eta_cut,
               gen=os.path.abspath(args.gen), wclip=args.wclip)
    tot = g["w"][sel].sum()
    rows = []
    for c in range(args.nclass):
        s = sel & (cl == c)
        u = -np.log(np.maximum(np.asarray(g["m_post"], float)[s]
                               / np.asarray(g["m_pre"], float)[s], 1e-300))
        ww = g["w"][s]
        rows.append(dict(c=c, frac=float(ww.sum() / tot),
                         kpred=float(np.sum(ww * kp[s]) / ww.sum()),
                         u=float(np.sum(ww * u) / ww.sum()),
                         n=int(s.sum())))
    out["selected"] = rows
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[classes] {args.nclass} classes of k_pred, {args.kin}-FSR "
          f"kinematics, pT > {cuts[0]}/{cuts[1]}, |eta| < {args.eta_cut}")
    print(f"  edges: {edges}")
    for r in rows:
        print(f"    class {r['c']}: frac {r['frac']:.4f}  <k_pred> "
              f"{r['kpred']:.5f}  <u> {r['u']*1e3:8.4f}e-3  n {r['n']}")
    print(f"[classes] -> {args.output}")


# --------------------------------------------------------------------------
# the reco-MC check: is k_pred-class conditioning enough for k conditioning?
# --------------------------------------------------------------------------
def load_reco(aux, pairs, wclip=100.0):
    """The reco MC caches, after the standard two-track selection.

    Returns a dict of per-candidate arrays; the selection cuts on the fit's own
    covariance (``sigma_m > 0``, ``chi2/ndof < 3``) and on the hit count, which
    is the standard Z selection, and requires both legs gen-matched.
    """
    a = np.load(aux)
    p = np.load(pairs)
    ok = (np.isfinite(a["sigma"]) & (a["sigma"] > 0)
          & (a["nv_p"] >= 8) & (a["nv_m"] >= 8) & (a["normchi2"] < 3.0)
          & (a["gpt_p"] > 0) & (a["gpt_m"] > 0)
          & (p["mpre"] > 0) & (a["mgen"] > 0) & np.isfinite(p["mtrk"])
          & (p["mtrk"] > 0))
    w = np.asarray(p["w"], float)
    aw = np.abs(w)
    w = np.clip(w, -wclip * np.median(aw), wclip * np.median(aw))
    w = w * (len(w) / w.sum())
    d = dict(w=w[ok],
             u=-np.log(np.asarray(a["mgen"], float)[ok]
                       / np.asarray(p["mpre"], float)[ok]),
             k=np.asarray(a["sigma"], float)[ok] / np.asarray(p["mtrk"], float)[ok],
             kcvh=0.5 * np.sqrt(np.asarray(p["sigrelp"], float)[ok] ** 2
                                + np.asarray(p["sigrelm"], float)[ok] ** 2),
             mreco=np.asarray(p["mtrk"], float)[ok],
             pt_p=np.asarray(a["pt_p"], float)[ok],
             pt_m=np.asarray(a["pt_m"], float)[ok],
             eta_p=np.asarray(p["etap"], float)[ok],
             eta_m=np.asarray(p["etam"], float)[ok],
             gpt_p=np.asarray(a["gpt_p"], float)[ok],
             gpt_m=np.asarray(a["gpt_m"], float)[ok],
             geta_p=np.asarray(a["geta_p"], float)[ok],
             geta_m=np.asarray(a["geta_m"], float)[ok])
    return d


def within_cell_slope(u, lnk, cell, nmin=50):
    """``d<u>/d ln k`` with the kinematic cell mean removed from both.

    The within-cell regression is the only form that answers the physics
    question: *given* the muon kinematics, does the residual of the per-track
    resolution (hit pattern, material, alignment) know anything about the
    radiation?  Pooling without the cell means measures the kinematic
    correlation instead, which is the thing the class is there to carry.
    """
    _, cell = np.unique(cell, return_inverse=True)
    n = cell.max() + 1
    cnt = np.bincount(cell, None, n)
    good = cnt >= nmin
    mk = np.where(good, np.bincount(cell, lnk, n) / np.maximum(cnt, 1), 0.0)
    mu = np.where(good, np.bincount(cell, u, n) / np.maximum(cnt, 1), 0.0)
    g = good[cell]
    dk = (lnk - mk[cell])[g]
    du = (u - mu[cell])[g]
    sxx = float(np.sum(dk * dk))
    slope = float(np.sum(dk * du) / sxx)
    r = du - slope * dk
    err = float(np.sqrt(np.sum(r * r) / sxx / (len(dk) - int(good.sum()))))
    rms = float(np.sqrt(sxx / len(dk)))
    return slope, err, rms, int(good.sum()), int(len(dk))


def _kin_cell(pt1, eta1, pt2, eta2, n_eta, pt_edges):
    """A symmetric fine kinematic cell index of the two legs."""
    ee = np.linspace(0.0, ETA_CUT, n_eta + 1)
    npt = len(pt_edges) - 1

    def one(pt, eta):
        return (np.clip(np.searchsorted(ee, np.abs(eta), "right") - 1, 0, n_eta - 1)
                * npt + np.clip(np.searchsorted(pt_edges, pt, "right") - 1, 0, npt - 1))

    i1, i2 = one(pt1, eta1), one(pt2, eta2)
    return np.minimum(i1, i2) * (n_eta * npt) + np.maximum(i1, i2)


#: the fine kinematic grids the within-cell slope is checked to converge on
CELL_GRIDS = (
    (15, np.array([10, 20, 25, 30, 35, 40, 45, 50, 60, 80, 2e4]), "|eta|/0.16, 10 pT"),
    (24, np.array([10, 20, 25, 28, 31, 34, 37, 40, 43, 46, 50, 55, 60, 70, 80,
                   100, 2e4]), "|eta|/0.10, 16 pT"),
    (48, np.array([10, 20, 25, 28, 31, 34, 37, 40, 43, 46, 50, 55, 60, 70, 80,
                   100, 2e4]), "|eta|/0.05, 16 pT"),
)


def _check_cmd(args):
    import ptres

    R = ptres.Resolution(args.res, mode=args.res_mode)
    d = load_reco(args.aux, args.pairs, args.wclip)
    cL, cT = (args.pt_cuts[0], args.pt_cuts[-1])
    sel = ((np.maximum(d["pt_p"], d["pt_m"]) > cL)
           & (np.minimum(d["pt_p"], d["pt_m"]) > cT)
           & (np.abs(d["eta_p"]) < args.eta_cut)
           & (np.abs(d["eta_m"]) < args.eta_cut))
    selT = ((np.maximum(d["gpt_p"], d["gpt_m"]) > cL)
            & (np.minimum(d["gpt_p"], d["gpt_m"]) > cT)
            & (np.abs(d["geta_p"]) < args.eta_cut)
            & (np.abs(d["geta_m"]) < args.eta_cut))
    kp = kpred(R, d["pt_p"], d["eta_p"], d["pt_m"], d["eta_m"])
    kpg = kpred(R, d["gpt_p"], d["geta_p"], d["gpt_m"], d["geta_m"])
    u, k, w = d["u"], d["k"], d["w"]
    print(f"[check] {len(u)} candidates, {int(sel.sum())} selected at "
          f"{cL}/{cT}, |eta| < {args.eta_cut}")

    # -- 1. the formula --------------------------------------------------
    print("\n--- k_pred against the two-track fit's own sigma_m ---")
    for lab, kk in (("1/2 sqrt(s+^2+s-^2), CVH per-leg sigma", d["kcvh"]),
                    ("1/2 sqrt(s+^2+s-^2), map @ reco (pT,eta)", kp),
                    ("1/2 sqrt(s+^2+s-^2), map @ gen  (pT,eta)", kpg)):
        r = k[sel] / kk[sel]
        q = np.quantile(r, [0.16, 0.5, 0.84])
        c = float(np.corrcoef(np.log(k[sel]), np.log(kk[sel]))[0, 1])
        print(f"    {lab:42s} k/k_pred median {q[1]:.4f}  68 % spread "
              f"{(q[2]-q[0])/2/q[1]*100:5.2f} %  corr(ln k, ln k_pred) {c:.4f}")

    # -- 2. <u> across the k quintiles, reco cut against true cut --------
    print(f"\n--- <u> across the k = sigma_m/m quantiles "
          f"({args.nq} bins, reco cut and true cut) ---")
    e = np.quantile(k, np.linspace(0.0, 1.0, args.nq + 1))
    e[0], e[-1] = -np.inf, np.inf
    kq = np.clip(np.searchsorted(e, k, "right") - 1, 0, args.nq - 1)
    act = np.zeros(args.nq)
    for i in range(args.nq):
        a_ = sel & (kq == i)
        b_ = selT & (kq == i)
        act[i] = float(np.sum(w[a_] * u[a_]) / np.sum(w[a_]))
        ub = float(np.sum(w[b_] * u[b_]) / np.sum(w[b_]))
        print(f"    {i*100//args.nq:3d}-{(i+1)*100//args.nq:<3d} %  <k> "
              f"{float(np.median(k[kq == i])):.5f}  n {int(a_.sum()):8d}  "
              f"<u> reco {act[i]*1e3:8.4f}e-3  true {ub*1e3:8.4f}e-3  "
              f"reco/true {act[i]/ub:.4f}")

    # -- 3. do the classes reproduce that run? ---------------------------
    print("\n--- <u | k quantile> from the k_pred classes alone ---")
    for nc in args.nclass:
        ce = _wquantile(kp[sel], w[sel], np.linspace(0, 1, nc + 1)[1:-1])
        cl = assign(kp, ce)
        ub = np.array([float(np.sum(w[sel & (cl == c)] * u[sel & (cl == c)])
                             / np.sum(w[sel & (cl == c)])) for c in range(nc)])
        pred = np.array([float(np.sum(w[sel & (kq == i)] * ub[cl[sel & (kq == i)]])
                               / np.sum(w[sel & (kq == i)])) for i in range(args.nq)])
        print(f"    {nc:3d} classes:  model " + " ".join(f"{v*1e3:6.2f}" for v in pred)
              + f"   max |model - MC| {np.abs(pred-act).max()*1e3:5.3f}e-3"
              + f"   rms {np.sqrt(((pred-act)**2).mean())*1e3:5.3f}e-3")
    print("    " + " " * 14 + "MC    " + " ".join(f"{v*1e3:6.2f}" for v in act))

    # -- 4. <u> against k WITHIN a class ---------------------------------
    nc = args.nclass[len(args.nclass) // 2]
    ce = _wquantile(kp[sel], w[sel], np.linspace(0, 1, nc + 1)[1:-1])
    cl = assign(kp, ce)
    rat = k / np.maximum(kp, 1e-12)
    print(f"\n--- <u> against the per-candidate k WITHIN each of {nc} classes "
          f"({args.nsub} bins of k/k_pred) ---")
    for c in range(nc):
        m = sel & (cl == c)
        ee = _wquantile(rat[m], w[m], np.linspace(0, 1, args.nsub + 1)[1:-1])
        sub = assign(rat, ee)
        xs, ys = [], []
        line = []
        for j in range(args.nsub):
            mm = m & (sub == j)
            ub = float(np.sum(w[mm] * u[mm]) / np.sum(w[mm]))
            xs.append(float(np.sum(w[mm] * np.log(k[mm])) / np.sum(w[mm])))
            ys.append(ub)
            line.append(f"{ub*1e3:7.3f}")
        sl = float(np.polyfit(xs, ys, 1)[0])
        sd = float(np.sqrt(np.sum(w[m] * (np.log(k[m]) - np.mean(xs)) ** 2)
                           / np.sum(w[m])))
        print(f"    class {c}: <k_pred> {float(np.sum(w[m]*kp[m])/np.sum(w[m])):.5f}"
              f"  <u> per k/k_pred bin " + " ".join(line)
              + f"   d<u>/dln k {sl*1e3:+7.3f}e-3  (x 1 sd of ln k in class,"
              f" {sd:.3f}) = {sl*sd*1e3:+6.3f}e-3")

    # -- 5. the same slope with the kinematics controlled exactly --------
    print("\n--- d<u>/d ln k with the muon kinematics controlled, grid by grid ---")
    for kk, lab in ((k, "k = sigma_m/m"), (d["kcvh"], "k = CVH per-leg")):
        for n_eta, pe, tag in CELL_GRIDS:
            cell = _kin_cell(d["pt_p"][sel], d["eta_p"][sel],
                             d["pt_m"][sel], d["eta_m"][sel], n_eta, pe)
            sl, er, rms, nc_, nu = within_cell_slope(u[sel], np.log(kk[sel]),
                                                     cell, args.nmin)
            print(f"    {lab:16s} {tag:20s} cells {nc_:6d}  slope "
                  f"{sl*1e3:+8.4f} +- {er*1e3:6.4f} e-3   rms d ln k {rms:.4f}"
                  f"  -> {sl*rms*1e3:+7.4f}e-3")

    if args.output:
        np.savez(args.output, kpred=kp, k=k, u=u, w=w, sel=sel, selT=selT,
                 kcvh=d["kcvh"], pt_p=d["pt_p"], pt_m=d["pt_m"],
                 eta_p=d["eta_p"], eta_m=d["eta_m"])
        print(f"\n[check] arrays -> {args.output}")


# --------------------------------------------------------------------------
# the per-class tables, kernels and acceptances
# --------------------------------------------------------------------------
def _one_kernel(job):
    """One `fsr_perleg` correlated kernel, in a worker process."""
    import fsr_perleg as FP

    (htpath, out, accout, cuts, run, pair, meta) = job
    t0 = time.time()
    ht = FP.load_htable(htpath)
    cells = FP.tabulated_cells(run, ht["bands"]) if run else None
    pr = FP.PassRegion(ht, cuts=cuts)
    ker, acc, info = FP.build_corr_kernel(ht, cells_by_band=cells,
                                          pair=tuple(pair), pr=pr, verbose=False)
    meta = dict(meta, kind="corr", selection=FP._sel_meta(pr, ht),
                htable=os.path.abspath(htpath), run=run, pair=list(pair),
                bands=info)
    np.savez(out, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"], m_hi=ker["m_hi"],
             provenance=np.array([json.dumps(meta)]))
    with open(accout, "w") as fh:
        json.dump(dict(kind="grid", m=acc["m"], a=acc["a"],
                       _meta=dict(selection=FP._sel_meta(pr, ht),
                                  eta_cut=ht["eta_cut"],
                                  source=os.path.abspath(out))), fh, indent=1)
    apk = float(np.interp(91.2, acc["m"], acc["a"]))
    return (os.path.basename(out), len(ker["r"]), apk, time.time() - t0)


def groups_of(nfine, ngroup):
    """Contiguous grouping of ``nfine`` fine classes into ``ngroup``.

    The fine classes are quantiles of ``k_pred``, so a contiguous grouping is
    again a set of ``k_pred`` intervals and the grouped table is the **sum** of
    the fine ones: ``h`` is linear in the events and every class is normalised
    to the same band total.  One build of the finest partition therefore serves
    the whole class-count scan.
    """
    e = np.linspace(0, nfine, ngroup + 1).round().astype(int)
    return [list(range(e[i], e[i + 1])) for i in range(ngroup)]


def clip_negative(ker, acc=None):
    """Clip the numerical negatives of a class kernel and its ``A(m)``.

    The staircase is an inclusion-exclusion of survival functions read
    bilinearly, so a band whose class content is empty -- every one of them is
    at ``m_pre`` above 170 GeV, far outside the 50-130 GeV Born grid -- can come
    back at ``-1e-12`` to ``-1e-3`` of the band.  The provider rejects a
    negative weight or acceptance outright, so they are clipped and the band is
    renormalised to the sum it had.
    """
    w = np.asarray(ker["w"], float).copy()
    w[~np.isfinite(w)] = 0.0
    lo, hi = np.asarray(ker["m_lo"], float), np.asarray(ker["m_hi"], float)
    for a, b in sorted(set(zip(lo[w < 0].tolist(), hi[w < 0].tolist()))):
        s = (lo == a) & (hi == b)
        tot = w[s].sum()
        c = np.maximum(w[s], 0.0)
        w[s] = c * (tot / c.sum()) if c.sum() > 0 and tot > 0 else c
    # a band whose content was entirely numerical noise now sums to zero, and
    # the provider rejects an empty band; drop its atoms outright
    keep = np.ones(len(w), bool)
    for a, b in sorted(set(zip(lo.tolist(), hi.tolist()))):
        s = (lo == a) & (hi == b)
        if w[s].sum() <= 0.0:
            keep &= ~s
    ker = dict(ker, r=np.asarray(ker["r"], float)[keep], w=w[keep],
               m_lo=lo[keep], m_hi=hi[keep])
    if acc is not None:
        acc = dict(acc, a=[max(float(v), 0.0) for v in acc["a"]])
    return ker, acc


def _one_kernel_pr(job):
    """One correlated kernel whose pass region carries the class."""
    import fsr_perleg as FP
    import ptres

    (htpath, h4path, out, accout, cuts, run, pair, resfile, resmode, ked,
     iclass, nknot, meta) = job
    t0 = time.time()
    ht = FP.load_htable(htpath)
    h4 = FP.load_h4table(h4path)
    R = ptres.Resolution(resfile, mode=resmode)
    pr = ClassPassRegion(ht, h4, R, cuts, ked, iclass, nknot=nknot)
    cells = FP.tabulated_cells(run, ht["bands"]) if run else None
    ker, acc, info = FP.build_corr_kernel(ht, cells_by_band=cells,
                                          pair=tuple(pair), pr=pr,
                                          verbose=False)
    ker, acc = clip_negative(ker, acc)
    meta = dict(meta, kind="corr", pt_lead=pr.cuts[0], pt_trail=pr.cuts[1],
                eta_cut=ht["eta_cut"], htable=os.path.abspath(htpath),
                h4=os.path.abspath(h4path), run=run, pair=list(pair),
                kedges=list(ked), iclass=iclass, nknot=nknot,
                n_terms=int(pr.n_terms), bands=info)
    np.savez(out, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"], m_hi=ker["m_hi"],
             provenance=np.array([json.dumps(meta)]))
    with open(accout, "w") as fh:
        json.dump(dict(kind="grid", m=acc["m"], a=acc["a"],
                       _meta=dict(source=os.path.abspath(out),
                                  iclass=iclass, cuts=list(cuts))), fh,
                  indent=1)
    return (os.path.basename(out), len(ker["r"]),
            float(np.interp(91.2, acc["m"], acc["a"])), time.time() - t0)


def _buildpr_cmd(args):
    """The per-class kernels with the class inside the PASS REGION.

    The only correct construction when the class is read off the observed
    (post-FSR) muons: the population ``h``/``h4`` tables are untouched and the
    class enters `ClassPassRegion` exactly where the ``p_T`` thresholds do.
    """
    from multiprocessing import Pool

    cfg = load_classes(args.classes)
    jobs = []
    for ng in args.ngroups:
        grp = groups_of(cfg["nclass"], ng)
        ked = [cfg["edges"][g[0] - 1] for g in grp[1:]]
        for cuts in args.cut_sets:
            cs = "".join(f"{int(c)}" for c in cuts)
            for j in range(ng):
                for lab, run, pair in (("mc", args.run, ()),
                                       ("data", None, ("e", "mu", "tau", "had"))):
                    if lab not in args.configs:
                        continue
                    out = f"data/kern_{args.tag}_{lab}_{cs}_n{ng}_c{j}.npz"
                    acc = f"data/acc_{args.tag}_{lab}_{cs}_n{ng}_c{j}.json"
                    if os.path.exists(out) and os.path.exists(acc) \
                            and not args.force:
                        continue
                    jobs.append((args.htable, args.h4, out, acc, list(cuts),
                                 run, pair, cfg["res"],
                                 cfg.get("res_mode", "shape"), ked, j,
                                 args.nknot,
                                 dict(kclass=dict(cfg, ngroup=ng, igroup=j),
                                      cuts=list(cuts), config=lab)))
    print(f"[buildpr] {len(jobs)} kernels on {args.nproc} workers")
    with Pool(args.nproc) as p:
        for name, na, apk, dt in p.imap_unordered(_one_kernel_pr, jobs):
            print(f"    {name:48s} {na:5d} atoms  A(91.2) {apk:.5f}"
                  f"  ({dt:.0f} s)")


def _build_cmd(args):
    import fsr_perleg as FP
    import fit_gen as FG
    import ptres

    cfg = load_classes(args.classes)
    nfine = cfg["nclass"]
    R = ptres.Resolution(cfg["res"], mode=cfg.get("res_mode", "shape"))
    g = _load_gen(args.gen, cfg.get("wclip", 100.0))
    sm = R if cfg.get("smear") else None
    kp = gen_kpred(g, R, kin=cfg["kin"], smear=sm, seed=cfg.get("smear_seed"))
    cl = assign(kp, cfg["edges"])
    m = np.asarray(g["m_pre"], float)
    w = g["w"]
    bands = FP.make_bands(50.0, 200.0, 1.0)
    be = FP.b_grid([0.0, float(np.log(25.0 / args.pt_ref))], b_min=args.b_min,
                   b_max=args.b_max)
    tag = args.tag

    # -- 1. the fine class-restricted h tables ---------------------------
    pt1, eta1, pt2, eta2 = (np.asarray(g[k], float) for k in
                            ("pt1_pre", "eta1_pre", "pt2_pre", "eta2_pre"))
    fine = []
    for c in range(nfine):
        path = f"data/ht_{tag}_f{c}.npz"
        fine.append(path)
        if os.path.exists(path) and not args.force:
            continue
        t0 = time.time()
        ht = class_htable(m, pt1, eta1, pt2, eta2, w, bands, cl == c,
                          pt_ref=args.pt_ref, eta_cut=cfg["eta_cut"],
                          b_edges=be)
        meta = dict(gen=os.path.abspath(args.gen), pt_ref=args.pt_ref,
                    anchor_cuts=[25.0], b_min=args.b_min, b_max=args.b_max,
                    eta_cut=cfg["eta_cut"], band_lo=50.0, band_hi=200.0,
                    band_width=1.0, wclip=cfg.get("wclip", 100.0), b_thin=1,
                    n_b=len(be), kclass=dict(cfg, iclass=c),
                    classes=os.path.abspath(args.classes))
        save_htable(path, ht, meta)
        kk = int(np.argmin(np.abs(ht["m_bar"] - 91.2)))
        print(f"[build] fine class {c}: band 91.2 GeV fraction "
              f"{ht['frac'][kk]:.4f}, sum h {ht['h'][kk].sum():.5f}"
              f"  ({time.time()-t0:.0f} s) -> {path}")

    # -- 2. the grouped tables: sums of the fine ones --------------------
    paths = {}
    for ng in args.ngroups:
        for j, grp in enumerate(groups_of(nfine, ng)):
            path = f"data/ht_{tag}_n{ng}_c{j}.npz"
            paths[(ng, j)] = path
            if os.path.exists(path) and not args.force:
                continue
            if len(grp) == 1:
                ht = FP.load_htable(fine[grp[0]])
            else:
                ht = FP.load_htable(fine[grp[0]])
                for i in grp[1:]:
                    ht["h"] = ht["h"] + FP.load_htable(fine[i])["h"]
            meta = dict(ht["provenance"])
            meta["kclass"] = dict(cfg, ngroup=ng, igroup=j, fine=grp)
            save_htable(path, ht, meta)
            kk = int(np.argmin(np.abs(ht["m_bar"] - 91.2)))
            print(f"[build] {ng} classes, group {j} = {grp}: sum h "
                  f"{np.asarray(ht['h'])[kk].sum():.5f} -> {path}")

    # -- 3. the correlated kernels, one job per (group, cut, config) -----
    jobs = []
    for ng in args.ngroups:
        for cuts in args.cut_sets:
            cs = "".join(f"{int(c)}" for c in cuts)
            for j in range(ng):
                for lab, run, pair in (("mc", args.run, ()),
                                       ("data", None, ("e", "mu", "tau", "had"))):
                    if lab not in args.configs:
                        continue
                    out = f"data/kern_{tag}_{lab}_{cs}_n{ng}_c{j}.npz"
                    acc = f"data/acc_{tag}_{lab}_{cs}_n{ng}_c{j}.json"
                    if os.path.exists(out) and os.path.exists(acc) \
                            and not args.force:
                        continue
                    jobs.append((paths[(ng, j)], out, acc, list(cuts), run,
                                 pair, dict(kclass=dict(cfg, ngroup=ng,
                                                        igroup=j),
                                            cuts=list(cuts), config=lab)))
    if jobs:
        from multiprocessing import Pool
        print(f"[build] {len(jobs)} kernels on {args.nproc} workers")
        with Pool(args.nproc) as p:
            for name, na, apk, dt in p.imap_unordered(_one_kernel, jobs):
                print(f"    {name:48s} {na:5d} atoms  A(91.2) {apk:.5f}"
                      f"  ({dt:.0f} s)")

    # -- 4. the MC's own conditional kernel per class, and its A(m) ------
    cb = list(zip(FP.COND_BANDS[:-1], FP.COND_BANDS[1:]))
    cb = [(0.0, FP.COND_BANDS[0])] + cb + [(FP.COND_BANDS[-1], np.inf)]
    mpost = np.asarray(g["m_post"], float)
    e = np.arange(50.0, 200.0 + 1e-9, 1.0)
    ib = np.clip(np.searchsorted(e, m, "right") - 1, 0, len(e) - 2)
    tot = np.bincount(ib, w, len(e) - 1)
    mb = np.bincount(ib, w * m, len(e) - 1) / np.maximum(tot, 1e-30)
    good = tot > 0
    for cuts in args.cut_sets:
        cs = "".join(f"{int(c)}" for c in cuts)
        sel = _fiducial(g, (cuts[0], cuts[-1]), cfg["eta_cut"], smear=sm,
                        seed=cfg.get("smear_seed"))
        for ng in args.ngroups:
            for j, grp in enumerate(groups_of(nfine, ng)):
                out = f"data/kern_{tag}_cond_{cs}_n{ng}_c{j}.npz"
                accout = f"data/acc_{tag}_cond_{cs}_n{ng}_c{j}.json"
                if os.path.exists(out) and not args.force:
                    continue
                s = sel & np.isin(cl, grp)
                ker, info = FG.build_banded_kernel(m[s], mpost[s], w[s], cb,
                                                   sigma_cap=3.3e-4)
                meta = dict(kind="condker", mode="true",
                            kclass=dict(cfg, ngroup=ng, igroup=j, fine=grp),
                            cuts=list(cuts), gen=os.path.abspath(args.gen),
                            n=int(s.sum()))
                np.savez(out, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"],
                         m_hi=ker["m_hi"],
                         provenance=np.array([json.dumps(meta)]))
                npass = np.bincount(ib, w * s, len(e) - 1)
                with open(accout, "w") as fh:
                    json.dump(dict(kind="grid", m=mb[good].tolist(),
                                   a=np.maximum(npass[good] / tot[good],
                                                0.0).tolist(),
                                   _meta=dict(source=os.path.abspath(out),
                                              ngroup=ng, igroup=j,
                                              cuts=list(cuts))), fh, indent=1)
                print(f"    {os.path.basename(out):48s} {int(s.sum()):9d} "
                      f"events, {len(ker['r']):5d} atoms")


def _ccheck_cmd(args):
    """``G_C`` off the table against a direct weighted event count.

    At fixed ``(u_+, u_-)`` the generator record is asked, event by event,
    whether the two muons with those losses applied would pass the cuts AND
    land in the class; the table's answer is the same quantity read through
    the ``h4`` binning and the staircase.  What this measures is the
    construction and its discretisation, nothing else.
    """
    import fsr_perleg as FP
    import ptres

    ht = FP.load_htable(args.htable)
    h4 = FP.load_h4table(args.h4)
    cfg = load_classes(args.classes)
    R = ptres.Resolution(cfg["res"], mode=cfg.get("res_mode", "shape"))
    grp = groups_of(cfg["nclass"], args.ngroup)
    ked = [cfg["edges"][g[0] - 1] for g in grp[1:]]
    cuts = (args.pt_cuts[0], args.pt_cuts[-1])
    d = np.load(args.gen)
    w = np.asarray(d["weight"], float)
    aw = np.abs(w)
    w = np.clip(w, -100 * np.median(aw), 100 * np.median(aw))
    w = w * (len(w) / w.sum())
    m = np.asarray(d["m_pre"], float)
    pt1, eta1, pt2, eta2 = (np.asarray(d[k], float) for k in
                            ("pt1_pre", "eta1_pre", "pt2_pre", "eta2_pre"))
    lo, hi = args.band
    s = (m >= lo) & (m < hi)
    tot = w[s].sum()
    kb = int(np.argmin(np.abs(np.asarray(ht["m_bar"]) - 0.5 * (lo + hi))))
    prs = [ClassPassRegion(ht, h4, R, cuts, ked, c, nknot=args.nknot)
           for c in range(args.ngroup)]
    full = ClassPassRegion(ht, h4, R, cuts, ked, None)
    print(f"[ccheck] band [{lo}, {hi}) -> table band {kb} "
          f"(m_bar {ht['m_bar'][kb]:.2f}), {args.ngroup} classes, "
          f"edges {[f'{v:.5f}' for v in ked]}")
    print(f"   terms per band: " + ", ".join(str(p.n_terms) for p in prs)
          + f" (plain pass region {full.n_terms})")
    G = [p.grid(kb) for p in prs]
    G0 = full.grid(kb)
    be = np.asarray(ht["b_edges"], float)
    print(f"   {'(u_+, u_-)':>16s} " + " ".join(f"{'class '+str(c):>10s}"
                                                for c in range(args.ngroup))
          + f" {'sum':>10s} {'G(all)':>10s}")
    for up, um in args.u:
        e1 = pt1 * np.exp(-up)
        e2 = pt2 * np.exp(-um)
        ok = s & (np.abs(eta1) < cfg["eta_cut"]) & (np.abs(eta2) < cfg["eta_cut"])
        ok &= (np.maximum(e1, e2) > cuts[0]) & (np.minimum(e1, e2) > cuts[1])
        kp = kpred(R, e1, eta1, e2, eta2)
        cl = assign(kp, ked)
        mc = [float(w[ok & (cl == c)].sum() / tot) for c in range(args.ngroup)]
        tb = [float(FP.eval_G_at(g, be, up, um)) for g in G]
        print(f"   {'(%.3g, %.3g)' % (up, um):>16s} "
              + " ".join(f"{t:10.5f}" for t in tb)
              + f" {sum(tb):10.5f} {float(FP.eval_G_at(G0, be, up, um)):10.5f}")
        print(f"   {'MC':>16s} " + " ".join(f"{v:10.5f}" for v in mc)
              + f" {sum(mc):10.5f} {float(w[ok].sum()/tot):10.5f}")
        print(f"   {'table - MC':>16s} "
              + " ".join(f"{t-v:10.2e}" for t, v in zip(tb, mc)))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def kernel_u_of(path):
    """Per-band ``(m, <u>, A)`` of a per-leg kernel file, or ``None``."""
    p = json.loads(str(np.load(path, allow_pickle=False)["provenance"][0]))
    b = p.get("bands")
    if not isinstance(b, list) or not b or not isinstance(b[0], dict):
        return None
    b = [x for x in b if x.get("natoms")]
    return (np.array([x["m_bar"] for x in b]),
            np.array([x["mean_u"] for x in b]),
            np.array([x["A"] for x in b]))


def cond_u_of(path):
    """Per-band ``(m, <u>)`` of a banded MC kernel: its own atoms."""
    with np.load(path, allow_pickle=False) as d:
        r, w, lo, hi = d["r"], d["w"], d["m_lo"], d["m_hi"]
    out = []
    for a, b in sorted(set(zip(lo.tolist(), hi.tolist()))):
        s = (lo == a) & (hi == b)
        if not s.any() or not np.isfinite(b) or a <= 0:
            continue
        ww = w[s] / w[s].sum()
        out.append((0.5 * (a + b), -float(np.sum(ww * np.log(r[s])))))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def _acc_of(path):
    with open(path) as fh:
        d = json.load(fh)
    return np.asarray(d["m"], float), np.asarray(d["a"], float)


def _figs_cmd(args):
    import matplotlib.pyplot as plt
    import mplhep as hep
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "resolution"))
    import pubhtml
    import ratiopanel

    hep.style.use(hep.style.ROOT)
    out = args.outpath or pubhtml.figdir("fsr_kclass")
    os.makedirs(out, exist_ok=True)
    cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, args.ngroup))

    # ---- the reco-MC figures -------------------------------------------
    if args.check:
        d = np.load(args.check)
        kp, k, u, w, sel = d["kpred"], d["k"], d["u"], d["w"], d["sel"]
        cfg = load_classes(args.classes) if args.classes else None
        fine = assign(kp, cfg["edges"]) if cfg else None
        grp = (groups_of(cfg["nclass"], args.ngroup) if cfg else None)

        # k against k_pred
        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
        e = np.geomspace(6e-3, 4e-2, 41)
        i = np.clip(np.searchsorted(e, kp[sel], "right") - 1, 0, len(e) - 2)
        n = np.bincount(i, w[sel], len(e) - 1)
        s1 = np.bincount(i, w[sel] * k[sel], len(e) - 1)
        c = 0.5 * (e[1:] + e[:-1])
        g = n > 0
        ax.plot(c[g], (s1[g] / n[g]) * 1e3, "o", ms=4, color="black",
                label=r"$\langle k\rangle$ of the CVH covariance")
        ax.plot(c[g], c[g] * 1e3, color="tab:red", lw=1.6,
                label=r"$k_{\rm pred}$ from the $\sigma_{p_T}/p_T$ map")
        rax.plot(c[g], (s1[g] / n[g]) / c[g], "o", ms=4, color="black")
        rax.axhline(1.0, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$k = \sigma_m/m$  [$10^{-3}$]")
        rax.set_ylabel(r"$k / k_{\rm pred}$")
        rax.set_xlabel(r"$k_{\rm pred}$")
        rax.set_ylim(0.9, 1.25)
        ax.legend(fontsize=13, frameon=False, loc="upper left")
        ax.set_title(r"the pair mass resolution from the two legs' "
                     r"$(p_T, \eta)$", fontsize=14, loc="left")
        pubhtml.savefig(fig, os.path.join(out, "10_kpred_vs_k.pdf"))
        plt.close(fig)

        # <u> across the k quantiles, reco cut and true cut
        selT = d["selT"]
        nq = 10
        e = np.quantile(k, np.linspace(0, 1, nq + 1))
        e[0], e[-1] = -np.inf, np.inf
        kq = np.clip(np.searchsorted(e, k, "right") - 1, 0, nq - 1)
        fig, ax = plt.subplots(figsize=(9.5, 6.4))
        for msk, lab, col, mk in ((sel, "cut on the reconstructed $p_T$",
                                   "black", "o"),
                                  (selT, "cut on the true (post-FSR) $p_T$",
                                   "tab:red", "s")):
            x = [float(np.median(k[kq == i])) for i in range(nq)]
            y = [float(np.sum(w[msk & (kq == i)] * u[msk & (kq == i)])
                       / np.sum(w[msk & (kq == i)])) for i in range(nq)]
            ax.plot(np.array(x) * 1e3, np.array(y) * 1e3, mk, ms=5, color=col,
                    label=lab, mfc="none" if mk == "s" else col)
        ax.set_xlabel(r"$k = \sigma_m/m$  [$10^{-3}$]")
        ax.set_ylabel(r"$\langle u\rangle$ of the selected sample  [$10^{-3}$]")
        ax.legend(fontsize=13, frameon=False)
        ax.set_title("the selected radiation against the per-candidate "
                     "resolution", fontsize=14, loc="left")
        pubhtml.savefig(fig, os.path.join(out, "11_u_vs_k.pdf"))
        plt.close(fig)

        # <u> against k WITHIN each class
        if grp is not None:
            fig, ax = plt.subplots(figsize=(9.5, 6.6))
            rat = k / np.maximum(kp, 1e-12)
            for j, gg in enumerate(grp):
                m = sel & np.isin(fine, gg)
                ee = _wquantile(rat[m], w[m], np.linspace(0, 1, 6)[1:-1])
                sub = assign(rat, ee)
                x, y = [], []
                for q in range(5):
                    mm = m & (sub == q)
                    x.append(float(np.sum(w[mm] * k[mm]) / np.sum(w[mm])))
                    y.append(float(np.sum(w[mm] * u[mm]) / np.sum(w[mm])))
                ub = float(np.sum(w[m] * u[m]) / np.sum(w[m]))
                ax.plot(np.array(x) * 1e3, np.array(y) / ub, "o-", ms=4,
                        color=cols[j], lw=1.4, label=f"class {j}")
            ax.axhline(1.0, color="grey", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(r"$k = \sigma_m/m$ within the class  [$10^{-3}$]")
            ax.set_ylabel(r"$\langle u | k\rangle\,/\,\langle u\rangle_{\rm class}$")
            ax.legend(fontsize=11, frameon=False, ncol=2)
            ax.set_title(r"is $k$ still informative once the class is fixed?",
                         fontsize=14, loc="left")
            pubhtml.savefig(fig, os.path.join(out, "12_u_vs_k_inclass.pdf"))
            plt.close(fig)

        # the within-cell slope against the fineness of the kinematic cell
        fig, ax = plt.subplots(figsize=(9.5, 6.4))
        xs, ys, es = [], [], []
        for n_eta, pe, lab in CELL_GRIDS:
            cell = _kin_cell(d["pt_p"][sel], d["eta_p"][sel], d["pt_m"][sel],
                             d["eta_m"][sel], n_eta, pe)
            sl, er, rms, ncell, _ = within_cell_slope(u[sel], np.log(k[sel]),
                                                      cell)
            xs.append(ncell)
            ys.append(sl * rms)
            es.append(er * rms)
        ax.errorbar(xs, np.array(ys) * 1e3, yerr=np.array(es) * 1e3, fmt="o",
                    ms=6, color="black", lw=0, elinewidth=1.2,
                    label="kinematic cells")
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("number of kinematic cells")
        ax.set_ylabel(r"$\Delta\langle u\rangle$ over $\pm 1\sigma$ of $\ln k$"
                      r"  [$10^{-3}$]")
        ax.legend(fontsize=13, frameon=False, loc="upper left")
        ax.set_title(r"given the kinematics, the residual of $k$ knows nothing "
                     r"about the radiation", fontsize=13, loc="left")
        pubhtml.savefig(fig, os.path.join(out, "13_slope_convergence.pdf"))
        plt.close(fig)

    # ---- the per-class kernels ------------------------------------------
    tag, cs, ng = args.tag, args.cuts, args.ngroup
    for mtag, name, title in ((tag, "14_meanu_class",
                               "the class in the pass region"),
                              (args.control, "17_meanu_class_restricted",
                               "the class as a restriction of the h table")):
        if not mtag or not all(
                os.path.exists(f"data/kern_{mtag}_mc_{cs}_n{ng}_c{c}.npz")
                for c in range(ng)):
            continue
        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
        for c in range(ng):
            mm, um = cond_u_of(f"data/kern_kcl_cond_{cs}_n{ng}_c{c}.npz")
            got = kernel_u_of(f"data/kern_{mtag}_mc_{cs}_n{ng}_c{c}.npz")
            ax.plot(mm, um * 1e3, "o", ms=4, color=cols[c], label=f"class {c}")
            ax.plot(got[0], got[1] * 1e3, color=cols[c], lw=1.6)
            rax.plot(mm, np.interp(mm, got[0], got[1]) / um, "o-", ms=3,
                     color=cols[c], lw=1.2)
        rax.axhline(1.0, color="grey", lw=0.8)
        rax.set_ylim(0.6, 1.9)
        ax.set_xlim(58, 122)
        ax.set_ylim(0.0, 70.0)
        ax.set_ylabel(r"$\langle u | m\rangle$ (selected)  [$10^{-3}$]")
        rax.set_ylabel("model / MC")
        rax.set_xlabel(r"$m_{\rm pre}$ [GeV]")
        ax.legend(fontsize=11, frameon=False, ncol=2)
        ax.set_title(f"{title}: {ng} classes, $p_T$ > {cs[:2]}/{cs[2:]} GeV "
                     f"(points: MC, lines: corr mc model)", fontsize=12,
                     loc="left")
        pubhtml.savefig(fig, os.path.join(out, f"{name}_{cs}.pdf"))
        plt.close(fig)
    if all(os.path.exists(f"data/acc_{tag}_mc_{cs}_n{ng}_c{c}.json")
           for c in range(ng)):

        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
        for c in range(ng):
            mm, am = _acc_of(f"data/acc_kcl_cond_{cs}_n{ng}_c{c}.json")
            ms, asr = _acc_of(f"data/acc_{tag}_mc_{cs}_n{ng}_c{c}.json")
            g = (mm > 58) & (mm < 122)
            ax.plot(mm[g], am[g], "o", ms=3.5, color=cols[c], label=f"class {c}")
            ax.plot(ms, asr, color=cols[c], lw=1.6)
            rax.plot(mm[g], np.interp(mm[g], ms, asr) / np.maximum(am[g], 1e-12),
                     color=cols[c], lw=1.2)
        rax.axhline(1.0, color="grey", lw=0.8)
        rax.set_ylim(0.95, 1.05)
        ax.set_xlim(58, 122)
        ax.set_ylabel(r"$A(m\,|\,{\rm class}) = P({\rm pass\ and\ class}\,|\,m)$")
        rax.set_ylabel("model / MC")
        rax.set_xlabel(r"$m_{\rm pre}$ [GeV]")
        ax.legend(fontsize=11, frameon=False, ncol=2)
        ax.set_title(f"the class acceptance, {ng} classes", fontsize=13,
                     loc="left")
        pubhtml.savefig(fig, os.path.join(out, f"15_acceptance_class_{cs}.pdf"))
        plt.close(fig)

    # ---- the fit shifts -------------------------------------------------
    for fj in args.fits or []:
        if not os.path.exists(fj):
            continue
        with open(fj) as fh:
            res = json.load(fh)
        keys = [k for k in res if "m_Z" in res[k]]
        fig, axs = plt.subplots(1, 2, figsize=(13.0, 0.42 * len(keys) + 2.6),
                                sharey=True)
        y = np.arange(len(keys))[::-1]
        for a, p in zip(axs, ("m_Z", "Gamma_Z")):
            v = np.array([res[k][p][0] for k in keys])
            e = np.array([res[k][p][1] for k in keys])
            col = ["tab:red" if "population" in k else
                   ("tab:blue" if "per class" in k else "black") for k in keys]
            a.errorbar(v, y, xerr=e, fmt="o", ms=5, lw=0, elinewidth=1.2,
                       ecolor="grey")
            a.scatter(v, y, c=col, s=26, zorder=3)
            a.axvline(0.0, color="grey", lw=0.8)
            a.set_xlabel((r"$\Delta m_Z$" if p == "m_Z" else
                          r"$\Delta \Gamma_Z$") + "  [MeV]")
        axs[0].set_yticks(y)
        axs[0].set_yticklabels(keys, fontsize=9)
        axs[0].set_title(os.path.basename(fj), fontsize=12, loc="left")
        pubhtml.savefig(fig, os.path.join(
            out, "16_fitshifts_" + os.path.basename(fj).replace(".json", "")
            + ".pdf"))
        plt.close(fig)

    pubhtml.ensure_index(out)
    print(f"[figs] -> {out}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classes", help="the k_pred class edges")
    c.add_argument("--gen", required=True)
    c.add_argument("--res", required=True)
    c.add_argument("--res-mode", default="shape", choices=("shape", "gauss"))
    c.add_argument("--nclass", type=int, default=5)
    c.add_argument("--kin", default="post", choices=("post", "pre"))
    c.add_argument("--smear", action="store_true",
                   help="assign the class on the SMEARED post-FSR pT, with "
                        "`fit_gen.fiducial`'s own draw")
    c.add_argument("--smear-seed", type=int, default=None)
    c.add_argument("--pt-cuts", nargs="*", type=float, default=[25.0, 10.0])
    c.add_argument("--eta-cut", type=float, default=ETA_CUT)
    c.add_argument("--wclip", type=float, default=100.0)
    c.add_argument("-o", "--output", required=True)
    c.set_defaults(func=_classes_cmd)

    k = sub.add_parser("check", help="the reco MC: k_pred against k")
    k.add_argument("--aux", required=True)
    k.add_argument("--pairs", required=True)
    k.add_argument("--res", required=True)
    k.add_argument("--res-mode", default="shape", choices=("shape", "gauss"))
    k.add_argument("--pt-cuts", nargs="*", type=float, default=[25.0, 10.0])
    k.add_argument("--eta-cut", type=float, default=ETA_CUT)
    k.add_argument("--nq", type=int, default=5)
    k.add_argument("--nsub", type=int, default=5)
    k.add_argument("--nclass", nargs="*", type=int, default=[3, 5, 10, 20])
    k.add_argument("--nmin", type=int, default=50)
    k.add_argument("--wclip", type=float, default=100.0)
    k.add_argument("-o", "--output", default=None)
    k.set_defaults(func=_check_cmd)

    b = sub.add_parser("build", help="the per-class tables and kernels")
    b.add_argument("--gen", required=True)
    b.add_argument("--classes", required=True)
    b.add_argument("--run", default="data/photos/gen_mcMix.npz")
    b.add_argument("--tag", default="kcl5")
    b.add_argument("--pt-ref", type=float, default=10.0)
    b.add_argument("--b-min", type=float, default=-0.3)
    b.add_argument("--b-max", type=float, default=4.0)
    b.add_argument("--cut-sets", nargs="*", default=["25,25", "25,10"])
    b.add_argument("--ngroups", nargs="*", type=int, default=[3, 5, 10],
                   help="class counts to build, as contiguous groupings of the "
                        "fine partition in --classes")
    b.add_argument("--configs", nargs="*", default=["mc", "data"])
    b.add_argument("--nproc", type=int, default=10)
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=_build_cmd)

    q = sub.add_parser("buildpr", help="per-class kernels with the class in "
                                       "the pass region (the correct one)")
    q.add_argument("--classes", required=True)
    q.add_argument("--htable", default="data/ht_ref10_1.0gev.npz")
    q.add_argument("--h4", default="data/h4_ref10_1.0gev.npz")
    q.add_argument("--run", default="data/photos/gen_mcMix.npz")
    q.add_argument("--tag", default="kpr")
    q.add_argument("--cut-sets", nargs="*", default=["25,25", "25,10"])
    q.add_argument("--ngroups", nargs="*", type=int, default=[3, 5, 10])
    q.add_argument("--configs", nargs="*", default=["mc", "data"])
    q.add_argument("--nknot", type=int, default=300)
    q.add_argument("--nproc", type=int, default=12)
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=_buildpr_cmd)

    cc = sub.add_parser("ccheck", help="G of pass AND class against a count")
    cc.add_argument("--gen", required=True)
    cc.add_argument("--htable", required=True)
    cc.add_argument("--h4", required=True)
    cc.add_argument("--classes", required=True)
    cc.add_argument("--ngroup", type=int, default=5)
    cc.add_argument("--pt-cuts", nargs="*", type=float, default=[25.0, 10.0])
    cc.add_argument("--band", nargs=2, type=float, default=(90.0, 92.0))
    cc.add_argument("--nknot", type=int, default=90)
    cc.add_argument("--u", nargs="*", type=float, default=None)
    cc.set_defaults(func=_ccheck_cmd)

    p = sub.add_parser("figs", help="the figures")
    p.add_argument("--check", default=None, help="a `check` npz")
    p.add_argument("--classes", default=None)
    p.add_argument("--tag", default="kpr")
    p.add_argument("--control", default="kcl",
                   help="the class-restricted-table build, for the control "
                        "figure that shows it is not a partition")
    p.add_argument("--cuts", default="2510")
    p.add_argument("--ngroup", type=int, default=5)
    p.add_argument("--fits", nargs="*", default=None,
                   help="`fit_gen.py fit` output jsons")
    p.add_argument("--outpath", default=None)
    p.set_defaults(func=_figs_cmd)

    args = ap.parse_args()
    if args.cmd in ("build", "buildpr"):
        args.cut_sets = [tuple(float(x) for x in s.split(","))
                         for s in args.cut_sets]
    if args.cmd == "ccheck":
        u = args.u or [0, 0, 0.01, 0.01, 0.05, 0.0, 0.2, 0.2, 0.5, 0.2]
        args.u = list(zip(u[::2], u[1::2]))
    args.func(args)


if __name__ == "__main__":
    main()
