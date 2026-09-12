#!/usr/bin/env python3
"""Shared loader and estimators for the CONVERGENCE variants (base / tight /
damp) of the 20-60 GeV tight-stepper muon gun.

WHAT IS BEING MEASURED. `z = (refParms[0] - genParms[0]) / sqrt(refCov[0])` is
the truth-referenced pull of the signed curvature q/p. Its CHARGE-EVEN mean is
the observable: a charge-even shift of z is a bias of |q/p| that does not flip
with the charge, i.e. a scale bias of the momentum, and it is the -4.9 / -3.6 /
+0.1 e-3 per-eta pattern the CF hit term does not describe.

THE TRUTH-REFERENCED TRANSFORM. `sigma` is the FITTED error, so it fluctuates
with the fit; a pull normalised by its own fluctuating error has a spurious odd
moment. `x = z / (1 - a q z)` with `a = sigma * p * (1 - vgf)` undoes the
leading part of that (`sigma = sigma_bar (1 + a q z)` to first order, from the
1/p^2 dependence of the multiple-scattering term, with `vgf` the HIT share of
the variance which does NOT scale that way). It is the same recipe as
`s1_trim.py` / `s6_mixture.py`, reproduced here verbatim so the convergence
variants are compared on exactly the statistic the baseline was measured with.
"""
import os

import numpy as np

BANDS = ["barrel |eta|<0.9", "middle 0.9-1.6", "endcap 1.6-2.4"]
BANDS_SHORT = ["barrel", "middle", "endcap"]
EDGES = [0.9, 1.6]
HERE = os.path.dirname(os.path.abspath(__file__))


class Var:
    """One variant's per-track arrays plus the derived statistic."""

    def __init__(self, path, name=None):
        self.name = name or os.path.basename(path).replace(".npz", "")
        d = np.load(path)
        self.d = {k: d[k] for k in d.files}
        z = self.z = self.d["z"].astype(np.float64)
        self.sigma = self.d["sigma"].astype(np.float64)
        self.eta = self.d["eta"].astype(np.float64)
        self.aeta = np.abs(self.eta)
        self.q = self.d["q"].astype(np.float64)
        self.vgf = self.d["vgf"].astype(np.float64)
        self.pt = self.d["pt"].astype(np.float64)
        # full momentum from the RECO state, as in s1/s6
        self.p = self.pt * np.cosh(self.eta)
        self.sigrel = self.sigma * self.p
        a = self.sigrel * (1. - self.vgf)
        den = 1. - a * self.q * z
        self.x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
        self.good = np.isfinite(self.x)
        self.band = np.digitize(self.aeta, EDGES)
        # the two step variables of the mixture split
        self.dq_seed = np.abs((self.d["qop_ref"] - self.d["qop_seed"])
                              / np.abs(self.d["qop_seed"]))
        self.dq_it0 = np.abs((self.d["qop_ref"] - self.d["qop_it0"])
                             / np.abs(self.d["qop_ref"]))
        # PAIRING KEY. The gun puts TWO muons of OPPOSITE charge in each
        # event, so (run, lumi, event, charge) is unique -- and unlike the
        # within-event slot it does not shift if one candidate of a pair is
        # dropped by the covariance-consistency cut in one variant and not the
        # other. `slot` is kept as the cross-check.
        self.key = np.stack([self.d["run"], self.d["lumi"], self.d["event"],
                             self.q.astype(np.int64)], axis=1).astype(np.int64)

    def __len__(self):
        return len(self.z)


def even_mean(x, q, mask, rng, nboot=300):
    """Charge-EVEN mean 0.5(<x>_+ + <x>_-) and its bootstrap error."""
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return np.nan, np.nan, len(idx)
    xp = x[idx][q[idx] > 0]
    xm = x[idx][q[idx] < 0]
    if len(xp) < 5 or len(xm) < 5:
        return np.nan, np.nan, len(idx)
    v = 0.5 * (xp.mean() + xm.mean())
    bs = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        qk = q[k]
        bs[i] = 0.5 * (x[k][qk > 0].mean() + x[k][qk < 0].mean())
    return v, bs.std(), len(idx)


def odd_mean(x, q, mask, rng, nboot=300):
    """Charge-ODD mean 0.5(<x>_+ - <x>_-), the control channel."""
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return np.nan, np.nan, len(idx)
    v = 0.5 * (x[idx][q[idx] > 0].mean() - x[idx][q[idx] < 0].mean())
    bs = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        qk = q[k]
        bs[i] = 0.5 * (x[k][qk > 0].mean() - x[k][qk < 0].mean())
    return v, bs.std(), len(idx)


def combine(vs, es):
    """Inverse-variance combination + chi2 of the spread."""
    vs, es = np.asarray(vs, float), np.asarray(es, float)
    m = np.isfinite(vs) & np.isfinite(es) & (es > 0)
    if m.sum() == 0:
        return np.nan, np.nan, np.nan
    w = 1. / es[m] ** 2
    mu = (vs[m] * w).sum() / w.sum()
    return mu, 1. / np.sqrt(w.sum()), float((w * (vs[m] - mu) ** 2).sum())


def pair(a, b):
    """Indices into a and b of the tracks present in BOTH, keyed on
    (run, lumi, event, charge).

    A PAIRED comparison is what makes the sub-1e-3 differences visible: the
    stochastic content of a track (its hits, its scatters) is COMMON to the
    two refits, so it cancels in the difference and only what the estimator
    did differently survives.
    """
    ka = [tuple(r) for r in a.key]
    kb = {tuple(r): i for i, r in enumerate(b.key)}
    ia, ib = [], []
    for i, k in enumerate(ka):
        j = kb.get(k)
        if j is not None:
            ia.append(i)
            ib.append(j)
    return np.asarray(ia, int), np.asarray(ib, int)
