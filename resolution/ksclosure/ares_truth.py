#!/usr/bin/env python3
"""The sigma-artefact slope measured from MC truth.

``sigma_i = sigma_bar_i + a_i (m_i - mu_i)`` DEFINES ``a_i``: it is the
regression of the reported width on the candidate's own mass fluctuation.
Writing ``A = a m / sigma`` (the dimensionless prefactor the closed forms
predict, ``1 + f_hit`` in the J/psi one),

    ln sigma_i  =  ln sigma_bar_i  +  A * (m_i - m_gen,i) / m_gen,i

because ``a (m - mu) / sigma = A (m - mu)/m``.  The regressor is therefore the
RELATIVE MASS RESIDUAL and contains no reported width at all -- there is no
mechanical ``sigma`` on ``sigma`` correlation to worry about.

``ln sigma_bar_i`` is a smooth function of the candidate's TRUE kinematics and
of the detector (hit counts, decay vertex).  It is removed by the
Frisch-Waugh-Lovell construction: inside cells of the true momenta, both
``ln sigma`` and the regressor are residualised against a linear design in the
true kinematics and the detector controls, and the pooled residuals are
regressed on each other.  That is algebraically identical to one joint fit
with a full set of per-cell controls and a single shared slope.

EVERY CONTROL IS TRUTH OR DETECTOR.  Binning or controlling on a reconstructed
kinematic variable would be a cut on the very residual being measured and
would attenuate the slope.

SO IS EVERY CUT, and for a sharper reason.  A cut on the PULL
``|m - m_gen| < k sigma`` uses the candidate's own reported width as the
threshold, and that width is exactly what carries the signal: a candidate whose
mass fluctuated up has a larger ``sigma``, hence a looser threshold, and
survives where its mirror image does not.  The cut therefore MANUFACTURES the
correlation being measured.  Measured on a toy built from this sample
(``--toy``), at the true ``A = 1.05``: no cut ``-0.02 +- 0.04``, a truth-width
cut ``|m - m_gen| < 3 sigma_bar`` ``+0.00 +- 0.06``, an absolute-residual cut
``|m - m_gen| < 60 MeV`` ``+0.03 +- 0.04`` -- and the pull cut
``|m - m_gen| < 3 sigma`` ``+0.10 to +0.25``, i.e. 10-25 % high.  The tail
control must be an ABSOLUTE residual window.
"""
import numpy as np


def cells(x1, x2, nb1, nb2):
    q1 = np.quantile(x1, np.linspace(0, 1, nb1 + 1))
    q2 = np.quantile(x2, np.linspace(0, 1, nb2 + 1))
    q1[0], q1[-1] = -np.inf, np.inf
    q2[0], q2[-1] = -np.inf, np.inf
    i1 = np.clip(np.searchsorted(q1, x1, "right") - 1, 0, nb1 - 1)
    i2 = np.clip(np.searchsorted(q2, x2, "right") - 1, 0, nb2 - 1)
    return i1 * nb2 + i2


def fwl_residuals(y, x, ctrl, cell, minn=60):
    """Residualise ``y`` and ``x`` on ``[1, *ctrl]`` inside every cell."""
    ey = np.full(len(y), np.nan)
    ex = np.full(len(y), np.nan)
    for cid in np.unique(cell):
        sel = np.where(cell == cid)[0]
        if len(sel) < minn:
            continue
        cols = [np.ones(len(sel))] + [cv[sel] - cv[sel].mean() for cv in ctrl]
        X = np.stack(cols, axis=1)
        Q, _ = np.linalg.qr(X)
        for src, dst in ((y, ey), (x, ex)):
            v = src[sel]
            dst[sel] = v - Q @ (Q.T @ v)
    return ey, ex


def measure(y, x, ctrl, cell, pred=None, nboot=200, seed=0, minn=60):
    """Slope of ``y`` on ``x`` after the cell/control residualisation.

    ``pred`` is an optional ``(n,)`` of per-candidate PREDICTED slopes; the same
    projection is applied to ``pred * x`` so that the predicted population
    slope is formed with the SAME weights and the SAME realised fluctuations as
    the measurement, and the two are directly comparable.
    """
    ey, ex = fwl_residuals(y, x, ctrl, cell, minn)
    ok = np.isfinite(ey) & np.isfinite(ex)
    ey, ex = ey[ok], ex[ok]
    out = {"n": int(ok.sum())}
    den = (ex ** 2).sum()
    out["slope"] = float((ey * ex).sum() / den)
    if pred is not None:
        _, epx = fwl_residuals(y, pred * x, ctrl, cell, minn)
        out["pred"] = float((epx[ok] * ex).sum() / den)
    if nboot > 1:
        rng = np.random.default_rng(seed)
        n = len(ey)
        bs = np.empty(nboot)
        for b in range(nboot):
            j = rng.integers(0, n, n)
            bs[b] = (ey[j] * ex[j]).sum() / (ex[j] ** 2).sum()
        out["err"] = float(bs.std(ddof=1))
    else:
        out["err"] = np.nan
    return out
