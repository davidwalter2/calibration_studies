"""Gate A2 of the conditional-Kalman recursion: an independent Monte Carlo of
the leg likelihood with the Gaussian parts integrated exactly,

    L_leg(x) = E_eta[ N(rho_leg(x) - D eta; Sigma_G) ],

`rho_leg` the leg's rows of the reduced model, `D` the kicks' columns
`G_b u_b`, `Sigma_G = sum_b G_b C_res,b G_b^T + V_hit`, eta_b drawn from the
blocks' Urban compound processes (the toy's own sampler, vectorised here).

Two estimators, both unbiased in L:
  `logL_prior`  eta from the Urban laws (the plain prior estimator);
  `logL_rb`     Rao-Blackwellised over one block: that block's kick is
                integrated exactly through its Gaussian-smoothed density
                (`smoothed_density`, from the law's characteristic function),
                the others drawn -- for the replicas whose weight one large
                kick carries.
"""
from __future__ import annotations

import numpy as np



def sample_urban_many(steps, wstd, rng, n, max_knockons=4_000_000):
    """`n` draws of a block's centred kick coordinate (the compound process of
    `marginal.sample_urban_leg`), vectorised in chunks small enough that a
    chunk holds at most `max_knockons` explicit knock-ons (the spectrum starts
    at e0 ~ 10 eV, so a block has 1e3-1e4 of them per draw)."""
    st = np.asarray(steps, dtype=float)
    m = st[:, 0] != 0
    if m.any():
        gam = st[m, 9]
        e0, tmax = st[m, 7] * gam, st[m, 8] * gam
        xi = np.where((st[m, 0] == 2) | (st[m, 0] == 3), st[m, 6] * gam, st[m, 6] * e0)
        per = float(np.sum(xi * (1.0 / np.maximum(e0, 1e-300) - 1.0 / np.maximum(tmax, 1e-300)))) * 1.2
        chunk = int(max(1, min(n, max_knockons / max(per, 1.0))))
        if chunk < n:
            return np.concatenate([_sample_urban_chunk(st, wstd, rng, min(chunk, n - s))
                                   for s in range(0, n, chunk)])
    return _sample_urban_chunk(st, wstd, rng, n)


def _sample_urban_chunk(st, wstd, rng, n):
    g = wstd * st[:, 10] * 1e-3
    x = np.zeros(n)
    m0 = st[:, 0] == 0
    if m0.any():
        sd = float(np.sqrt(np.sum(st[m0, 1] * g[m0] ** 2)))
        x += sd * rng.normal(size=n)
    m = ~m0
    if not m.any():
        return x
    gam = st[m, 9]
    gm = g[m]
    for ja, je in ((2, 3), (4, 5)):
        a = st[m, ja]
        e = st[m, je] * gam
        act = (a > 0) & (e > 0)
        if act.any():
            cnt = rng.poisson(a[act][None, :], size=(n, int(act.sum())))
            x += (cnt - a[act][None, :]) @ (gm[act] * e[act])
    reg = st[m, 0]
    a3 = st[m, 6]
    e0 = st[m, 7] * gam
    tmax = st[m, 8] * gam
    act = (a3 > 0) & (tmax > e0) & (e0 > 0)
    ex = (reg == 2) | (reg == 3)
    m1 = act & ~ex
    if m1.any():
        nu = a3[m1] * (1.0 - e0[m1] / tmax[m1])
        mean = a3[m1] * e0[m1] * np.log(tmax[m1] / e0[m1])
        ns1 = int(m1.sum())
        cnt = rng.poisson(nu[None, :], size=(n, ns1))
        tot = int(cnt.sum())
        if tot:
            flat = cnt.ravel()
            who = np.repeat(np.repeat(np.arange(n), ns1), flat)
            idx = np.repeat(np.tile(np.arange(ns1), n), flat)
            e0r, tmr = e0[m1][idx], tmax[m1][idx]
            T = e0r / (1.0 - rng.random(tot) * (1.0 - e0r / tmr))
            np.add.at(x, who, gm[m1][idx] * T)
        x -= float(np.sum(gm[m1] * mean))
    m2 = act & ex
    if m2.any():
        # the exact spin-1/2 spectrum by THINNING the Rutherford intensity
        # xi/T^2 (closed-form inverse CDF): accept with probability
        # f(T)/fmax, f(T) = 1 - beta^2 T/Tmax + half T^2/(2E^2) <= fmax
        xi = a3[m2] * gam[m2]
        e0b, tmb = e0[m2], tmax[m2]
        b2, et = st[m, 11][m2], st[m, 12][m2]
        half = (reg[m2] == 2).astype(float)
        fmax = 1.0 + half * tmb ** 2 / (2 * et ** 2)
        nu_dom = xi * (1.0 / e0b - 1.0 / tmb) * fmax
        mean = xi * (np.log(tmb / e0b) - b2 * (tmb - e0b) / tmb
                     + half * (tmb ** 2 - e0b ** 2) / (4 * et ** 2))
        ns2 = int(m2.sum())
        cnt = rng.poisson(np.maximum(nu_dom, 0.0)[None, :], size=(n, ns2))
        tot = int(cnt.sum())
        gb = gm[m2]
        if tot:
            flat = cnt.ravel()
            who = np.repeat(np.repeat(np.arange(n), ns2), flat)
            idx = np.repeat(np.tile(np.arange(ns2), n), flat)
            e0r, tmr = e0b[idx], tmb[idx]
            T = e0r / (1.0 - rng.random(tot) * (1.0 - e0r / tmr))
            f = 1.0 - b2[idx] * T / tmr + half[idx] * T * T / (2 * et[idx] ** 2)
            acc = rng.random(tot) * fmax[idx] < f
            np.add.at(x, who[acc], gb[idx[acc]] * T[acc])
        x -= float(np.sum(gb * mean))
    return x


class LegMC:
    """The leg's Gaussian pieces for the Monte Carlo: rows, D, Sigma_G, and
    the map x_ref -> rho on those rows."""

    def __init__(self, ml, leg):
        red = ml.red
        rows = np.concatenate([r for r in leg.rows if len(r)])
        self.rows = rows
        self.blk = list(leg.blk)
        Sg = red.Vhit[np.ix_(rows, rows)].copy()
        for i in self.blk:
            Gb = red.Gb[i][rows]
            Sg += Gb @ ml.blocks[i].Cres @ Gb.T
        self.Sg = 0.5 * (Sg + Sg.T)
        self.L = np.linalg.cholesky(self.Sg)
        self.ld = 2.0 * float(np.sum(np.log(np.diag(self.L))))
        self.D = np.column_stack([red.Gb[i][rows] @ ml.blocks[i].u for i in self.blk])
        self.Dw = np.linalg.solve(self.L, self.D)            # whitened columns
        self.Fref = red.F_ref[rows]
        self.red = red
        self.laws = [ml.blocks[i].urban for i in self.blk]

    def rho(self, x10):
        return self.Fref @ x10 - self.red.c0[self.rows]

    def logL_prior(self, x10, eta):
        """ln mean_s N(rho - D eta_s; Sigma_G) and its MC standard error (in
        ln units), eta (n, nb) prior draws."""
        rw = np.linalg.solve(self.L, self.rho(x10))
        q = rw[None, :] - eta @ self.Dw.T
        lf = -0.5 * np.sum(q * q, axis=1) - 0.5 * self.ld - 0.5 * len(rw) * np.log(2 * np.pi)
        M = float(np.max(lf))
        w = np.exp(lf - M)
        mw = float(np.mean(w))
        se = float(np.std(w, ddof=1) / np.sqrt(len(w))) / mw
        return M + np.log(mw), se, float(np.sum(w) ** 2 / np.sum(w * w))

    def smoothed_density(self, k, lam, mlo, mhi):
        """h(m) = (g_k * N(0, 1/lam))(m) for block k's kick law g_k on a grid
        covering [mlo, mhi] (the range the data probe), from its characteristic
        function: the exact 1-D integral the Rao-Blackwellised estimator needs,
        INT g(eta) exp(beta eta - lam eta^2/2) = exp(beta^2/2lam) sqrt(2pi/lam)
        h(beta/lam).  The FFT period is twice the span plus a margin, so the
        law's tail beyond it aliases in at the level of its mass out there.
        Returns (m0, dm, h)."""
        law = self.laws[k]
        s = 1.0 / np.sqrt(lam)
        dm = s / 16.0
        span = (mhi - mlo) + 40.0 * s
        n = int(2 ** np.ceil(np.log2(2.0 * span / dm)))
        n = min(n, 2 ** 22)
        t = 2.0 * np.pi * np.fft.rfftfreq(n, dm)
        with np.errstate(all="ignore"):
            phi = np.exp(law._S(t) - 0.5 * t * t / lam)
        h = np.fft.irfft(phi, n=n) / dm                  # on m = dm * i, periodic
        centre = 0.5 * (mlo + mhi)
        i0 = int(np.round((centre - 0.5 * n * dm) / dm))
        h = np.roll(h, -i0)
        m0 = i0 * dm
        return m0, dm, h

    def logL_rb(self, x10s, eta, k, hk, lamk):
        """Rao-Blackwellised over block k: eta[:, k] is ignored and integrated
        exactly.  Returns, per x, (ln L, standard error in ln units, ESS)."""
        out = []
        dk = self.Dw[:, k]
        lam = float(dk @ dk)
        assert abs(lam / lamk - 1.0) < 1e-10
        m0, dm, h = hk
        other = eta.copy()
        other[:, k] = 0.0
        Dq = other @ self.Dw.T
        for x10 in x10s:
            rw = np.linalg.solve(self.L, self.rho(x10))
            q = rw[None, :] - Dq
            beta = q @ dk
            mloc = beta / lam
            pos = (mloc - m0) / dm
            i0 = np.floor(pos).astype(int)
            fr = pos - i0
            ok = (i0 >= 1) & (i0 + 2 < len(h))
            ic = np.clip(i0, 1, len(h) - 3)
            # 4-point cubic (Lagrange) interpolation of the smooth table
            hm1, h0, h1, h2 = h[ic - 1], h[ic], h[ic + 1], h[ic + 2]
            hv = (-fr * (fr - 1) * (fr - 2) / 6 * hm1 + (fr + 1) * (fr - 1) * (fr - 2) / 2 * h0
                  - (fr + 1) * fr * (fr - 2) / 2 * h1 + (fr + 1) * fr * (fr - 1) / 6 * h2)
            hv = np.where(ok, hv, 0.0)
            lf = (-0.5 * np.sum(q * q, axis=1) + 0.5 * beta * beta / lam
                  + 0.5 * np.log(2 * np.pi / lam)
                  - 0.5 * self.ld - 0.5 * len(rw) * np.log(2 * np.pi))
            with np.errstate(divide="ignore"):
                lw = lf + np.log(np.maximum(hv, 1e-300))
            M = float(np.max(lw))
            w = np.exp(lw - M)
            mw = float(np.mean(w))
            se = float(np.std(w, ddof=1) / np.sqrt(len(w))) / mw
            out.append((M + np.log(mw), se, float(np.sum(w) ** 2 / np.sum(w * w)),
                        float(np.mean(~ok))))
        return out
