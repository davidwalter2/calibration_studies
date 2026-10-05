"""The exact hit likelihood with ionisation tails: a Rao-Blackwellised
(conditional-Kalman) recursion over the kick-induced shift of the filter.

THE MODEL.  Per leg, in the frozen linear model (deviations from the exported
reference), with the hit states s_i on the hit surfaces i = 0..N-1,

    s_0 = T0 x_ref + n_0,     s_i = Phi_i s_{i-1} + n_i,
    n_i = w_i + u_i eta_i,    w_i ~ N(0, Q^G_i),   eta_i ~ g_i,
    y_i = H_i s_i + e_i,      e_i ~ N(0, V_i)      (measured directions only)

`T0 = -F[prop rows of hit 0, 0:10]` and `Phi_i = -F[prop rows of hit i,
state of hit i-1]` are read off the exported design (its propagation rows are
`+I` on the hit's own state), `H_i` is the measurement row rotated onto the
eigenvectors of its weight with the unmeasured direction of a strip dropped
(exactly as `marginal.ReducedModel` does, so the rows and the data `y = -c0`
are the reduced model's), `u_i` is the dominant eigenvector of `fm_QI`,
`Q^G_i = Q_i - lambda_i u_i u_i^T` the Gaussian remainder of the exported block
covariance (`marginal.BlockCGF.Cres`), and `g_i` the block's Urban law
(`marginal.UrbanCGF`: the step records' exponent, Kokoulin off) -- the same
law `marginal.sample_urban_leg` draws. The two legs are independent given
`x_ref`; the gun records carry no beam-line or pointing rows.

THE RECURSION.  Every non-Gaussian input is a 1-D kick along a known
direction and everything else is linear-Gaussian, so the Kalman covariances
`P`, the innovation covariances `S_i`, the gains `K_i` and `A_i = I - K_i H_i`
do not depend on the kicks: only the filtered mean moves, linearly.  With the
filter run once at `eta = 0` (innovations `nubar_i`, linear in `x_ref`), the
kick-induced shift of the filtered mean obeys

    D_{i|i-1} = Phi_i D_{i-1} + u_i eta_i,    D_i = A_i D_{i|i-1},

and the leg likelihood is exactly

    L(x_ref) = E_eta[ prod_i N(nubar_i(x_ref) - H_i D_{i|i-1}; 0, S_i) ].

See the README section "The exact marginal: the conditional-Kalman
recursion" for the numerical construction and its gates.
"""
from __future__ import annotations

import numpy as np

from .marginal import MarginalLikelihood

LOG2PI = float(np.log(2.0 * np.pi))


# ---- the per-leg state-space model ------------------------------------------
class LegSSM:
    """One leg of a frozen model as a linear state-space model with 1-D kicks.

    Attributes (lists over the leg's hits in propagation order):
        Phi[i]   (5, 5)   transport hit i-1 -> hit i (Phi[0] unused)
        T0       (5, 10)  the predicted state at hit 0 per unit x_ref
        H[i]     (m_i, 5) measured directions of hit i (m_i = 0 for a hitless one)
        V[i]     (m_i,)   their variances
        rows[i]  (m_i,)   their positions in the reduced model's row vector
        QG[i]    (5, 5)   the Gaussian remainder of the block covariance
        u[i]     (5,)     the kick direction (dominant fm_QI eigenvector)
        lam[i]           the fit's own (truncated) variance along u
        blk[i]           index of the hit's block in `MarginalLikelihood.blocks`
        D[i]     (5, 5)   the whitening of hit i's state (`_precondition`): all of
                          the above are in the whitened states D_i s_i
    """

    def __init__(self, leg, Phi, T0, H, V, rows, QG, u, lam, blk, cgfs):
        self.leg = leg
        self.Phi, self.T0, self.H, self.V, self.rows = Phi, T0, H, V, rows
        self.QG, self.u, self.lam, self.blk, self.cgf = QG, u, lam, blk, cgfs
        self.n = len(Phi)

    @property
    def nmeas(self):
        return int(sum(len(r) for r in self.rows))


def build_legs(ml):
    """The two `LegSSM` of a `MarginalLikelihood` (its reduced model, its block
    CGFs: the same `u`, `lambda`, `C_res` and Urban laws the saddle-point
    marginal and the toy sampler use)."""
    fm = ml.fm
    F = fm.F.tocsr()
    nb = {0: 5, 1: 2, 2: 3, 3: 1}
    # the reduced model's row bookkeeping, rebuilt with the same rule
    meas_rows, off = {}, 0
    for ib, t in enumerate(fm.rowtype):
        if t != 1:
            continue
        w, W = np.linalg.eigh(fm.Vinv_blocks[ib])
        keep = w > 1e-12 * max(w.max(), 0.0)
        Wk = W[:, keep]
        r0 = fm.row0[ib]
        meas_rows[(int(fm.rowtrk[ib]), int(fm.rowhit[ib]))] = (
            Wk, 1.0 / w[keep], off + np.arange(int(keep.sum())), r0)
        off += int(keep.sum())
    if off != ml.red.nrows:
        raise ValueError("measurement-row bookkeeping does not match the reduced model")
    blocks_of = {}
    for i, ib in enumerate(ml.red.hits):
        blocks_of[(int(fm.rowtrk[ib]), int(fm.rowhit[ib]))] = (i, ib)
    legs = []
    for leg in (0, 1):
        nh = int(fm.nhits[leg])
        c0 = int(fm.trkstate[leg])
        Phi, H, V, rows, QG, u, lam, blk, cgfs = [], [], [], [], [], [], [], [], []
        T0 = None
        for ih in range(nh):
            i, ib = blocks_of[(leg, ih)]
            pr = fm.row0[ib] + np.arange(5)
            cols = c0 + 5 * ih + np.arange(5)
            Fp = F[pr].toarray()
            if np.max(np.abs(Fp[:, cols] - np.eye(5))) > 1e-12:
                raise ValueError("propagation row is not +I on its own hit state")
            if ih == 0:
                T0 = -Fp[:, :10]
                Phi.append(np.eye(5))
            else:
                Phi.append(-Fp[:, cols - 5])
            key = (leg, ih)
            if key in meas_rows:
                Wk, vv, rr, r0 = meas_rows[key]
                Fm = F[r0 + np.arange(2)].toarray()
                if np.max(np.abs(np.delete(Fm, cols, axis=1))) != 0.0:
                    raise ValueError("a measurement row touches another state")
                H.append(Wk.T @ Fm[:, cols])
                V.append(vv)
                rows.append(rr)
            else:
                H.append(np.zeros((0, 5)))
                V.append(np.zeros(0))
                rows.append(np.zeros(0, int))
            b = ml.blocks[i]
            QG.append(np.array(b.Cres, float))
            u.append(np.array(b.u, float))
            lam.append(float(b.lam))
            blk.append(i)
            cgfs.append(b.urban)
        legs.append(_precondition(LegSSM(leg, Phi, T0, H, V, rows, QG, u, lam, blk, cgfs),
                                  ml.red.nrows))
    return legs


def _precondition(leg, nrows, rfloor=1e-12):
    """Re-express the leg's states in coordinates whitened by the information
    the whole leg has on them (s_i -> D_i s_i, D_i = Lambda_i^1/2): the
    information the hits outside of i have on s_i (the backward filter's,
    which carries the local precision of the positions) plus the leg's final
    information at hit 0 carried outwards along the transports (which carries
    the scale of the directions the outer hits barely measure, the momentum
    first).  The likelihood is invariant (the transition densities and the
    volume element carry the same Jacobian); what changes is the
    conditioning.  In the raw units the information after the first few hits
    spans ~12 orders of magnitude and the weights the backward filter forms,
    `W = Om' - Om' Om^+ Om'`, lose all their digits in the weak directions
    (O(1) negative eigenvalues, data terms tilting the measure by e^500
    across a lattice); whitened, a direction is weak only by the fraction of
    its final information it has seen, and `backward_filter`'s pseudo-inverse
    cuts at a fraction of it."""
    n = leg.n
    Ob = [None] * n
    Om = np.zeros((5, 5))
    for i in range(n - 1, -1, -1):
        if len(leg.H[i]):
            Om = Om + (leg.H[i].T / leg.V[i][None, :]) @ leg.H[i]
        Ob[i] = 0.5 * (Om + Om.T)
        M = np.eye(5) + Om @ leg.QG[i]
        Om = np.linalg.solve(M, Om)
        Om = 0.5 * (Om + Om.T)
        if i > 0:
            Om = leg.Phi[i].T @ Om @ leg.Phi[i]
    D, Di = [], []
    Lp = Om                                       # the final information, at hit 0
    for i in range(n):
        if i > 0:
            Pinv = np.linalg.inv(leg.Phi[i])
            Lp = Pinv.T @ Lp @ Pinv
            Lp = 0.5 * (Lp + Lp.T)
        w, V = np.linalg.eigh(Ob[i] + Lp)
        w = np.maximum(w, rfloor * float(w.max()))
        D.append((V * np.sqrt(w)) @ V.T)
        Di.append((V / np.sqrt(w)) @ V.T)
    leg.T0 = D[0] @ leg.T0
    leg.Phi = [np.eye(5)] + [D[i] @ leg.Phi[i] @ Di[i - 1] for i in range(1, n)]
    leg.H = [h @ Di[i] for i, h in enumerate(leg.H)]
    leg.QG = [D[i] @ q @ D[i] for i, q in enumerate(leg.QG)]
    leg.u = [D[i] @ u for i, u in enumerate(leg.u)]
    leg.D = D
    return leg


# ---- the Gaussian filter -----------------------------------------------------
def kalman(leg, y, x, extra_var=None):
    """The Kalman filter of one leg at `eta = 0`.

    `y` is the reduced model's data vector (`-c0`), `x` the 10 reference
    parameters.  `extra_var` (per hit) adds `v u u^T` to the process noise --
    the Gaussian limit of the kicks.  Returns a dict with, per hit, the
    innovation `nu`, its covariance `S`, the gain `K`, `A = I - K H`, the
    predicted and filtered covariances, and the Gaussian log-likelihood."""
    n = leg.n
    m = leg.T0 @ x
    P = leg.QG[0].copy()
    if extra_var is not None:
        P = P + extra_var[0] * np.outer(leg.u[0], leg.u[0])
    out = dict(nu=[], S=[], K=[], A=[], Ppred=[], Pfilt=[], mpred=[], mfilt=[])
    ll = 0.0
    for i in range(n):
        if i > 0:
            m = leg.Phi[i] @ m
            P = leg.Phi[i] @ P @ leg.Phi[i].T + leg.QG[i]
            if extra_var is not None:
                P = P + extra_var[i] * np.outer(leg.u[i], leg.u[i])
        out["mpred"].append(m.copy())
        out["Ppred"].append(P.copy())
        Hi = leg.H[i]
        if len(Hi):
            S = Hi @ P @ Hi.T + np.diag(leg.V[i])
            S = 0.5 * (S + S.T)
            PHt = P @ Hi.T
            K = np.linalg.solve(S, PHt.T).T
            nu = y[leg.rows[i]] - Hi @ m
            A = np.eye(5) - K @ Hi
            m = m + K @ nu
            # Joseph form: P stays symmetric positive semi-definite
            P = A @ P @ A.T + K @ np.diag(leg.V[i]) @ K.T
            sgn, ld = np.linalg.slogdet(S)
            ll += -0.5 * (float(nu @ np.linalg.solve(S, nu)) + ld + len(nu) * LOG2PI)
        else:
            S, K, nu, A = np.zeros((0, 0)), np.zeros((5, 0)), np.zeros(0), np.eye(5)
        out["nu"].append(nu)
        out["S"].append(S)
        out["K"].append(K)
        out["A"].append(A)
        out["mfilt"].append(m.copy())
        out["Pfilt"].append(P.copy())
    out["ll"] = ll
    return out


# ---- the backward information filter -----------------------------------------
def _gauss_conv(Om, th, c, Q):
    """exp(-1/2 s^T Om s + th^T s + c) convolved with N(0, Q): information form,
    valid for a rank-deficient Om."""
    M = np.eye(len(Om)) + Om @ Q
    Mi = np.linalg.inv(M)
    Om2 = Mi @ Om
    th2 = Mi @ th
    sgn, ld = np.linalg.slogdet(M)
    c2 = c - 0.5 * ld + 0.5 * float(th @ (Q @ th2))
    return 0.5 * (Om2 + Om2.T), th2, c2


def backward_filter(leg, y, extra_var=None, rtol=1e-9):
    """The backward information filter of one leg at eta = 0, from the last hit
    to the predicted state at hit 0, and the per-step objects the kick measure
    needs.  Processing order: hit N-1, ..., hit 0, then block 0.

    beta(s) = INT dmu(z) exp(-1/2 (s-z)^T Om (s-z) + (s-z)^T th + c)

    Steps (`ops`, in processing order), each a dict:
      kind 'kick'    block j: z -> z - u_j eta_j      (frame of s_j)
      kind 'map'     z -> Phi_j^-1 z                   (s_j -> s_{j-1})
      kind 'meas'    hit i: weight w(z) = exp(-1/2 z^T W z + a^T z), then
                     z -> R z, R = Om_i^+ Om'          (Om' before the update)
    Returns (ops, (Om_f, th_f, c_f)) with the final Gaussian on the predicted
    state at hit 0, `s = T0 x_ref`.  In the whitened states of `_precondition`
    a direction holding less than `rtol` of its final information is treated
    as unconstrained by the pseudo-inverse."""
    n = leg.n
    Om = np.zeros((5, 5))
    th = np.zeros(5)
    c = 0.0
    ops = []
    for i in range(n - 1, -1, -1):
        # measurement at hit i
        Hi = leg.H[i]
        if len(Hi):
            Vi = leg.V[i]
            HtVi = Hi.T / Vi[None, :]
            Om_n = Om + HtVi @ Hi
            th_n = th + HtVi @ y[leg.rows[i]]
            c_n = c - 0.5 * float(np.sum(y[leg.rows[i]] ** 2 / Vi)) \
                - 0.5 * float(np.sum(np.log(2.0 * np.pi * Vi)))
            Om_n = 0.5 * (Om_n + Om_n.T)
            Op = np.linalg.pinv(Om_n, rcond=rtol, hermitian=True)
            R = Op @ Om
            W = Om - Om @ Op @ Om
            a = Om @ Op @ th_n - th
            ops.append(dict(kind="meas", hit=i, W=0.5 * (W + W.T), a=a, R=R,
                            Om_prev=Om.copy(), Om=Om_n.copy()))
            Om, th, c = Om_n, th_n, c_n
        # the block-i noise: Gaussian remainder, the kick, then the transport
        Qg = leg.QG[i]
        if extra_var is not None:
            Qg = Qg + extra_var[i] * np.outer(leg.u[i], leg.u[i])
        Om, th, c = _gauss_conv(Om, th, c, Qg)
        ops.append(dict(kind="kick", block=i, u=leg.u[i].copy(), Om=Om.copy()))
        if i > 0:
            Pi = leg.Phi[i]
            Pinv = np.linalg.inv(Pi)
            Om = Pi.T @ Om @ Pi
            Om = 0.5 * (Om + Om.T)
            th = Pi.T @ th
            ops.append(dict(kind="map", block=i, M=Pinv))
    return ops, (Om, th, c)


def remaining_information(ops, fin, kick_var=None):
    """Per op t, what everything AFTER op t knows about the shift leaving op t:
    `J[t]` (the inner weights and the final Gaussian, mapped to op t's frame),
    and `Jw[t], bw[t]` (the inner weights alone, with their data-dependent
    linear terms).  Index `t` refers to the state right after op t; `J[-1]`
    is the state before op 0.

    `kick_var(block)` smooths the inner weights by every LATER kick, as a
    Gaussian of that variance along its direction (`np.inf`: whatever a later
    kick can move, the weights beyond it do not constrain), so that the
    importance `_trim` forms does not credit the atoms already on the lattice
    with explaining a deviation a later kick can explain.  `J` is never
    smoothed: it sets the lattice's resolution."""
    nops = len(ops)
    J = [None] * (nops + 1)
    Jw = [None] * (nops + 1)
    bw = [None] * (nops + 1)
    Jc = fin[0].copy()
    Wc = np.zeros((5, 5))
    bc = np.zeros(5)
    J[nops], Jw[nops], bw[nops] = Jc.copy(), Wc.copy(), bc.copy()
    for t in range(nops - 1, -1, -1):
        o = ops[t]
        if o["kind"] == "map":
            Jc = o["M"].T @ Jc @ o["M"]
            Wc = o["M"].T @ Wc @ o["M"]
            bc = o["M"].T @ bc
        elif o["kind"] == "meas":
            Jc = o["R"].T @ Jc @ o["R"] + o["W"]
            Wc = o["R"].T @ Wc @ o["R"] + o["W"]
            bc = o["R"].T @ bc + o["a"]
        elif o["kind"] == "kick" and kick_var is not None:
            v = float(kick_var(o["block"]))
            u = o["u"]
            Wu = Wc @ u
            d = float(u @ Wu) + (1.0 / v if (np.isfinite(v) and v > 0) else 0.0)
            if v > 0 and d > 0:
                Wc = Wc - np.outer(Wu, Wu) / d
                bc = bc - Wu * (float(u @ bc) / d)
        J[t] = 0.5 * (Jc + Jc.T)
        Jw[t] = 0.5 * (Wc + Wc.T)
        bw[t] = bc.copy()
    # entry t holds what is known BEFORE op t; shift by one so that [t] is after op t
    return J[1:] + [J[nops]], Jw[1:] + [Jw[nops]], bw[1:] + [bw[nops]]


# ---- the kick laws -------------------------------------------------------------
def _core_step(a):
    """The core's share of a collision of size |y| = a xc: 1 up to xc, a
    quintic smooth step down to 0 at 2 xc, 0 beyond.  Compact support keeps
    every core cumulant bounded, |kappa_n| <= (2 xc)^(n-2) kappa_2; a step
    with a power-law tail (1/(1 + a^4)) lets the Rutherford tail (~1/y^2)
    into kappa_n for n >= 5 and turns the moment stencil into an
    alternating comb."""
    t = np.clip(np.asarray(a, float) - 1.0, 0.0, 1.0)
    return 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


class KickLaw:
    """The law of one block's kick coordinate eta (the x of `UrbanCGF`, the
    noise is `eta u`): the step records' exponent (`UrbanCGF`, Kokoulin off --
    the law `marginal.sample_urban_leg` draws), or a Gaussian of variance
    `var` (the Gaussian limit).  `_levy_kernel` reads the Urban law's Levy
    measure directly from the step records; `split` < 1 keeps only that
    fraction of a Gaussian law's variance in the analytic kernel (the lattice
    path of gate A1)."""

    def __init__(self, urban=None, var=None, lam=None):
        self.urban = urban
        self.var = None if var is None else float(var)
        # the core scale: the fit's alpha = 0.999-truncated variance along u
        self.lam = float(lam if lam is not None else var)
        self.split = 1.0
        self._q = {}

    @property
    def gaussian(self):
        return self.urban is None

    @property
    def kappa2(self):
        """The law's exact variance (for an Urban law the whole exponent's,
        ~2 orders above the truncated `lam`)."""
        if self.gaussian:
            return self.var
        if "_k2" not in self.__dict__:
            self._k2 = float(self.urban.terms(0.0, 2)[2])
        return self._k2

    def split_at(self, xc, nq=24):
        """(mu, v): the Gaussian the kernel takes when the lattice resolves
        kicks above |x| ~ xc.  `v` is the variance of everything below the
        lattice scale -- the regime-0 steps, the excitations and the knock-ons
        weighted by the compact step `_core_step(|x|/xc)` -- and `mu` minus
        the mean of the knock-ons above it (weighted by 1 - w), so that the
        lattice's residual, phi(t) exp(-i mu t + v t^2/2), is a spike on its
        origin (third-order core structure only) plus a smooth tail.  Any
        (mu, v) is exact; this one keeps the lattice compact."""
        if self.gaussian:
            # an infinite cut is the request to put the whole kick in the kernel
            return 0.0, (self.var if not np.isfinite(xc) else self.split * self.var)
        key = round(float(np.log(xc)), 3)
        cache = self.__dict__.setdefault("_split", {})
        if key in cache:
            return cache[key]
        st = self.urban.steps
        g = self.urban.wstd * st[:, 10] * 1e-3              # x per MeV, signed
        v = 0.0
        m0 = st[:, 0] == 0
        v += float(np.sum(st[m0, 1] * g[m0] ** 2))
        m = ~m0
        gam = st[m, 9]
        gm = g[m]
        for ja, je in ((2, 3), (4, 5)):
            a = st[m, ja]
            e = st[m, je] * gam
            act = (a > 0) & (e > 0)
            v += float(np.sum(a[act] * (gm[act] * e[act]) ** 2))
        reg = st[m, 0]
        a3 = st[m, 6]
        e0 = st[m, 7] * gam
        tmax = st[m, 8] * gam
        act = (a3 > 0) & (tmax > e0) & (e0 > 0)
        ex = (reg == 2) | (reg == 3)
        mean_tail = 0.0
        zg, wg = np.polynomial.legendre.leggauss(nq)
        if act.any():
            sel = act
            lo, hi = e0[sel], tmax[sel]
            gs = gm[sel]
            Tc = xc / np.maximum(np.abs(gs), 1e-300)
            # panels in ln T, with edges at the cut's transition
            edges = [np.log(lo)]
            for f in (0.25, 0.5, 1.0, 2.0, 4.0):
                edges.append(np.clip(np.log(f * Tc), np.log(lo), np.log(hi)))
            edges.append(np.log(hi))
            edges = np.sort(np.stack(edges, axis=1), axis=1)
            npan = edges.shape[1] - 1
            T = np.concatenate([np.exp(edges[:, k:k + 1] + 0.5 * (edges[:, k + 1:k + 2] - edges[:, k:k + 1])
                                       * (zg[None, :] + 1.0)) for k in range(npan)], axis=1)
            jac = np.concatenate([0.5 * (edges[:, k + 1:k + 2] - edges[:, k:k + 1]) * wg[None, :]
                                  for k in range(npan)], axis=1) * T
            exs = ex[sel]
            xi = np.where(exs, a3[sel] * gam[sel], a3[sel] * e0[sel])
            b2 = np.where(exs, st[m, 11][sel], 0.0)
            et = np.where(exs, st[m, 12][sel], 1.0)
            half = np.where(exs, (reg[sel] == 2).astype(float), 0.0)
            lam = (xi[:, None] / T ** 2) * (1.0 - b2[:, None] * T / hi[:, None]
                                            + half[:, None] * T ** 2 / (2 * et[:, None] ** 2))
            x = gs[:, None] * T
            w = _core_step(np.abs(x) / xc)
            v += float(np.sum(jac * lam * w * x * x))
            mean_tail = float(np.sum(jac * lam * (1.0 - w) * x))
        out = (-mean_tail, v)
        cache[key] = out
        return out

    def bounds(self, p):
        """(eta_lo, eta_hi): the kick coordinate's support at tail probability p
        on each side.  The heavy side is the knock-on intensity's count above
        |x| (`rb_census.jump_scale`); the light side is the core, ~Gaussian
        below the mean, bounded by 12 truncated sigmas."""
        if p in self._q:
            return self._q[p]
        if self.gaussian:
            s = np.sqrt(self.var)
            q = (-12.0 * s, 12.0 * s)
        else:
            from .rb_census import jump_scale
            J = max(jump_scale(self.urban, p), 12.0 * np.sqrt(self.lam))
            # the light side is the core's: lighter than a Gaussian of the
            # truncated width, bounded generously at 12 of them
            lo = -12.0 * np.sqrt(self.lam)
            sg = np.sign(self.urban.wstd) or 1.0
            q = (lo, J) if sg > 0 else (-J, -lo)
        self._q[p] = q
        return q

    def tail_prob(self, x):
        """Expected number of knock-ons whose kick exceeds |x| (the heavy side)."""
        if self.gaussian or not np.isfinite(x):
            return 0.0
        st = self.urban.steps
        m = st[:, 0] >= 2
        if not m.any():
            return 0.0
        gam = st[m, 9]
        xi, e0, tmax = st[m, 6] * gam, st[m, 7] * gam, st[m, 8] * gam
        b2, et = st[m, 11], st[m, 12]
        half = (st[m, 0] == 2).astype(float)
        cc = np.abs(self.urban.wstd * st[m, 10] * 1e-3)
        Tc = abs(x) / np.maximum(cc, 1e-300)
        hi = np.maximum(tmax, Tc)
        lo = np.minimum(np.maximum(Tc, e0), hi)
        return float(np.sum(xi * ((1.0 / lo - 1.0 / hi) - (b2 / tmax) * np.log(hi / lo)
                                  + half * (hi - lo) / (2 * et ** 2))))


# ---- the lattice measure ----------------------------------------------------
class LegMixture:
    """The leg likelihood as an explicit Gaussian mixture over the lattice of
    kick-induced shifts zeta = o + E z of the hit-0 state:

        L(v) = exp(logscale) sum_z s_z exp(-1/2 (v - zeta_z)^T Om (v - zeta_z)
                                         + th^T (v - zeta_z) + c)

    `v = T0 x_ref`.  Value, gradient and Hessian are exact for this mixture:
    grad = -Om (v - zbar) + th, hess = -Om + Om E Cov(z) E^T Om under the
    mixture's posterior weights."""

    def __init__(self, o, E, Z, s, logscale, fin):
        self.o, self.E = o, E
        self.Om, self.th, self.c = fin
        self.Z = Z                        # (N, r) lattice coordinates
        self.s = s                        # (N,) signed masses
        self.logscale = float(logscale)
        zeta = o[None, :] + Z @ E.T       # (N, 5)
        self.kappa = (-0.5 * np.einsum("ni,ij,nj->n", zeta, self.Om, zeta)
                      - zeta @ self.th)
        self.OmE = self.Om @ E            # (5, r)
        self.Omo = self.Om @ o
        pos = s > 0
        self.lns = np.where(pos, np.log(np.where(pos, s, 1.0)),
                            np.log(np.where(pos, 1.0, -s)))
        self.sgn = np.where(pos, 1.0, -1.0)

    def logL(self, v, order=2):
        v = np.asarray(v, float)
        w = self.OmE.T @ v
        e = self.kappa + self.lns + self.Z @ w
        M = float(np.max(e))
        p = self.sgn * np.exp(e - M)
        S0 = float(np.sum(p))
        if not S0 > 0:
            return -np.inf, None, None
        base = (-0.5 * float(v @ (self.Om @ v)) + float(self.th @ v) + self.c
                + float(v @ self.Omo) + self.logscale)
        val = base + M + np.log(S0)
        if order == 0:
            return val, None, None
        pi = p / S0
        zbar = pi @ self.Z
        g = -self.Om @ v + self.th + self.Omo + self.OmE @ zbar
        if order < 2:
            return val, g, None
        dz = self.Z - zbar
        C = (dz * pi[:, None]).T @ dz
        H = -self.Om + self.OmE @ C @ self.OmE.T
        return val, g, 0.5 * (H + H.T)


def _grid_axes(lo, shape):
    return [lo[k] + np.arange(shape[k]) for k in range(len(shape))]


def _quad_on_grid(lo, shape, A, b, c0):
    """q(z) = -1/2 z^T A z + b^T z + c0 on the tensor lattice, by broadcasting."""
    r = len(shape)
    ax = _grid_axes(lo, shape)
    q = np.full(shape, float(c0))
    for k in range(r):
        sh = [1] * r
        sh[k] = shape[k]
        zk = ax[k].reshape(sh).astype(float)
        q = q + (b[k] * zk - 0.5 * A[k, k] * zk * zk)
        for l in range(k + 1, r):
            sl = [1] * r
            sl[l] = shape[l]
            zl = ax[l].reshape(sl).astype(float)
            q = q - A[k, l] * zk * zl
    return q


class LatticeConfig:
    """Numerical settings of the recursion (lengths in units of the remaining
    information, i.e. whitened sigma, unless stated).

    h            lattice spacing along an axis when the axis is created (the
                 remaining information only decreases afterwards, so the
                 lattice stays resolved)
    rmax         the largest number of lattice axes; the measure's spread in
                 the directions beyond is collapsed (`dropped` records its rms)
    tol_dim      a direction is kept only if the measure's spread along it
                 exceeds this; a kick whose big-jump reach is below it goes
                 to the analytic kernel whole
    tail_p       the tail probability that sets a kick's big-jump reach
    trim         atoms whose importance |mu(z)| x (remaining weights) is below
                 exp(-trim) of the largest are trimmed after every weight
    lookahead    how the remaining weights of that importance see the later
                 kicks: "kicks" smooths them by each kick's exact variance,
                 "flat" leaves everything a later kick can move unconstrained,
                 "none" ignores the later kicks (see `remaining_information`)
    margin       padding of a lattice box, in nodes
    floor        atoms below floor x the largest |mass| are dropped
    moment_floor the same, for the moments that choose the frame
    mix_floor    the same, for the final mixture
    xcut         the core/tail split of a kick's collisions, in lattice units:
                 collisions below xcut are core (their variance exact in the
                 analytic kernel, their cumulants 3..order on a stencil), above
                 2 xcut tail (the Levy measure on the lattice), a smooth step
                 between (`_core_step`)
    rbox         the region of interest: collisions, and atoms, farther than
                 this from the measure's peak are dropped
    kfloor       a kick kernel is cut where it falls below kfloor x its max
    order        the Lagrange order of every assignment to the lattice (the
                 moments up to it are exact) ...
    order_small  ... on an axis along which the measure sits within
                 `small_extent` nodes of the origin
    moment_match collapse the dropped directions onto their regression on the
                 kept transverse axes, with the residual covariance in the
                 analytic kernel (instead of onto the peak's plane)
    rtol         the backward filter's pseudo-inverse cut: a whitened direction
                 holding less than this fraction of its final information is
                 unconstrained
    trace        print the lattice's shape at every kick
    """

    def __init__(self, h=0.3, rmax=3, tol_dim=1e-3, tail_p=1e-6, trim=25.0, margin=2,
                 lookahead="kicks", floor=1e-11, moment_floor=1e-4, mix_floor=1e-12,
                 xcut=0.5, rbox=20.0, kfloor=1e-14, order=9, order_small=3,
                 small_extent=3.0, moment_match=False, rtol=1e-9, trace=False):
        if lookahead not in ("kicks", "flat", "none"):
            raise ValueError(f"lookahead {lookahead!r}")
        self.h, self.rmax, self.tol_dim, self.tail_p = h, rmax, tol_dim, tail_p
        self.trim, self.margin, self.lookahead = trim, margin, lookahead
        self.floor, self.moment_floor, self.mix_floor = floor, moment_floor, mix_floor
        self.xcut, self.rbox, self.kfloor = xcut, rbox, kfloor
        self.order, self.order_small, self.small_extent = order, order_small, small_extent
        self.moment_match, self.rtol, self.trace = moment_match, rtol, trace


def _whiten(J, rtol=1e-12):
    """(J^1/2, J^-1/2) on the range of the symmetric PSD J."""
    w, V = np.linalg.eigh(0.5 * (J + J.T))
    keep = w > rtol * max(float(w.max()), 1e-300)
    Vr, wr = V[:, keep], w[keep]
    return (Vr * np.sqrt(wr)) @ Vr.T, (Vr / np.sqrt(wr)) @ Vr.T, Vr


def _atoms(vals, lo, floor):
    """Lattice coordinates (N, r) and masses of the atoms above `floor` x max."""
    av = np.abs(vals)
    keep = av > floor * float(np.max(av))
    idx = np.nonzero(keep)
    Z = np.stack([lo[k] + idx[k] for k in range(vals.ndim)], axis=1).astype(float)
    return Z, vals[keep]



def _lagrange_weights(frac, p):
    """1-D Lagrange assignment of a point at offset `frac` in [0, 1) from its
    floor node onto the p+1 nodes -(p//2) .. p - p//2: the weights reproduce
    every moment up to order p exactly.  Returns (nodes, weights (n, p+1))."""
    nodes = np.arange(-(p // 2), p - p // 2 + 1)
    w = np.ones((len(frac), len(nodes)))
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i != j:
                w[:, i] *= (frac - nj) / (ni - nj)
    return nodes, w


def _assign_local(Y, m, p, lo, shape, chunk=4096):
    """Point masses m at lattice coordinates Y (n, r), assigned to the lattice
    (offset lo, shape) by tensor-product Lagrange weights of order p (one
    order per axis when p is a sequence)."""
    r = Y.shape[1]
    p = [int(p)] * r if np.ndim(p) == 0 else [int(x) for x in p]
    out = np.zeros(int(np.prod(shape)))
    strides = np.array([int(np.prod(shape[k + 1:])) for k in range(r)])
    for s in range(0, len(m), chunk):
        y = Y[s:s + chunk]
        mm = m[s:s + chunk]
        fl = np.floor(y).astype(int)
        fr = y - fl
        idx = np.zeros((len(mm), 1), int)
        w = mm[:, None].copy()
        for k in range(r):
            nodes, wk = _lagrange_weights(fr[:, k], p[k])
            ik = (fl[:, k:k + 1] + nodes[None, :] - lo[k]) * strides[k]
            idx = (idx[:, :, None] + ik[:, None, :]).reshape(len(mm), -1)
            w = (w[:, :, None] * wk[:, None, :]).reshape(len(mm), -1)
        out += np.bincount(idx.ravel(), weights=w.ravel(), minlength=len(out))
    return out.reshape(shape)



def _moments_from_cumulants(kap, p):
    """Raw moments m_0..m_p from cumulants kap_0..kap_p (kap_0 ignored), by
    the recursion m_n = sum_{k=1}^{n} C(n-1, k-1) kap_k m_{n-k}."""
    from math import comb
    m = np.zeros(p + 1)
    m[0] = 1.0
    for n in range(1, p + 1):
        m[n] = sum(comb(n - 1, k - 1) * kap[k] * m[n - k] for k in range(1, n + 1))
    return m

def _levy_kernel(law, c1, cfg, nq=12):
    """The kick's law on the lattice along axis 1 (c1 lattice units per unit
    eta), LOCAL and moment-exact: every collision size y = c1 x is assigned to
    the lattice nodes with the order-p Lagrange weights of `_assign_local`
    (the discretised Levy measure reproduces every moment up to p of the
    exact one), and the compound-Poisson exponent sum_j k_j (e^{-i w j} - 1 +
    i w j) is exponentiated by FFT.  The Gaussian parts (regime-0 steps; a
    Gaussian law) stay in the analytic kernel; collisions beyond the region of
    interest (`rbox`) are dropped as jumps -- they cannot bring the measure
    back -- and their mean is kept as a deterministic shift.
    Returns (j0, kernel, mu, v): offsets j0.., the kernel, and the analytic
    kernel's mean and variance (eta units).  Cached per (law, c1)."""
    key = (round(float(np.log(c1)), 6), cfg.order, round(cfg.rbox / cfg.h, 3), cfg.kfloor, cfg.xcut)
    cache = law.__dict__.setdefault("_levy", {})
    if key in cache:
        return cache[key]
    p = cfg.order
    reach = int(np.ceil(cfg.rbox / cfg.h)) + cfg.margin
    nodes0 = np.arange(-(p // 2), p - p // 2 + 1)
    dep_i, dep_v = [], []

    def deposit(y, w):
        """Intensity w at lattice positions y (signed), Lagrange-assigned."""
        if not len(y):
            return
        fl = np.floor(y)
        nodes, L = _lagrange_weights(y - fl, p)
        dep_i.append((fl[:, None] + nodes[None, :]).astype(int).ravel())
        dep_v.append((w[:, None] * L).ravel())

    def cells_quadrature(ylo, yhi, dens, nq_=nq):
        """Gauss-Legendre per lattice cell of [ylo, yhi] for a density in y."""
        edges = np.unique(np.concatenate([[ylo, yhi], np.arange(np.ceil(ylo), np.floor(yhi) + 1)]))
        a_, b_ = edges[:-1], edges[1:]
        y = 0.5 * (a_ + b_)[:, None] + 0.5 * (b_ - a_)[:, None] * zg[None, :]
        wy = 0.5 * (b_ - a_)[:, None] * wq[None, :] * dens(y)
        return y.ravel(), wy.ravel()

    zg, wq = np.polynomial.legendre.leggauss(nq)

    if law.gaussian:
        # a Gaussian law on the lattice: its own Lagrange assignment by
        # Gauss-Hermite quadrature (moments up to p exact); the kernel's share
        mu_k, vk = law.split_at(np.inf if law.split >= 1.0 else 1.0)
        s_lat = np.sqrt(max(law.var - vk, 0.0)) * c1
        if s_lat <= 0:
            out = (0, np.ones(1), mu_k, vk)
            cache[key] = out
            return out
        y, wy = cells_quadrature(-12.0 * s_lat, 12.0 * s_lat,
                                 lambda u: np.exp(-0.5 * (u / s_lat) ** 2) / (np.sqrt(2 * np.pi) * s_lat))
        deposit(y, wy)
        idx, val = np.concatenate(dep_i), np.concatenate(dep_v)
        lo_, hi_ = int(idx.min()), int(idx.max())
        k = np.zeros(hi_ - lo_ + 1)
        np.add.at(k, idx - lo_, val)
        out = (lo_, k / k.sum(), mu_k, vk)
        cache[key] = out
        return out
    st = law.urban.steps
    g = law.urban.wstd * st[:, 10] * 1e-3              # eta per MeV, signed
    xc = cfg.xcut                                      # the split, lattice units
    # the CORE (everything below xc lattice units, the compact step
    # `_core_step` to 2 xc) is summarised by its cumulants: kappa_2 goes to the
    # analytic kernel (exact in every direction of the state), kappa_3..p to a
    # local moment-matched stencil; the TAIL (weight 1 - w) is Levy-discretised
    kc = np.zeros(p + 1)                               # core cumulants, lattice units
    m0 = st[:, 0] == 0
    kc[2] += float(np.sum(st[m0, 1] * (c1 * g[m0]) ** 2))
    m = ~m0
    gam = st[m, 9]
    gm = g[m]
    far_mean = 0.0
    nu_far = 0.0

    def core_cumulants(y, w):
        for n_ in range(2, p + 1):
            kc[n_] += float(np.sum(w * y ** n_))

    for ja, je in ((2, 3), (4, 5)):
        a = st[m, ja]
        e = st[m, je] * gam
        act = (a > 0) & (e > 0)
        y = c1 * gm[act] * e[act]
        wc = _core_step(np.abs(y) / xc)
        core_cumulants(y, a[act] * wc)
        deposit(y, a[act] * (1.0 - wc))
    reg = st[m, 0]
    a3 = st[m, 6]
    e0 = st[m, 7] * gam
    tmax = st[m, 8] * gam
    act = (a3 > 0) & (tmax > e0) & (e0 > 0)
    ex = (reg == 2) | (reg == 3)
    for s in np.flatnonzero(act):
        gs = c1 * gm[s]                                    # lattice units per MeV
        if gs == 0:
            continue
        if ex[s]:
            xi, b2, et, half = a3[s] * gam[s], st[m, 11][s], st[m, 12][s], float(reg[s] == 2)
        else:
            xi, b2, et, half = a3[s] * e0[s], 0.0, 1.0, 0.0

        def lam(T):
            return (xi / T ** 2) * (1.0 - b2 * T / tmax[s] + half * T ** 2 / (2 * et ** 2))
        Tr = reach / abs(gs)                               # the region of interest
        lo_T, hi_T = e0[s], min(tmax[s], Tr)
        if hi_T > lo_T:
            # panels: per decade below one lattice unit, per lattice unit above
            Tu = 1.0 / abs(gs)
            edges = [lo_T]
            t = lo_T
            while t * 10 < min(Tu, hi_T):
                t *= 10
                edges.append(t)
            for f_ in (0.25, 0.5, 0.75):
                if lo_T < f_ * Tu < hi_T:
                    edges.append(f_ * Tu)
            if Tu < hi_T:
                ncell = int(np.ceil((hi_T - max(Tu, lo_T)) / Tu))
                edges.extend(list(np.minimum(max(Tu, lo_T) + Tu * np.arange(ncell + 1), hi_T)))
            else:
                edges.append(hi_T)
            edges = np.unique(np.asarray(edges))
            a_, b_ = np.log(edges[:-1]), np.log(edges[1:])
            T = np.exp(0.5 * (a_[:, None] + b_[:, None]) + 0.5 * (b_ - a_)[:, None] * zg[None, :])
            wT = 0.5 * (b_ - a_)[:, None] * wq[None, :] * T * lam(T)
            y = (gs * T).ravel()
            wT = wT.ravel()
            wc = _core_step(np.abs(y) / xc)
            core_cumulants(y, wT * wc)
            deposit(y, wT * (1.0 - wc))
        if tmax[s] > Tr:
            # beyond the region of interest: no jumps, the mean as a shift
            a_, b_ = np.log(Tr), np.log(tmax[s])
            T = np.exp(0.5 * (a_ + b_) + 0.5 * (b_ - a_) * zg)
            w = 0.5 * (b_ - a_) * wq * T * lam(T)
            far_mean += float(np.sum(w * T)) * gm[s]
            nu_far += float(np.sum(w))
    if dep_i:
        idx, val = np.concatenate(dep_i), np.concatenate(dep_v)
        js, inv = np.unique(idx, return_inverse=True)
        kv_ = np.bincount(inv, weights=val)
        keep = js != 0
        js, kv_ = js[keep], kv_[keep]
    else:
        js, kv_ = np.zeros(0, int), np.zeros(0)
    # the compound Poisson on a periodic grid wide enough for two reaches,
    # compensated by the INTEGER part n0 of the tail's mean only: a
    # fractional shift on the lattice is a band-limited sinc that rings
    # around the no-collision spike; the fractional part is exact in the
    # analytic kernel's mean instead
    ymean = float(kv_ @ js)
    n0 = int(np.round(ymean))
    M = int(2 ** np.ceil(np.log2(4 * (reach + p + 8))))
    w = 2.0 * np.pi * np.fft.fftfreq(M)
    S = (np.exp(-1j * np.outer(w, js)) - 1.0) @ kv_ + 1j * w * n0
    k = np.real(np.fft.ifft(np.exp(S)))
    k = np.fft.fftshift(k)
    jj = np.arange(M) - M // 2
    big = np.abs(k) > cfg.kfloor * float(np.max(np.abs(k)))
    jl, jh = int(jj[np.argmax(big)]), int(jj[len(jj) - 1 - np.argmax(big[::-1])])
    sel = (jj >= jl) & (jj <= jh)
    kt = k[sel] * np.exp(-nu_far)
    # the core's non-Gaussian part: a stencil on nodes -p//2..p-p//2 with the
    # moments of the core deconvolved by its Gaussian (cumulants 0, 0, k3..kp)
    nodes = np.arange(-(p // 2), p - p // 2 + 1)
    mom = _moments_from_cumulants(np.concatenate([[0.0, 0.0, 0.0], kc[3:]]), p)
    Vm = np.vander(nodes.astype(float), p + 1, increasing=True).T
    wst = np.linalg.solve(Vm, mom)
    kern = np.convolve(kt, wst)
    jl2 = jl + int(nodes[0])
    # the analytic kernel: the core's variance, the far collisions' mean and
    # the fractional part of the tail's
    out = (jl2, kern, -far_mean - (ymean - n0) / c1, kc[2] / c1 ** 2)
    cache[key] = out
    return out


def _reframe_kick(vals, lo, E, o, kv, law, J, cfg, dg):
    """One kick's non-Gaussian residual, with the lattice re-framed: axis 1
    along the kick in the remaining-information-whitened frame, the other axes
    the measure's principal directions orthogonal to it (up to `rmax`), the
    origin on the measure's peak (the core spike stays on a lattice point).
    The old atoms are assigned to the new lattice locally, by tensor-product
    Lagrange weights (every moment up to the order exact, nothing band-limited
    to ring); atoms farther than `rbox` from the peak cannot reach the region
    of interest and are dropped; the kick is the exact linear convolution with
    its Levy-discretised kernel along axis 1.
    Returns (vals, lo, E, o, (mu, v, dSig)): the analytic kernel's share of
    the kick (mean and variance in eta) and, with `moment_match`, the dropped
    directions' residual covariance."""
    from scipy import fft as sfft
    r_old = vals.ndim
    Jh, Jih, Vr = _whiten(J)
    yk = Jh @ kv
    ny = float(np.linalg.norm(yk))
    elo, ehi = law.bounds(cfg.tail_p)
    big = max(abs(elo), abs(ehi))
    if ny * big < cfg.tol_dim:
        # the kick cannot move anything that remains to be seen: its whole
        # variance goes to the kernel (a Gaussian it does not need to be)
        dg["skipped"] += 1
        return vals, lo, E, o, law.split_at(np.inf)
    q1 = yk / ny
    if r_old > 0:
        Zm, mm = _atoms(vals, lo, cfg.moment_floor)
        wm = np.abs(mm)
        Ym = (o[None, :] + Zm @ E.T) @ Jh.T
        ybar = (wm @ Ym) / wm.sum()
        dY = Ym - ybar
        Cm = (dY * wm[:, None]).T @ dY / wm.sum()
        ipk = int(np.argmax(np.abs(vals)))
        zpk = np.array(np.unravel_index(ipk, vals.shape)) + lo
        o_new = o + E @ zpk
        Pp = np.eye(5) - np.outer(q1, q1)
        ev, U = np.linalg.eigh(Pp @ Cm @ Pp)
        order = np.argsort(ev)[::-1]
        ev, U = ev[order], U[:, order]
        nperp = int(min(cfg.rmax - 1, np.sum(ev > cfg.tol_dim ** 2)))
        if nperp < 5 and nperp < len(ev) and ev[min(nperp, 4)] > 0:
            dg["dropped"] = max(dg["dropped"], float(np.sqrt(max(np.sum(ev[nperp:]), 0.0))))
        Q = np.column_stack([q1] + [U[:, k] for k in range(nperp)])
    else:
        o_new = o.copy()
        Q = q1[:, None]
    r_new = Q.shape[1]
    h = cfg.h
    E_new = Jih @ Q * h
    P_new = Q.T @ Jh / h
    dSig = None
    if r_old > 0 and cfg.moment_match and r_new < Vr.shape[1]:
        # the directions the lattice does not keep: moment matching of the
        # measure's spread there (the brief's "moment-matched into P").  The
        # dropped whitened coordinates d are regressed on the kept ones a over
        # the atoms (signed masses: exact first and second joint moments),
        # d = b + B a + residual; the linear part rides on the lattice
        # embedding, the residual covariance goes to the analytic kernel.
        Za, ma = _atoms(vals, lo, cfg.floor)
        Ya = (o[None, :] - o_new[None, :] + Za @ E.T) @ Jh.T          # whitened, rel. o_new
        Dd = Vr - Q @ (Q.T @ Vr)                                       # range of J minus kept
        ud, sd_, _ = np.linalg.svd(Dd, full_matrices=False)
        D = ud[:, sd_ > 1e-8]
        if D.shape[1]:
            # regress on the kept axes ORTHOGONAL to the kick only: the kick
            # moves atoms along axis 1, and that axis must carry no shear
            A_ = Ya @ Q[:, 1:]
            Dv = Ya @ D
            w_ = np.abs(ma) / np.abs(ma).sum()
            abar, dbar = w_ @ A_, w_ @ Dv
            Ac, Dc = A_ - abar, Dv - dbar
            Cda = (Dc * w_[:, None]).T @ Ac
            Cdd = (Dc * w_[:, None]).T @ Dc
            if A_.shape[1]:
                Caa = (Ac * w_[:, None]).T @ Ac
                Bm = Cda @ np.linalg.pinv(Caa, rcond=1e-10)
            else:
                Bm = np.zeros((D.shape[1], 0))
            Cres = 0.5 * ((Cdd - Bm @ Cda.T) + (Cdd - Bm @ Cda.T).T)
            wr, Vres = np.linalg.eigh(Cres)
            Cres = (Vres * np.maximum(wr, 0.0)) @ Vres.T
            Bm = np.column_stack([np.zeros(D.shape[1]), Bm])
            abar = np.concatenate([[0.0], abar])
            E_new = Jih @ (Q + D @ Bm) * h
            o_new = o_new + Jih @ D @ (dbar - Bm @ abar)
            dSig = Jih @ D @ Cres @ D.T @ Jih
            dg["matched"] = max(dg.get("matched", 0.0), float(np.sqrt(max(np.trace(Cres), 0.0))))
    c1 = ny / h
    jl, kern, mu_k, vk = _levy_kernel(law, c1, cfg)
    jh = jl + len(kern) - 1
    rb = cfg.rbox / h
    far = None
    # the old atoms in the new frame; those beyond the region of interest go
    if r_old > 0:
        Zb, mb = _atoms(vals, lo, cfg.floor)
        Znew = (o[None, :] - o_new[None, :] + Zb @ E.T) @ P_new.T
        far = np.sqrt(np.sum(Znew ** 2, axis=1)) > rb
        if far.any():
            dg["cut"] = max(dg.get("cut", 0.0), float(np.sum(np.abs(mb[far]))) /
                            float(np.sum(np.abs(mb))))
            vals = vals.copy()
            idx = tuple((Zb[far] - lo).astype(int).T)
            vals[idx] = 0.0
            Znew = Znew[~far]
        zmin, zmax = Znew.min(axis=0), Znew.max(axis=0)
    else:
        zmin = zmax = np.zeros(r_new)
    lo_new = np.zeros(r_new, int)
    shape = []
    for k in range(r_new):
        a = int(np.floor(zmin[k])) - cfg.margin
        b = int(np.ceil(zmax[k])) + cfg.margin
        if k == 0:
            a, b = a + jl, b + jh
        lo_new[k] = a
        shape.append(b - a + 1)
    shape = tuple(shape)
    dg["maxN"] = max(dg["maxN"], int(np.prod(shape)))
    # the old atoms, assigned locally (moments up to cfg.order exact)
    if r_old > 0:
        Zl, ml_ = _atoms(vals, lo, cfg.floor)
        Yl = (o[None, :] - o_new[None, :] + Zl @ E.T) @ P_new.T
        if far is not None:
            keep = np.sqrt(np.sum(Yl ** 2, axis=1)) <= rb
            Yl, ml_ = Yl[keep], ml_[keep]
        # the order per axis: the full order where the measure extends,
        # a short stencil where it sits within a node or two of the origin
        ext = np.max(np.abs(Yl), axis=0)
        orders = [cfg.order if (k == 0 or ext[k] > cfg.small_extent) else cfg.order_small
                  for k in range(r_new)]
        pads = np.array([pk // 2 + 1 for pk in orders])
        lo_new = np.minimum(lo_new, np.floor(Yl.min(axis=0)).astype(int) - pads)
        hi_new = lo_new + np.array(shape) - 1
        hi_new = np.maximum(hi_new, np.ceil(Yl.max(axis=0)).astype(int) + pads)
        shape = tuple(int(x) for x in (hi_new - lo_new + 1))
        base = _assign_local(Yl, ml_, orders, lo_new, shape)
    else:
        base = np.zeros(shape)
        base[tuple(-lo_new)] = float(vals)
    dg["maxN"] = max(dg["maxN"], int(np.prod(shape)))
    # the kick: exact linear convolution with the truncated kernel along
    # axis 0 (the box was extended by the kernel's support)
    n0 = shape[0]
    kfull = np.zeros(n0)
    jj = jl + np.arange(len(kern))
    kfull[jj % n0] += kern
    Vb = sfft.rfft(base, axis=0, workers=1)
    Kf = sfft.rfft(kfull, workers=1)
    Vb = Vb * Kf.reshape((-1,) + (1,) * (r_new - 1))
    vals_new = sfft.irfft(Vb, n=n0, axis=0, workers=1)
    return vals_new, lo_new, E_new, o_new, (mu_k, vk, dSig)


def _kernel_weight(Sig, W, a):
    """A Gaussian weight exp(-1/2 z^T W z + a^T z) acting on atoms that carry
    the kernel N(0, Sig): the atoms' weights become w~(m) = exp(-1/2 m^T Wt m
    + at^T m + ct), the atoms move by m -> T m + t0, the kernel becomes Sig'.
    Exact for a singular Sig."""
    n = len(W)
    M = np.eye(n) + Sig @ W
    T = np.linalg.inv(M)
    Wt = W @ T
    Wt = 0.5 * (Wt + Wt.T)
    at = T.T @ a
    sgn, ld = np.linalg.slogdet(M)
    ct = -0.5 * ld + 0.5 * float(a @ (T @ (Sig @ a)))
    t0 = T @ (Sig @ a)
    Sig2 = T @ Sig
    return Wt, at, ct, T, t0, 0.5 * (Sig2 + Sig2.T)


def leg_measure(leg, y, laws, cfg=None, diag=None):
    """Run the backward recursion of one leg and return its `LegMixture`.

    The measure is a lattice of atoms that all carry a common Gaussian kernel
    N(0, Sig): the kicks' Gaussian cores go into the kernel (exactly), their
    non-Gaussian residuals onto the lattice.  `laws[i]` is the `KickLaw` of
    hit i's block.  `diag` receives: `maxN` the largest lattice, `dropped`
    the largest rms spread collapsed beyond `rmax` axes, `cut` the largest
    share of |mass| beyond `rbox`, `neg` the negative share of the final
    mixture's |mass| (the Lagrange assignments make the lattice measure
    signed; the Gaussians it is integrated against smooth that away),
    `garbage` the largest factor by which a weight promoted an atom over the
    previous peak, `nkick`/`skipped` the kicks on the lattice / whole in the
    kernel, `trimguard` the trims whose importance sat entirely below the
    floor (none expected), `r` and `shape` of the final lattice."""
    cfg = LatticeConfig() if cfg is None else cfg
    ops, fin = backward_filter(leg, y, rtol=cfg.rtol)
    if cfg.lookahead == "kicks":
        kick_var = lambda j: laws[j].kappa2          # noqa: E731
    elif cfg.lookahead == "flat":
        kick_var = lambda j: np.inf                  # noqa: E731
    else:
        kick_var = None
    Jrem, Jw, bw = remaining_information(ops, fin, kick_var)
    E = np.zeros((5, 0))
    o = np.zeros(5)
    lo = np.zeros(0, int)
    vals = np.ones(())
    Sig = np.zeros((5, 5))
    logscale = 0.0
    dg = dict(dropped=0.0, cut=0.0, neg=0.0, maxN=0, nkick=0, skipped=0, garbage=1.0,
              trimguard=0)
    for t, op in enumerate(ops):
        kind = op["kind"]
        r = vals.ndim
        if kind == "map":
            M = op["M"]
            o, E = M @ o, M @ E
            Sig = M @ Sig @ M.T
            continue
        if kind == "meas":
            Wt, at, ct, T, t0, Sig = _kernel_weight(Sig, op["W"], op["a"])
            A = E.T @ Wt @ E
            b = E.T @ (at - Wt @ o)
            c0 = -0.5 * float(o @ (Wt @ o)) + float(at @ o) + ct
            if r == 0:
                logscale += c0
            else:
                lw = _quad_on_grid(lo, vals.shape, A, b, c0)
                pre = np.abs(vals)
                mx = float(np.max(lw))
                vals = vals * np.exp(lw - mx)
                logscale += mx
                m = float(np.max(np.abs(vals)))
                k = int(np.argmax(np.abs(vals)))
                dg["garbage"] = max(dg["garbage"], float(np.max(pre)) / max(float(pre.flat[k]), 1e-300))
                vals = vals / m
                logscale += np.log(m)
            o, E = T @ o + t0, T @ E
            R = op["R"]
            o, E = R @ o, R @ E
            Sig = R @ Sig @ R.T
            if r > 0:
                vals, lo = _trim(vals, lo, E, o, Jw[t], bw[t], cfg, dg)
            continue
        # a kick: the Gaussian core into the kernel, the residual onto the lattice
        law = laws[op["block"]]
        kv = -op["u"]
        vals, lo, E, o, split = _reframe_kick(vals, lo, E, o, kv, law, Jrem[t], cfg, dg)
        mu, vk = split[0], split[1]
        o = o + mu * kv
        Sig = Sig + vk * np.outer(kv, kv)
        if len(split) > 2 and split[2] is not None:
            Sig = Sig + split[2]
        if vals.ndim:
            dg["nkick"] += 1
            m = float(np.max(np.abs(vals)))
            vals = vals / m
            logscale += np.log(m)
            if cfg.trace:
                print(f"  kick {op['block']:2d} -> r {vals.ndim} shape {vals.shape} "
                      f"v/lam {vk / law.lam:.3f} mu/sqrt(lam) {mu / np.sqrt(law.lam):+.3f}")
    r = vals.ndim
    if r == 0:
        Z = np.zeros((1, 0))
        s = np.array([float(vals)])
    else:
        keep = np.abs(vals) > cfg.mix_floor * np.max(np.abs(vals))
        idx = np.nonzero(keep)
        Z = np.stack([lo[k] + idx[k] for k in range(r)], axis=1).astype(float)
        s = vals[keep]
        dg["neg"] = float(np.sum(-s[s < 0]) / np.sum(np.abs(s)))
    if diag is not None:
        diag.update(dg)
        diag["r"] = vals.ndim
        diag["shape"] = tuple(vals.shape)
    finK = _gauss_conv(fin[0], fin[1], fin[2], Sig)
    return LegMixture(o, E, Z, s, logscale, finK)


def _trim(vals, lo, E, o, Jw, bw, cfg, dg):
    """Drop lattice slices whose importance |mu(z)| F_rem(z) is below
    exp(-trim) of the largest, F_rem the remaining weights' Gaussian (the
    final Gaussian, whose centre T0 x is free, does not trim)."""
    r = vals.ndim
    if r == 0:
        return vals, lo
    A = E.T @ Jw @ E
    b = E.T @ (bw - Jw @ o)
    lf = _quad_on_grid(lo, vals.shape, A, b, 0.0)
    av = np.abs(vals)
    with np.errstate(divide="ignore"):
        li = np.where(av > 1e-300, np.log(np.maximum(av, 1e-300)), -np.inf) + lf
    mx = float(np.max(li))
    # below the round-off floor a mass is numerically zero, whatever the
    # remaining weights would make of it
    keep = (li > mx - cfg.trim) & (av > cfg.floor * float(np.max(av)))
    if not keep.any():
        # the importance sits entirely below the floor: the remaining weights
        # want what the lattice cannot resolve; keep the box (counted)
        dg["trimguard"] += 1
        return vals, lo
    sl = []
    for k in range(r):
        other = tuple(x for x in range(r) if x != k)
        anyk = np.any(keep, axis=other) if other else keep
        nz = np.nonzero(anyk)[0]
        a0 = max(int(nz[0]) - cfg.margin, 0)
        a1 = min(int(nz[-1]) + 1 + cfg.margin, vals.shape[k])
        sl.append(slice(a0, a1))
    lo = lo + np.array([s.start for s in sl])
    return vals[tuple(sl)].copy(), lo


# ---- the candidate objective (the interface profile.py and minimise.py use) --
class RBGridLikelihood:
    """-ln p(h | x_ref) of a candidate, exact for the frozen model with Urban
    kicks: the two legs' lattice mixtures in their hit-0 states, `v = T0 x`.

    `set_data(c0)` re-runs the recursion for new data (a toy replica);
    everything else is the objective interface of `marginal.MarginalLikelihood`
    (value, gradient, Hessian on the 9 free reference parameters; the Gaussian
    Fisher matrix as damping metric; the mixed information is the Hessian,
    the model being a location family in rho = F_ref x - c0)."""

    freeidx = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9])

    def __init__(self, cand, gaussian=False, cfg=None, ml=None, run=True):
        self.cand = cand
        self.fm = cand.fm
        self.ml = MarginalLikelihood(cand, gaussian=gaussian) if ml is None else ml
        self.red = self.ml.red
        self.nfree = len(self.freeidx)
        self.is_gaussian = bool(gaussian)
        self.cfg = LatticeConfig() if cfg is None else cfg
        self.legs = build_legs(self.ml)
        self.laws = []
        for leg in self.legs:
            if gaussian:
                self.laws.append([KickLaw(var=lam, lam=lam) for lam in leg.lam])
            else:
                self.laws.append([KickLaw(urban=cg, lam=lam) for cg, lam in zip(leg.cgf, leg.lam)])
        self._A = self.ml._A
        self.diag = [dict(), dict()]
        self.T0 = [leg.T0 for leg in self.legs]
        if run:
            self.set_data(self.red.c0)

    def set_data(self, c0):
        self.red.c0 = np.asarray(c0, float)
        y = -self.red.c0
        self.mix = [leg_measure(leg, y, laws, self.cfg, diag=dg)
                    for leg, laws, dg in zip(self.legs, self.laws, self.diag)]
        self.T0 = [leg.T0 for leg in self.legs]

    def embed(self, x_free):
        x = np.zeros(10)
        x[self.freeidx] = x_free
        return x

    def free_position(self, full_indices):
        pos = np.full(10, -1, dtype=int)
        pos[self.freeidx] = np.arange(self.nfree)
        out = pos[np.asarray(full_indices, dtype=int)]
        if np.any(out < 0):
            raise ValueError("state index not free")
        return out

    def logp(self, x_free, order=2):
        x = self.embed(x_free)
        val, g, H = 0.0, np.zeros(10), np.zeros((10, 10))
        for mx, T0 in zip(self.mix, self.T0):
            v, gv, Hv = mx.logL(T0 @ x, order)
            if not np.isfinite(v):
                return -np.inf, None, None
            val += v
            if order:
                g += T0.T @ gv
                if order >= 2:
                    H += T0.T @ Hv @ T0
        f = self.freeidx
        return val, (g[f] if order else None), (H[np.ix_(f, f)] if order >= 2 else None)

    def value(self, x_free):
        return -self.logp(x_free, 0)[0]

    def value_grad(self, x_free):
        v, g, _ = self.logp(x_free, 1)
        return -v, (-g if g is not None else None)

    def value_grad_hess(self, x_free, order=2):
        v, g, H = self.logp(x_free, order)
        if g is None:
            return np.inf, np.full(self.nfree, np.nan), np.full((self.nfree, self.nfree), np.nan)
        return -v, -g, (-H if H is not None else None)

    def fisher(self):
        return self._A

    def fisher_at(self, delta):
        return self.value_grad_hess(delta)[2]

    def mixed_hessian(self, delta, hess=None):
        return self.value_grad_hess(delta)[2] if hess is None else hess
