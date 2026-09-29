"""Scattering at the track's own momentum: the momentum as a state variable of
the CF model, for the directions a radiative loss does not enter linearly.

The model's channels are independent: every scattering row is evaluated at
the REFERENCE momentum.  A track that radiated a fraction v of its energy
scatters afterwards at p(1 - v), i.e. with its angles scaled by 1/(1 - v).
For beta = 1 a scattering row is a function of theta*p only, so a row whose
log-CF at the reference is L_e(t) is L_e(t x_e) on a track with
x_e = p_ref,e / p_e, and

    Phi(t) = E[ exp sum_e L_e(t x_e) ]

over the loss history.  With multiplicative increments independent of the
state (the emission spectrum in v taken at the reference energy),
x_{e+1}/x_e = (1 - m_e)/(1 - v) (m_e the reference's mean radiative
fraction of entry e, v the emitted one), and Phi = Psi_0(t) from the
backward recursion

    Psi_e(t) = exp L_e(t) * E_v[ Psi_{e+1}(t (1 - m_e)/(1 - v)) ],

one function of t per entry -- the scaling makes the state drop out.  The
expectation is first order in the entry's emission count (checked).  All other
channels stay as they are (their log-CFs added unchanged).

    python cleanprop/momentum_state_ms.py <pdg> <func> [--mode coupled|standard|lambda1d]
    python cleanprop/momentum_state_ms.py ditrack <realmat_ditrack args>
                                          (its legclosure, coupled)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import realmat_closure as rc                          # noqa: E402  (sets the env)
import cf_brems_exact as cbe                          # noqa: E402
import cf_knockon                                     # noqa: E402
import cf_propagation_test as cpt                     # noqa: E402
import cf_rows                                        # noqa: E402
import geom_closure as gc                             # noqa: E402
import hbasis as hb                                   # noqa: E402
from toy_loader import plane_frames                   # noqa: E402

STATS = {"nmax": 0.0}


def _interp(psi, tau, t):
    return (np.interp(t, tau, psi.real, right=psi.real[-1])
            + 1j * np.interp(t, tau, psi.imag, right=psi.imag[-1]))


def _expect(psi, tau, v, dNb, dNp, recoil=None):
    """E_v[psi(t (1 - m)/(1 - v)) G(t, v)] over one entry's emissions, first
    order in its emission count; G the recoil's 2D kick CF for a
    bremsstrahlung emission of fraction v (`recoil`, (nt, nv), None = 1)."""
    w = np.zeros(len(v))
    dv = np.diff(v)
    w[:-1] += 0.5 * dv
    w[1:] += 0.5 * dv
    qb, qp = w * dNb, w * dNp
    q = qb + qp
    if not np.any(q > 0):
        return psi
    N, m = q.sum(), (q * v).sum()
    STATS["nmax"] = max(STATS["nmax"], N)
    base = _interp(psi, tau, tau * (1.0 - m))
    out = (1.0 - N) * base
    for i in np.flatnonzero((q > 0) & (v < 1.0)):
        jump = _interp(psi, tau, tau * (1.0 - m) / (1.0 - v[i]))
        g = qp[i] + qb[i] * (1.0 if recoil is None else 1.0 + recoil[:, i])
        out += g * jump
    return out


JOINT_RECOIL = True   # the recoil inside the recursion, with its emission


def coupled_phi(legs, k, avec, sigma, tau):
    """The momentum-state CF.  With JOINT_RECOIL the radiative recoil kick
    rides on the emission that lowers the momentum, inside the recursion; the
    emission's q/p change stays in the independent product (the recursion
    scales the angles with the momentum, which the logarithmic q/p change
    does not), so its joint with the recoil is dropped -- exact for a
    functional without a q/p component."""
    A_ms, A_ioni, A_ms_start, A_ioni_start = cpt.step_transports(legs, k, ioni_start=True)
    S = np.zeros(len(tau), dtype=np.complex128)
    entries = []                                  # (L_e, v, dNb, dNp, recoil) in track order
    chans = ("ioni", "ms") if JOINT_RECOIL else ("ioni", "ms", "rad")
    for j in range(k + 1):
        leg = legs[j]
        S += cpt.leg_exponent(leg, A_ms[j], A_ioni[j], A_ms_start[j],
                              A_ioni_start[j], avec, sigma, tau, channels=chans)
        if not len(leg["ms"]):
            continue
        rid, _, wb, frac = cf_rows.transport_entries(leg, A_ms[j], A_ms_start[j], avec,
                                                     sigma, cpt.MS_NSUB, "length")
        have_rad = leg.get("rad") is not None and len(leg["rad"]) and cpt.RAD_CHANNEL
        if have_rad and JOINT_RECOIL:
            _, wq_r, wb_r, _ = cf_rows.transport_entries(leg, A_ms[j], A_ms_start[j], avec,
                                                         sigma, cpt.MS_NSUB, "vector")
            # the q/p side of every emission stays in the independent product,
            # its recoil moves into the recursion
            S += cf_rows.rad_rows(tau, leg["rad"], leg["radspec"], leg["radvgrid"],
                                  rid, wq_r, np.zeros_like(wb_r), frac)
        for n, (r, w, f) in enumerate(zip(rid, wb, frac)):
            L = cf_rows.ms_rows(tau, leg["ms"], np.array([r]), np.array([w]),
                                np.array([f]), scale=cpt.KMS_SCALE)
            S -= L
            v = dNb = dNp = rec_cf = None
            if have_rad:
                rec = np.array(leg["rad"][r], dtype=np.float64, copy=True)
                rec[cbe.R_STEPCM] *= f
                v, dNb, dNp = cbe.step_spectrum_parts(rec, leg["radspec"][r],
                                                      leg["radvgrid"])
                if JOINT_RECOIL and wb_r[n] > 0.0:
                    E, p = rec[cbe.R_ETOT], rec[cbe.R_P]
                    M = cbe.species_mass(E, p)
                    T = v * E
                    pp = np.sqrt(np.maximum((E - T) ** 2 - M * M, (1e-3 * p) ** 2))
                    rec_cf = cbe.rad_angle_cfm1(np.outer(tau * wb_r[n], T / pp * M / E), M)
            entries.append((L, v, dNb, dNp, rec_cf))
    psi = np.ones(len(tau), dtype=np.complex128)
    for L, v, dNb, dNp, rec_cf in reversed(entries):
        if v is not None:
            psi = _expect(psi, tau, v, dNb, dNp, rec_cf)
        psi = np.exp(L) * psi
    return np.exp(S) * psi


# ---------------------------------------------------------------------------
# the momentum state for any functional: (t, l) recursion
# ---------------------------------------------------------------------------
# A functional with a q/p component does not scale with the momentum: in the
# logarithmic variable an emission moves q/p by |q/p|_ref ln(1/(1-v)) whatever
# the momentum it happens at, while every angle scales as p_ref/p.  So the
# state stays explicit, l = ln(p_ref/p) on LGRID, and the CF argument t stays
# the q/p one:
#
#   Psi_e(t, l) = e^{L_e(t e^l)} E_v[ e^{i t w_q X(v)} G(t e^l, v)
#                                     Psi_{e+1}(t, l + d_e - ln(1 - v)) ],
#   Phi(t) = Psi_0(t, 0),
#
# d_e = ln(1 - m_e) + s_e the drift (the reference's mean radiative fraction
# m_e, the soft emissions' mean log loss s_e).  Emissions below V_SOFT move l
# by < 1e-3 and stay in the independent product (their mean in the drift);
# the hard ones are rebinned to NVBIN log bins per record and applied in
# sub-steps of at most NSUB_MAX expected emissions, so that with the state
# frozen (STATE = False) the recursion is the independent-channel product.
# The MS rows' angle and the recoil scale with e^l; the knock-on, ionisation
# and every other channel stay at the reference (their log-CFs unchanged).
LGRID = np.arange(-1.5, 5.0 + 1e-9, 0.05)
V_SOFT = 1e-3
NVBIN = 40
NSUB_MAX = 0.01
NT_MODEL = None
STATE = True


def _shift(psi, s):
    """psi(t, l + s) on LGRID, linear, clamped at the ends."""
    if not STATE or s == 0.0:
        return psi
    x = (LGRID + s - LGRID[0]) / (LGRID[1] - LGRID[0])
    i0 = np.clip(np.floor(x).astype(int), 0, len(LGRID) - 2)
    f = np.clip(x - i0, 0.0, 1.0)
    return psi[:, i0] * (1.0 - f) + psi[:, i0 + 1] * f


def _rebin(v, qb, qp, nb):
    """Hard emissions (v >= V_SOFT) in nb bins, log-spaced in the jump
    u = -ln(1 - v) (which diverges as v -> 1, where log bins in v would lump
    u in [1.4, inf) together): counts summed, u at the count-weighted mean
    (brems and pair kept apart for the recoil).  Returns v of the bins."""
    # v -> 1 (the grid's endpoint, the primary left at the p' floor) is a
    # hard emission like any other: its jump is capped at the floor
    h = (v >= V_SOFT) & ((qb + qp) > 0)
    if not h.any():
        return None
    u = -np.log1p(-np.minimum(v[h], 1.0 - 1e-3))
    edges = np.geomspace(-np.log1p(-V_SOFT), max(u.max(), 1e-2) * (1 + 1e-9), nb + 1)
    ib = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, nb - 1)
    q = qb[h] + qp[h]
    Q = np.bincount(ib, q, nb)
    B = np.bincount(ib, qb[h], nb)
    P = np.bincount(ib, qp[h], nb)
    Um = np.bincount(ib, q * u, nb)
    ok = Q > 0
    # the fine emissions of every kept bin, for the bins' exact q/p phase sums
    remap = -np.ones(nb, dtype=int)
    remap[ok] = np.arange(ok.sum())
    return -np.expm1(-Um[ok] / Q[ok]), B[ok], P[ok], np.flatnonzero(h), remap[ib]


def coupled_phi2d(legs, k, avec, sigma, tau):
    """The momentum-state CF of the functional `avec` at plane k on `tau`."""
    A_ms, A_ioni, A_ms_start, A_ioni_start = cpt.step_transports(legs, k, ioni_start=True)
    S = np.zeros(len(tau), dtype=np.complex128)
    tmax = float(np.max(tau))
    # the recursion on the caller's own grid (NT_MODEL None, exact) or on a
    # coarser one interpolated back (the emissions' q/p phases rotate the
    # coupled factor fast in t: the coarse grid is a diagnostic only)
    tg = (np.asarray(tau, dtype=float) if NT_MODEL is None else
          np.concatenate([[0.0], np.geomspace(1e-4, tmax, NT_MODEL)]))
    TL = np.concatenate([[0.0], np.geomspace(1e-5, tmax * np.exp(LGRID[-1]) * 1.01, 800)])
    ents = []
    c1 = 0.0          # slope of the radiative centring phase, -i c1 t in S
    for j in range(k + 1):
        leg = legs[j]
        S += cpt.leg_exponent(leg, A_ms[j], A_ioni[j], A_ms_start[j],
                              A_ioni_start[j], avec, sigma, tau)
        if not len(leg["ms"]):
            continue
        rid, _, wb, frac = cf_rows.transport_entries(leg, A_ms[j], A_ms_start[j], avec,
                                                     sigma, cpt.MS_NSUB, "length")
        have_rad = (cpt.RAD_CHANNEL and leg.get("rad") is not None and len(leg["rad"]))
        if have_rad:
            _, wq_r, wb_r, _ = cf_rows.transport_entries(leg, A_ms[j], A_ms_start[j], avec,
                                                         sigma, cpt.MS_NSUB, "vector")
            S -= cf_rows.rad_rows(tau, leg["rad"], leg["radspec"], leg["radvgrid"],
                                  rid, wq_r, wb_r, frac)
        for n, (r, w, f) in enumerate(zip(rid, wb, frac)):
            one = (np.array([r]), np.array([w]), np.array([f]))
            S -= cf_rows.ms_rows(tau, leg["ms"], *one, scale=cpt.KMS_SCALE)
            ent = dict(L=cf_rows.ms_rows(TL, leg["ms"], *one, scale=cpt.KMS_SCALE).real,
                       d=0.0, hard=None)
            if have_rad:
                rec = np.array(leg["rad"][r], dtype=np.float64, copy=True)
                rec[cbe.R_STEPCM] *= f
                v, dNb, dNp = cbe.step_spectrum_parts(rec, leg["radspec"][r],
                                                      leg["radvgrid"])
                wt = np.zeros(len(v))
                dv = np.diff(v)
                wt[:-1] += 0.5 * dv
                wt[1:] += 0.5 * dv
                qb, qp = wt * dNb, wt * dNp
                E, p = rec[cbe.R_ETOT], rec[cbe.R_P]
                M = cbe.species_mass(E, p)
                cs = rec[cbe.R_CS] * wq_r[n]
                q = qb + qp
                m = float((q * v).sum())
                soft = v < V_SOFT
                # the radiative row, soft part: independent (the recoil at the
                # reference; it is ~1e-6 rad there)
                T = v * E
                pp = np.sqrt(np.maximum((E - T) ** 2 - M * M, (1e-3 * p) ** 2))
                X = cf_knockon.qop_map(T, E, p, pp)
                a = tau * cs
                ph = np.exp(1j * np.outer(a, X[soft]))
                g = 1.0 + cbe.rad_angle_cfm1(np.outer(tau * wb_r[n], (T / pp * M / E)[soft]), M)
                S += (ph * g) @ qb[soft] + ph @ qp[soft] - q[soft].sum()
                # the whole row's centring (linear, state-free)
                S -= 1j * a * float((q * T).sum())
                c1 += cs * float((q * T).sum())
                reb = _rebin(v, qb, qp, NVBIN)
                s_soft = float((q[soft] * -np.log1p(-v[soft])).sum())
                ent["d"] = np.log1p(-m) + s_soft
                if reb is not None:
                    vh, bh, ph_, fine, fbin = reb
                    Th = vh * E
                    pph = np.sqrt(np.maximum((E - Th) ** 2 - M * M, (1e-3 * p) ** 2))
                    # the jump and the recoil at the bin's mean; the q/p phase
                    # summed over the bin's fine emissions (it oscillates
                    # within a bin: a X reaches hundreds of radians)
                    # the recoil CF averaged over each bin's fine brems
                    # emissions, tabulated in the scaled argument on TL
                    cf = (T / pp * M / E)[fine]
                    gt = np.zeros((len(vh), len(TL)))
                    wbin = np.bincount(fbin, qb[fine], len(vh))
                    for ib in range(len(vh)):
                        sel = fbin == ib
                        if wbin[ib] > 0:
                            gt[ib] = (cbe.rad_angle_cfm1(np.outer(TL * wb_r[n], cf[sel]), M)
                                      @ qb[fine][sel]) / wbin[ib]
                    ent["hard"] = dict(v=vh, qb=bh, qp=ph_, Xf=X[fine], qbf=qb[fine],
                                       qpf=qp[fine], fbin=fbin, gt=gt, cs=cs)
            ents.append(ent)
    el = np.exp(LGRID) if STATE else np.ones_like(LGRID)
    tau2 = np.outer(tg, el)                                  # the angles' argument
    lt2 = np.log(np.maximum(tau2, 1e-300)).ravel()
    lTL = np.log(np.maximum(TL, 1e-300))
    psi = np.ones((len(tg), len(LGRID)), dtype=np.complex128)
    for e in reversed(ents):
        h = e["hard"]
        if h is not None:
            N = float((h["qb"] + h["qp"]).sum())
            K = max(1, int(np.ceil(N / NSUB_MAX)))
            phf = np.exp(1j * np.outer(tg * h["cs"], h["Xf"]))     # (nt, nfine)
            nb = len(h["v"])
            PB = np.zeros((len(tg), nb), dtype=np.complex128)
            PP = np.zeros((len(tg), nb), dtype=np.complex128)
            np.add.at(PB.T, h["fbin"], (phf * h["qbf"]).T)
            np.add.at(PP.T, h["fbin"], (phf * h["qpf"]).T)
            gm = [PP[:, i:i + 1] + PB[:, i:i + 1]
                  * (1.0 + np.interp(lt2, lTL, h["gt"][i]).reshape(tau2.shape))
                  for i in range(nb)]
            jumps = -np.log1p(-h["v"])
            for _ in range(K):
                base = _shift(psi, e["d"] / K)
                out = (1.0 - N / K) * base
                for i in range(nb):
                    out += (gm[i] / K) * _shift(psi, e["d"] / K + jumps[i])
                psi = out
        elif e["d"]:
            psi = _shift(psi, e["d"])
        Lv = np.interp(lt2, lTL, e["L"]).reshape(tau2.shape)
        psi = np.exp(Lv) * psi
    i0 = int(np.argmin(np.abs(LGRID)))
    # the emissions' q/p phases rotate the coupled factor fast in t; with the
    # (exact, linear) centring taken out it is smooth enough for the grid
    if NT_MODEL is None:
        return np.exp(S) * psi[:, i0]
    cen = _interp_cf(psi[:, i0] * np.exp(-1j * c1 * tg), tg, tau)
    return np.exp(S + 1j * c1 * tau) * cen


def _interp_cf(ph, tg, tau):
    """A CF on `tg` onto `tau`: ln|phi| and the unwrapped phase, linear in
    ln t (both smooth there; linear in phi itself is not converged at the
    closure's 1e-4 on the model grid).  Where |phi| is below 1e-12 linear."""
    lt, lq = np.log(tg[1:]), np.log(np.maximum(tau, tg[1]))
    amp = np.abs(ph[1:])
    good = amp > 1e-12
    la = np.log(np.maximum(amp, 1e-300))
    ang = np.unwrap(np.angle(ph[1:]))
    out = np.exp(np.interp(lq, lt, la) + 1j * np.interp(lq, lt, ang))
    lin = np.interp(tau, tg, ph.real) + 1j * np.interp(tau, tg, ph.imag)
    tiny = np.interp(lq, lt, good.astype(float)) < 1.0
    out = np.where(tiny, lin, out)
    return np.where(tau <= tg[1], ph[0] + (ph[1] - ph[0]) * tau / tg[1], out)


LAM = np.array([0.0, 1.0, 0.0, 0.0, 0.0])   # curvilinear lambda


def add_lambda(pdg):
    """`lam`: the global dip angle, which the field does not turn -- the
    non-bending direction without the local frame's dependence on the
    crossing angle (dy/dz = tan(lambda) sqrt(1 + dx/dz^2) on these planes).
    Sim from the local slopes through the plane frames, reference likewise,
    model functional the curvilinear lambda."""
    sim = rc._sim(pdg)
    legs = cpt.load_model(rc.model_path(pdg))
    ns = {}
    exec(open(rc.species_planes_path(pdg)).read(), ns)
    _, R = plane_frames(ns["origin"], ns["normal"], ns["uaxis"])

    def lam(dxdz, dydz):
        ld = np.stack([dxdz, dydz, np.ones_like(dxdz)], axis=-1)
        d = np.einsum("...pi,pij->...pj", ld, R)
        return np.arcsin(np.clip(d[..., 2] / np.linalg.norm(d, axis=-1), -1, 1))
    sim["lam"] = lam(sim["dxdz"], sim["dydz"])
    reflam = lam(np.array([l["refdxdz"] for l in legs]), np.array([l["refdydz"] for l in legs]))
    for m in (cpt, gc):
        m.SIM_BRANCH["lam"] = "lam"
        m.REF_BRANCH["lam"] = "reflam"
    real_load = cpt.load_model

    def load_with_lam(path):
        out = real_load(path)
        for l, r in zip(out, reflam):
            l["reflam"] = float(r)
        return out
    rc.cpt.load_model = load_with_lam
    real_avecs = hb.avecs
    hb.avecs = lambda legs, func: ([LAM] * len(legs) if func == "lam"
                                   else real_avecs(legs, func))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdg", type=int)
    ap.add_argument("func")
    ap.add_argument("--mode", choices=("coupled", "standard", "lambda1d"), default="coupled",
                    help="coupled: the (t, l) momentum state; lambda1d: the one-variable "
                         "recursion (functionals without q/p only)")
    ap.add_argument("--arm", default="off", help="sim arm (norad: eBrem off)")
    ap.add_argument("--model-tag", default="_mat", help="_ion with --arm norad")
    a = ap.parse_args()
    rc.MODEL_TAG = a.model_tag
    rc.ARM = a.arm
    if a.func == "lam":
        add_lambda(a.pdg)
    if a.mode == "coupled":
        gc.model_phi = coupled_phi2d
    elif a.mode == "lambda1d":
        gc.model_phi = coupled_phi
    c = rc.cell(a.pdg, a.func)
    r, e = np.array(c["rows"]), np.array(c["errs"])
    print(f"{a.pdg} {a.func} {a.mode}  u = {list(gc.UCURVE)}  max entry count "
          f"{STATS['nmax']:.3f}")
    for k in range(len(r)):
        print(f"{c['ks'][k]:2d} " + " ".join(f"{x:+.4f}" for x in r[k])
              + f"   err(u=0.1) {e[k][4]:.4f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ditrack":
        import realmat_ditrack as rd
        # the leg closure (geom_closure) and the vertex-mass product (cpt)
        gc.model_phi = coupled_phi2d
        cpt.model_phi = coupled_phi2d
        _variant = rd.variant
        rd.variant = lambda: _variant() + "_mstate"      # own result files
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        rd.main()
    else:
        main()
