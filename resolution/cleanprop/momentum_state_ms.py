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

    python cleanprop/momentum_state_ms.py <pdg> <func> [--mode standard]
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
    ap.add_argument("--mode", choices=("coupled", "standard"), default="coupled")
    ap.add_argument("--arm", default="off", help="sim arm (norad: eBrem off)")
    ap.add_argument("--model-tag", default="_mat", help="_ion with --arm norad")
    a = ap.parse_args()
    rc.MODEL_TAG = a.model_tag
    rc.ARM = a.arm
    if a.func == "lam":
        add_lambda(a.pdg)
    if a.mode == "coupled":
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
        gc.model_phi = coupled_phi
        _variant = rd.variant
        rd.variant = lambda: _variant() + "_mstate"      # own result files
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        rd.main()
    else:
        main()
