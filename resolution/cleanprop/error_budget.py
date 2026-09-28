#!/usr/bin/env python3
"""The error budget of the clean-propagation closure: what a closure residual
is worth on the fitted momentum scale of the Z and J/psi mass likelihoods.

THE FIT SIDE (this file, `response`)
------------------------------------
The mass likelihood of candidate i is the density of the observed mass

    f_i(m; alpha) = int dmu K(mu; alpha) r_i(m - mu),     r_i(x) = phi_i(x/sigma_i)/sigma_i

normalised in the fit window W, with the physics kernel K (Z: the generator's
post-FSR dimuon mass within acceptance, i.e. Born x FSR; J/psi: a delta at the
PDG mass convolved with the Photos FSR kernel of the fit) evaluated at the
scaled mass mu (1 + alpha), and r_i the candidate's CF resolution model,
phi_i(t) = exp S_i(t),

    S_i(t) = -vgf_i t^2/2 + Sms_i(t) + Sio_i(t) + Srad_i(t)

exactly the per-candidate blocks of the fit's pairs cache (`cf_inmaker.py`),
standardised to the candidate's own sigma_i.

If the truth differs from the model by a small change of the blocks,
S_i -> S_i + eps g_i(t), the maximum-likelihood alpha moves, to first order in
eps, by the expected score of the unperturbed model under the perturbed truth:

    d alpha = sum_i int_W df_i s_i / F_i  /  sum_i I_i ,
    s_i = d ln f_i / d alpha - d ln F_i / d alpha ,     I_i = int_W f_i s_i^2 / F_i

(F_i the window integral).  This is what an Asimov injection -- generate from
the perturbed model, fit with the unperturbed one -- returns in the limit of
small eps, with no statistical noise; `asimov` verifies it against the direct
maximisation of the expected log-likelihood for two families.  With Gamma_Z floated (`--float-gamma`)
the score is the profiled one, s_alpha - (I_aG / I_GG) s_G.

THE FAMILIES (per channel c in ms, ioni, rad, hit; `families`)
  W_c   width:     every kick of channel c larger by sqrt(1 + eps), the
                   channel's variance x (1 + eps):     g = (t/2) S_c'(t)
  R_c   rate:      the channel's collision rate x (1 + eps):     g = S_c(t)
  T_c   tail:      R_c - W_c, a change of shape at FIXED mean and variance
                   (zero for a Gaussian channel)
  O_c   odd part:  the channel's odd cumulants x (1 + eps):   g = i Im S_c(t)
  shift            every candidate's mass shifted by eps m_ref:
                   g = i t m_ref/sigma_i, d alpha = eps up to the spread of the
                   true mass over the window -- the machinery's unit test
The same families act on the closure frames (`error_budget_frames.py`): an
eps defined on the physics of a channel (its rate, its kick size) is the same
number in every linear functional that channel feeds.

usage:
  python error_budget.py response --channel z    [--n 100000] [--float-gamma]
  python error_budget.py response --channel jpsi [--n 100000]
  python error_budget.py asimov   --channel z|jpsi [--n 3000] [--eps 0.02]
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLUTION = os.path.dirname(HERE)
CALIB = os.path.dirname(RESOLUTION)
RUNS = os.path.join(RESOLUTION, "runs", "error_budget_260928")

MJPSI = 3.0969
JPSI_WINDOW = 0.35
MZ_GEN, GZ_GEN = 91.153509740726733, 2.4932018986110700   # the generator's
Z_WINDOW = (60.0, 120.0)

SAMPLES = {"z": os.path.join(RUNS, "sample_z.npz"),
           "jpsi": os.path.join(RUNS, "sample_jpsi.npz")}
ZGEN = os.path.join(CALIB, "zchannel", "data", "genmerged_full.npz")
JPSI_FSR = os.path.join(CALIB, "zchannel", "data", "jpsi_kern_mc.npz")

CHANNELS = ("ms", "ioni", "rad", "hit")
FAMILIES = ("W", "R", "T", "O")


# =========================================================================
# physics kernels as characteristic functions  phi_K(w) = E[e^{i w (mu - m0)}]
# =========================================================================

def z_kernel(m0, dm, N, acc_pt=5.0, acc_eta=2.4):
    """The Z kernel on the mass grid from the generator record: post-FSR
    dimuon mass within acceptance, histogrammed; also mu K (for the alpha
    score) and d ln BW / d Gamma K (for the Gamma score), all as numpy FFTs of
    the grid so that phi_K = conj(fft(K)) etc."""
    g = np.load(ZGEN)
    acc = ((g["pt1"] > acc_pt) & (g["pt2"] > acc_pt) & (np.abs(g["eta1"]) < acc_eta)
           & (np.abs(g["eta2"]) < acc_eta))
    mpost = g["m_post"][acc]
    mpre = g["m_pre"][acc]
    w = np.sign(g["weight"][acc]).astype(np.float64)
    s = mpre ** 2
    M2 = MZ_GEN ** 2
    den = (s - M2) ** 2 + s ** 2 * GZ_GEN ** 2 / M2
    dlnbw = -2.0 * s ** 2 * GZ_GEN / M2 / den
    edges = m0 + dm * (np.arange(N + 1) - 0.5)
    K, _ = np.histogram(mpost, bins=edges, weights=w)
    KG, _ = np.histogram(mpost, bins=edges, weights=w * dlnbw)
    norm = K.sum()
    return K / norm, KG / norm, int(acc.sum())


# =========================================================================
# the response
# =========================================================================

def build_grid(channel):
    if channel == "z":
        m0, N = 30.0, 1 << 14
        dm = 0.010
        win = Z_WINDOW
    else:
        m0, N = MJPSI - 1.024, 1 << 13
        dm = 0.00025
        win = (MJPSI - JPSI_WINDOW, MJPSI + JPSI_WINDOW)
    m = m0 + dm * np.arange(N)
    w = 2.0 * np.pi * np.fft.fftfreq(N, d=dm)
    return m0, dm, N, m, w, win


def kernel_cf(channel, m0, dm, N, w):
    """phi_K(w), E[mu e^{i w (mu - m0)}] and the Gamma derivative (Z) on the
    FFT frequencies."""
    if channel == "z":
        K, KG, nacc = z_kernel(m0, dm, N)
        m = m0 + dm * np.arange(N)
        # numpy fft: sum_j K_j e^{-i w_k x_j}  = conj(phi_K(w_k))
        phiK = np.conj(np.fft.fft(K))
        muK = np.conj(np.fft.fft(K * m))
        phiG = np.conj(np.fft.fft(KG))
        info = dict(kernel="generator m_post, pT > 5, |eta| < 2.4", n_gen=nacc)
        return phiK, muK, phiG, info
    k = np.load(JPSI_FSR, allow_pickle=True)
    tk = k["phik_t"]
    pk = k["phik_re"] + 1j * k["phik_im"]
    aw = np.abs(w)
    re = np.interp(aw, tk, pk.real, right=0.0)
    im = np.interp(aw, tk, pk.imag, right=0.0) * np.sign(w)
    phik = re + 1j * im
    dpk = np.gradient(pk, tk)
    dre = np.interp(aw, tk, dpk.real, right=0.0) * np.sign(w)
    dim = np.interp(aw, tk, dpk.imag, right=0.0)
    dphik = dre + 1j * dim                  # d phik / d w  (odd real, even imag)
    ph = np.exp(1j * w * (MJPSI - m0))
    phiK = ph * phik
    # E[mu e^{i w(mu - m0)}], mu = M + Delta:  M phi_K - i e^{i w (M - m0)} phik'(w)
    muK = MJPSI * phiK - 1j * ph * dphik
    info = dict(kernel="delta at the PDG mass x Photos FSR (jpsi_kern_mc.npz)")
    return phiK, muK, None, info


def to_density(Phi, dm, N):
    """f_j = (1/(N dm)) sum_k Phi_k e^{-i w_k x_j}  (Phi a CF sampled on the
    FFT frequencies), last axis."""
    return np.real(np.fft.fft(Phi, axis=-1)) / (N * dm)


def response(channel, n=100000, batch=500, float_gamma=False, log=print):
    d = np.load(SAMPLES[channel])
    sel = np.arange(min(n, len(d["z"])))
    sig = d["sigma"][sel].astype(np.float64)
    vgf = d["vgf"][sel].astype(np.float64)
    good = np.isfinite(sig) & (sig > 0) & (sig / d["mtrk"][sel] < 0.10)
    if "chisqval" in d.files:
        good &= d["chisqval"][sel] / np.maximum(d["ndof"][sel], 1) < 3.0
    sel = sel[good]
    sig, vgf = sig[good], vgf[good]
    tg = np.asarray(d["tgrid"], dtype=np.float64)
    tmax = float(tg[-1])
    blocks = {"ms": d["Sms"][sel].astype(np.float64) + 0j,
              "ioni": d["Sio_re"][sel].astype(np.float64)
              + 1j * d["Sio_im"][sel].astype(np.float64),
              "rad": d["Srad_re"][sel].astype(np.float64)
              + 1j * d["Srad_im"][sel].astype(np.float64)}
    m0, dm, N, m, w, win = build_grid(channel)
    mref = 91.1876 if channel == "z" else MJPSI
    inwin = (m >= win[0]) & (m <= win[1])
    phiK, muK, phiG, kinfo = kernel_cf(channel, m0, dm, N, w)
    dphiK_da = 1j * w * muK                         # d/d alpha of phi_K(w(1+alpha))
    fams = [(f, c) for c in CHANNELS for f in FAMILIES
            if not (c == "hit" and f in ("T", "O"))] + [("shift", "all")]
    num = {fc: 0.0 for fc in fams}
    numG = {fc: 0.0 for fc in fams}
    Iaa = IaG = IGG = 0.0
    sig_over_m = []
    # the spline of the stored exponents in t, one coefficient array for all
    # candidates: evaluate S_c(sigma_i w) by the per-candidate t values
    cs = {c: CubicSpline(tg, blocks[c], axis=1) for c in blocks}
    t0 = time.time()
    per = dict(Iaa=np.zeros(len(sel)), num={fc: np.zeros(len(sel)) for fc in fams})
    for b0 in range(0, len(sel), batch):
        sl = slice(b0, min(b0 + batch, len(sel)))
        nb = sl.stop - sl.start
        s = sig[sl]
        tt = np.abs(s[:, None] * w[None, :])            # (nb, N)
        live = tt <= tmax
        sgn = np.sign(w)[None, :]
        S = -0.5 * vgf[sl, None] * tt ** 2 + 0j
        Sc, dSc = {}, {}
        for c, spl in cs.items():
            v = np.zeros((nb, N), dtype=np.complex128)
            dv = np.zeros((nb, N), dtype=np.complex128)
            # evaluate each candidate's own spline row at its own t
            for j in range(nb):
                lj = live[j]
                x = tt[j, lj]
                idx = sl.start + j
                v[j, lj] = _eval_row(spl, idx, x, 0)
                dv[j, lj] = _eval_row(spl, idx, x, 1)
            Sc[c], dSc[c] = v, dv
            S = S + v
        Sc["hit"] = -0.5 * vgf[sl, None] * tt ** 2 + 0j
        dSc["hit"] = -vgf[sl, None] * tt + 0j
        # phi(-t) = conj phi(t): the stored exponents are for t >= 0
        def signed(x):
            return np.where(sgn < 0, np.conj(x), x)
        phir = np.where(live, np.exp(S), 0.0)
        phir = signed(phir)
        Phi = phiK[None, :] * phir
        f = to_density(Phi, dm, N)
        fa = to_density(dphiK_da[None, :] * phir, dm, N)
        fw = f[:, inwin]
        F = fw.sum(axis=1) * dm
        Fa = fa[:, inwin].sum(axis=1) * dm
        # the FFT's round-off floor (~1e-16 of the peak) is not a density:
        # points below 1e-10 of the candidate's peak carry no likelihood
        ok = fw > 1e-10 * fw.max(axis=1, keepdims=True)
        fpos = np.where(ok, fw, 1.0)
        sa = np.where(ok, fa[:, inwin] / fpos - (Fa / F)[:, None], 0.0)
        if float_gamma:
            fg = to_density(phiG[None, :] * phir, dm, N)
            Fg = fg[:, inwin].sum(axis=1) * dm
            sg = np.where(ok, fg[:, inwin] / fpos - (Fg / F)[:, None], 0.0)
        wt = np.where(ok, fw, 0.0) / F[:, None] * dm
        ia = (wt * sa * sa).sum(axis=1)
        Iaa += ia.sum()
        per["Iaa"][sl] = ia
        if float_gamma:
            IaG += (wt * sa * sg).sum()
            IGG += (wt * sg * sg).sum()
        for (fam, c) in fams:
            if fam == "shift":
                # every candidate's mass shifted by eps * m_ref: d alpha = eps
                g = 1j * tt * (mref / s)[:, None]
            elif fam == "W":
                g = 0.5 * tt * dSc[c]
            elif fam == "R":
                g = Sc[c]
            elif fam == "T":
                g = Sc[c] - 0.5 * tt * dSc[c]
            elif fam == "O":
                g = 1j * Sc[c].imag
            dPhi = phiK[None, :] * signed(np.where(live, np.exp(S) * g, 0.0))
            df = to_density(dPhi, dm, N)[:, inwin]
            nn = (df * sa).sum(axis=1) * dm / F
            per["num"][(fam, c)][sl] = nn
            num[(fam, c)] += nn.sum()
            if float_gamma:
                numG[(fam, c)] += ((df * sg).sum(axis=1) * dm / F).sum()
        sig_over_m.append(s / d["mtrk"][sel][sl])
        if b0 == 0:
            log(f"  first batch {time.time() - t0:.1f} s")
    som = np.concatenate(sig_over_m)
    res = {}
    for fc in fams:
        da = num[fc] / Iaa
        rec = dict(dalpha=da)
        if float_gamma:
            det = Iaa * IGG - IaG ** 2
            rec["dalpha_gprof"] = (IGG * num[fc] - IaG * numG[fc]) / det
            rec["dgamma_MeV"] = 1e3 * (Iaa * numG[fc] - IaG * num[fc]) / det
        res[f"{fc[0]}_{fc[1]}"] = rec
    out = dict(channel=channel, n=int(len(sel)), kernel=kinfo, window=list(win),
               mean_sigma_over_m=float(np.mean(som)),
               fisher_weighted_sigma_over_m=float(np.sum(per["Iaa"] * som)
                                                  / np.sum(per["Iaa"])),
               shares={c: float(np.median(-2.0 * blocks[c][:, 1].real / tg[1] ** 2))
                       for c in blocks} | {"hit": float(np.median(vgf))},
               float_gamma=bool(float_gamma), families=res,
               elapsed_s=time.time() - t0)
    return out, per, sel


def _eval_row(spl, i, x, nu):
    """Candidate i's spline (a slice of the batched CubicSpline) at x."""
    c = spl.c[:, :, i]                              # (4, nint)
    br = spl.x
    k = np.clip(np.searchsorted(br, x, side="right") - 1, 0, len(br) - 2)
    dx = x - br[k]
    a3, a2, a1, a0 = c[0, k], c[1, k], c[2, k], c[3, k]
    if nu == 0:
        return ((a3 * dx + a2) * dx + a1) * dx + a0
    return (3.0 * a3 * dx + 2.0 * a2) * dx + a1


def kernel_alpha(channel, alpha, m0, dm, N, w, _cache={}):
    """phi_K of the kernel at the scaled mass mu (1 + alpha), exactly (no
    linearisation): the Asimov check's model."""
    if channel == "z":
        if "K" not in _cache:
            _cache["K"] = z_kernel(m0, dm, N)[0]
        K = _cache["K"]
        m = m0 + dm * np.arange(N)
        Ka = np.interp(m / (1.0 + alpha), m, K, left=0.0, right=0.0)
        return np.conj(np.fft.fft(Ka / Ka.sum()))
    k = np.load(JPSI_FSR, allow_pickle=True)
    tk, pk = k["phik_t"], k["phik_re"] + 1j * k["phik_im"]
    aw = np.abs(w) * (1.0 + alpha)
    phik = (np.interp(aw, tk, pk.real, right=0.0)
            + 1j * np.interp(aw, tk, pk.imag, right=0.0) * np.sign(w))
    return np.exp(1j * w * (MJPSI * (1.0 + alpha) - m0)) * phik


def asimov(channel, n=3000, eps=0.02, fams=(("W", "ms"), ("shift", "all")), log=print):
    """Direct check of the linear response: the expected log-likelihood of the
    unperturbed model under the perturbed truth (exact exponent, S + eps g),
    maximised in alpha on a scan, against `response` on the same candidates."""
    lin, _, sel = response(channel, n=n, float_gamma=False, log=lambda *a: None)
    d = np.load(SAMPLES[channel])
    tg = np.asarray(d["tgrid"], dtype=np.float64)
    tmax = float(tg[-1])
    sig = d["sigma"][sel].astype(np.float64)
    vgf = d["vgf"][sel].astype(np.float64)
    blocks = {"ms": d["Sms"][sel].astype(np.float64) + 0j,
              "ioni": d["Sio_re"][sel].astype(np.float64)
              + 1j * d["Sio_im"][sel].astype(np.float64),
              "rad": d["Srad_re"][sel].astype(np.float64)
              + 1j * d["Srad_im"][sel].astype(np.float64)}
    m0, dm, N, m, w, win = build_grid(channel)
    mref = 91.1876 if channel == "z" else MJPSI
    inwin = (m >= win[0]) & (m <= win[1])
    cs = {c: CubicSpline(tg, blocks[c], axis=1) for c in blocks}
    tt = np.abs(sig[:, None] * w[None, :])
    live = tt <= tmax
    sgn = np.sign(w)[None, :]
    S = -0.5 * vgf[:, None] * tt ** 2 + 0j
    Sc, dSc = {}, {}
    for c, spl in cs.items():
        v = np.zeros(tt.shape, dtype=np.complex128)
        dv = np.zeros(tt.shape, dtype=np.complex128)
        for j in range(len(sel)):
            lj = live[j]
            v[j, lj] = _eval_row(spl, j, tt[j, lj], 0)
            dv[j, lj] = _eval_row(spl, j, tt[j, lj], 1)
        Sc[c], dSc[c] = v, dv
        S = S + v

    def signed(x):
        return np.where(sgn < 0, np.conj(x), x)

    def dens(phiK, Sx):
        f = to_density(phiK[None, :] * signed(np.where(live, np.exp(Sx), 0.0)), dm, N)
        f = f[:, inwin]
        ok = f > 1e-10 * f.max(axis=1, keepdims=True)
        f = np.where(ok, f, 0.0)
        return f / (f.sum(axis=1, keepdims=True) * dm), ok

    out = {}
    for fam, c in fams:
        # the shift is a mass offset of eps m_ref: a small one, so that the
        # check is of the linearisation and not of a 2 % mass offset
        eps_f = 1e-4 if fam == "shift" else eps
        if fam == "shift":
            g = 1j * tt * (mref / sig)[:, None]
        elif fam == "W":
            g = 0.5 * tt * dSc[c]
        elif fam == "R":
            g = Sc[c]
        pred = lin["families"][f"{fam}_{c}"]["dalpha"] * eps_f
        ftrue, _ = dens(kernel_alpha(channel, 0.0, m0, dm, N, w), S + eps_f * g)
        span = max(abs(pred), 2e-6) * 2.0
        al = pred + span * np.linspace(-1, 1, 7)
        ll = []
        for a_ in al:
            fa, ok = dens(kernel_alpha(channel, a_, m0, dm, N, w), S)
            lf = np.log(np.where(ok & (fa > 0), fa, 1.0))
            ll.append(float((ftrue * lf).sum() * dm))
        cfit = np.polyfit(al - pred, ll, 2)
        amax = pred - cfit[1] / (2.0 * cfit[0])
        out[f"{fam}_{c}"] = dict(eps=eps_f, linear=pred, asimov=float(amax))
        log(f"  {fam}_{c}: eps {eps_f:+.4f}  linear d alpha {pred:+.4e}  "
            f"Asimov {amax:+.4e}  ratio {amax / pred:.4f}")
    return out


def cmd_asimov(a):
    out = asimov(a.channel, n=a.n, eps=a.eps)
    p = os.path.join(RUNS, f"asimov_{a.channel}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")


def cmd_response(a):
    os.makedirs(RUNS, exist_ok=True)
    out, per, sel = response(a.channel, n=a.n, float_gamma=a.float_gamma)
    tag = f"response_{a.channel}{'_gprof' if a.float_gamma else ''}.json"
    p = os.path.join(RUNS, tag)
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "families"}, indent=1))
    m = 91.1876e3 if a.channel == "z" else MJPSI * 1e3
    print(f"{'family':<10} {'d alpha / eps':>14} {'MeV / eps':>11}"
          + ("  profiled(Gamma)  dGamma MeV/eps" if a.float_gamma else ""))
    for k, r in out["families"].items():
        line = f"{k:<10} {r['dalpha']:+14.4e} {m * r['dalpha']:+11.3f}"
        if a.float_gamma:
            line += f"  {r['dalpha_gprof']:+14.4e}  {r['dgamma_MeV']:+10.3f}"
        print(line)
    print(f"-> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("response")
    r.add_argument("--channel", choices=("z", "jpsi"), required=True)
    r.add_argument("--n", type=int, default=100000)
    r.add_argument("--float-gamma", action="store_true")
    s = sub.add_parser("asimov")
    s.add_argument("--channel", choices=("z", "jpsi"), required=True)
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--eps", type=float, default=0.02)
    a = ap.parse_args()
    {"response": cmd_response, "asimov": cmd_asimov}[a.cmd](a)


if __name__ == "__main__":
    main()
