#!/usr/bin/env python3
"""The MASS model's own odd moment, per candidate, from its CF.

At track level the model's `<z e^{-u z^2}>` is ~0 (measured 5e-5), so a raw
data odd moment is `data - model` there. At MASS level it is not: the model
carries the Jensen mean shift AND the ionisation and radiative skews of the
CF. This evaluates all three, so that a mass-level odd moment becomes an
ATTRIBUTION rather than a correlation.

THE ASSEMBLY, read off `unbinned.MassCFTerm._family_parts`. `tgrid` is the
STANDARDIZED argument (the Gaussian family enters as `-0.5 vgf t^2`, which is
a unit-variance Gaussian at `vgf = 1`), so the CF of `z` is directly

    log phi_z(t) = k_ms Sms + k_ioni (Sio_re + i Sio_im)
                            + k_rad (Srad_re + i Srad_im) - 0.5 k_hit vgf t^2

with every `k` at its default 1, plus the Jensen displacement `i t d_i/sigma_i`
with `d_i/sigma_i = 1.5 (sigma_i/m_i)(1 + f_ang,i)`. Then

    <z e^{-u z^2}> = 1/sqrt(pi u) Int_0^inf (t/2u) e^{-t^2/4u} Im phi(t) dt

which is `cf_skew_closure.weier_odd`'s formula on the term's own grid.

    python3 model_odd_mass.py --cache .../gzpairs_dyv2_n50.npz
"""
import argparse

import numpy as np

PROBES = (0.05, 0.2)


def model_odd(d, u, jensen=True, kioni=1.0, krad=1.0, idx=None):
    """`<z e^{-u z^2}>` of the model, per candidate."""
    t = np.asarray(d["tgrid"], dtype=np.float64).ravel()
    nt = len(t)
    n = len(np.asarray(d["sigma"]).ravel())
    sel = slice(None) if idx is None else idx

    def blk(k):
        if k not in d.files:
            return None
        a = np.asarray(d[k], dtype=np.float64).reshape(-1, nt)
        return a[sel]

    vgf = np.asarray(d["vgf"], dtype=np.float64)[sel]
    sig = np.asarray(d["sigma"], dtype=np.float64)[sel]
    z = np.asarray(d["z"], dtype=np.float64)[sel]
    mg = np.asarray(d["mgen"], dtype=np.float64)[sel]
    fang = (np.asarray(d["fang"], dtype=np.float64)[sel]
            if "fang" in d.files else np.zeros(len(sig)))
    m = z * sig + mg

    s_re = -0.5 * vgf[:, None] * t[None, :] ** 2
    for k, sc in (("Sms", 1.0), ("Sio_re", kioni), ("Srad_re", krad)):
        b = blk(k)
        if b is not None:
            s_re = s_re + sc * b
    s_im = np.zeros_like(s_re)
    for k, sc in (("Sio_im", kioni), ("Srad_im", krad)):
        b = blk(k)
        if b is not None:
            s_im = s_im + sc * b
    if jensen:
        # the Jensen mean shift, as a displacement of the standardized variable
        s_im = s_im + (1.5 * (sig / m) * (1.0 + fang))[:, None] * t[None, :]

    phi_im = np.exp(s_re) * np.sin(s_im)
    out = np.empty((len(sig), len(PROBES)))
    for j, uu in enumerate(PROBES):
        wt = (t / (2.0 * uu)) * np.exp(-t * t / (4.0 * uu)) / np.sqrt(np.pi * uu)
        out[:, j] = np.trapezoid(phi_im * wt[None, :], t, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/work/submit/david_w/ZMass/"
                    "calibration_studies/fullscale/runs/gzpairs_dyv2_n50.npz")
    ap.add_argument("--nmax", type=int, default=120000)
    a = ap.parse_args()
    d = np.load(a.cache, allow_pickle=True)
    n = len(np.asarray(d["sigma"]).ravel())
    idx = np.arange(min(n, a.nmax))
    etal = np.maximum(np.abs(np.asarray(d["etap"], dtype=np.float64)[idx]),
                      np.abs(np.asarray(d["etam"], dtype=np.float64)[idx]))
    w = (np.asarray(d["w"], dtype=np.float64)[idx]
         if "w" in d.files else np.ones(len(idx)))
    full = model_odd(d, None, jensen=True, idx=idx)
    jen = model_odd(d, None, jensen=True, kioni=0.0, krad=0.0, idx=idx)
    nojen = model_odd(d, None, jensen=False, idx=idx)
    print(f"{len(idx)} candidates. The model's OWN odd moment, 1e-3.\n")
    hdr = (f"{'band':14s} {'n':>8s}" +
           "".join(f"{'u='+str(u)+': full / Jensen / skew':>34s}" for u in PROBES))
    print(hdr); print("-" * len(hdr))
    for lo, hi, lab in ((0, 0.9, "|eta|<0.9"), (0.9, 1.6, "0.9-1.6"),
                        (1.6, 3.0, "1.6-3.0"), (0, 9, "inclusive")):
        b = (etal >= lo) & (etal < hi)
        if b.sum() < 100:
            continue
        row = f"{lab:14s} {int(b.sum()):8d}"
        for j in range(len(PROBES)):
            row += (f"  {1e3*np.average(full[b, j], weights=w[b]):+10.2f}"
                    f" {1e3*np.average(jen[b, j], weights=w[b]):+10.2f}"
                    f" {1e3*np.average(nojen[b, j], weights=w[b]):+10.2f}")
        print(row)
    print("\n`Jensen` = the mean shift alone (ionisation and radiative off); "
          "`skew` = the CF's\nionisation+radiative odd content alone (Jensen "
          "off). `full` is what must be subtracted.")


if __name__ == "__main__":
    main()
