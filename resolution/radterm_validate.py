#!/usr/bin/env python3
"""Validation of the RADIATIVE (brems + pair) CF term added to the offline
model on 2026-09-03 (`cf_track_resolution.extract` / `cf_mass_likelihood`).

Four checks, all on the production files themselves -- no toy:

 1. MEAN IDENTITY.  The modelled spectrum must reproduce the exported
    dE_rad = sum_steps dedxRad * step.  It is true by construction (each
    process is renormalized to its own ComputeDEDXPerVolume) so what this
    actually tests is the v grid, the trapezoid and the kinematic limit --
    i.e. that the offline reader is reading the export correctly.  Runs
    `cf_brems_exact.validate_mean` on legs assembled by POOLING radstepidx,
    which is also a test that the pooling is the one the extraction does.

 2. SMALL-t BEHAVIOUR.  The term is CENTRED, so S_rad(t) = -kappa2 t^2/2 +
    O(t^4) and Im S_rad = kappa3 t^3/6 + O(t^5).  Both exponents are measured
    on a decade-spaced t grid; a mean that survived the centring would show up
    as Im S / t -> const instead of -> 0.

 3. kappa2 FRACTION, against the in-fit numbers.  NOTES_CGFFIT s47/s52 quote
    the radiative channel as "2.5 % of the q/p kappa2 at pT = 3 and 18.6 % at
    pT = 40".  Here the same ratio is formed offline from the two exponents'
    own small-t curvature, per track, binned in gen pT -- the offline term and
    the in-fit channel are built from the same records, so this is a
    cross-implementation check of both.

 4. QUADRATURE.  The CF is tabulated on TG (448 points on [0, 14]) and every
    downstream statistic is a trapezoid over it.  The radiative exponent is
    the newest and the least smooth (a compound Poisson with a 1/v spectrum),
    so this refines TG by x4 for a sample of tracks, rebuilds ALL exponents on
    the fine grid, and compares <e^{-u z^2}> and <z e^{-u z^2}>.

usage:
  python radterm_validate.py [--file F] [--file-hi F] [--ntracks N]
"""
import argparse
import os

import numpy as np
import uproot

import cf_brems_exact as cbe
import cf_track_resolution as cft
import prodfiles

RS = cbe.RADV_STRIDE
NV = cbe.NRADV
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
# single-threaded productions, so one stream file per task; single_file
# finds it without naming stream 0 and warns if there are several.
F_LOW = prodfiles.single_file(
    f"{CEPH}/resolution_trackres_mugun_lowpt_260903x_m0/task_0000", "globalcor_resclosure")
F_HI = prodfiles.single_file(
    f"{CEPH}/resolution_trackres_mugun_ul16_260903x_m0/task_0000", "globalcor_resclosure")

BR = ["refParms", "refCov", "genParms", "resinfcov", "resinfvarv", "reseigidx",
      "msmoliidx", "msmoliv", "ioniurbanidx", "ioniurbanv",
      "radstepidx", "radstepv", "radstepspecv", "radvgrid", "genPt"]


def read(fn, n):
    t = uproot.open(fn)["tree"]
    pt = uproot.open(fn)["runtree"]["parmtype"].array(library="np")
    br = list(BR)
    for b in ("ioniqscaleidx", "ioniqscalev"):
        if b in t.keys():
            br.append(b)
    return t.arrays(br, library="np", entry_stop=n), pt


def blocks(a, pt, ic):
    """[(g, wstd*charge, ioni steps, rad recs, rad spec)] for one track."""
    qg = a["genParms"][ic][0]
    if qg == 0.:
        return None
    c00 = float(a["refCov"][ic][0])
    cov = float(a["resinfcov"][ic])
    if not (c00 > 0.) or abs(cov / c00 - 1.) > cft.COVTOL:
        return None
    sig = np.sqrt(c00)
    chg = np.sign(qg)
    gi = np.asarray(a["reseigidx"][ic])
    vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
    fam = pt[gi]
    uvi = np.asarray(a["ioniurbanv"][ic], dtype=np.float64)
    uii = np.asarray(a["ioniurbanidx"][ic])
    uvi = uvi.reshape(-1, len(uvi) // max(len(uii), 1)) if len(uii) else uvi.reshape(0, 11)
    qsi = np.asarray(a["ioniqscaleidx"][ic]) if "ioniqscaleidx" in a else None
    qsv = (np.asarray(a["ioniqscalev"][ic], dtype=np.float64).reshape(-1, 2)
           if "ioniqscalev" in a else None)
    ridx = np.asarray(a["radstepidx"][ic])
    rrec = np.asarray(a["radstepv"][ic], dtype=np.float64).reshape(-1, RS)
    rspc = np.asarray(a["radstepspecv"][ic], dtype=np.float64).reshape(-1, 2 * NV)
    out = []
    sel = fam == 11
    for g in np.unique(gi[sel]):
        vpool = vb[sel & (gi == g)].sum()
        if vpool <= 0.:
            continue
        steps = uvi[uii == g]
        if not len(steps):
            return None
        sq2 = cft.ioni_sq2(steps, qsv[qsi == g] if qsv is not None else None)
        if sq2 <= 0.:
            continue
        w = chg * np.sqrt(vpool / sq2) / sig
        rm = ridx == g
        out.append((int(g), w, steps, rrec[rm], rspc[rm]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=F_LOW)
    p.add_argument("--file-hi", default=F_HI)
    p.add_argument("--ntracks", type=int, default=40)
    p.add_argument("--nquad", type=int, default=5)
    p.add_argument("--nkappa", type=int, default=0,
                   help="tracks per file for check 3 (default: --ntracks)")
    args = p.parse_args()

    a, pt = read(args.file, args.ntracks)
    vg = np.asarray(a["radvgrid"][0], dtype=float)
    print(f"# {os.path.basename(os.path.dirname(args.file))}  "
          f"{len(a['genPt'])} entries read, v grid {len(vg)} pts "
          f"[{vg[0]:.1e}, {vg[-1]:.3f}]\n")

    # ---------------- 1. mean identity, on POOLED legs
    print("## 1. mean identity: spectrum integral vs exported dedxRad * step")
    legs = []
    for ic in range(min(6, len(a["genPt"]))):
        b = blocks(a, pt, ic)
        if not b:
            continue
        for _, _, _, rr, rp in b:
            if len(rr):
                legs.append((rr, rp))
    cbe.validate_mean(legs, vg)
    print()

    # ---------------- 2. small-t behaviour
    print("## 2. centring: Re S_rad = O(t^2), Im S_rad = O(t^3)")
    print(f"{'track':>6} {'t':>10} {'Re S':>14} {'-2ReS/t^2':>14} "
          f"{'Im S':>14} {'6 ImS/t^3':>14}")
    tt = np.array([1e-3, 1e-2, 1e-1, 1.0])
    nshown = 0
    for ic in range(len(a["genPt"])):
        b = blocks(a, pt, ic)
        if not b:
            continue
        S = np.zeros(len(tt), dtype=np.complex128)
        for _, w, _, rr, rp in b:
            if len(rr):
                S += cbe.rad_exponent(tt, rr, rp, vg, weights=np.full(len(rr), w))
        for j in range(len(tt)):
            print(f"{ic:6d} {tt[j]:10.1e} {S.real[j]:+14.5e} "
                  f"{-2*S.real[j]/tt[j]**2:+14.5e} {S.imag[j]:+14.5e} "
                  f"{6*S.imag[j]/tt[j]**3:+14.5e}")
        nshown += 1
        if nshown >= 2:
            break
    print("   -> the two ratio columns must be FLAT in t at small t "
          "(kappa2 and kappa3 of the block); a surviving mean would make\n"
          "      Im S / t constant instead.\n")

    # ---------------- 3. kappa2 fraction vs the in-fit numbers
    print("## 3. radiative share of the q/p kappa2, per gen pT "
          "(NOTES_CGFFIT s47: 2.5 % at pT = 3, 18.6 % at pT = 40)")
    t0 = 1e-3
    rows = []
    nk = args.nkappa or args.ntracks
    for fn, tag in ((args.file, "lowpt"), (args.file_hi, "ul16")):
        aa, ptt = read(fn, nk)
        vgg = np.asarray(aa["radvgrid"][0], dtype=float)
        for ic in range(len(aa["genPt"])):
            b = blocks(aa, ptt, ic)
            if not b:
                continue
            kr = ki = 0.
            for _, w, steps, rr, rp in b:
                ki += -2. * cft.ioni_step_exponent(
                    steps, w, np.array([t0])).real[0] / t0 ** 2
                if len(rr):
                    kr += -2. * cbe.rad_exponent(
                        np.array([t0]), rr, rp, vgg,
                        weights=np.full(len(rr), w)).real[0] / t0 ** 2
            rows.append((float(aa["genPt"][ic]), kr, ki, tag))
    rows = np.array([(r[0], r[1], r[2]) for r in rows])
    edges = [0., 4., 8., 15., 25., 35., 45., 100.]
    print("   The radiative kappa2 is TAIL DOMINATED (dsigma/dv ~ 1/v, so the")
    print("   second moment sits at v -> 1: one hard radiator carries a bin),")
    print("   which is why the fit's Q excludes it and why the MEDIAN of the")
    print("   per-track ratio and its SUM differ by a factor of a few. Both")
    print("   are given; the toy number quoted above is a plane mean.")
    print(f"{'pT bin':>14} {'n':>6} {'kappa2_rad':>13} {'kappa2_ioni':>13} "
          f"{'sum r/i':>9} {'med r/i':>9} {'med r/(r+i)':>12}")
    for j in range(len(edges) - 1):
        m = (rows[:, 0] >= edges[j]) & (rows[:, 0] < edges[j + 1])
        if not m.any():
            continue
        kr, ki = rows[m, 1].sum(), rows[m, 2].sum()
        rat = rows[m, 1] / np.maximum(rows[m, 2], 1e-300)
        print(f"{edges[j]:6.0f}-{edges[j+1]:<7.0f} {int(m.sum()):6d} "
              f"{kr:13.5e} {ki:13.5e} {100*kr/ki:8.2f}% "
              f"{100*np.median(rat):8.2f}% "
              f"{100*np.median(rat/(1+rat)):11.2f}%")
    print()

    # ---------------- 4. quadrature: refine TG x4
    print("## 4. TG quadrature: full model on 448 pts vs x4 refinement")
    TG = cft.TG
    TG4 = np.linspace(TG[0], TG[-1], 4 * (len(TG) - 1) + 1)
    print(f"{'track':>6} {'u':>6} {'<e^-uz2> 448':>15} {'x4':>15} {'diff':>11} "
          f"{'<ze^-uz2> 448':>15} {'x4':>15} {'diff':>11}")
    nsh = 0
    for ic in range(len(a["genPt"])):
        b = blocks(a, pt, ic)
        if not b:
            continue
        gi = np.asarray(a["reseigidx"][ic])
        vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
        fam = pt[gi]
        c00 = float(a["refCov"][ic][0])
        sig = np.sqrt(c00)
        uvm = np.asarray(a["msmoliv"][ic], dtype=np.float64)
        uim = np.asarray(a["msmoliidx"][ic])
        uvm = uvm.reshape(-1, len(uvm) // max(len(uim), 1))
        out = {}
        for name, tg in (("448", TG), ("x4", TG4)):
            S = np.zeros(len(tg), dtype=np.complex128)
            S += -0.5 * (vb[(fam == 8) | (fam == 9)].sum() / c00) * tg ** 2
            for g in np.unique(gi[fam == 10]):
                vp = vb[(fam == 10) & (gi == g)].sum()
                st = uvm[uim == g]
                s2 = st[:, 5].sum()
                if vp <= 0. or s2 <= 0.:
                    continue
                S += cft.ms_step_exponent(st, np.sqrt(vp / s2) / sig, tg)
            for _, w, steps, rr, rp in b:
                S += cft.ioni_step_exponent(steps, w, tg)
                if len(rr):
                    S += cbe.rad_exponent(tg, rr, rp, vg,
                                          weights=np.full(len(rr), w))
            out[name] = np.exp(S)
        for u in (0.05, 0.2, 1.0, 2.0):
            e = {}
            o = {}
            for name, tg in (("448", TG), ("x4", TG4)):
                ph = out[name]
                w = np.exp(-tg ** 2 / (4. * u)) / np.sqrt(np.pi * u)
                e[name] = float(np.trapezoid(ph.real * w, tg))
                wo = (tg / (2. * u)) * np.exp(-tg ** 2 / (4. * u)) / np.sqrt(np.pi * u)
                o[name] = float(np.trapezoid(ph.imag * wo, tg))
            print(f"{ic:6d} {u:6.2f} {e['448']:15.9f} {e['x4']:15.9f} "
                  f"{e['448']-e['x4']:+11.2e} {o['448']:+15.9f} {o['x4']:+15.9f} "
                  f"{o['448']-o['x4']:+11.2e}")
        nsh += 1
        if nsh >= args.nquad:
            break
    print("   -> the diff columns bound the quadrature error of every closure "
          "number reported from these caches.")


if __name__ == "__main__":
    main()
