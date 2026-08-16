#!/usr/bin/env python3
"""Direct measurement of the SYSTEMATIC (coherent) energy-loss transmission.

CRITIQUE_260813 action 1.2 / NOTES_GATE item 4. `NOTES.md` carries two values
under the single name "the transmission":

  * 2026-08-05  T = 0.413 +- 0.009 -- regression of the fitted q/p residual on
    the TRUE per-track in-tracker loss (simPabsFirst/Last), i.e. the response
    to a RANDOM per-track FLUCTUATION;
  * 2026-08-12 (RETRACTION) T ~ 1 -- asserted, not measured, for the response
    to a COHERENT shift of the reference.

This measures the second one directly and theory-free. `CVH_DEDX_SCALE` scales
the reference mean dE/dx table coherently (same sign, every step, every track)
and touches only the MEAN -- the ionisation variance entering Q comes from the
Urban model in Geant4ePropagator::computeErrorIoni, not from the scaled table.
The build exports, per leg,

  Mu{plus,minus}_dEref = total mean energy loss the reference trajectory
                         actually applied between the PCA and the outermost
                         hit, last iteration of the unconstrained pass [GeV]

so the response needs no external normalisation:

  T_sys = d(p_fit at PCA) / d(assumed total energy loss)     (1 = full inherit)

Both numerator and denominator are per track and paired across configurations,
so the statistical error is set by the track-to-track spread of T_sys itself.

usage:
  python transmission_probe.py [--tag 260813] [--nfiles 0] [--obs leg|pair]
"""
import argparse
import glob
import os

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
MMU = 0.1056583745

BRANCHES = ["run", "lumi", "event",
            "Muplus_pt", "Muplus_eta", "Muminus_pt", "Muminus_eta",
            "Muplusgen_pt", "Muplusgen_eta", "Muminusgen_pt", "Muminusgen_eta",
            "Muplus_dEref", "Muminus_dEref", "Jpsi_mass", "Jpsigen_mass"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="260813")
    p.add_argument("--nfiles", type=int, default=0, help="0 = all")
    p.add_argument("--nboot", type=int, default=400)
    p.add_argument("--nbins", type=int, default=6)
    p.add_argument("--ref", type=float, default=1.000,
                   help="scale treated as the nominal reference")
    p.add_argument("--kappa", default="",
                   help="weighted_gap.py output; multiplies the reference "
                        "fractions by the measured weighted/unweighted gap "
                        "ratio kappa(p) before forming c")
    return p.parse_args()


def load(tag, scale, nfiles):
    stag = f"{int(round(scale * 1000)):04d}"
    fs = sorted(glob.glob(f"{CEPH}/resolution_transmission_{tag}_s{stag}/task_*/globalcor_0.root"))
    fs = [f for f in fs if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    if nfiles:
        fs = fs[:nfiles]
    if not fs:
        return None
    cols = {k: [] for k in BRANCHES}
    for fn in fs:
        try:
            a = uproot.open(fn)["tree"].arrays(BRANCHES, library="np")
        except Exception as exc:                     # noqa: BLE001
            print(f"  [warn] {fn}: {exc}")
            continue
        for k in BRANCHES:
            cols[k].append(np.asarray(a[k], dtype=np.float64))
    if not cols["run"]:
        return None
    d = {k: np.concatenate(v) for k, v in cols.items()}
    # run/lumi/event can repeat when an event yields >1 candidate, so pin the
    # gen kinematics too (same convention as cgf_dedx_scan.py). NEVER index-match:
    # the configurations emit candidates in a different order.
    d["key"] = np.array([f"{int(r)}:{int(l)}:{int(e)}:{gp:.5f}"
                         for r, l, e, gp in zip(d["run"], d["lumi"], d["event"],
                                                d["Muplusgen_pt"])])
    return d


def legs(d):
    """Per-leg (p_fit, p_gen, dEref) stacked as 2N rows, plus the pair momentum."""
    out = {}
    for leg in ("Muplus", "Muminus"):
        out[leg] = dict(
            pfit=d[f"{leg}_pt"] * np.cosh(d[f"{leg}_eta"]),
            pgen=d[f"{leg}gen_pt"] * np.cosh(d[f"{leg}gen_eta"]),
            dE=d[f"{leg}_dEref"],
        )
    pp, pm = out["Muplus"]["pgen"], out["Muminus"]["pgen"]
    out["phar"] = 2.0 / (1.0 / pp + 1.0 / pm)
    return out


def boot(x, f, nboot, rng):
    if len(x) < 20:
        return f(x), np.nan
    idx = rng.integers(0, len(x), (nboot, len(x)))
    return f(x), float(np.std(f(x[idx], axis=1)))


def med(x, axis=None):
    return np.median(x, axis=axis)


def main():
    args = parse_args()
    rng = np.random.default_rng(11)

    scales = []
    for sdir in sorted(glob.glob(f"{CEPH}/resolution_transmission_{args.tag}_s????")):
        scales.append(int(sdir[-4:]) / 1000.0)
    scales = sorted(scales)
    data = {}
    for s in scales:
        d = load(args.tag, s, args.nfiles)
        if d is None:
            print(f"[skip] s={s:.3f}: no complete tasks")
            continue
        data[s] = d
        print(f"loaded s={s:.3f}  n={len(d['key'])}")
    scales = sorted(data)
    if args.ref not in data:
        raise SystemExit(f"reference scale {args.ref} not available")

    common = set(data[scales[0]]["key"])
    for s in scales[1:]:
        common &= set(data[s]["key"])
    common = np.array(sorted(common))
    print(f"\nkey-matched candidates common to all {len(scales)} configs: {len(common)}")
    if len(common) < 200:
        raise SystemExit("too few matched candidates")

    L = {}
    for s in scales:
        pos = {k: i for i, k in enumerate(data[s]["key"])}
        i = np.array([pos[k] for k in common])
        L[s] = legs({k: v[i] for k, v in data[s].items() if k != "key"})

    ref = args.ref
    # stack the two legs; the gen momentum and the momentum used for binning are
    # per leg, so the leg observable is the natural one for a per-track response
    pgen = np.concatenate([L[ref]["Muplus"]["pgen"], L[ref]["Muminus"]["pgen"]])
    dEnom = np.concatenate([L[ref]["Muplus"]["dE"], L[ref]["Muminus"]["dE"]])
    P = {s: np.concatenate([L[s]["Muplus"]["pfit"], L[s]["Muminus"]["pfit"]]) for s in scales}
    E = {s: np.concatenate([L[s]["Muplus"]["dE"], L[s]["Muminus"]["dE"]]) for s in scales}

    good = np.ones(len(pgen), bool)
    for s in scales:
        good &= np.isfinite(P[s]) & np.isfinite(E[s]) & (E[s] > 1e-4)
    good &= np.isfinite(pgen) & (pgen > 0) & (np.abs(P[ref] / pgen - 1.0) < 0.2)
    print(f"legs after quality: {good.sum()} / {len(good)}")

    print("\n" + "=" * 78)
    print("0. THE KNOB: does the coherent scale move the applied reference loss?")
    print("=" * 78)
    print(f"{'s':>7} {'median dEref [MeV]':>20} {'ratio to s=1':>14} {'expected':>10}")
    for s in scales:
        m = np.median(E[s][good])
        print(f"{s:7.3f} {m*1e3:20.3f} {m/np.median(E[ref][good]):14.5f} {s/ref:10.3f}")

    print("\n" + "=" * 78)
    print("1. T_sys = dp_fit / d(applied eloss), per scale PAIR (linearity in the shift)")
    print("=" * 78)
    print(f"{'pair':>16} {'d(dEref) [MeV]':>16} {'T_sys (median)':>18} {'boot err':>10} "
          f"{'T_sys (mean)':>14} {'n':>8}")
    pair_T = {}
    for s in scales:
        if s == ref:
            continue
        dp = P[s][good] - P[ref][good]
        de = E[s][good] - E[ref][good]
        m = np.abs(de) > 1e-5
        T = dp[m] / de[m]
        # robust: 3-sigma-clip on the ratio (a handful of legs get a different
        # local minimum / anchoring recovery under the perturbed reference)
        keep = np.abs(T - np.median(T)) < 5.0 * (np.quantile(T, 0.84) - np.quantile(T, 0.16))
        Tm, Te = boot(T[keep], med, args.nboot, rng)
        pair_T[s] = (Tm, Te)
        print(f"{ref:6.3f}->{s:<9.3f} {np.median(de)*1e3:16.3f} {Tm:18.4f} {Te:10.4f} "
              f"{T[keep].mean():14.4f} {keep.sum():8d}")

    # straight-line check: T should be independent of the size AND SIGN of the shift
    ss = np.array([s for s in pair_T])
    tt = np.array([pair_T[s][0] for s in ss])
    te = np.array([pair_T[s][1] for s in ss])
    w = 1.0 / te ** 2
    tbar = np.sum(w * tt) / np.sum(w)
    chi2 = np.sum(w * (tt - tbar) ** 2)
    print(f"\nconstant fit over the {len(ss)} shifts: T_sys = {tbar:.4f} +- "
          f"{1/np.sqrt(np.sum(w)):.4f}   chi2/ndf = {chi2:.2f}/{len(ss)-1}")

    print("\n" + "=" * 78)
    print("2. T_sys vs momentum (leg observable, fixed-edge bins in gen p)")
    print("=" * 78)
    smax = max(scales, key=lambda s: abs(s - ref))
    dp = P[smax][good] - P[ref][good]
    de = E[smax][good] - E[ref][good]
    pg = pgen[good]
    en = dEnom[good]
    edges = np.array([0., 4., 8., 12., 20., 35., 1e9])
    print(f"{'p range':>14} {'<p>':>8} {'n':>8} {'dEref [MeV]':>13} {'T_sys med':>10} "
          f"{'err':>8} {'T_sys pop':>10}")
    for i in range(len(edges) - 1):
        sel = (pg >= edges[i]) & (pg < edges[i + 1]) & (np.abs(de) > 1e-5)
        if sel.sum() < 200:
            continue
        T = dp[sel] / de[sel]
        keep = np.abs(T - np.median(T)) < 5.0 * (np.quantile(T, 0.84) - np.quantile(T, 0.16))
        Tm, Te = boot(T[keep], med, args.nboot, rng)
        # population estimator: ratio of sums = the loss-weighted mean of the
        # per-track transmission, which is what enters the population response R
        Tpop = dp[sel][keep].sum() / de[sel][keep].sum()
        hi = "inf" if edges[i + 1] > 1e8 else f"{edges[i+1]:.0f}"
        print(f"{edges[i]:6.0f} - {hi:>5} {np.median(pg[sel]):8.2f} {keep.sum():8d} "
              f"{np.median(en[sel])*1e3:13.2f} {Tm:10.4f} {Te:8.4f} {Tpop:10.4f}")
    Tall = dp[np.abs(de) > 1e-5] / de[np.abs(de) > 1e-5]
    kall = np.abs(Tall - np.median(Tall)) < 5.0 * (np.quantile(Tall, 0.84) - np.quantile(Tall, 0.16))
    print(f"{'ALL':>14} {np.median(pg):8.2f} {kall.sum():8d} {np.median(en)*1e3:13.2f} "
          f"{np.median(Tall[kall]):10.4f} {'':8s} "
          f"{dp[np.abs(de)>1e-5][kall].sum()/de[np.abs(de)>1e-5][kall].sum():10.4f}")
    print(f"per-track spread of T_sys: q16/q50/q84 = "
          f"{np.quantile(Tall[kall],0.16):.3f} / {np.median(Tall[kall]):.3f} / "
          f"{np.quantile(Tall[kall],0.84):.3f}")

    print("\n" + "=" * 78)
    print("3. Does T_sys cancel in ds_req?  ds_req = bias(s=ref)/R from EACH scale")
    print("   pair separately, and R against the measured T_sys * <dEref/p>.")
    print("   Momentum-binned: dEref/p spans a factor ~20 over the sample, so an")
    print("   all-leg median of a product is not the product of the medians.")
    print("=" * 78)
    bias_ref_all = np.median(P[ref][good] / pgen[good] - 1.0)
    print(f"bias(s={ref:.3f}) = median dp/p = {bias_ref_all*1e4:+.3f} e-4   "
          f"(all legs, n={good.sum()})\n")
    print(f"{'<p>':>7} {'n':>7} {'bias e-4':>10} {'R e-4':>9} {'ds_req':>9} "
          f"{'T_sys':>8} {'<dE/p> e-4':>11} {'R/(T*dE/p)':>11}")
    binres = []
    for i in range(len(edges) - 1):
        selb = good & (pgen >= edges[i]) & (pgen < edges[i + 1])
        if selb.sum() < 400:
            continue
        rel = {s: P[s][selb] / pgen[selb] - 1.0 for s in scales}
        b0 = np.median(rel[ref])
        sarr = np.array(scales)
        marr = np.array([np.median(rel[s]) for s in scales])
        R = float(np.polyfit(sarr, marr, 1)[0])
        de_over_p = np.median(dEnom[selb] / pgen[selb])
        # T_sys in this bin, from the widest shift
        dpb = P[smax][selb] - P[ref][selb]
        deb = E[smax][selb] - E[ref][selb]
        mm = np.abs(deb) > 1e-5
        Tb = np.median(dpb[mm] / deb[mm])
        pmed = float(np.median(pgen[selb]))
        binres.append((pmed, selb, b0, R, b0 / R, Tb))
        print(f"{pmed:7.2f} {selb.sum():7d} {b0*1e4:10.2f} {R*1e4:9.2f} "
              f"{b0/R:9.4f} {Tb:8.4f} {de_over_p*1e4:11.1f} {R/(Tb*de_over_p):11.4f}")
    print("\nR/(T_sys*<dE/p>) = 1 confirms R is nothing but the transmitted")
    print("reference shift, so ds_req = bias/R divides T_sys straight back out:")
    print("T_sys CANNOT be applied again on top of ds_req.")

    # -------------------------------------------------------------- 3b
    print("\n" + "=" * 78)
    print("3b. CLOSURE: median dp/p against the applied coherent shift, all legs.")
    print("    Straight line through ALL points; residuals test linearity out to")
    print("    the physically relevant shift size (|1-s| ~ 0.15-0.25).")
    print("=" * 78)
    ss_all = np.array(scales)
    b_all = np.array([np.median(P[s][good] / pgen[good] - 1.0) for s in scales])
    co = np.polyfit(ss_all, b_all, 1)
    print(f"{'s':>7} {'median dp/p e-4':>16} {'line e-4':>10} {'resid e-4':>11}")
    for s, b in zip(ss_all, b_all):
        print(f"{s:7.3f} {b*1e4:16.3f} {np.polyval(co, s)*1e4:10.3f} "
              f"{(b-np.polyval(co,s))*1e4:11.3f}")
    szero = -co[1] / co[0]
    print(f"\nzero crossing of the all-leg bias: s = {szero:.4f}  "
          f"(ds_req = {1-szero:.4f});  slope R = {co[0]*1e4:.2f} e-4 per unit s")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 78)
    print("4. ds_req against the G4 mode / median reference fractions")
    print("   (NOTES_GATE gate_refs.json; pt10 dropped -- known mis-targeted")
    print("    model/sim pair with no shared detid)")
    print("=" * 78)
    import json
    refs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "gate_refs.json")))
    refs = [r for r in refs if r.get("f_g4mode") is not None and r["label"] != "pt10"]
    rp = np.array([r["p"] for r in refs])
    fmo = np.array([r["f_g4mode"] for r in refs])
    fme = np.array([r["f_g4med"] for r in refs])
    o = np.argsort(rp)
    rp, fmo, fme = rp[o], fmo[o], fme[o]

    # INFLUENCE-WEIGHT CORRECTION (2026-08-14). ds_req is the gap of the fit's
    # influence-WEIGHTED loss sum; gate_refs holds the gap of the UNWEIGHTED
    # total. kappa(p) = f_weighted/f_unweighted, measured on Geant4 truth with
    # the nested-cylinder weight profile (weighted_gap.py).
    if args.kappa:
        wg = json.load(open(args.kappa))
        kp = np.array([r["p"] for r in wg["rows"]])
        kk = np.array([r["kappa"] for r in wg["rows"]])
        ok = np.argsort(kp)
        kp, kk = kp[ok], kk[ok]
        kfac = np.interp(np.log(rp), np.log(kp), kk)
        print(f"applying kappa(p) = " +
              ", ".join(f"{p:.1f}:{k:.3f}" for p, k in zip(rp, kfac)))
        fmo = fmo * kfac
        fme = fme * kfac

    # bootstrap the per-bin ds_req by resampling CANDIDATES (the two legs of a
    # candidate share the vertex constraint, so legs are not independent)
    ncand = len(common)
    legcand = np.concatenate([np.arange(ncand), np.arange(ncand)])
    print(f"{'<p>':>7} {'ds_req':>16} {'f_mode(G4)':>11} {'pull':>7} "
          f"{'f_med(G4)':>11} {'pull':>7}")
    rows = []
    for pmed, selb, b0, R, dsr, Tb in binres:
        cand_in = legcand[selb]
        vals = []
        for _ in range(args.nboot):
            pick = rng.integers(0, ncand, ncand)
            cnt = np.bincount(pick, minlength=ncand)
            wsel = cnt[cand_in]
            k = np.repeat(np.arange(selb.sum()), wsel)
            if len(k) < 100:
                continue
            rr = {s: (P[s][selb] / pgen[selb] - 1.0)[k] for s in scales}
            mb = np.array([np.median(rr[s]) for s in scales])
            Rb = float(np.polyfit(np.array(scales), mb, 1)[0])
            vals.append(np.median(rr[ref]) / Rb)
        err = float(np.std(vals))
        fm = float(np.interp(np.log(pmed), np.log(rp), fmo))
        fd = float(np.interp(np.log(pmed), np.log(rp), fme))
        rows.append((pmed, dsr, err, fm, fd))
        print(f"{pmed:7.2f} {dsr:9.4f} +- {err:.4f} {fm:11.4f} "
              f"{(dsr-fm)/err:7.1f} {fd:11.4f} {(dsr-fd)/err:7.1f}")

    print("\nglobal scale factor c in  ds_req = c * f_ref  (inverse-variance):")
    for name, col in (("G4 mode", 3), ("G4 median", 4)):
        for lo, tag in ((0.0, "all p"), (4.0, "p > 4"), (8.0, "p > 8")):
            rr = [r for r in rows if r[0] > lo]
            if len(rr) < 2:
                continue
            y = np.array([r[1] for r in rr])
            e = np.array([r[2] for r in rr])
            f = np.array([r[col] for r in rr])
            w = (f / e) ** 2
            c = np.sum(w * y / f) / np.sum(w)
            ce = 1.0 / np.sqrt(np.sum(w))
            chi1 = np.sum(((y - f) / e) ** 2)
            print(f"  {name:10s} {tag:6s} nbin={len(rr)}  c = {c:.3f} +- {ce:.3f}"
                  f"   chi2(c=1) = {chi1:.1f}/{len(rr)}")


if __name__ == "__main__":
    main()
