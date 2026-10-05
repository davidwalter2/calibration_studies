"""Gate A3 of the conditional-Kalman recursion: the closure of arm R (the exact
marginal by the lattice recursion) on the tail toy's replicas, paired replica
by replica with the Gaussian marginal G, the published CF estimator H and its
Gaussian-kernel form K -- and, by (file, entry) key, with the tail marginal A
of an earlier run on the same seeds.

    python solver/rb_tail_report.py 'solver/runs/rbgrid/a3/part_*.npz' \
        [--with-A runs/tail_toy_lap_260925.npz]

Per replica i and arm k, y_ik = -S_ik / mean_i(C_ik) (the replica's share of
alpha_k; truth 0); paired differences y_ik - y_il carry the error of the
replicas the arms share.  The arms are normalised by their own mean curvature
except where noted.
"""
import argparse
import glob

import numpy as np

ARMS = ("A", "B", "C", "G", "M", "H", "K", "R")


def load(pattern, diag=None):
    S, C, ok, keys, RD, X = [], [], [], [], [], []
    dg = {k: [] for k in (diag or ())}
    for f in sorted(glob.glob(pattern)):
        d = np.load(f, allow_pickle=True)
        nrep = np.bincount(d["model"]).astype(int)
        S.append(d["S"])
        C.append(d["C"])
        ok.append(d["ok"].astype(bool))
        RD.append(d["RD"] if d["RD"].size else np.full((len(d["S"]), 10), np.nan))
        X.append(d["X"])
        for m, (p, e) in enumerate(zip(d["paths"], d["entries"])):
            keys.extend([(str(p), int(e), r) for r in range(nrep[m])])
        for k in dg:
            dg[k].append(np.repeat(d["diag_" + k], nrep))
    out = (np.concatenate(S), np.concatenate(C), np.concatenate(ok), keys,
           np.concatenate(RD), np.concatenate(X))
    if diag:
        return out + ({k: np.concatenate(v) for k, v in dg.items()},)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts")
    ap.add_argument("--with-A", dest="with_A", default=None)
    ap.add_argument("--ab", default=None,
                    help="parts of a second arm-R run on the same seeds: R - R' paired by key")
    ap.add_argument("--forms", action="store_true",
                    help="R - G per form of the elimination and per correction term (runs with FS/FC)")
    ap.add_argument("--bins", action="store_true",
                    help="R - G and R - H in quartiles of the model's sigma_m/m and ionisation share")
    args = ap.parse_args()
    S, C, ok, keys, RD, X, DG = load(args.parts, diag=("sig_rel", "f_ioni"))
    col = {k: i for i, k in enumerate(ARMS)}
    good = ok & np.all(np.isfinite(S[:, [col[k] for k in "BCGHKR"]]), axis=1) \
        & np.all(np.isfinite(C[:, [col[k] for k in "BCGHKR"]]), axis=1)
    n = int(good.sum())
    print(f"{len(S)} replicas, {n} with every arm finite ({len(S) - n} not); "
          f"{len(set(k[:2] for k in keys))} models")
    if RD.size and np.isfinite(RD).any():
        print(f"arm R: EDM median {np.nanmedian(RD[:, 0]):.1e}, max {np.nanmax(RD[:, 0]):.1e}; "
              f"not converged {int(np.nansum(RD[:, 2]))}; lattice maxN median "
              f"{np.nanmedian(RD[:, 3]):.0f}, max {np.nanmax(RD[:, 3]):.0f}; dropped max "
              f"{np.nanmax(RD[:, 4]):.3f}; negative share median {np.nanmedian(RD[:, 5]):.1e}, "
              f"max {np.nanmax(RD[:, 5]):.1e}; cut max {np.nanmax(RD[:, 6]):.1e}")
    y = {}
    for k in "BCGHKR":
        Cm = C[good, col[k]].mean()
        y[k] = -S[good, col[k]] / Cm
    f = lambda v: f"{1e3 * v.mean():+.4f} +- {1e3 * v.std(ddof=1) / np.sqrt(len(v)):.4f}"  # noqa: E731
    print("\n== single arms, alpha [e-3], truth 0")
    for k in "BCGHKR":
        print(f"   {k}: {f(y[k])}")
    print("\n== paired [e-3]")
    for i, j in (("R", "G"), ("R", "H"), ("R", "K"), ("H", "G"), ("K", "G"), ("H", "K")):
        print(f"   {i} - {j}: {f(y[i] - y[j])}")
    # robustness: the paired R - G with the curvature of G for both (same
    # normalisation), and trimmed on the per-replica difference
    dRG = (-S[good, col["R"]] + S[good, col["G"]]) / C[good, col["G"]].mean()
    med = np.median(dRG)
    mad = 1.4826 * np.median(np.abs(dRG - med))
    print("\n== R - G with G's curvature, trimmed on the per-replica difference [e-3]")
    for cut in (np.inf, 1000, 100, 30):
        m = np.abs(dRG - med) < cut * mad
        print(f"   |dev| < {cut:g} MAD ({int((~m).sum())} out): {f(dRG[m])}")
    # the curvature ratio: a non-convex R scan would show here
    cr = C[good, col["R"]] / C[good, col["G"]]
    print(f"\ncurvature R/G per replica: median {np.median(cr):.4f}, p1 {np.quantile(cr, 0.01):.3f}, "
          f"p99 {np.quantile(cr, 0.99):.3f}, min {cr.min():.3f}")
    if args.forms:
        FS, FC = [], []
        for fn in sorted(glob.glob(args.parts)):
            d = np.load(fn, allow_pickle=True)
            FS.append(d["FS"])
            FC.append(d["FC"])
        FS, FC = np.concatenate(FS)[good], np.concatenate(FC)[good]
        names = ("profile", "laplace", "modprofile", "reduced", "1/2 ln det j_ll", "ln |l_l;l^|")
        nf = len(names)
        Cm = FC[:, 2].mean()                    # G's modified-profile curvature
        print(f"\n== the elimination, form by form: alpha share -S/C (C = G's modified-profile "
              f"mean curvature) [e-3]")
        print(f"   {'form':18s} {'G':>20s} {'R':>20s} {'R - G':>20s}")
        for k, nm in enumerate(names):
            yG = -FS[:, k] / Cm
            yR = -FS[:, nf + k] / Cm
            print(f"   {nm:18s} {f(yG):>20s} {f(yR):>20s} {f(yR - yG):>20s}")
    if args.bins:
        CmG = C[good, col["G"]].mean()
        yR = -S[good, col["R"]] / CmG
        yG = -S[good, col["G"]] / CmG
        yH = -S[good, col["H"]] / CmG
        for name in ("sig_rel", "f_ioni"):
            v = DG[name][good]
            qs = np.quantile(v, [0, 0.25, 0.5, 0.75, 1.0])
            print(f"\n== by quartile of {name} (G's curvature) [e-3]")
            for lo_, hi_ in zip(qs[:-1], qs[1:]):
                m = (v >= lo_) & (v <= hi_)
                print(f"   [{lo_:.2e}, {hi_:.2e}] {int(m.sum()):6d}: R - G {f(yR[m] - yG[m])}   "
                      f"H - G {f(yH[m] - yG[m])}   R - H {f(yR[m] - yH[m])}")
    if args.with_A:
        a = np.load(args.with_A, allow_pickle=True)
        nrep = np.bincount(a["model"]).astype(int)
        akeys = []
        for m, (p, e) in enumerate(zip(a["paths"], a["entries"])):
            akeys.extend([(str(p), int(e), r) for r in range(nrep[m])])
        aidx = {k: i for i, k in enumerate(akeys)}
        sel = [(i, aidx[k]) for i, k in enumerate(keys) if k in aidx and good[i]]
        if sel:
            ii, jj = np.array(sel).T
            okA = a["ok"][jj].astype(bool) & np.isfinite(a["S"][jj, 0]) & np.isfinite(a["C"][jj, 0])
            ii, jj = ii[okA], jj[okA]
            # the same replicas? the linear fluctuation X1 is a function of the noise only
            same = np.nanmax(np.abs(X[ii, 0] - a["X"][jj, 0]))
            CmG = C[ii, col["G"]].mean()
            yA = -a["S"][jj, 0] / CmG
            yR = -S[ii, col["R"]] / CmG
            yG = -S[ii, col["G"]] / CmG
            print(f"\n== with arm A of {args.with_A}: {len(ii)} common replicas (max |X1 diff| {same:.1e}); "
                  f"G's curvature for all [e-3]")
            print(f"   A - G: {f(yA - yG)}   R - G: {f(yR - yG)}   A - R: {f(yA - yR)}")
    if args.ab:
        S2, C2, ok2, keys2, RD2, X2 = load(args.ab)
        idx2 = {k: i for i, k in enumerate(keys2)}
        sel = [(i, idx2[k]) for i, k in enumerate(keys) if k in idx2 and good[i]]
        ii, jj = np.array(sel).T
        fin2 = ok2[jj] & np.isfinite(S2[jj, col["R"]]) & np.isfinite(C2[jj, col["R"]])
        ii, jj = ii[fin2], jj[fin2]
        same = np.nanmax(np.abs(X[ii, 0] - X2[jj, 0]))
        CmG = C[ii, col["G"]].mean()
        d = (-S[ii, col["R"]] + S2[jj, col["R"]]) / CmG
        dg = (-S[ii, col["G"]] + S2[jj, col["G"]]) / CmG
        print(f"\n== R - R' ({args.ab}): {len(ii)} common replicas (max |X1 diff| {same:.1e}, "
              f"G - G' max {np.max(np.abs(dg)):.1e}); G's curvature [e-3]")
        print(f"   R - R': {f(d)}   |R - R'| per replica: median {1e3 * np.median(np.abs(d)):.4f}, "
              f"max {1e3 * np.max(np.abs(d)):.4f}")
        print(f"   R - G: {f((-S[ii, col['R']] + S[ii, col['G']]) / CmG)}   "
              f"R' - G: {f((-S2[jj, col['R']] + S[ii, col['G']]) / CmG)}")
        print(f"   curvature R'/R: median {np.median(C2[jj, col['R']] / C[ii, col['R']]):.5f}")


if __name__ == "__main__":
    main()
