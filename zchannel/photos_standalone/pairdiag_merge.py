#!/usr/bin/env python3
"""Merge `photos_pairdiag` outputs and print the pair phase-space tables.

The binary is a flat block of additive sums plus two small blocks of running
minima/maxima, so merging is a sum for the first and a min/max for the others.
With ``--pd`` the per-call `trypar` bookkeeping printed by the instrumented
Photos build (``#PD...`` lines) is summed alongside.
"""
import argparse
import glob
import struct
import sys

import numpy as np

MAGIC = b"PDIAG02"

# ---- layout, mirroring photos_pairdiag.cc ---------------------------------
G_NEV, G_NEVPAIR, G_SUMU, G_SUMU2, G_SUMX, G_NUPOS, G_NADD_OTHER, G_NBAD = range(8)
G_NSCAL = 8
S_N, S_SUMU, S_SUMQ, S_SUMXE, S_SUMX = 0, 1, 2, 3, 4
S_NQ, S_SUMUQ, S_NX = 5, 11, 17
S_COSP, S_COSM, S_COSPHI, S_COSMHI = 22, 23, 24, 25
S_SUMR, S_SUMABSR, S_SUMRELR, S_SUMRELR2, S_NR = 26, 27, 28, 29, 30
S_NSCAL = 31

QCUT = [2 * 0.000511, 0.01, 0.1, 1.0, 3.0, 10.0]
XCUT = [1e-4, 1e-3, 1e-2, 0.1, 0.5]
SPEC = ["e+e-", "mu+mu-"]


def read(fn):
    with open(fn, "rb") as f:
        b = f.read()
    if b[:7] != MAGIC:
        raise ValueError(f"{fn}: bad magic {b[:8]!r}")
    nq, nx, nu, n2, nc, nsum, nmin, nmax = struct.unpack_from("<8q", b, 8)
    o = 8 + 64
    s = np.frombuffer(b, "<f8", nsum, o).copy()
    o += 8 * nsum
    mn = np.frombuffer(b, "<f8", nmin, o).copy()
    o += 8 * nmin
    mx = np.frombuffer(b, "<f8", nmax, o).copy()
    return (nq, nx, nu, n2, nc), s, mn, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--pd", default="", help="glob of logs with #PD... lines")
    a = ap.parse_args()

    files = sorted(sum([glob.glob(p) for p in a.inputs], []))
    if not files:
        raise SystemExit("no input files")
    shp, S, MN, MX = read(files[0])
    for fn in files[1:]:
        sh, s, mn, mx = read(fn)
        assert sh == shp
        S += s
        MN = np.minimum(MN, mn)
        MX = np.maximum(MX, mx)
    nq, nx, nu, n2, nc = shp
    nhist = nq + nx + nu + nc + n2 * n2

    def spec(i):
        o = G_NSCAL + i * (S_NSCAL + nhist)
        return S[o:o + S_NSCAL], S[o + S_NSCAL:o + S_NSCAL + nhist]

    nev = S[G_NEV]
    print(f"files = {len(files)}, N_events = {nev:.6e}, N_bad = {S[G_NBAD]:.0f}, "
          f"non-lepton/non-photon additions = {S[G_NADD_OTHER]:.0f}")
    print()

    # ---- (a) totals -------------------------------------------------------
    print("(a) pairs per event")
    tot = 0.0
    for i in range(2):
        sc, _ = spec(i)
        r = sc[S_N] / nev
        tot += sc[S_N]
        print(f"    {SPEC[i]:8s}  N = {sc[S_N]:12.0f}   rate = {r:.5e} +- {np.sqrt(sc[S_N])/nev:.2e}")
    print(f"    {'total':8s}  N = {tot:12.0f}   rate = {tot/nev:.5e} +- {np.sqrt(tot)/nev:.2e}")
    print(f"    events with >=1 pair: {S[G_NEVPAIR]:.0f}  ({S[G_NEVPAIR]/nev:.5e}/ev)")
    print()

    # ---- (b) q cuts -------------------------------------------------------
    print("(b) rate(q > q_cut) per event, and <u> over pairs with q > q_cut")
    print(f"    {'q_cut [GeV]':>12s} | " + " | ".join(
        f"{s:>28s}" for s in SPEC) + " |" + f"{'both: rate':>14s}")
    for k, qc in enumerate(QCUT):
        row = f"    {qc:12.5g} | "
        cells, nsum, usum = [], 0.0, 0.0
        for i in range(2):
            sc, _ = spec(i)
            n, su = sc[S_NQ + k], sc[S_SUMUQ + k]
            nsum += n
            usum += su
            rate = n / nev
            err = np.sqrt(max(n, 1)) / nev
            mu = su / n if n > 0 else 0.0
            cells.append(f"{rate:.4e}+-{err:.1e} <u>={mu:.3e}")
        row += " | ".join(f"{c:>28s}" for c in cells)
        row += f" |{nsum/nev:14.4e}"
        print(row + f"   (rel.err {1/np.sqrt(max(nsum,1)):.3f})")
    print()

    # ---- (c) x_E cuts -----------------------------------------------------
    print("(c) rate(x_E > x_cut) per event")
    print(f"    {'x_cut':>10s} | {'e+e-':>22s} | {'mu+mu-':>22s} | {'total':>12s}")
    for k, xc in enumerate(XCUT):
        cells, nsum = [], 0.0
        for i in range(2):
            sc, _ = spec(i)
            n = sc[S_NX + k]
            nsum += n
            cells.append(f"{n/nev:.4e}+-{np.sqrt(max(n,1))/nev:.1e}")
        print(f"    {xc:10.4g} | {cells[0]:>22s} | {cells[1]:>22s} | {nsum/nev:12.4e}")
    print()

    # ---- (d) which leg ----------------------------------------------------
    print("(d) cos(theta) of the pair w.r.t. the pre-FSR mu- direction")
    for i in range(2):
        sc, _ = spec(i)
        n = sc[S_N]
        if n == 0:
            continue
        p, m = sc[S_COSP], sc[S_COSM]
        ph, mh = sc[S_COSPHI], sc[S_COSMHI]
        print(f"    {SPEC[i]:8s} cos>0 {p/n:.4f} +- {np.sqrt(p)/n:.4f}   "
              f"cos<0 {m/n:.4f}   |cos|>0.9: cos>+0.9 {ph/n:.4f} +- {np.sqrt(max(ph,1))/n:.4f}, "
              f"cos<-0.9 {mh/n:.4f} +- {np.sqrt(max(mh,1))/n:.4f}  "
              f"(sum |cos|>0.9 = {(ph+mh)/n:.4f})")
    print()

    # ---- (e) IR-safe <u> --------------------------------------------------
    print("(e) mean mass loss over ALL events")
    var = S[G_SUMU2] / nev - (S[G_SUMU] / nev) ** 2
    print(f"    <u>      = {S[G_SUMU]/nev:.6e} +- {np.sqrt(max(var,0)/nev):.2e}")
    print(f"    <1-e^-2u>= {S[G_SUMX]/nev:.6e}")
    print(f"    fraction of events with u > 0: {S[G_NUPOS]/nev:.6e}")
    for i in range(2):
        sc, _ = spec(i)
        if sc[S_N] == 0:
            continue
        print(f"    {SPEC[i]:8s} contribution to <u>: {sc[S_SUMU]/nev:.6e}"
              f"   <u|pair> = {sc[S_SUMU]/sc[S_N]:.6e}"
              f"   <q> = {sc[S_SUMQ]/sc[S_N]:.5f} GeV"
              f"   <x_E> = {sc[S_SUMXE]/sc[S_N]:.6e}")
    print()

    # ---- (f) closure ------------------------------------------------------
    print("(f) closure of 1-exp(-2u) against x_E - q^2/m^2 (single-pair events)")
    for i in range(2):
        sc, _ = spec(i)
        n = sc[S_NR]
        if n == 0:
            continue
        mr, ma = sc[S_SUMR] / n, sc[S_SUMABSR] / n
        rr = sc[S_SUMRELR] / n
        rr2 = sc[S_SUMRELR2] / n
        print(f"    {SPEC[i]:8s} <resid> = {mr:+.4e}  <|resid|> = {ma:.4e}  "
              f"max|resid| = {MX[3*i+2]:.4e}  <rel> = {rr:+.4e}  "
              f"rms(rel) = {np.sqrt(max(rr2,0)):.4e}")
    print()

    # ---- minima / maxima --------------------------------------------------
    print("    minimum / maximum generated")
    for i in range(2):
        sc, _ = spec(i)
        if sc[S_N] == 0:
            continue
        print(f"    {SPEC[i]:8s} q in [{MN[2*i]:.6e}, {MX[3*i]:.6e}] GeV   "
              f"x_E in [{MN[2*i+1]:.6e}, {MX[3*i+1]:.6e}]")
    print()

    # ---- #PD bookkeeping --------------------------------------------------
    if a.pd:
        pd_cell, pd_par, hdr = {}, {}, {}
        for fn in sorted(sum([glob.glob(p) for p in a.pd.split(",")], [])):
            for line in open(fn):
                w = line.split()
                if not w:
                    continue
                if w[0] == "#PDIAG":
                    hdr["nev"] = hdr.get("nev", 0) + int(w[2])
                    hdr["streng"], hdr["wtdiv"] = w[4], w[6]
                elif w[0] == "#PDPHOPAR":
                    k = (w[1], w[2])
                    d = pd_par.setdefault(k, {})
                    for j in range(3, len(w), 2):
                        d[w[j]] = d.get(w[j], 0) + int(w[j + 1])
                elif w[0] == "#PDCELL":
                    k = (w[1], w[2], w[3])
                    d = pd_cell.setdefault(k, {})
                    for j in range(4, len(w), 2):
                        v = float(w[j + 1])
                        if w[j].startswith("max_"):
                            d[w[j]] = max(d.get(w[j], -1e300), v)
                        else:
                            d[w[j]] = d.get(w[j], 0.0) + v
        if hdr:
            n = hdr["nev"]
            print(f"#PD bookkeeping: nev = {n}, STRENG = {hdr['streng']}, WTDIV = {hdr['wtdiv']}")
            print("  PHOPAR  sp  blk   nphopar   legs_seen   qedrad_vetoed")
            for k in sorted(pd_par):
                d = pd_par[k]
                print(f"          {k[0]:3s} {k[1]:5s} {d['nphopar']:10d} {d['legseen']:11d} "
                      f"{d['qedveto']:13d}   (early returns {d['early']})")
            print("  trypar  sp  blk leg     ncall   nactive    ncrude    nphase    nwt  "
                  "nfinal  nover   <PRHARD>   maxPRHARD    <WT>   maxWT  <WT|crude>  "
                  "<max(WT-1,0)>  f(WT>1)")
            tot = {}
            for k in sorted(pd_cell):
                d = pd_cell[k]
                na, nc_, nw, nf = d["nactive"], d["ncrude"], d["nwt"], d["nfinal"]
                print(f"          {k[0]:3s} {k[1]:5s} {k[2]:1s} {d['ncall']:10.0f} "
                      f"{na:9.0f} {nc_:9.0f} {d['nphase']:9.0f} {nw:6.0f} {nf:7.0f} "
                      f"{d['nover']:6.0f} {d['sum_prhard']/max(na,1):10.4e} "
                      f"{d['max_prhard']:10.4e} {d['sum_wt']/max(nw,1):8.4f} "
                      f"{d['max_wt']:7.4f} {d['sum_wt']/max(nc_,1):10.4f} "
                      f"{d['sum_excess']/max(nw,1):13.3e} {d['nover']/max(nw,1):8.3e}")
                for f in ("ncall", "nactive", "ncrude", "nphase", "nwt", "nfinal",
                          "nover", "sum_prhard", "sum_wt", "sum_excess"):
                    tot[(k[0], f)] = tot.get((k[0], f), 0.0) + d[f]
                tot[(k[0], "max_wt")] = max(tot.get((k[0], "max_wt"), 0.0), d["max_wt"])
            for sp in ("e", "mu"):
                if (sp, "ncall") not in tot:
                    continue
                print(f"  TOTAL {sp:3s}: calls {tot[(sp,'ncall')]:.0f} active {tot[(sp,'nactive')]:.0f} "
                      f"crude {tot[(sp,'ncrude')]:.0f} ({tot[(sp,'ncrude')]/n:.5e}/ev) "
                      f"phase {tot[(sp,'nphase')]:.0f} final {tot[(sp,'nfinal')]:.0f} "
                      f"({tot[(sp,'nfinal')]/n:.5e}/ev) over {tot[(sp,'nover')]:.0f} "
                      f"maxWT {tot[(sp,'max_wt')]:.5f} "
                      f"<WT|crude> {tot[(sp,'sum_wt')]/max(tot[(sp,'ncrude')],1):.5f} "
                      f"lost_to_WT>1 {tot[(sp,'sum_excess')]/n:.4e}/ev")
        print()

    if a.output:
        out = {"sums": S, "mins": MN, "maxs": MX,
               "shape": np.array(shp), "nfiles": np.array(len(files))}
        o = G_NSCAL
        for i in range(2):
            h = S[o + S_NSCAL:o + S_NSCAL + nhist]
            out[f"hq_{i}"] = h[:nq]
            out[f"hx_{i}"] = h[nq:nq + nx]
            out[f"hu_{i}"] = h[nq + nx:nq + nx + nu]
            out[f"hcos_{i}"] = h[nq + nx + nu:nq + nx + nu + nc]
            out[f"h2d_{i}"] = h[nq + nx + nu + nc:].reshape(n2, n2)
            o += S_NSCAL + nhist
        np.savez_compressed(a.output, **out)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
