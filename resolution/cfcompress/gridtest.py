#!/usr/bin/env python3
"""The OTHER compression axis: the t grid itself.

Storage per candidate is (#t points) x (#families) x 4 B, so decimating or
truncating TG = linspace(0, 14, 448) is an alternative to a low-rank basis --
with no basis to fit and no reconstruction cost, but with a quadrature error
instead of a reconstruction error.  This refits the same 4-parameter mass
likelihood on decimated / truncated grids and reports the shift of alpha.

Usage
  python3 gridtest.py --tag jpsigun --nfit 20000 --nproc 8
"""
import argparse, json, time
import numpy as np
import massnll_np as MN

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/cfcompress")
FAMS = ["Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jpsigun")
    ap.add_argument("--n0", type=int, default=40000)
    ap.add_argument("--nfit", type=int, default=20000)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    d = np.load(f"{SCRATCH}/sub_{a.tag}_n60000_s1234.npz")
    TG = d["tgrid"].astype(np.float64)
    sl = slice(a.n0, a.n0 + a.nfit)
    sig = d["sigma"][sl].astype(np.float64)
    mobs = d["z"][sl] * sig + (d["eta"][sl] - MN.MJPSI)
    vgf = d["vgf"][sl].astype(np.float64)
    M = {f: np.ascontiguousarray(d[f][sl]) for f in FAMS}
    pKre, pKim = MN.phik(a.tag, TG, sig)

    def build(idx):
        return MN.Model(sig, mobs, vgf, M["Sms"][:, idx], M["Sio_re"][:, idx],
                        M["Sio_im"][:, idx], M["Srad_re"][:, idx],
                        M["Srad_im"][:, idx], pKre[:, idx], pKim[:, idx],
                        TG[idx], nproc=a.nproc)

    ref = build(np.arange(448)).fit()
    print(f"full grid 448 pts (t<=14): NLL={ref['nll']:.6f}  " +
          "  ".join(f"{p}={v:+.6f}+-{e:.6f}" for p, v, e
                    in zip(MN.PARNAMES, ref["x"], ref["err"])))
    rows = [dict(name="448 / t<=14", npts=448, nll=ref["nll"],
                 x=ref["x"].tolist(), err=ref["err"].tolist())]
    for stride, tmax in [(2, 14.), (4, 14.), (8, 14.), (16, 14.),
                         (1, 8.), (2, 8.), (4, 8.), (1, 6.), (2, 6.), (4, 6.),
                         (1, 5.), (2, 5.)]:
        idx = np.arange(0, 448, stride)
        idx = idx[TG[idx] <= tmax + 1e-9]
        if idx[-1] != np.searchsorted(TG, tmax):
            pass
        m = build(idx)
        f0 = m.nll_grad(ref["x"], want_grad=False)[0]
        fr = m.fit(th0=ref["x"])
        dv = fr["x"] - ref["x"]
        print(f"stride {stride:2d} t<={tmax:4.1f}: {len(idx):4d} pts "
              f"({4*5*len(idx):6d} B/cand, x{8960/(4*5*len(idx)):5.1f})  "
              f"dNLL(th*)={f0-ref['nll']:+.3e}  " +
              "  ".join(f"d{p}={x:+.3e}({x/e:+.3f}s)" for p, x, e
                        in zip(MN.PARNAMES, dv, ref["err"])))
        rows.append(dict(name=f"stride{stride}/t<={tmax}", npts=int(len(idx)),
                         nll=fr["nll"], x=fr["x"].tolist(),
                         d=dv.tolist(), dnll_at_ref=float(f0 - ref["nll"])))
        del m
    json.dump(rows, open(f"{SCRATCH}/gridtest_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
