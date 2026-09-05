#!/usr/bin/env python3
"""Text tables for the CF-exponent compression study (stdout + <out>/tables.txt)."""
import datetime, os, sys
import numpy as np

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/cfcompress")
OUT = os.path.expanduser("~/public_html/cvh/%s_cfcompress"
                         % datetime.date.today().strftime("%y%m%d"))
TAGS = ["jpsigun", "btojpsix", "trk_lowpt", "trk_ul16"]
BUF = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); BUF.append(s)


def minrank(ranks, err, thr):
    """smallest r in the scan with err(r) < thr (interpolation-free)."""
    ok = np.where(np.asarray(err) < thr)[0]
    return int(ranks[ok[0]]) if len(ok) else None


def main():
    P("=" * 96)
    P("A. JOINT basis (ALL families -> ONE coefficient vector per candidate)")
    P("   smallest rank r in the scan {1,2,3,4,6,8,12,16,24,32,48,64,96,128} with")
    P("   max over candidates / 99.9 % of  max_t W(t)|dS|  below the threshold,")
    P("   W = exp(Re S_tot) [x |phi_K| for the mass caches];  worst family quoted.")
    P("")
    P(f"{'cache':12s} {'n':>7s} {'max<1e-3':>9s} {'max<1e-4':>9s} "
      f"{'q999<1e-3':>10s} {'q999<1e-4':>10s} {'unweighted max<1e-4':>20s}")
    for tag in TAGS:
        d = np.load(f"{SCRATCH}/spec_{tag}.npz", allow_pickle=True)
        fams = [str(x) for x in d["fams"]]
        r = d["ALL/ranks"]
        wmax = np.max([d[f"ALL/ewgt_{f}_max"] for f in fams], axis=0)
        wq = np.max([d[f"ALL/ewgt_{f}_q999"] for f in fams], axis=0)
        amax = np.max([d[f"ALL/eabs_{f}_max"] for f in fams], axis=0)
        P(f"{tag:12s} {int(d['n']):7d} {str(minrank(r,wmax,1e-3)):>9s} "
          f"{str(minrank(r,wmax,1e-4)):>9s} {str(minrank(r,wq,1e-3)):>10s} "
          f"{str(minrank(r,wq,1e-4)):>10s} {str(minrank(r,amax,1e-4)):>20s}")
    P("")
    P("=" * 96)
    P("B. per-block bases (MS | IO = [Sio_re|Sio_im] | RAD), rank r EACH")
    P(f"{'cache':12s} {'block':6s} " + " ".join(f"{x:>9s}" for x in
      ["max<1e-3", "max<1e-4", "q999<1e-4", "sv2/sv1", "sv5/sv1"]))
    for tag in TAGS:
        d = np.load(f"{SCRATCH}/spec_{tag}.npz", allow_pickle=True)
        for b, bf in (("Sms", ["Sms"]), ("IO", ["Sio_re", "Sio_im"]),
                      ("RAD", ["Srad_re", "Srad_im"]), ("Sdel", ["Sdel"])):
            if f"{b}/ranks" not in d.files:
                continue
            r = d[f"{b}/ranks"]
            wmax = np.max([d[f"{b}/ewgt_{f}_max"] for f in bf], axis=0)
            wq = np.max([d[f"{b}/ewgt_{f}_q999"] for f in bf], axis=0)
            sv = d[f"{b}/sv"]
            P(f"{tag:12s} {b:6s} {str(minrank(r,wmax,1e-3)):>9s} "
              f"{str(minrank(r,wmax,1e-4)):>9s} {str(minrank(r,wq,1e-4)):>9s} "
              f"{sv[1]/sv[0]:9.2e} {sv[4]/sv[0]:9.2e}")
    P("")
    P("=" * 96)
    P("C. full error tables, JOINT basis, weighted metric  max_t W|dS| (worst cand)")
    for tag in TAGS:
        d = np.load(f"{SCRATCH}/spec_{tag}.npz", allow_pickle=True)
        fams = [str(x) for x in d["fams"]]
        r = d["ALL/ranks"]
        P(f"-- {tag}   (float32 cache ulp at max|S| = "
          f"{float(d['q32']):.1e})")
        P(f"{'r':>4s} " + " ".join(f"{f:>10s}" for f in fams) + "   bytes/cand")
        for i, rr in enumerate(r):
            P(f"{rr:4d} " + " ".join(f"{d[f'ALL/ewgt_{f}_max'][i]:10.2e}"
                                     for f in fams) + f"   {4*rr:6d}")
    P("")
    P("=" * 96)
    P("D. physics dictionary vs PCA (J/psi gun, test split, 99.9 % of max_t W|dS|)")
    f = f"{SCRATCH}/phys_jpsigun.npz"
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        k = d["klist"]
        for b in ("MS", "IO", "RAD"):
            P(f"-- {b}   full physics dictionary (K={int(d[f'{b}/fine/K'])}): "
              f"q999={float(d[f'{b}/fine/ewgt_q999']):.2e}  "
              f"max={float(d[f'{b}/fine/ewgt_max']):.2e}  "
              f"(eabs max={float(d[f'{b}/fine/eabs_max']):.2e})")
            P(f"{'k':>4s} {'phys':>10s} {'poly':>10s} {'pca':>10s}")
            for i, kk in enumerate(k):
                P(f"{kk:4d} {d[f'{b}/phys/ewgt_q999'][i]:10.2e} "
                  f"{d[f'{b}/poly/ewgt_q999'][i]:10.2e} "
                  f"{d[f'{b}/pca/ewgt_q999'][i]:10.2e}")
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/tables.txt", "w").write("\n".join(BUF) + "\n")
    print("\nwrote", f"{OUT}/tables.txt")


if __name__ == "__main__":
    main()
