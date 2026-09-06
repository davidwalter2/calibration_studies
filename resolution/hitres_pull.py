#!/usr/bin/env python3
"""Hit-pull study for the CVH fit: is the Gaussian hit likelihood right?

The fit consumes exactly one thing per hit, `preciseHit->localPositionError()`,
inverted into Vinvfull. This asks whether that covariance describes the actual
hit errors -- a DISTRIBUTIONAL question, not a scale question. If the pull

    (rec - sim) / sigma_CPE

is Gaussian with a width != 1 the fix is a rescaling and the 2022 parmtype-8/9
attempt would have worked; it did not, which is evidence that the pull is not
Gaussian. So core width and tail fraction are reported SEPARATELY, never a
single rms.

Frames. dxrecsim and dxerr are both written by the maker in the frame the fit
actually uses: local x for pixels and 1D strips, local PHI (radians) for
radial/wedge strip topologies. The pull is therefore their plain ratio and
needs no conversion here -- see ResidualGlobalCorrectionMakerG4e.cc, the
`dxrecsimval = phihit - phisim` branch.

Binning. Strips are binned in (subdet, N, uProj), which are the CPE's OWN
variables: for N<=4 sigma = P0*uProj*exp(-uProj*P1)+P2 in pitch units, three
numbers for the whole strip tracker; for N>4 sigma = P0(subdet)+N*P1(subdet).
Pixels are binned in (qbin, |cot alpha| via localdxdz, edge/size class), which
is how the template is indexed.

usage:
  python hitres_pull.py --tag mugun_lowpt --outdir <dir>
  python hitres_pull.py --compare mugun_lowpt pion_ul16 kaon_ul16 proton_ul16
"""
import argparse
import datetime
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wums import logging as _wums_logging                        # noqa: E402
import prodfiles

logger = _wums_logging.child_logger(__name__)

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"

# Per-hit branches. hitDetId/hitUProj/hitPitch/hitThickness were added for
# this study (the CPE's uProj cannot be rebuilt offline: it carries the
# per-module Lorentz drift).
HITBR = ["dxrecsim", "dyrecsim", "dxerr", "dyerr",
         "clusterSize", "clusterSizeX", "clusterSizeY", "clusterCharge",
         "clusterChargeBin", "clusterOnEdge", "clusterProbXY", "clusterSN",
         "stripsToEdge", "localdxdz", "localdydz", "localqop",
         "hitDetId", "hitUProj", "hitPitch", "hitThickness"]
TRKBR = ["genPt", "genEta", "genParms", "refParms", "refCov", "nValidHits",
         "chisqval", "ndof"]

SUBDETS = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}


def subdet_of(detid):
    """DetId::subdetId() = bits 25-27 of the raw id (tracker dets only)."""
    return (np.asarray(detid, np.uint32) >> 25) & 0x7


def robust_stats(x, ntail=(3.0, 5.0)):
    """Core width and tail fraction, reported separately and never merged.

    core   -- half of the 68.27 % interquantile range, which for a Gaussian
              equals sigma and for a heavy-tailed distribution measures the
              CORE only (the quantity a rescaling can fix);
    sigmaG -- 1.4826 * MAD, a second core estimator with a different
              breakdown point, quoted so a disagreement between the two is
              visible rather than averaged away;
    rms    -- the second moment, i.e. what a variance-based estimator such
              as parmtype 8/9 actually sees. core != rms IS the result.
    tail_k -- fraction beyond k core widths; the Gaussian expectation is
              printed alongside so the excess is readable directly.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    out = {"n": n}
    if n < 20:
        return out
    q16, q50, q84 = np.percentile(x, [15.865, 50.0, 84.135])
    core = 0.5 * (q84 - q16)
    out["median"] = q50
    out["core"] = core
    out["mad"] = 1.4826 * np.median(np.abs(x - q50))
    out["rms"] = float(np.sqrt(np.mean((x - q50) ** 2)))
    out["mean"] = float(np.mean(x))
    # 10 % trimmed mean: a bias estimator that survives the tail
    xs = np.sort(x)
    lo, hi = int(0.1 * n), max(int(0.9 * n), int(0.1 * n) + 1)
    out["trimmean"] = float(xs[lo:hi].mean())
    from math import erfc, sqrt
    for k in ntail:
        out[f"tail{k:g}"] = float(np.mean(np.abs(x - q50) > k * core))
        out[f"gaus{k:g}"] = float(erfc(k / sqrt(2.0)))
    # The far tail is a DIFFERENT object from the shape: |pull| > 10 cannot be
    # a mis-parametrised sigma, it is a wrong hit (or, on the truth side, a
    # sim hit from a different crossing of the same module). Kept separate so
    # it never contaminates a width.
    out["patho"] = float(np.mean(np.abs(x - q50) > 10 * core))
    # second moment with the pathological component removed, i.e. what a
    # variance estimator with any outlier protection at all would see
    inl = np.abs(x - q50) <= 10 * core
    out["rms_trim"] = float(np.sqrt(np.mean((x[inl] - q50) ** 2))) if inl.any() else np.nan
    # SHAPE, independent of any scale. w_p is the half-width of the central
    # p-quantile band; the ratio w90/w68 is 1.645 for a Gaussian, 1.318 for a
    # uniform and larger than 1.645 for anything heavy-tailed. It is the one
    # number that separates "the width is wrong" from "the SHAPE is wrong",
    # which is the whole question here.
    def _w(p_):
        a, b = np.percentile(x, [50 - 50 * p_, 50 + 50 * p_])
        return 0.5 * (b - a)
    w68 = _w(0.6827)
    out["R90"] = _w(0.90) / w68 if w68 > 0 else np.nan
    out["R99"] = _w(0.99) / w68 if w68 > 0 else np.nan
    out["absmax"] = float(np.max(np.abs(x - q50)) / core) if core > 0 else np.nan
    # bootstrap error on the core width (the tails make the analytic one wrong)
    rng = np.random.default_rng(12345)
    nb = 200
    idx = rng.integers(0, n, size=(nb, min(n, 20000)))
    samp = x[idx]
    q = np.percentile(samp, [15.865, 84.135], axis=1)
    out["core_err"] = float(np.std(0.5 * (q[1] - q[0])))
    out["median_err"] = float(np.std(np.median(samp, axis=1)))
    return out


def load(tag, nfiles=0, subdir="hitres"):
    """Flatten the per-hit branches of one production into one dict of arrays.

    Track-level quantities are broadcast to their hits so any hit selection
    can also cut on the track (pT, eta, chi2).
    """
    import uproot
    pat = f"{CEPH}/{subdir}_{tag}/task_*/globalcor_resclosure_*.root"
    fs = prodfiles.resolve(pat, nfiles)
    if not fs:
        raise SystemExit(f"no files matching {pat}")
    cols = {k: [] for k in HITBR}
    tcols = {k: [] for k in ("genPt", "genEta", "chisq_ndof", "qop_res")}
    nfileok = 0
    for fn in fs:
        # Resume/partial-analysis guard: cmsRun creates its output at START,
        # so a still-running or killed task leaves a non-empty but TRUNCATED
        # file. Trust the completion sentinel, never the .root's existence.
        if not os.path.exists(os.path.join(os.path.dirname(fn), ".complete")):
            continue
        try:
            t = uproot.open(fn)["tree"]
            have = set(t.keys())
            miss = [k for k in HITBR if k not in have]
            if miss:
                raise SystemExit(f"{fn} is missing {miss}: rebuild the plugin "
                                 f"or point --subdir at the right production")
            a = t.arrays(HITBR + TRKBR, library="np")
        except SystemExit:
            raise
        except Exception as e:                                    # noqa: BLE001
            logger.warning(f"skipping {fn}: {e}")
            continue
        nfileok += 1
        nh = np.array([len(v) for v in a["dxrecsim"]])
        for k in HITBR:
            cols[k].append(np.concatenate(a[k]) if len(a[k]) else np.array([]))
        qg = np.array([g[0] for g in a["genParms"]])
        qf = np.array([r[0] for r in a["refParms"]])
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(qg != 0, (qf - qg) / np.abs(qg), np.nan)
            c2 = a["chisqval"] / np.maximum(a["ndof"], 1)
        tcols["genPt"].append(np.repeat(a["genPt"], nh))
        tcols["genEta"].append(np.repeat(a["genEta"], nh))
        tcols["chisq_ndof"].append(np.repeat(c2, nh))
        tcols["qop_res"].append(np.repeat(res, nh))
    d = {k: np.concatenate(v) for k, v in cols.items() if v}
    d.update({k: np.concatenate(v) for k, v in tcols.items() if v})
    d["subdet"] = subdet_of(d["hitDetId"])
    d["ispixel"] = d["subdet"] <= 2
    # The strip CPE's ONE species-sensitive switch, rebuilt exactly:
    #   dQdx = charge * invThickness * |cos(local polar angle)|
    # and `dQdx > maxChgOneMIP` (6000) sends the hit to the 3-constant LEGACY
    # formula instead of the tuned one. A low-momentum kaon or proton crosses
    # that threshold where a muon does not, which is one of only two ways the
    # hit error model can depend on the particle at all.
    with np.errstate(divide="ignore", invalid="ignore"):
        absdz = 1.0 / np.sqrt(1.0 + d["localdxdz"] ** 2 + d["localdydz"] ** 2)
        d["dQdx"] = np.where(d["hitThickness"] > 0,
                             d["clusterCharge"] / d["hitThickness"] * absdz, np.nan)
    # the fit's own pull, in the fit's frame
    with np.errstate(divide="ignore", invalid="ignore"):
        d["pullx"] = np.where(d["dxerr"] > 0, d["dxrecsim"] / d["dxerr"], np.nan)
        d["pully"] = np.where(d["dyerr"] > 0, d["dyrecsim"] / d["dyerr"], np.nan)
    d["matched"] = d["dxrecsim"] > -98.0
    d["nfiles"] = nfileok
    return d


def _fmt(s, label, width=26):
    if s.get("n", 0) < 20:
        return f"  {label:<{width}} {s.get('n', 0):>8d}   (too few)"
    return (f"  {label:<{width}} {s['n']:>8d}  {s['median']:+.4f}  "
            f"{s['core']:.4f}+-{s['core_err']:.4f}  {s['rms_trim']/s['core']:6.3f}  "
            f"{s['R90']:6.3f}  {s['R99']:6.3f}  "
            f"{100*s['tail3']:6.2f}%  {100*s['tail5']:6.3f}%  {100*s['patho']:7.4f}%")


HDR = (f"  {'bin':<26} {'nhits':>8}  {'median':>7}  {'core (68%)':>15}  "
       f"{'rmsT/cr':>6}  {'R90':>6}  {'R99':>6}  {'>3core':>7}  {'>5core':>7}  "
       f"{'>10core':>8}")
GAUS = ("  Gaussian reference:  median 0   core 1   rmsT/cr 1.000   R90 1.645   "
        "R99 2.576   0.27 %   0.00006 %   0 %\n"
        "  uniform  reference:                         0.845            1.318   "
        "1.450   0        0           0\n"
        "  (rmsT = rms with |pull| > 10 core removed; the last column IS that "
        "removed fraction --\n"
        "   a wrong hit, not a mis-scaled sigma, and it is kept separate for "
        "exactly that reason.)")


def report_strips(d, sel, title, out=print):
    """Strips, binned the way the CPE is parameterised."""
    out("")
    out(f"== {title}: STRIPS, by subdetector ==")
    out(HDR)
    for sd in (3, 4, 5, 6):
        m = sel & (d["subdet"] == sd)
        out(_fmt(robust_stats(d["pullx"][m]), SUBDETS[sd]))
    out(GAUS)

    out("")
    out(f"== {title}: STRIPS, by cluster width N (the CPE's branch variable) ==")
    out("  N<=4 uses the 3-parameter uProj form; N>4 the 2-parameter linear one.")
    out(HDR)
    strip = sel & (~d["ispixel"])
    for n in (1, 2, 3, 4):
        m = strip & (d["clusterSizeX"] == n)
        out(_fmt(robust_stats(d["pullx"][m]), f"N = {n}"))
    for lo, hi in ((5, 6), (7, 9), (10, 99)):
        m = strip & (d["clusterSizeX"] >= lo) & (d["clusterSizeX"] <= hi)
        out(_fmt(robust_stats(d["pullx"][m]), f"N = {lo}-{hi}"))

    out("")
    out(f"== {title}: STRIPS, by uProj (projected path in pitch units) ==")
    out(HDR)
    edges = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 99.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = strip & (d["hitUProj"] >= lo) & (d["hitUProj"] < hi)
        out(_fmt(robust_stats(d["pullx"][m]), f"uProj {lo:.1f}-{hi:.1f}"))

    out("")
    out(f"== {title}: STRIPS, across the dQdx > maxChgOneMIP branch switch ==")
    out("  above 6000 ADC/cm the CPE abandons the tuned form for the legacy")
    out("  3-constant one. This is one of only two ways the hit error model")
    out("  can depend on the particle species at all.")
    out(HDR)
    for lab, m in (("dQdx < 3000", strip & (d["dQdx"] < 3000)),
                   ("dQdx 3000-6000", strip & (d["dQdx"] >= 3000) & (d["dQdx"] < 6000)),
                   ("dQdx 6000-9000", strip & (d["dQdx"] >= 6000) & (d["dQdx"] < 9000)),
                   ("dQdx > 9000 (legacy)", strip & (d["dQdx"] >= 9000))):
        out(_fmt(robust_stats(d["pullx"][m]), lab))
    out(f"  fraction of strip hits on the legacy branch (dQdx>6000): "
        f"{100*np.mean(d['dQdx'][strip] > 6000):.2f} %")

    out("")
    out(f"== {title}: STRIPS, N x uProj (the 2D parametrisation) ==")
    out(HDR)
    for n in (1, 2, 3, 4):
        for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 99.0)):
            m = strip & (d["clusterSizeX"] == n) & (d["hitUProj"] >= lo) & (d["hitUProj"] < hi)
            out(_fmt(robust_stats(d["pullx"][m]), f"N={n}  uProj {lo:.0f}-{hi:.0f}"))


def report_pixels(d, sel, title, out=print):
    pix = sel & d["ispixel"]
    out("")
    out(f"== {title}: PIXELS, local x and local y ==")
    out(HDR)
    for sd in (1, 2):
        m = sel & (d["subdet"] == sd)
        out(_fmt(robust_stats(d["pullx"][m]), f"{SUBDETS[sd]} x"))
        out(_fmt(robust_stats(d["pully"][m]), f"{SUBDETS[sd]} y"))

    out("")
    out(f"== {title}: PIXELS, by template charge bin (qbin) ==")
    out(HDR)
    out("  qbin 0 is the HIGHEST charge (fq > fbin[0]), i.e. the delta-ray end;")
    out("  4 and 5 are the low-charge flags.")
    for q in (0, 1, 2, 3, 4, 5):
        m = pix & (d["clusterChargeBin"] == q)
        out(_fmt(robust_stats(d["pullx"][m]), f"qbin {q}  x"))
        out(_fmt(robust_stats(d["pully"][m]), f"qbin {q}  y"))

    out("")
    out(f"== {title}: PIXELS, by hit class ==")
    out("  clusterOnEdge and sizeX==1 are the two hardcoded CPE escapes.")
    out(HDR)
    for lab, m in (("on edge", pix & (d["clusterOnEdge"] == 1)),
                   ("not on edge", pix & (d["clusterOnEdge"] == 0)),
                   ("sizeX == 1", pix & (d["clusterSizeX"] == 1)),
                   ("sizeX >= 2", pix & (d["clusterSizeX"] >= 2)),
                   ("sizeY == 1", pix & (d["clusterSizeY"] == 1)),
                   ("sizeY >= 2", pix & (d["clusterSizeY"] >= 2))):
        out(_fmt(robust_stats(d["pullx"][m]), f"{lab}  x"))
        out(_fmt(robust_stats(d["pully"][m]), f"{lab}  y"))

    out("")
    out(f"== {title}: PIXELS, by |local dx/dz| (template angle index) ==")
    out(HDR)
    ad = np.abs(d["localdxdz"])
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 99.0)):
        m = pix & (ad >= lo) & (ad < hi)
        out(_fmt(robust_stats(d["pullx"][m]), f"|dx/dz| {lo:.1f}-{hi:.1f} x"))


def report_overall(d, sel, title, out=print):
    out("")
    out(f"== {title}: OVERALL ==")
    out(f"  files {int(d['nfiles'])}   hits {int(sel.sum())}   "
        f"sim-matched {100*np.mean(d['matched'][sel]):.2f} %")
    out(HDR)
    out(_fmt(robust_stats(d["pullx"][sel & d["ispixel"]]), "pixel  x"))
    out(_fmt(robust_stats(d["pully"][sel & d["ispixel"]]), "pixel  y"))
    out(_fmt(robust_stats(d["pullx"][sel & (~d["ispixel"])]), "strip  x/phi"))
    out(GAUS)
    out("")
    out("  TWO prescriptions, and they disagree. A LIKELIHOOD wants the core;")
    out("  an optimal LINEAR weighting -- and any second-moment estimator such")
    out("  as the 2022 parmtype 8/9 -- wants the variance. The gap between them")
    out("  is the size of the non-Gaussianity, and it is why one number cannot")
    out("  serve both:")
    out(f"    {'':<12} {'core^2':>9} {'rmsT^2':>9} {'raw rms^2':>12}   "
        f"(all = the exp(parm) each would converge to)")
    for lab, m in (("pixel x", sel & d["ispixel"]),
                   ("pixel y", sel & d["ispixel"]),
                   ("strip x/phi", sel & (~d["ispixel"]))):
        s = robust_stats(d["pully" if lab == "pixel y" else "pullx"][m])
        if s.get("n", 0) > 20:
            out(f"    {lab:<12} {s['core']**2:9.3f} {s['rms_trim']**2:9.3f} "
                f"{s['rms']**2:12.1f}")
    out("")
    out("  core^2 is to be compared with the dead-code hypothesis at")
    out("  ResidualGlobalCorrectionMakerG4e.cc: 0.8 pixel / 1.2 strip.")
    out("  The raw rms^2 column is not a candidate for anything -- it is the")
    out("  pathological >10-core component, and it is shown only to make the")
    out("  point that an UNPROTECTED variance fit has nothing to converge to.")


def select(d, args):
    sel = d["matched"] & np.isfinite(d["pullx"])
    if args.maxchi2:
        sel &= d["chisq_ndof"] < args.maxchi2
    if args.ptmin:
        sel &= d["genPt"] > args.ptmin
    if args.ptmax:
        sel &= d["genPt"] < args.ptmax
    if args.etamax:
        sel &= np.abs(d["genEta"]) < args.etamax
    return sel


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="mugun_lowpt")
    p.add_argument("--subdir", default="hitres")
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--maxchi2", type=float, default=0.)
    p.add_argument("--ptmin", type=float, default=0.)
    p.add_argument("--ptmax", type=float, default=0.)
    p.add_argument("--etamax", type=float, default=0.)
    p.add_argument("--outdir", default="")
    p.add_argument("--npz", default="", help="also cache the flattened arrays here")
    args = p.parse_args()

    today = datetime.date.today().strftime("%y%m%d")
    outdir = args.outdir or os.path.expanduser(
        f"~/public_html/calibration_studies/{today}_hitres/")
    os.makedirs(outdir, exist_ok=True)
    txt = os.path.join(outdir, f"pull_{args.tag}.txt")
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    d = load(args.tag, args.nfiles, args.subdir)
    sel = select(d, args)
    out(f"# hit pull study, sample {args.tag} ({args.subdir})")
    out(f"# selection: chi2/ndof<{args.maxchi2 or 'inf'}  "
        f"pT {args.ptmin or 0}-{args.ptmax or 'inf'}  |eta|<{args.etamax or 'inf'}")
    report_overall(d, sel, args.tag, out)
    report_strips(d, sel, args.tag, out)
    report_pixels(d, sel, args.tag, out)
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"wrote {txt}")
    if args.npz:
        np.savez_compressed(args.npz, **{k: v for k, v in d.items()
                                         if isinstance(v, np.ndarray)})
        logger.info(f"cached arrays to {args.npz}")


if __name__ == "__main__":
    main()
