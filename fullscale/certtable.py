#!/usr/bin/env python3
"""The CERTIFIED table: every fit of every card, and which row may be quoted.

The acceptance test of STATE sec. 0f.16, applied mechanically to EVERY stored
result rather than to a chosen one:

1. **the value** -- what the fit returns;
2. **the NLL** -- the same minimum? Every fit of a card is compared with the
   LOWEST NLL any fit of that card has reached. A fit more than `--nll-tol`
   above it is at a different (worse) stationary point, or is not at one;
3. **the EDM** -- `0.5 g^T H^-1 g < --edm-tol` (1e-3 is 0.045 sigma in the
   worst direction);
4. **the full Newton step of each POI in units of its own error** -- the
   interpretable form of (3), and the one that catches sec. 0f.19's failure
   mode: a fit that never moved sits at its start, which in every closure test
   here is the MC truth, and so reads as a perfect closure.

Nothing is re-fitted. The verdict is BEST when a row passes all four and is
the lowest NLL of its card; `no` otherwise, with the reason.

    python3 certtable.py              # the per-card ledger + the summary
    python3 certtable.py --summary    # the summary alone
"""
import argparse
import glob
import json
import os

import numpy as np

FS = os.path.dirname(os.path.abspath(__file__))
POIS = ("m_Z", "Gamma_Z")

# card -> (label, form).  A card missing here is still listed, unlabelled.
CARDS = {
    "z_full380_fl":     ("inclusive, K(m) 5 terms",  "m"),
    "z_full380_fl_s6":  ("K(m) 6 terms",             "m"),
    "z_full380_fl_s7":  ("K(m) 7 terms",             "m"),
    "z_M_etaB":         ("|eta_lead| < 0.9",         "m"),
    "z_M_etaT":         ("0.9 - 1.6",                "m"),
    "z_M_etaE":         ("1.6 - 3.0",                "m"),
    "z_F_toy":          ("the assembly toy",         "m"),
    "z_F_dc8":          ("F_dc8, sigma-reweighted",  "m"),
    "z_F_toydc":        ("F_toydc",                  "m"),
    "z_F_w70110":       ("70-110 GeV window",        "m"),
    "z_V_full":         ("inclusive, K(m) 5 terms",  "v"),
    "z_V_s6":           ("K(m) 6 terms",             "v"),
    "z_V_s7":           ("K(m) 7 terms",             "v"),
    "z_V_etaB":         ("|eta_lead| < 0.9",         "v"),
    "z_V_etaT":         ("0.9 - 1.6",                "v"),
    "z_V_etaE":         ("1.6 - 3.0",                "v"),
    "z_V_toy":          ("the assembly toy",         "v"),
    "z_VK_etaB":        ("per-band FSR kernel, B",   "v"),
    "z_VK_etaT":        ("per-band FSR kernel, T",   "v"),
    "z_full380_fl_s9":  ("K(m) 9 terms",              "m"),
    "z_full380_fl_s12": ("K(m) 12 terms",             "m"),
    "z_V_s9":           ("K(m) 9 terms",              "v"),
    "z_V_s12":          ("K(m) 12 terms",             "v"),
}
ORDER = ["z_full380_fl", "z_full380_fl_s6", "z_full380_fl_s7",
         "z_full380_fl_s9", "z_full380_fl_s12",
         "z_M_etaB", "z_M_etaT", "z_M_etaE",
         "z_F_toy", "z_F_dc8", "z_F_toydc", "z_F_w70110",
         "z_V_full", "z_V_s6", "z_V_s7", "z_V_s9", "z_V_s12",
         "z_V_etaB", "z_V_etaT", "z_V_etaE", "z_V_toy",
         "z_VK_etaB", "z_VK_etaT"]

# a rabbit result carries no card name; this is the map
RABBIT_CARD = {
    "f380ref": "z_full380_fl", "f380refX": "z_full380_fl",
    "f380krylovfix": "z_full380_fl", "f380refS": "z_full380_fl",
    "Rdc8": "z_F_dc8", "Rdc8X": "z_F_dc8", "Sdc8": "z_F_dc8",
    "Rvfull": "z_V_full", "RvfullX": "z_V_full", "SVfull": "z_V_full",
    "Rvtoy": "z_V_toy", "SVtoy": "z_V_toy",
    "Rs6": "z_full380_fl_s6", "Ss6": "z_full380_fl_s6",
    "Rs7": "z_full380_fl_s7", "Ss7": "z_full380_fl_s7",
    "SVs6": "z_V_s6", "SVs7": "z_V_s7",
    "SVetaB": "z_V_etaB", "SVetaT": "z_V_etaT", "SVetaE": "z_V_etaE",
    "SVKetaB": "z_VK_etaB", "SVKetaT": "z_VK_etaT",
    "SMetaB": "z_M_etaB", "SMetaT": "z_M_etaT", "SMetaE": "z_M_etaE",
    "Stoy": "z_F_toy", "Stoydc": "z_F_toydc", "Sw70110": "z_F_w70110",
    # the warm twins of the three cold controls, and the preconditioning control
    "f380refW": "z_full380_fl", "SVfullW": "z_V_full", "Sdc8W": "z_F_dc8",
    "f380refP": "z_full380_fl",
    # the extended K(m) ladder
    "Ss9": "z_full380_fl_s9", "SVs9": "z_V_s9",
    "Ss12": "z_full380_fl_s12", "SVs12": "z_V_s12",
}


def rows():
    """Every stored fit, as dicts with a common schema."""
    out = []
    # (a) `fit.py` json results
    for p in sorted(glob.glob(f"{FS}/results/eng/fit_*.json")) + \
             sorted(glob.glob(f"{FS}/results/fit_*.json")):
        d = json.load(open(p))
        if "fitted" not in d or "params" not in d:
            continue
        card = os.path.basename(d.get("card", "")).replace(".hdf5", "")
        nm = d["params"]
        sw = d.get("sandwich_err") or d["err"]
        ratio = d.get("sandwich_ratio")
        r = {"src": "fit.py", "tag": os.path.basename(p)[4:-5], "card": card,
             "swratio": (dict(zip(nm, ratio)) if ratio else None),
             "nll": d.get("nll"), "edm": d.get("edm"), "nit": d.get("nit"),
             "method": d.get("method") or "trust-exact",
             "step": d.get("worst_poi_step_sigma"), "n": d.get("n"),
             # the fit-time model switches. `fit.py` turns one card into a scan
             # (--ares / --jensen / --corr-clip / --fix), so two results of the
             # same card are only comparable in NLL when these agree -- e.g.
             # `f380fl_noboth` sits 2512 NLL units BELOW `f380fl_base` on
             # exactly the same candidates because it is a different likelihood
             "model": (d.get("jensen_mode"), bool(d.get("fluct_active")),
                       float(d.get("corr_clip") or 0.0),
                       tuple(sorted(d.get("fixed") or [])), d.get("n"))}
        for q in POIS:
            if q in nm:
                r[q] = (d["fitted"][nm.index(q)], sw[nm.index(q)])
        # a verdict computed after the fact by checkconv.py
        cv = f"{FS}/results/conv_" + os.path.basename(p)
        if os.path.exists(cv):
            c = json.load(open(cv))
            r["edm"] = c.get("edm", r["edm"])
            r["step"] = c.get("worst_poi_sigma", r["step"])
        out.append(r)
    # (b) `rabbit_fit.py` results, via native_dump.py
    for p in sorted(glob.glob(f"{FS}/results/nativejson/rabbit_*.json")):
        d = json.load(open(p))
        tag = os.path.basename(p)[len("rabbit_"):-5]
        nm = d["params"]
        r = {"src": "rabbit", "tag": tag, "card": RABBIT_CARD.get(tag, ""),
             "swratio": None,
             "nll": d.get("nllvalreduced"), "edm": d.get("edmval"),
             "nit": None, "method": "rabbit", "step": None, "n": None,
             "model": None}  # the card's own defaults
        for q in POIS:
            if q in nm:
                r[q] = (d["fitted"][nm.index(q)], d["err"][nm.index(q)])
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edm-tol", type=float, default=1e-3)
    ap.add_argument("--nll-tol", type=float, default=0.01)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    R = rows()
    # the DEFAULT model of each card: what a `rabbit_fit.py` run of it does.
    # Taken from the `fit.py` rows themselves -- the modal model over the rows
    # of that card, which is the un-switched one.
    default = {}
    for r in R:
        if r["model"] is not None:
            default.setdefault(r["card"], {}).setdefault(r["model"], 0)
            default[r["card"]][r["model"]] += 1
    default = {c: max(v, key=v.get) for c, v in default.items()}
    for r in R:
        if r["model"] is None:
            r["model"] = default.get(r["card"])
        r["is_default"] = (r["model"] == default.get(r["card"]))
    by = {}
    for r in R:
        by.setdefault(r["card"], []).append(r)

    # A `rabbit_fit.py` row carries the INVERSE-HESSIAN error; the table quotes
    # the sandwich, and the factor is not flat (1.088 to 1.177 across these
    # cards). Where a `fit.py` row of the same card sits at the SAME point --
    # which is what a warm start seeded from that row and converged there
    # means -- its measured `sandwich_ratio` is transported onto the rabbit
    # row and the row is marked `~`. Where no such row exists the rabbit row
    # keeps its Hessian error and is marked `H`; it needs `sandwich.sbatch`
    # before it is quoted.
    for c, v in by.items():
        donors = [x for x in v if x.get("swratio")]
        for r in v:
            if r["src"] != "rabbit" or not donors:
                continue
            near = [x for x in donors
                    if "m_Z" in x and "m_Z" in r
                    and abs(x["m_Z"][0] - r["m_Z"][0]) < 0.01
                    and x["nll"] is not None and r["nll"] is not None
                    and abs(x["nll"] - r["nll"]) < 0.01]
            if not near:
                r["errkind"] = "H"
                continue
            k = near[0]["swratio"]
            for q in POIS:
                if q in r and q in k:
                    r[q] = (r[q][0], r[q][1] * k[q])
            r["errkind"] = "~"
    for r in R:
        r.setdefault("errkind", "s" if r["src"] == "fit.py" else "H")

    # the reference NLL of a card is the best over its DEFAULT-model fits only
    best = {c: min((x["nll"] for x in v
                    if x["nll"] is not None and x["is_default"]), default=None)
            for c, v in by.items()}

    def verdict(r):
        why = []
        if r["edm"] is None or not np.isfinite(r["edm"]):
            why.append("no EDM")
        elif r["edm"] > a.edm_tol:
            why.append(f"EDM {r['edm']:.3g}")
        b = best.get(r["card"])
        if r["nll"] is not None and b is not None and r["nll"] > b + a.nll_tol:
            why.append(f"NLL +{r['nll'] - b:.4g}")
        if r.get("step") is not None and abs(r["step"]) > 0.045:
            why.append(f"POI step {r['step']:.3g} sig")
        if not r["is_default"]:
            why.append("model switched")
        return ("QUOTE" if not why else "no"), "; ".join(why)

    if not a.summary:
        print("PER-CARD LEDGER -- every stored fit, and the four-part test\n")
        for card in ORDER + [c for c in by if c not in ORDER]:
            if card not in by:
                continue
            lab, form = CARDS.get(card, ("", "?"))
            print(f"== {card}   [{form} form]  {lab}")
            for r in sorted(by[card], key=lambda x: (x["nll"] is None, x["nll"])):
                if not r["is_default"]:
                    continue
                vd, why = verdict(r)
                mz = r.get("m_Z", (np.nan, np.nan))
                gz = r.get("Gamma_Z", (np.nan, np.nan))
                print(f"   {r['src']:7s} {r['tag']:22s} "
                      f"{mz[0]:+8.2f} +-{mz[1]:5.2f} {gz[0]:+8.2f} +-{gz[1]:5.2f} "
                      f"{r['nll'] if r['nll'] is not None else np.nan:18.4f} "
                      f"{r['edm'] if r['edm'] is not None else np.nan:10.2e}  "
                      f"{vd}{(' (' + why + ')') if why else ''}")
            print()

    print("\nCERTIFIED TABLE -- the best QUOTABLE fit of each card\n")
    hdr = (f"{'row':28s} {'form':4s} {'from':8s}{'m_Z [MeV]':>16s} "
           f"{'Gamma_Z [MeV]':>16s} {'NLL':>18s} {'EDM':>9s} {'POI step':>9s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for card in ORDER:
        if card not in by:
            print(f"{CARDS[card][0]:28s} {CARDS[card][1]:4s} {'(no fit of ' + card + ')'}")
            continue
        ok = [r for r in by[card] if verdict(r)[0] == "QUOTE"]
        pick = (min(ok, key=lambda x: x["nll"]) if ok
                else min((r for r in by[card]
                          if r["nll"] is not None and r["is_default"]),
                         key=lambda x: x["nll"], default=None))
        if pick is None:
            continue
        vd, why = verdict(pick)
        mz, gz = pick.get("m_Z", (np.nan,) * 2), pick.get("Gamma_Z", (np.nan,) * 2)
        st = pick.get("step")
        print(f"{CARDS[card][0]:28s} {CARDS[card][1]:4s} "
              f"{pick['src'] + pick['errkind']:8s}"
              f"{mz[0]:+8.2f} +-{mz[1]:5.2f} {gz[0]:+8.2f} +-{gz[1]:5.2f} "
              f"{pick['nll']:18.4f} {pick['edm'] if pick['edm'] is not None else np.nan:9.2e} "
              f"{(f'{st:9.1e}' if st is not None else '        -')}  "
              f"{'QUOTABLE' if vd == 'QUOTE' else 'NOT QUOTABLE (' + why + ')'}")
    print("\nOffsets in MeV from the generator: m_Z = 91.153509740726733 GeV, "
          "Gamma_Z = 2.4932018986110700 GeV.")
    print("Error column: `s` = the measured sandwich; `~` = the sandwich RATIO "
          "transported from a\n`fit.py` fit of the same card at the same point "
          "(same m_Z to 0.01 MeV and same NLL to 0.01);\n`H` = the "
          "inverse-Hessian error only -- that row still needs `sandwich.sbatch`.")


if __name__ == "__main__":
    main()
