#!/usr/bin/env python3
"""Per-file census of the CVH resolution productions: how many of the THROWN
particles/candidates reach the ntuple, and what the survivors look like.

Written for the 2026-09-04 selection-censoring test.  The question it answers
is the first one any censoring hypothesis has to pass: is anything actually
being lost, and is what is lost correlated with the energy loss?

  --single   one entry per RECO TRACK (mugun, piongun, ...): thrown = evt/file
             x n_pdg, so the loss fraction is 1 - n_tree/(nev*npart).  Also
             reports the fraction with an unmatched gen (`genPt < 0`), the
             niter cap, and the low quantiles of trackPt/genPt (the only place
             a large-loss track could be truncated).
  --twotrack one entry per CANDIDATE (jpsigun, btojpsix): thrown = evt/file,
             and the survivors' mass/pT/chi2 tails say whether the maker cuts
             on the OBSERVED mass (it does not: see the note in NOTES.md).

The counts are the whole point, so nothing is subsampled and the branches read
are scalars only (the jagged CF exports are never touched).
"""
import argparse, glob, os, sys
import numpy as np
import uproot
import prodfiles

SC = ["run", "lumi", "event"]


def single(files, nev, npart, log):
    tot = nun = ncap = 0
    ev = []
    rows = []
    q = []
    for fn in files:
        t = uproot.open(fn)["tree"]
        a = t.arrays(["event", "genPt", "trackPt", "normalizedChi2",
                      "nValidHits", "niter", "genDR", "genCharge"],
                     library="np")
        n = len(a["event"])
        tot += n
        nun += int((a["genPt"] <= 0).sum())
        ncap += int((a["niter"] >= 10).sum())
        u, c = np.unique(a["event"], return_counts=True)
        ev.append(np.bincount(c, minlength=npart + 2)[: npart + 2])
        good = a["genPt"] > 0
        q.append(np.stack([a["trackPt"][good] / a["genPt"][good],
                           a["normalizedChi2"][good],
                           a["nValidHits"][good].astype(float),
                           a["genDR"][good],
                           a["genCharge"][good].astype(float)]))
        rows.append((os.path.basename(os.path.dirname(fn)), n, len(u)))
    ev = np.sum(ev, axis=0)
    q = np.concatenate(q, axis=1)
    thrown = len(files) * nev * npart
    log(f"files {len(files)}  thrown {thrown}  in tree {tot}  "
        f"LOSS {thrown - tot} = {100.*(thrown-tot)/thrown:.3f} %")
    log(f"  unmatched gen (genPt<=0) {nun} = {100.*nun/max(tot,1):.3f} % ; "
        f"niter>=10 (cap) {ncap} = {100.*ncap/max(tot,1):.3f} %")
    log(f"  tracks/event multiplicity {list(enumerate(ev))}")
    for nm, i in (("trackPt/genPt", 0), ("normchi2", 1), ("nValidHits", 2),
                  ("genDR", 3)):
        v = q[i]
        log(f"  {nm:>14s}  min {v.min():.4f}  q0.1% {np.quantile(v,1e-3):.4f}  "
            f"q1% {np.quantile(v,.01):.4f}  med {np.median(v):.4f}  "
            f"q99% {np.quantile(v,.99):.4f}  max {v.max():.4f}")
    r = q[0]
    for c in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
        log(f"  frac trackPt/genPt < {c:.2f}: {(r<c).mean():.3e}  "
            f"({int((r<c).sum())} tracks)")
    return dict(thrown=thrown, ntree=tot, nunmatched=nun, ncap=ncap, mult=ev,
                ratio=r, normchi2=q[1], nvalid=q[2], gendr=q[3], charge=q[4])


def twotrack(files, nev, log):
    tot = 0
    ncap = nnan = nunmatched = 0
    keep = {k: [] for k in ("m", "mgen", "sig", "c2n", "niter", "ptp", "ptm",
                            "ptpg", "ptmg", "etap", "etam", "nvp", "nvm",
                            "event")}
    for fn in files:
        t = uproot.open(fn)["tree"]
        a = t.arrays(["event", "Jpsi_mass", "Jpsigen_mass", "Jpsi_sigmamass",
                      "chisqval", "ndof", "niter", "Muplus_pt", "Muminus_pt",
                      "Muplusgen_pt", "Muminusgen_pt", "Muplus_eta",
                      "Muminus_eta", "Muplus_nvalid", "Muminus_nvalid"],
                     library="np")
        n = len(a["event"])
        tot += n
        ncap += int((a["niter"] >= 10).sum())
        s = a["Jpsi_sigmamass"]
        nnan += int((~np.isfinite(s)).sum() + (s <= 0).sum())
        nunmatched += int(((a["Muplusgen_pt"] <= 0)
                           | (a["Muminusgen_pt"] <= 0)).sum())
        keep["m"].append(a["Jpsi_mass"]); keep["mgen"].append(a["Jpsigen_mass"])
        keep["sig"].append(s); keep["c2n"].append(a["chisqval"] / a["ndof"])
        keep["niter"].append(a["niter"].astype(float))
        keep["ptp"].append(a["Muplus_pt"]); keep["ptm"].append(a["Muminus_pt"])
        keep["ptpg"].append(a["Muplusgen_pt"]); keep["ptmg"].append(a["Muminusgen_pt"])
        keep["etap"].append(a["Muplus_eta"]); keep["etam"].append(a["Muminus_eta"])
        keep["nvp"].append(a["Muplus_nvalid"].astype(float))
        keep["nvm"].append(a["Muminus_nvalid"].astype(float))
        keep["event"].append(a["event"].astype(float))
    K = {k: np.concatenate(v) for k, v in keep.items()}
    thrown = len(files) * nev
    log(f"files {len(files)}  thrown (gen J/psi) {thrown}  candidates {tot}  "
        f"LOSS {thrown-tot} = {100.*(thrown-tot)/thrown:.3f} %")
    log(f"  NaN/<=0 sigma_m {nnan} = {100.*nnan/tot:.3f} % ; "
        f"unmatched gen muon {nunmatched} = {100.*nunmatched/tot:.3f} % ; "
        f"niter>=10 {ncap} = {100.*ncap/tot:.3f} %")
    for nm, v in (("Jpsi_mass", K["m"]), ("Jpsigen_mass", K["mgen"]),
                  ("sigma_m", K["sig"]), ("chi2/ndof", K["c2n"]),
                  ("min mu pT", np.minimum(K["ptp"], K["ptm"])),
                  ("max |eta|", np.maximum(np.abs(K["etap"]),
                                           np.abs(K["etam"]))),
                  ("min nvalid", np.minimum(K["nvp"], K["nvm"]))):
        v = v[np.isfinite(v)]
        log(f"  {nm:>13s}  min {v.min():.4f}  q0.1% {np.quantile(v,1e-3):.4f}  "
            f"med {np.median(v):.4f}  q99.9% {np.quantile(v,1-1e-3):.4f}  "
            f"max {v.max():.4f}")
    return K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True)
    p.add_argument("--nev", type=int, required=True, help="events per file")
    p.add_argument("--npart", type=int, default=2, help="thrown particles/event")
    p.add_argument("--mode", choices=["single", "twotrack"], required=True)
    p.add_argument("--ntasks", type=int, default=100000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    # --ntasks caps TASKS (a numberOfThreads=N task is N stream files)
    files = prodfiles.resolve(a.files, a.ntasks, logger=lambda m: print(m, flush=True))
    if not files:
        sys.exit(f"no files match {a.files}")
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"# census_probe {a.mode}  {a.files}")
    r = (single(files, a.nev, a.npart, log) if a.mode == "single"
         else twotrack(files, a.nev, log))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out, **{k: v for k, v in r.items()
                                      if isinstance(v, np.ndarray)})
        with open(a.out.replace(".npz", ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        log(f"-> {a.out}")


if __name__ == "__main__":
    main()
