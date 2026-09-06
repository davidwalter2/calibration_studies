#!/usr/bin/env python3
"""Fast tree-level comparison of the WITH- and WITHOUT-clamp ditrack refits.

Reads the candidate trees directly (no pairs cache needed), so it gives the
chi2 / frozen / soft-daughter-mass answer as soon as the first tasks land.

usage: python3 clampfix_treecheck_260904.py --old <tag> --new <tag> [--ntasks N]
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import uproot
import prodfiles

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
BR = ["run", "lumi", "event", "Jpsi_mass", "Jpsigen_mass", "Jpsi_sigmamass",
      "Jpsikin_mass", "chisqval", "ndof", "niter",
      "Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta"]


def one(args):
    fn, itask = args
    try:
        a = uproot.open(fn)["tree"].arrays(BR, library="np")
    except Exception as e:
        print(f"[skip] {fn}: {type(e).__name__}")
        return None
    d = {k: np.asarray(v, dtype=np.float64) for k, v in a.items()}
    d["nc2"] = d["chisqval"] / np.maximum(d["ndof"], 1)
    d["ptmin"] = np.minimum(d["Muplus_pt"], d["Muminus_pt"])
    d["pmin"] = np.minimum(d["Muplus_pt"] * np.cosh(d["Muplus_eta"]),
                           d["Muminus_pt"] * np.cosh(d["Muminus_eta"]))
    d["frozen"] = (np.abs(d["Jpsi_mass"] - d["Jpsikin_mass"]) < 1e-6).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["z"] = (d["Jpsi_mass"] - d["Jpsigen_mass"]) / d["Jpsi_sigmamass"]
    d["task"] = np.full(len(d["event"]), float(itask))
    return {k: d[k] for k in
            ("task", "run", "lumi", "event", "Jpsi_mass", "Jpsigen_mass",
             "Jpsi_sigmamass", "nc2", "ptmin", "pmin", "frozen", "niter", "z")}


def done_tasks(tag, stem):
    """Task NUMBERS that are usable in this production (all streams present)."""
    return {int(os.path.basename(d).split("_")[1])
            for d in prodfiles.task_dirs(f"{CEPH}/resolution_trackres_{tag}", stem)
            if prodfiles.task_complete(d, stem)}


def load(tag, tasks, nproc, stem):
    # EVERY stream of each task: at numberOfThreads=N a task is N files, and
    # taking stream 0 would compare 1/N of one production with 1/N of another.
    files = [f for i in sorted(tasks)
             for f in prodfiles.stream_files(
                 f"{CEPH}/resolution_trackres_{tag}/task_{i:04d}", stem)]
    print(f"{tag}: {len(files)} stream files over {len(tasks)} complete tasks")
    with Pool(nproc) as p:
        parts = [q for q in p.map(one, [(f, int(os.path.basename(os.path.dirname(f))
                                              .split("_")[1])) for f in files])
                 if q is not None]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}, len(files)


def stats(nm, d):
    ok = np.isfinite(d["z"]) & np.isfinite(d["Jpsi_sigmamass"]) & (d["Jpsi_sigmamass"] > 0)
    sane = ok & (d["Jpsi_sigmamass"] < 0.15)
    c2 = d["nc2"]
    print(f"\n-- {nm}: n={len(c2)}  sane={int(sane.sum())} ({100.*sane.mean():.3f} %)")
    print("   chi2/ndof q50/q80/q87/q90/q95/q99/max = "
          + " ".join(f"{np.quantile(c2, q):.4g}" for q in (.5, .8, .87, .9, .95, .99, 1.)))
    print(f"   chi2>10 {np.mean(c2 > 10):.5f}  chi2>3 {np.mean(c2 > 3):.5f}  "
          f"frozen {d['frozen'].mean():.5f}  niter>=10 {np.mean(d['niter'] >= 10):.5f}")
    for cut, lab in ((2., "ptmin<2"), (0.5, "ptmin<0.5")):
        s = sane & (d["ptmin"] < cut)
        if s.sum():
            dm = d["Jpsi_mass"][s] - d["Jpsigen_mass"][s]
            z = d["z"][s]
            print(f"   {lab}: {100.*np.mean(d['ptmin'] < cut):6.3f} %  n={int(s.sum()):7d}  "
                  f"<m-mgen> {1e3*dm.mean():+8.3f} MeV  med {1e3*np.median(dm):+8.3f}  "
                  f"<z> {z.mean():+7.4f}  odd {np.mean(z*np.exp(-0.05*z**2)):+7.4f}  "
                  f"chi2>10 {np.mean(c2[s] > 10):.4f}")
    for lab, sel in (("all", sane), ("chi2<3", sane & (c2 < 3)),
                     ("ptmin>2", sane & (d["ptmin"] > 2))):
        z = d["z"][sel]
        print(f"   odd moment [{lab:>8}] n={int(sel.sum()):7d}  <z> {z.mean():+7.4f}  "
              f"<z e^-0.05z^2> {np.mean(z*np.exp(-0.05*z**2)):+7.5f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", default="jpsigun_ul16_260903x_m0")
    p.add_argument("--new", default="jpsigun_ul16_260904f_m0")
    p.add_argument("--ntasks", type=int, default=20)
    p.add_argument("--nproc", type=int, default=20)
    p.add_argument("--fname", default="globalcor",
                   help="output STEM (globalcor / globalcor_resclosure); every "
                        "stream of the task is read")
    a = p.parse_args()
    common_tasks = sorted(done_tasks(a.old, a.fname) & done_tasks(a.new, a.fname))
    if a.ntasks > 0:
        common_tasks = common_tasks[: a.ntasks]
    print(f"common complete tasks: {len(common_tasks)}")
    A, na = load(a.old, common_tasks, a.nproc, a.fname)
    B, nb = load(a.new, common_tasks, a.nproc, a.fname)
    stats(f"{a.old} ({na} tasks)", A)
    stats(f"{a.new} ({nb} tasks)", B)
    # paired comparison on the (event, gen-mass) key -- the RECO pt changes with
    # the fit, so it cannot be part of the key
    def key(D):
        return (D["task"].astype(np.int64) * 10**13
                + D["event"].astype(np.int64) * 10**7
                + np.round(D["Jpsigen_mass"] * 1e6).astype(np.int64))
    ka, kb = key(A), key(B)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    print(f"\npaired candidates: {len(common)}")
    if len(common):
        ch = A["Jpsi_mass"][ia] != B["Jpsi_mass"][ib]
        print(f"   changed refit mass: {ch.sum()} ({100.*ch.mean():.3f} %)")
        s = ch
        if s.sum():
            print(f"   on the changed ones: chi2/ndof mean {A['nc2'][ia][s].mean():.4g} "
                  f"-> {B['nc2'][ib][s].mean():.4g};  "
                  f"<m-mgen> {1e3*np.mean(A['Jpsi_mass'][ia][s]-A['Jpsigen_mass'][ia][s]):+.3f} "
                  f"-> {1e3*np.mean(B['Jpsi_mass'][ib][s]-B['Jpsigen_mass'][ib][s]):+.3f} MeV;  "
                  f"ptmin med {np.median(A['ptmin'][ia][s]):.3f} GeV")


if __name__ == "__main__":
    main()
