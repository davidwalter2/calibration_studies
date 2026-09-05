#!/usr/bin/env python3
"""Candidate-level comparison of the WITH- and WITHOUT-clamp productions.

Prints the tables the 2026-09-04 clamp-fix NOTES entry needs:
  chi2/ndof quantiles and tail fractions, frozen fraction, niter at the cap,
  the soft-daughter (ptmin < 2 / 0.5 GeV) fractions and their mass pull,
  and the odd moment <z e^{-0.05 z^2}> split by fit quality.

usage: python3 clampfix_summary_260904.py [gun|v3] ...
"""
import os
import sys

import numpy as np

U = 0.05
SIGMAX = 0.15
SPECS = {
    "gun": [("floor 2.0  (_260903x)", "runs/censoring260904/aux_jpsigun.npz",
             "runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz"),
            ("floor 0.25 (_260904f)", "runs/clampfix260904/aux_jpsigun_260904f.npz",
             "runs/cf_masspairs_jpsigun_ul16_260904f_m0.npz")],
    "v3": [("floor 2.0  (_260903x)", "runs/censoring260904/aux_btojpsix.npz",
            "runs/cf_masspairs_btojpsix_v3_260903x_m0.npz"),
           ("floor 0.25 (_260904f)", "runs/clampfix260904/aux_btojpsix_260904f.npz",
            "runs/cf_masspairs_btojpsix_v3_260904f_m0.npz")],
}


def load(aux, cache):
    d, c = np.load(aux), np.load(cache)
    z = d["z"].astype(np.float64)
    s = d["sigma"].astype(np.float64)
    ok = np.isfinite(s) & (s > 0) & (s < SIGMAX) & np.isfinite(z)
    return dict(z=z[ok], sig=s[ok], mgen=c["eta"].astype(np.float64)[ok],
                m=(z * s + c["eta"].astype(np.float64))[ok],
                c2=d["normchi2"].astype(np.float64)[ok],
                it=d["niter"].astype(np.float64)[ok],
                fz=(d["frozen"].astype(np.float64) > 0.5)[ok],
                ptmin=d["ptmin"].astype(np.float64)[ok],
                pmin=d["pmin"].astype(np.float64)[ok],
                n_all=len(z), n_sane=int(ok.sum()))


def odd(z):
    return float(np.mean(z * np.exp(-U * z ** 2))) if len(z) else np.nan


for tag in (sys.argv[1:] or list(SPECS)):
    rows = [(lab, a, c) for lab, a, c in SPECS[tag]
            if os.path.exists(a) and os.path.exists(c)]
    if not rows:
        print(f"[skip] {tag}: no caches"); continue
    print(f"\n################ {tag} ################")
    V = {}
    for lab, a, c in rows:
        V[lab] = load(a, c)
    print(f"{'production':<24} {'n(all)':>9} {'n(sane)':>9} "
          f"{'q50':>8} {'q90':>10} {'q99':>11} {'>10':>8} {'>3':>8} "
          f"{'frozen':>8} {'niter=cap':>10}")
    for lab, d in V.items():
        print(f"{lab:<24} {d['n_all']:9d} {d['n_sane']:9d} "
              f"{np.quantile(d['c2'], .5):8.3f} {np.quantile(d['c2'], .9):10.4g} "
              f"{np.quantile(d['c2'], .99):11.4g} {np.mean(d['c2'] > 10):8.5f} "
              f"{np.mean(d['c2'] > 3):8.5f} {d['fz'].mean():8.5f} "
              f"{np.mean(d['it'] >= 10):10.5f}")
    print(f"\n{'production':<24} {'ptmin<2':>9} {'ptmin<0.5':>10} "
          f"{'<m-mgen>|pt<2':>15} {'<z>|pt<2':>10} {'oddm|pt<2':>11} "
          f"{'<m-mgen>|pt<0.5':>17} {'oddm|pt<0.5':>12}")
    for lab, d in V.items():
        s2 = d["ptmin"] < 2.; s5 = d["ptmin"] < 0.5
        print(f"{lab:<24} {s2.mean():9.5f} {s5.mean():10.5f} "
              f"{1e3*np.mean(d['m'][s2]-d['mgen'][s2]):15.4f} "
              f"{np.mean(d['z'][s2]):10.4f} {odd(d['z'][s2]):11.5f} "
              f"{1e3*np.mean(d['m'][s5]-d['mgen'][s5]) if s5.any() else np.nan:17.4f} "
              f"{odd(d['z'][s5]):12.5f}")
    print(f"\nodd moment <z e^{{-{U}z^2}}> by selection")
    print(f"{'production':<24} {'all':>10} {'chi2<3':>10} {'chi2<10':>10} "
          f"{'ptmin>2':>10} {'ptmin>3':>10} {'not frozen':>11}")
    for lab, d in V.items():
        print(f"{lab:<24} {odd(d['z']):10.5f} {odd(d['z'][d['c2'] < 3]):10.5f} "
              f"{odd(d['z'][d['c2'] < 10]):10.5f} {odd(d['z'][d['ptmin'] > 2]):10.5f} "
              f"{odd(d['z'][d['ptmin'] > 3]):10.5f} {odd(d['z'][~d['fz']]):11.5f}")
    print(f"\nmean pull <z> by selection")
    print(f"{'production':<24} {'all':>10} {'chi2<3':>10} {'ptmin>2':>10}")
    for lab, d in V.items():
        print(f"{lab:<24} {np.mean(d['z']):10.4f} {np.mean(d['z'][d['c2'] < 3]):10.4f} "
              f"{np.mean(d['z'][d['ptmin'] > 2]):10.4f}")
