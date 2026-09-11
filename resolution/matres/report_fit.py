#!/usr/bin/env python3
"""Pulls, errors and degeneracy structure of a material-parameterised fit.

Reads one or more rabbit ``fitresults.hdf5`` and prints, split by parameter
family (``bfield_mode*`` / ``material_*`` / ``hitres_*`` / the rest):

* value, error and pull against the truth (0 on MC, or the injected vector
  read from the card's ``global_index_map`` bundle);
* the error RATIO between two fits, which is how "quadratic only vs mass only
  vs joint" is read;
* the eigen-decomposition of the covariance restricted to a family: which
  combinations are measured and which are not.

Usage::

    python report_fit.py --fit fits/joint/fitresults.hdf5 --card cards/joint.hdf5
    python report_fit.py --fit fits/joint/fitresults.hdf5 \\
        --compare quad=fits/quad/fitresults.hdf5 mass=fits/mass/fitresults.hdf5
"""

import argparse
import json
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", required=True)
    p.add_argument("--card", default=None,
                   help="datacard, for the injected vector and the units")
    p.add_argument("--compare", nargs="*", default=[], metavar="LABEL=PATH")
    p.add_argument("--eig", action="store_true",
                   help="eigen-analysis of the material block's covariance")
    p.add_argument("--top", type=int, default=60)
    return p.parse_args()


def read_fit(path):
    from rabbit import io_tools

    fr = io_tools.get_fitresult(path)
    h = fr["parms"].get()
    names = [str(s) for s in np.array(h.axes["parms"])]
    val = np.asarray(h.values(), np.float64)
    err = np.sqrt(np.asarray(h.variances(), np.float64))
    cov = np.asarray(fr["cov"].get().values(), np.float64) if "cov" in fr else None
    extra = {}
    for k in ("nllvalfull", "nllvalreduced", "edmval"):
        if k in fr:
            try:
                extra[k] = float(np.asarray(fr[k].get()))
            except Exception:  # noqa: BLE001
                pass
    return names, val, err, cov, extra


def read_card(path):
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        g = f.get("auxiliary/global_index_map")
        if g is None:
            return out
        for k in g.keys():
            v = g[k][...]
            if v.dtype.kind in "SO":
                out[k] = [s.decode() if isinstance(s, bytes) else str(s) for s in v]
            else:
                out[k] = np.asarray(v)
    return out


FAMILIES = (("field modes", lambda n: n.startswith("bfield_mode")),
            ("material groups", lambda n: n.startswith("material_")),
            ("hit classes", lambda n: n.startswith("hitres_")),
            ("other", lambda n: True))


def main():
    args = parse_args()
    names, val, err, cov, extra = read_fit(args.fit)
    truth = np.zeros(len(names))
    card = read_card(args.card) if args.card else {}
    if card and "injected" in card and "params" in card:
        inj = dict(zip(card["params"], card["injected"]))
        for i, n in enumerate(names):
            truth[i] = inj.get(n, 0.0)
        if np.any(truth):
            print("injected truth: " + ", ".join(
                f"{n}={truth[i]:+.6g}" for i, n in enumerate(names) if truth[i]))
    # the card may float a rescaled variable; `physical = value * units`
    units = None
    if card and "params" in card and "units" in card:
        units = dict(zip(card["params"], np.asarray(card["units"])))
    elif card and "params" in card and "scale" in card:
        units = {n: 1.0 / max(sc, 1e-300)
                 for n, sc in zip(card["params"], np.asarray(card["scale"]))}
    others = {}
    for spec in args.compare:
        lab, path = spec.split("=", 1)
        others[lab] = read_fit(path)
    print(f"fit {args.fit}")
    for k, v in extra.items():
        print(f"  {k:<16} {v!r}")

    assigned = set()
    for label, pred in FAMILIES:
        idx = [i for i, n in enumerate(names)
               if n not in assigned and pred(n)]
        idx = [i for i in idx if names[i] not in assigned]
        if not idx:
            continue
        for i in idx:
            assigned.add(names[i])
        pulls = (val[idx] - truth[idx]) / np.maximum(err[idx], 1e-300)
        print(f"\n=== {label} ({len(idx)}) ===")
        print(f"  pulls: rms {np.sqrt((pulls**2).mean()):.3f}  "
              f"max |{np.abs(pulls).max():.3f}| at "
              f"{names[idx[int(np.argmax(np.abs(pulls)))]]}   "
              f"mean {pulls.mean():+.3f}")
        hdr = f"  {'parameter':<28} {'value':>13} {'error':>12} {'pull':>7}"
        if units is not None:
            hdr += f" {'k_phys':>11} {'+-':>10}"
            if label == "material groups":
                hdr += f" {'dmat [%]':>10}"
        for lab in others:
            hdr += f" {'err/'+lab:>12}"
        print(hdr)
        order = sorted(idx, key=lambda i: -abs(pulls[idx.index(i)]))[: args.top]
        for i in order:
            j = idx.index(i)
            line = (f"  {names[i]:<28} {val[i]:+13.6f} {err[i]:12.6f} "
                    f"{pulls[j]:+7.2f}")
            if units is not None:
                u = units.get(names[i], 1.0)
                kp, ke = val[i] * u, err[i] * u
                line += f" {kp:+11.5f} {ke:10.5f}"
                if label == "material groups":
                    line += f" {100.*(np.exp(kp)-1.):+10.2f}"
            for lab, (n2, v2, e2, _, _) in others.items():
                if names[i] in n2:
                    k = n2.index(names[i])
                    line += f" {err[i]/max(e2[k],1e-300):12.4f}"
                else:
                    line += f" {'-':>12}"
            print(line)

    if args.eig and cov is not None:
        for label, pred in FAMILIES[:2]:
            idx = [i for i, n in enumerate(names) if pred(n)]
            if len(idx) < 2:
                continue
            C = cov[np.ix_(idx, idx)]
            w, V = np.linalg.eigh(C)
            print(f"\n=== {label}: covariance eigen-analysis ===")
            print(f"  sigma per eigenmode (largest first): " + "  ".join(
                f"{np.sqrt(max(x,0)):.4g}" for x in w[::-1][:8]))
            print(f"  cond {w.max()/max(w.min(),1e-300):.3e}")
            for k in range(min(3, len(w))):
                v = V[:, -1 - k]
                o = np.argsort(-np.abs(v))[:6]
                print(f"  worst-measured #{k+1} (sigma {np.sqrt(max(w[-1-k],0)):.4g}): "
                      + "  ".join(f"{names[idx[i]].split('_',1)[1]} {v[i]:+.3f}"
                                  for i in o))


if __name__ == "__main__":
    main()
