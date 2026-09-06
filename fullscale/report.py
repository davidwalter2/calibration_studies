#!/usr/bin/env python3
"""Turn the per-variant fit JSONs into the phase tables.

Every row is the same card and the same candidates with ONE thing changed, so
the differences are the modelling effects themselves and carry no statistical
scatter between rows beyond what the change itself does.
"""
import argparse
import glob
import json
import os


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            r = json.load(fh)
        out[r.get("label") or os.path.basename(f)] = r
    return out


def get(r, name, key="fitted"):
    if name not in r["params"]:
        return None
    return r[key][r["params"].index(name)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/fit_*.json")
    p.add_argument("--order", nargs="*", default=None)
    p.add_argument("--pois", nargs="*", default=["m_Z", "Gamma_Z"])
    p.add_argument("--ref", default="base", help="row the deltas are taken from")
    a = p.parse_args()
    res = load(a.results)
    if not res:
        raise SystemExit(f"nothing matched {a.results}")
    order = [k for k in (a.order or res) if k in res]

    print("\n## Fitted minus generator  [MeV]\n")
    hdr = f"| {'variant':<12s} | {'n':>9s} |"
    for poi in a.pois:
        hdr += f" {poi + ' fit-truth':>18s} | {'stat':>8s} |"
    print(hdr)
    print("|" + "|".join(["-" * 14, "-" * 11]
                         + ["-" * 20, "-" * 10] * len(a.pois)) + "|")
    for k in order:
        r = res[k]
        row = f"| {k:<12s} | {r['n']:9d} |"
        for poi in a.pois:
            v = get(r, poi)
            e = get(r, poi, "sandwich_err") or get(r, poi, "err")
            t = r.get("truth", {}).get(poi)
            if v is None:
                row += f" {'fixed':>18s} | {'-':>8s} |"
            else:
                d = v - (t or 0.0)
                row += f" {d:+18.3f} | {e:8.3f} |"
        print(row)

    if a.ref in res:
        print(f"\n## Effect of each change, relative to `{a.ref}`  [MeV]\n")
        base = res[a.ref]
        print(f"| {'variant':<12s} |" + "".join(
            f" {'d' + poi:>14s} |" for poi in a.pois)
            + f" {'dNLL':>12s} |")
        print("|" + "|".join(["-" * 14] + ["-" * 16] * len(a.pois) + ["-" * 14]) + "|")
        for k in order:
            if k == a.ref:
                continue
            r = res[k]
            row = f"| {k:<12s} |"
            for poi in a.pois:
                v, b = get(r, poi), get(base, poi)
                row += (f" {v - b:+14.3f} |" if (v is not None and b is not None)
                        else f" {'-':>14s} |")
            row += f" {r.get('nll', float('nan')) - base.get('nll', float('nan')):+12.3f} |"
            print(row)

    print("\n## Cost\n")
    print(f"| {'variant':<12s} | {'load':>7s} | {'grad':>7s} | {'hess':>8s} | "
          f"{'sandwich':>9s} | {'fit':>8s} | {'iter':>5s} | {'peak RSS':>9s} |")
    print("|" + "|".join(["-" * 14, "-" * 9, "-" * 9, "-" * 10, "-" * 11,
                          "-" * 10, "-" * 7, "-" * 11]) + "|")
    for k in order:
        r = res[k]
        print(f"| {k:<12s} | {r.get('t_load', 0):6.0f}s | {r.get('t_grad', 0):6.1f}s "
              f"| {r.get('t_hess', 0):7.1f}s | {r.get('t_sandwich', 0):8.1f}s | "
              f"{r.get('t_fit', 0):7.0f}s | {r.get('nit', 0):5d} | "
              f"{r.get('rss_gb', 0):8.1f}G |")

    for k in order:
        r = res[k]
        if "corr" not in r:
            continue
        print(f"\n## Correlations, `{k}`\n")
        nm = r["params"]
        print("| |" + "".join(f" {s:>10s} |" for s in nm))
        print("|" + "|".join(["-" * 12] * (len(nm) + 1)) + "|")
        for i, s in enumerate(nm):
            print(f"| {s:<10s} |" + "".join(
                f" {r['corr'][i][j]:10.4f} |" for j in range(len(nm))))
        break


if __name__ == "__main__":
    main()
