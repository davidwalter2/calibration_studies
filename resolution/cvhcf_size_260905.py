#!/usr/bin/env python3
"""Per-branch and per-candidate volume of a CVH maker output.

The slimming switch is only worth anything if the saving is MEASURED on the
file rather than estimated from strides, so this reports the compressed bytes
ROOT actually wrote, grouped into

  cf       the in-maker resolution-CF exponents (cfqop_* / cfmass_*)
  raw      the per-step export they replace (ioniurbanv, msmoliv, radstepv,
           radstepspecv, reseigv, resinfv, resinfbv)
  keep     the small per-block columns that stay either way (reseigidx,
           reshitidx, resinfvarv, resinfcov, ioniqscale*, radvgrid, ...)
  other    everything else in the tree (hits, gradients, Hessian, kinematics)

usage:
  python3 cvhcf_size_260905.py <file.root> [<file.root> ...]
"""
import os
import sys

import uproot

RAW = {"ioniurbanidx", "ioniurbanv", "msmoliidx", "msmoliv",
       "radstepidx", "radstepv", "radstepspecv",
       "reseigv", "resinfv", "resinfbv"}
KEEP = {"reseigidx", "reshitidx", "resinfvarv", "resinfcov",
        "ioniqscaleidx", "ioniqscalev", "radvgrid", "radstepstride",
        "radstepnv"}


def group(name):
    if name.startswith("cfqop_") or name.startswith("cfmass_"):
        return "cf"
    if name in RAW:
        return "raw"
    if name in KEEP:
        return "keep"
    return "other"


def main():
    for fn in sys.argv[1:]:
        f = uproot.open(fn)
        t = f["tree"]
        n = t.num_entries
        tot = {"cf": 0, "raw": 0, "keep": 0, "other": 0}
        per = {}
        for b in t.branches:
            try:
                nb = int(b.compressed_bytes)
            except Exception:
                nb = 0
            g = group(b.name)
            tot[g] += nb
            per[b.name] = (g, nb)
        allb = sum(tot.values())
        size = os.path.getsize(fn)
        print(f"\n{fn}")
        print(f"  {n} entries, file {size/1e6:.2f} MB, branches {allb/1e6:.2f} MB")
        print(f"  {'group':<8} {'MB':>10} {'B/entry':>10}")
        for g in ("cf", "raw", "keep", "other"):
            print(f"  {g:<8} {tot[g]/1e6:>10.3f} {tot[g]/max(n,1):>10.1f}")
        print(f"  {'TOTAL':<8} {size/1e6:>10.3f} {size/max(n,1):>10.1f}")
        print(f"  top branches:")
        for name, (g, nb) in sorted(per.items(), key=lambda kv: -kv[1][1])[:14]:
            print(f"    {name:<26} {g:<6} {nb/max(n,1):>10.1f} B/entry")


if __name__ == "__main__":
    main()
