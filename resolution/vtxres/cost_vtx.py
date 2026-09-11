#!/usr/bin/env python3
"""Cost and export bill of the vertex term.

--bill     compressed bytes per candidate of every branch of a production
           file, grouped by what they belong to (the vertex block, the mass
           block, the shared influence vectors, the D rows, the rest), and
           what each costs at full scale (34 M J/psi + 7 M Z candidates).
--timing   NLL, NLL+gradient and one HVP on ONE PINNED CPU for a card, i.e.
           what one iteration of the joint fit costs per candidate.
"""
import argparse, os, sys, time
import numpy as np

FULLSCALE = 41e6   # 34 M J/psi + 7 M Z candidates

GROUPS = [
    ("vertex: the exponents (per group)", ("cfvtx_grp_ms", "cfvtx_grp_ioni_re",
                                           "cfvtx_grp_ioni_im", "cfvtx_grp_rad_re",
                                           "cfvtx_grp_rad_im", "cfvtx_grp",
                                           "cfvtx_grp_vqms", "cfvtx_grp_vqio",
                                           "cfvtx_grp_closure")),
    ("vertex: the exponents (flat)", ("cfvtx_ms", "cfvtx_del", "cfvtx_ioni_re",
                                      "cfvtx_ioni_im", "cfvtx_rad_re",
                                      "cfvtx_rad_im")),
    ("vertex: shares + scalars", ("vtxvarv", "vtxsgnv", "cfvtx_hitcls",
                                  "cfvtx_hitv", "Jpsi_vtxres", "Jpsi_vtxsig",
                                  "Jpsi_vtxz", "Jpsi_vtxb6", "Jpsi_vtxdchi2",
                                  "Jpsi_vtxvchk", "Jpsi_vtxvgf", "Jpsi_vtxbfree",
                                  "Jpsi_vtxfree", "Jpsi_vtxok", "Jpsi_vtxsgnchk",
                                  "Jpsi_vtxfirstplus", "Jpsi_vtxvhit",
                                  "Jpsi_vtxvms", "Jpsi_vtxvioni")),
    ("vertex: the influence a_b", ("resinfvtxv",)),
    ("vertex: the D row", ("Jpsi_jacVtx",)),
    ("mass: the exponents (per group)", ("cfmass_grp_ms", "cfmass_grp_ioni_re",
                                         "cfmass_grp_ioni_im", "cfmass_grp_rad_re",
                                         "cfmass_grp_rad_im", "cfmass_grp",
                                         "cfmass_grp_vqms", "cfmass_grp_vqio",
                                         "cfmass_grp_closure")),
    ("mass: the exponents (flat)", ("cfmass_ms", "cfmass_del", "cfmass_ioni_re",
                                    "cfmass_ioni_im", "cfmass_rad_re",
                                    "cfmass_rad_im")),
    ("mass: shares + scalars", ("resinfvarv", "reseigidx", "reshitidx",
                                "reshitcls", "cfmass_hitcls", "cfmass_hitv",
                                "cfmass_vgf", "cfmass_ok", "resinfcov",
                                "resinfcovhit", "resinfcovgrp", "Jpsi_sigmamass")),
    ("mass: the influence a_b", ("resinfv",)),
    ("mass: the D row", ("Jpsi_jacMass",)),
]


def bill(a):
    import uproot
    fh = uproot.open(a.file)
    t = fh["tree"]
    n = t.num_entries
    tot = os.path.getsize(a.file)
    sizes = {}
    for k in t.keys():
        b = t[k]
        try:
            sizes[k.split(";")[0]] = float(b.compressed_bytes)
        except Exception:
            sizes[k.split(";")[0]] = 0.0
    print(f"{a.file}")
    print(f"{n} candidates, {tot/1e6:.1f} MB, {tot/n/1024:.1f} kB/candidate "
          f"(all branches, compressed)\n")
    claimed = set()
    print(f"{'group':<36}{'kB/cand':>10}{'TB at 41 M':>13}")
    print("-" * 59)
    for lab, br in GROUPS:
        s = sum(sizes.get(x, 0.0) for x in br)
        claimed |= set(br)
        print(f"{lab:<36}{s/n/1024:>10.2f}{s/n*FULLSCALE/1e12:>13.3f}")
    rest = sum(v for k, v in sizes.items() if k not in claimed)
    print(f"{'everything else (fit output, kinematics)':<36}"
          f"{rest/n/1024:>10.2f}{rest/n*FULLSCALE/1e12:>13.3f}")
    print("-" * 59)
    s_all = sum(sizes.values())
    print(f"{'sum of branches':<36}{s_all/n/1024:>10.2f}{s_all/n*FULLSCALE/1e12:>13.3f}")
    # the marginal cost of the VERTEX term on top of the mass one
    vtx = sum(sizes.get(x, 0.0) for lab, br in GROUPS if lab.startswith("vertex")
              for x in br)
    mass = sum(sizes.get(x, 0.0) for lab, br in GROUPS if lab.startswith("mass")
               for x in br)
    print(f"\nthe VERTEX block alone: {vtx/n/1024:.2f} kB/cand "
          f"({vtx/n*FULLSCALE/1e12:.3f} TB at 41 M)")
    print(f"the MASS block alone  : {mass/n/1024:.2f} kB/cand "
          f"({mass/n*FULLSCALE/1e12:.3f} TB at 41 M)")
    print(f"=> the vertex term costs {vtx/max(mass,1):.2f} x the mass term's export")
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:12]
    print("\nthe 12 largest branches:")
    for k, v in top:
        print(f"   {k:<28}{v/n/1024:>9.2f} kB/cand")


def timing(a):
    import tensorflow as tf
    import h5py
    sys.path.insert(0, os.environ.get("RABBIT", "/work/submit/david_w/ZMass/rabbit-vmass"))
    from rabbit import inputdata
    ws = inputdata.FitInputData(a.card)
    print("cannot rebuild terms from the card here; use --bill")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default=None)
    p.add_argument("--card", default=None)
    p.add_argument("--bill", action="store_true")
    a = p.parse_args()
    if a.bill or a.file:
        bill(a)


if __name__ == "__main__":
    main()
