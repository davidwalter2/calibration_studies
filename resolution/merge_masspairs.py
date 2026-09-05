"""Concatenate per-task cf_mass_likelihood --pairs-tt caches into one.

`--pairs-tt` is a strictly per-file loop with no cross-file state (the
TwoTrack tree is already per candidate, so there is no pairing that could
span files), which makes it embarrassingly parallel -- but it runs serially
at ~12 min per 2.7k-candidate task file, i.e. ~10 h for a 48-task
production.  Running one invocation per task file and concatenating here
gives a byte-equivalent cache in the wall time of a single task.

All arrays are concatenated along axis 0 in the order the inputs are given;
the scalars `tgrid` and `ioni_sign_fixed` (the provenance flag saying the
ionization blocks carry the physical mass-functional sign -1) must be
identical across inputs and are copied through.  A part set that disagrees on
either is a mix of code versions and is rejected rather than merged.

usage: python merge_masspairs.py --out runs/cf_masspairs_<tag>.npz \
           'runs/parts/cf_masspairs_<tag>_task*.npz'
"""
import argparse
import glob
import sys

import numpy as np

CONCAT = ("z", "sigma", "eta", "vgf", "Sms", "Sio_re", "Sio_im")
# per-candidate arrays present only on parts built after the radiative term
# (2026-09-03). Concatenated when present; a part set that has them in some
# parts and not others is a mix of code versions and is rejected below, the
# same rule KEEPONE applies to the scalars.
CONCAT_OPT = ("Srad_re", "Srad_im")
# scalars kept as one copy, asserted identical across parts. `rad_model` is
# the 0/1 provenance value saying whether the radiative block was built from
# the `radstepv` export; it IS read downstream (cf_masslik_fit's k_rad), so a
# part set that disagrees on it is a mixed-model cache and is rejected.
KEEPONE = ("tgrid", "ioni_sign_fixed", "rad_model")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("parts", nargs="+")
    a = p.parse_args()

    files = []
    for pat in a.parts:
        files += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    if not files:
        sys.exit("no input parts matched")
    print(f"{len(files)} parts")

    have_opt = None
    acc = {k: [] for k in CONCAT}
    one = {}
    for fn in files:
        d = np.load(fn)
        opt = tuple(k for k in CONCAT_OPT if k in d.files)
        if have_opt is None:
            have_opt = opt
            for k in opt:
                acc[k] = []
        elif opt != have_opt:
            sys.exit(f"{fn} has optional arrays {opt} against {have_opt} in "
                     f"{files[0]} -- parts were built by different code "
                     f"versions")
        for k in KEEPONE:
            have = k in d.files
            if fn == files[0]:
                if have:
                    one[k] = d[k]
            elif have != (k in one):
                sys.exit(f"{k} present in some parts and not others "
                         f"({fn}) -- parts were built by different code "
                         f"versions")
            elif have and not np.array_equal(one[k], d[k]):
                sys.exit(f"{k} mismatch in {fn}")
        for k in list(CONCAT) + list(have_opt):
            acc[k].append(d[k])
        print(f"  {fn}: {len(d['z'])}")
    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out.update(one)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} ({len(out['z'])} candidates)")


if __name__ == "__main__":
    main()
