#!/usr/bin/env python3
"""The candidate sample the error budget runs on: the first N rows of a fit
pairs cache (`cf_inmaker.py` output), read without decompressing the whole
member.

A pairs cache is a zip of .npy members of several GB each; `np.load` inflates
a member completely before slicing.  Here each member is streamed and only
its first N rows are inflated, which for 100 000 rows of the 3.7 M-candidate
Z cache takes seconds instead of minutes.  The rows of a cache follow the
production's task order, and the tasks are random slices of the dataset, so
the first N rows are an unbiased subsample.

usage: python error_budget_sample.py --cache <pairs.npz> --n 100000 -o out.npz
"""

import argparse
import zipfile

import numpy as np

KEYS = ("z", "sigma", "vgf", "Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im",
        "mtrk", "w", "eta", "pt", "chisqval", "ndof", "fang", "ptp", "ptm",
        "etap", "etam")
META = ("tgrid", "cf_model", "cf_source", "rad_model", "ioni_sign_fixed")


def head_rows(zf, name, n):
    """The first n rows of member `name` (a C-ordered .npy), streamed."""
    with zf.open(name) as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
        if fortran:
            raise ValueError(f"{name}: Fortran-ordered member")
        n = min(n, shape[0])
        row = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        buf = fh.read(n * row * dtype.itemsize)
    return np.frombuffer(buf, dtype=dtype).reshape((n,) + tuple(shape[1:])).copy()


def load_head(path, n, keys=KEYS):
    zf = zipfile.ZipFile(path)
    names = {i.filename[:-4] for i in zf.infolist()}
    out = {}
    for k in keys:
        if k in names:
            out[k] = head_rows(zf, k + ".npy", n)
    for k in META:
        if k in names:
            with zf.open(k + ".npy") as fh:
                out[k] = np.lib.format.read_array(fh, allow_pickle=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    d = load_head(a.cache, a.n)
    d["source"] = np.array(a.cache)
    np.savez(a.output, **d)
    print(f"{len(d['z'])} candidates -> {a.output}")


if __name__ == "__main__":
    main()
