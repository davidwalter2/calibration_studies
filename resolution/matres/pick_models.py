"""Which rabbit param models a card needs, from what it actually declares."""
import sys

import h5py

with h5py.File(sys.argv[1], "r") as f:
    m = []
    if "unbinned_terms" in f:
        m.append("--paramModel UnbinnedParams")
    if "auxiliary" in f and "global_params" in f["auxiliary"]:
        m.append("--paramModel ExternalParams bundle:global_params")
print(" ".join(m))
