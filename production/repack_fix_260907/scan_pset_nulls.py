"""Scan an EDM file's ParameterSets tree for runs of NUL bytes.

The 2026-09-07 finding: four of the 620 repacked JPsiToMuMu ALCARECO files
carry NUL runs INSIDE the uncompressed ParameterSets payload -- the pset value
content replaced by zeros with its byte length exactly preserved.  cmsRun then
dies while constructing PoolSource with

    FormatIncompatibility ... EntryError can not convert representation of
    <param>: <NULs> to value of type vector<string_hex>          (exit 91)

The baskets decompress cleanly, so the zeros were WRITTEN that way: it is
write-side memory corruption in our own repack process, not storage or
transfer damage.  `edmProvDump` is the expensive way to find it; this is the
cheap one (ParameterSets is a few MB even for a 2 GB ALCARECO).

  python3 scan_pset_nulls.py <file> [more files ...]
prints one line per file:  OK | NULS n_entries=<..> n_runs=<..> bytes=<..>
"""
import sys
import numpy as np
import uproot

MINRUN = 8   # below this, an isolated 0 byte is ordinary payload


def scan(path):
    try:
        f = uproot.open(path)
    except Exception as e:                       # noqa: BLE001
        return "OPENFAIL", f"{type(e).__name__}: {e}"
    if "ParameterSets" not in f:
        return "NOPSETS", ""
    b = f["ParameterSets"]["IdToParameterSetsBlobs"]
    nent = f["ParameterSets"].num_entries
    nrun = nbyte = nbad = 0
    for i in range(nent):
        arr = np.asarray(b.debug_array(i), dtype=np.uint8)
        nz = arr == 0
        if not nz.any():
            continue
        d = np.diff(np.concatenate(([0], nz.astype(np.int8), [0])))
        L = np.where(d == -1)[0] - np.where(d == 1)[0]
        long = L[L >= MINRUN]
        if long.size:
            nbad += 1
            nrun += int(long.size)
            nbyte += int(long.sum())
    if nbad:
        return "NULS", f"entries={nbad}/{nent} runs={nrun} bytes={nbyte}"
    return "OK", f"entries={nent}"


if __name__ == "__main__":
    rc = 0
    for p in sys.argv[1:]:
        verdict, detail = scan(p)
        if verdict != "OK":
            rc = 1
        print(f"{verdict:9s} {detail:34s} {p}", flush=True)
    sys.exit(rc)
