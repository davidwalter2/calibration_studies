#!/usr/bin/env python3
"""Rebuild the nll_impact json from a compress log that is still running.

`nll_impact.py --mode compress` only writes its json at the very end, so this
recovers the rows that are already in the log (same schema, minus the fields
that only the json carries).  Used to make the figures without waiting for the
tail of the scan.
"""
import json, re, sys

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/cfcompress")
ROW = re.compile(
    r"^(pca_all r=\d+|pca_fam r=\d+/fam|levy k=[\d+]+)\s+dNLL=(\S+)\s+"
    r"bytes/cand=\s*(\d+)\s+dalpha\[1e-3\]=(\S+?)\(.*?dk_hit=(\S+?)\(.*?"
    r"dk_ms=(\S+?)\(.*?dk_ioni=(\S+?)\(")
REF = re.compile(r"^full \(uncompressed\)\s+NLL=(\S+)\s+alpha\[1e-3\]=(\S+?)\+-(\S+)"
                 r"\s+k_hit=(\S+?)\+-(\S+)\s+k_ms=(\S+?)\+-(\S+)\s+"
                 r"k_ioni=(\S+?)\+-(\S+)")


def main(log, out):
    ref = None
    rows = []
    for ln in open(log):
        m = REF.match(ln)
        if m and ref is None:
            g = [float(x) for x in m.groups()]
            ref = dict(name="full", nll=g[0], x=[g[1], g[3], g[5], g[7]],
                       err=[g[2], g[4], g[6], g[8]], bytes=5 * 448 * 4)
            rows.append(ref)
            continue
        m = ROW.match(ln)
        if m and ref is not None:
            nm, dn, nb = m.group(1), float(m.group(2)), int(m.group(3))
            d = [float(m.group(i)) for i in (4, 5, 6, 7)]
            kind = ("pca_all" if nm.startswith("pca_all")
                    else "pca_fam" if nm.startswith("pca_fam") else "levy")
            r = int(re.search(r"\d+", nm).group()) if kind != "levy" else nb // 4
            rows.append(dict(name=nm, kind=kind, r=r, bytes=nb,
                             nll=ref["nll"] + dn,
                             x=[ref["x"][i] + d[i] for i in range(4)],
                             err=ref["err"]))
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{len(rows)-1} compressed rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else f"{SCRATCH}/compress_jpsigun.log",
         sys.argv[2] if len(sys.argv) > 2
         else f"{SCRATCH}/nllimpact_jpsigun_n20000.json")
