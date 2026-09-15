#!/usr/bin/env python3
"""Print the rows of a fit as a table: a finished `.json` or a running `.log`.

    python3 show_fit.py data/fit_machinery_table.json data/00_fit_conv.log
"""
import json
import re
import sys

#: the block `fit_gen.summarise` prints for one row, so a run can be followed
#: before its json exists
PAT = re.compile(
    r"^  (\S.*?)  \[(\d+) events.*?\n\s+m_Z\s+([-+0-9.]+) \+-\s+([0-9.]+)"
    r".*?\n\s+Gamma_Z\s+([-+0-9.]+) \+-\s+([0-9.]+)", re.M | re.S)


def rows(path):
    if path.endswith(".json"):
        with open(path) as fh:
            d = json.load(fh)
        return [(k, v["m_Z"][0], v["m_Z"][1], v["Gamma_Z"][0], v["Gamma_Z"][1])
                for k, v in d.items() if "m_Z" in v]
    return [(r[0], float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in PAT.findall(open(path).read())]


def main():
    for path in sys.argv[1:]:
        r = rows(path)
        print(f"\n=== {path} ===")
        if not r:
            continue
        w = max(len(x[0]) for x in r) + 2
        print(f"{'row':{w}s} {'d m_Z [MeV]':>18s} {'d Gamma_Z [MeV]':>18s}")
        for lab, m, dm, g, dg in r:
            print(f"{lab:{w}s} {m:>+10.3f} +- {dm:5.3f} {g:>+10.3f} +- {dg:5.3f}")


if __name__ == "__main__":
    main()
