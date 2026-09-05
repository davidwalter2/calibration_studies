#!/usr/bin/env python3
"""One table out of the radiative-term closure arms.

Pulls the TOTAL rows out of the `skew_closure_*_krad{0,1}.txt` files and lays
the two arms side by side, so "what the radiative term does to the odd
moment" can be read without opening sixteen files.

  section 1  <z e^{-u z^2}> data / model / data-model at every probe
  section 4  <z> against the trim (|z| < 1, 2, 3, 5) and the median

usage: python summarise_rad_260903x.py [skew-output-dir]
"""
import glob
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/public_html/cvh/%s_skew" % __import__("datetime").date.today().strftime("%y%m%d"))


def section_rows(path, head, rowpat="^TOTAL"):
    txt = open(path).read().split("\n")
    out, on = [], False
    for i, L in enumerate(txt):
        if L.startswith("## "):
            on = L.startswith(head)
            continue
        if on and re.match(rowpat, L):
            out.append((L, txt[i + 1] if i + 1 < len(txt) else ""))
    return out


def f(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def main():
    files = sorted(glob.glob(os.path.join(OUT, "skew_closure_*_krad1.txt")))
    if not files:
        sys.exit(f"no *_krad1.txt in {OUT}")
    print(f"# radiative-term closure summary from {OUT}\n")
    print("## 1. <z e^{-u z^2}> TOTAL, data vs model, radiative term OFF/ON")
    print(f"{'arm':<44}{'u':>6}{'data':>10}{'mod krad0':>11}{'mod krad1':>11}"
          f"{'d-m krad0':>11}{'d-m krad1':>11}{'boot':>9}")
    probes = [0.05, 0.2, 0.5, 1.0, 2.0]
    for f1 in files:
        f0 = f1.replace("_krad1.txt", "_krad0.txt")
        if not os.path.exists(f0):
            continue
        arm = os.path.basename(f1)[len("skew_closure_"):-len("_krad1.txt")]
        r1 = section_rows(f1, "## 1. <z e^(-u z^2)>")
        r0 = section_rows(f0, "## 1. <z e^(-u z^2)>")
        if not r1 or not r0:
            continue
        # first occurrence = the pT binning (or the only binning)
        c1, e1 = r1[0]
        c0, _ = r0[0]
        v1 = c1.split()[2:]
        v0 = c0.split()[2:]
        eb = e1.split()
        for iu, u in enumerate(probes):
            if 3 * iu + 2 >= len(v1):
                break
            print(f"{arm:<44}{u:>6}{f(v1[3*iu]):>10.5f}"
                  f"{f(v0[3*iu+1]):>11.5f}{f(v1[3*iu+1]):>11.5f}"
                  f"{f(v0[3*iu+2]):>11.5f}{f(v1[3*iu+2]):>11.5f}"
                  f"{f(eb[iu].replace('+-','')) if iu < len(eb) else float('nan'):>9.5f}")
        print()
    print("\n## 4. <z> vs trim, TOTAL, data / model(krad0) / model(krad1)")
    print(f"{'arm':<44}{'trim':>8}{'data':>10}{'mod krad0':>11}{'mod krad1':>11}")
    for f1 in files:
        f0 = f1.replace("_krad1.txt", "_krad0.txt")
        if not os.path.exists(f0):
            continue
        arm = os.path.basename(f1)[len("skew_closure_"):-len("_krad1.txt")]
        r1 = section_rows(f1, "## 4. LOCATION")
        r0 = section_rows(f0, "## 4. LOCATION")
        if not r1 or not r0:
            continue
        v1 = r1[0][0].split()[2:]
        v0 = r0[0][0].split()[2:]
        for it, tr in enumerate(("|z|<1", "|z|<2", "|z|<3", "|z|<5")):
            print(f"{arm:<44}{tr:>8}{f(v1[3*it]):>10.5f}"
                  f"{f(v0[3*it+2]):>11.5f}{f(v1[3*it+2]):>11.5f}")
        print(f"{arm:<44}{'median':>8}{f(v1[12]):>10.5f}"
              f"{f(v0[13]):>11.5f}{f(v1[13]):>11.5f}")
        print()


if __name__ == "__main__":
    main()
