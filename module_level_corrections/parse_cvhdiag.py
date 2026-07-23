"""Parse CVHDIAG lines from the instrumented CVH refit to see WHICH Geant4e
step diverges for failing muons at the hotspot (eta~-1.6, phi~-1.05).

CVHDIAG reason=<PROP|HITUPD|PARUPD> ihit=.. nhit=.. detid=.. subdet=.. layer=..
        [sx sy sz] trkpt=.. trketa=.. trkphi=.. iiter=.. [edmval lamupd]
subdet here = DetId.subdetId(): 1=PXB 2=PXF 3=TIB 4=TID 5=TOB 6=TEC
"""
import re
import sys
from collections import Counter

import numpy as np

SUB = {1: "PXB", 2: "PXF", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
LOG = sys.argv[1] if len(sys.argv) > 1 else \
    "/work/submit/david_w/ZMass/calibration_studies/module_level_corrections/diag_run/cvhdiag.log"

pat = re.compile(r"CVHDIAG reason=(\w+) ihit=(-?\d+) nhit=(\d+) detid=(\d+) "
                 r"subdet=(-?\d+) layer=(-?\d+).*?trkpt=([\d.eE+-]+) "
                 r"trketa=([\d.eE+-]+) trkphi=([\d.eE+-]+) iiter=(-?\d+)")

rows = []
for line in open(LOG):
    m = pat.search(line)
    if not m:
        continue
    rows.append(dict(reason=m.group(1), ihit=int(m.group(2)), nhit=int(m.group(3)),
                     detid=int(m.group(4)), subdet=int(m.group(5)), layer=int(m.group(6)),
                     pt=float(m.group(7)), eta=float(m.group(8)), phi=float(m.group(9)),
                     iiter=int(m.group(10))))
print(f"total CVHDIAG failures parsed: {len(rows)}")

eta = np.array([r["eta"] for r in rows])
phi = np.array([r["phi"] for r in rows])
hot = (eta > -1.75) & (eta < -1.45) & (phi > -1.30) & (phi < -0.80)
mir = (eta > 1.45) & (eta < 1.75) & (phi > -1.30) & (phi < -0.80)
print(f"failures in hotspot cell: {hot.sum()}  (mirror cell: {mir.sum()})")

hs = [r for r, h in zip(rows, hot) if h]
print("\n=== hotspot failures by REASON ===")
for k, v in Counter(r["reason"] for r in hs).most_common():
    print(f"  {k}: {v}")

print("\n=== hotspot PROP/HITUPD failures: which surface (subdet, layer) diverges ===")
sl = Counter((SUB.get(r["subdet"], r["subdet"]), r["layer"]) for r in hs if r["reason"] in ("PROP", "HITUPD"))
for (sd, ly), v in sl.most_common(12):
    print(f"  {sd} layer {ly}: {v}")

print("\n=== hotspot: exact failing modules (detid) top 12 ===")
for did, v in Counter(r["detid"] for r in hs if r["reason"] in ("PROP", "HITUPD")).most_common(12):
    r0 = next(r for r in hs if r["detid"] == did)
    print(f"  detid {did}  {SUB.get(r0['subdet'], r0['subdet'])} L{r0['layer']}  x{v}")

print("\n=== hotspot: ihit at failure (how far into the track) ===")
ih = np.array([r["ihit"] for r in hs if r["reason"] in ("PROP", "HITUPD")])
nh = np.array([r["nhit"] for r in hs if r["reason"] in ("PROP", "HITUPD")])
if len(ih):
    print(f"  ihit: median {np.median(ih):.0f} mean {ih.mean():.1f}  (nhit median {np.median(nh):.0f})")
    print(f"  frac failing at ihit==0: {np.mean(ih == 0):.2f}   at ihit>=nhit-1: {np.mean(ih >= nh - 1):.2f}")

# baseline: whole-detector reason mix for context
print("\n=== ALL failures (any location) by reason ===")
for k, v in Counter(r["reason"] for r in rows).most_common():
    print(f"  {k}: {v}")
