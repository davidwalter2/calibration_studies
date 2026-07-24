#!/usr/bin/env python
"""Lorentz-angle response of the on-track pixel CPE, per cluster class.

Reads the muon-gun GEN-SIM-RECO files (nominal / shifted true Lorentz
angle, identical seeds) and compares each on-track BPix rec-hit position
with the matched muon PSimHit (mid-plane truth). The difference of the
per-class mean residuals between the two variants, normalised to the
size-1 class, gives the response weights w(class) used by the dtanLA
parameter (parmtype 22) in the CVH fit.

Run inside the CMSSW_10_6_20_patch1 environment (FWLite).
BPix only: phase-0 barrel modules are uniformly 160 rows x 416 columns.
"""
from __future__ import print_function
import sys
from collections import defaultdict

from DataFormats.FWLite import Events, Handle
import ROOT

NROWS, NCOLS = 160, 416
CM2UM = 1e4


def classify(cl):
    bits = 0
    if cl.minPixelRow() == 0:
        bits |= 1        # -x edge
    if cl.maxPixelRow() == NROWS - 1:
        bits |= 2        # +x edge
    if cl.minPixelCol() == 0 or cl.maxPixelCol() == NCOLS - 1:
        bits |= 4        # y edge (either side; not used for x response)
    if cl.sizeX() <= 1:
        bits |= 8
    if cl.sizeY() <= 1:
        bits |= 16
    return bits


def main(fname, out):
    events = Events(fname)
    hhits = Handle("edm::OwnVector<TrackingRecHit,edm::ClonePolicy<TrackingRecHit> >")
    hsim = Handle("std::vector<PSimHit>")

    # accumulate sum/sumsq/n of dx per (layer, category)
    acc = defaultdict(lambda: [0.0, 0.0, 0])

    nev = 0
    for ev in events:
        nev += 1
        if nev % 2000 == 0:
            print("  event", nev)
        # the on-track hits of all generalTracks (final-fit CPE positions)
        ev.getByLabel("generalTracks", hhits)
        ev.getByLabel("g4SimHits", "TrackerHitsPixelBarrelLowTof", hsim)
        # simhits indexed by detid
        simbydet = defaultdict(list)
        for sh in hsim.product():
            if abs(sh.particleType()) == 13:
                simbydet[sh.detUnitId()].append(sh)
        for hit in hhits.product():
            if not hit.isValid():
                continue
            det = hit.geographicalId().rawId()
            if (det >> 25) & 0x7 != 1:      # PXB only
                continue
            layer = (det >> 16) & 0xF
            if not hasattr(hit, "cluster_pixel"):
                continue
            clref = hit.cluster_pixel()
            if clref.isNull():
                continue
            cl = clref.get()
            if not cl:
                continue
            sims = simbydet.get(det)
            if sims:
                lp = hit.localPosition()
                # closest muon simhit on this module
                best, bestd = None, 0.05     # 500 um match window
                for sh in sims:
                    d = abs(sh.localPosition().x() - lp.x())
                    if d < bestd:
                        best, bestd = sh, d
                if best is None:
                    continue
                dx = lp.x() - best.localPosition().x()
                # local track angle in the drift plane from the simhit
                p = best.momentumAtEntry()
                tanax = p.x() / p.z() if abs(p.z()) > 1e-6 else 999.
                bits = classify(cl)
                if bits & 8:
                    cat = "sizeX1"
                elif bits & 3:
                    cat = "edgeXlo" if bits & 1 else "edgeXhi"
                else:
                    cat = "clean"
                for key in ((layer, cat), (0, cat)):   # 0 = all layers
                    a = acc[key]
                    a[0] += dx
                    a[1] += dx * dx
                    a[2] += 1
                # angle-binned clean response (drift response depends on
                # |tan(alpha) - tanLA|): coarse bins of tan(alpha_x)
                if cat == "clean" and abs(tanax) < 2.:
                    ab = int((tanax + 2.) / 0.5)
                    a = acc[("ang%d" % ab, "clean")]
                    a[0] += dx
                    a[1] += dx * dx
                    a[2] += 1

    print("events:", nev)
    with open(out, "w") as f:
        for key in sorted(acc, key=str):
            s, s2, n = acc[key]
            if n < 20:
                continue
            m = s / n
            e = ((s2 / n - m * m) / n) ** 0.5
            f.write("%s %s %d %+.4f %.4f\n"
                    % (key[0], key[1], n, m * CM2UM, e * CM2UM))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
