"""Count `slimmedMuonTrackExtras` TrackExtras and classify their rechits by
subdetector.  Used to tell apart a MiniAOD that carries the muon *tracker*
hits (CVH-refittable) from one that carries only muon-system hits.

  python3 rechitprobe.py <file-or-root://url> <nevents>
"""
from __future__ import print_function
import sys, ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
try: ROOT.FWLiteEnabler.enable()
except AttributeError: ROOT.AutoLibraryLoader.enable()
from DataFormats.FWLite import Events, Handle
fn = sys.argv[1]; nmax = int(sys.argv[2])
ev = Events(fn)
hh = Handle("edm::OwnVector<TrackingRecHit,edm::ClonePolicy<TrackingRecHit> >")
he = Handle("std::vector<reco::TrackExtra>")
SUBDET = {1:"PXB",2:"PXF",3:"TIB",4:"TID",5:"TOB",6:"TEC"}
DET = {1:"Tracker",2:"Muon",3:"Ecal",4:"Hcal",6:"Calo"}
counts = {}
nev=0; nhit=0; nextra=0
for e in ev:
    if nev>=nmax: break
    e.getByLabel("slimmedMuonTrackExtras", hh)
    e.getByLabel("slimmedMuonTrackExtras", he)
    if he.isValid(): nextra += he.product().size()
    if hh.isValid():
        hits = hh.product()
        for i in range(hits.size()):
            h = hits[i]
            rid = h.geographicalId().rawId()
            det = (rid>>28)&0xF
            sub = (rid>>25)&0x7
            key = "%s/%s" % (DET.get(det,"det%d"%det), SUBDET.get(sub,"sub%d"%sub) if det==1 else "sub%d"%sub)
            counts[key] = counts.get(key,0)+1
            nhit += 1
    nev+=1
print("events=%d trackextras=%d rechits=%d" % (nev, nextra, nhit))
for k in sorted(counts): print("   %-14s %d" % (k, counts[k]))
