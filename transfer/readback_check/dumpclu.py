"""Release-agnostic dump of SiStrip/SiPixel clusters via FWLite.
usage: dumpclu.py <file> <nevents> <outfile> [label]
Deterministic sorted text so the same file read in two CMSSW releases diffs byte-for-byte.
"""
from __future__ import print_function
import sys, hashlib
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
try:
    ROOT.FWLiteEnabler.enable()
except AttributeError:
    ROOT.AutoLibraryLoader.enable()
from DataFormats.FWLite import Events, Handle

fn    = sys.argv[1]
nmax  = int(sys.argv[2])
outfn = sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else "slimmedMuonTrackExtras"

def _b(v):
    return ord(v) if isinstance(v, str) else int(v)

def _int(v):
    try:
        return int(v)
    except Exception:
        return int(v.load())          # std::atomic<int>

def detsets(dsv):
    """Yield (detid, [clusters]) release-agnostically (PyROOT 6.14 cannot
    iterate edmNew::DetSetVector's boost transform_iterator)."""
    ids, data = dsv.ids(), dsv.data()
    for k in range(ids.size()):
        it = ids[k]
        off, n = _int(it.offset), _int(it.size)
        yield int(it.id), [data[j] for j in range(off, off + n)]

def amps_of(cl):
    a = cl.amplitudes()
    try:
        n = a.size()
        if n is not None:
            return [_b(a[i]) for i in range(int(n))]
    except Exception:
        pass
    n = int(cl.endStrip()) - int(cl.firstStrip())   # 15_X: amplitudes() returns *this
    return [_b(cl[i]) for i in range(n)]

events = Events(fn)
hs = Handle("edmNew::DetSetVector<SiStripCluster>")
hp = Handle("edmNew::DetSetVector<SiPixelCluster>")

out = open(outfn, "w")
nev = nstrip = npix = nempty = nbad = 0
ampsum_all = 0
for ev in events:
    if nev >= nmax:
        break
    aux = ev.eventAuxiliary()
    out.write("EVENT %d:%d:%d\n" % (aux.run(), aux.luminosityBlock(), aux.event()))
    ev.getByLabel(label, hs)
    rows = []
    for detid, cls in detsets(hs.product()):
        for cl in cls:
            v = amps_of(cl)
            s = sum(v)
            nstrip += 1
            ampsum_all += s
            if len(v) == 0 or s == 0:
                nempty += 1
            if any(x < 0 or x > 255 for x in v):
                nbad += 1
            rows.append("  S det=%d first=%d n=%d sum=%d q=%d bary=%.5f amps=%s" %
                        (detid, cl.firstStrip(), len(v), s, cl.charge(), cl.barycenter(),
                         ",".join(str(x) for x in v)))
    rows.sort()
    if rows:
        out.write("\n".join(rows) + "\n")
    ev.getByLabel(label, hp)
    rows = []
    for detid, cls in detsets(hp.product()):
        for cl in cls:
            npix += 1
            rows.append("  P det=%d size=%d sizeX=%d sizeY=%d q=%.1f x=%.5f y=%.5f minX=%d minY=%d" %
                        (detid, cl.size(), cl.sizeX(), cl.sizeY(), cl.charge(),
                         cl.x(), cl.y(), cl.minPixelRow(), cl.minPixelCol()))
    rows.sort()
    if rows:
        out.write("\n".join(rows) + "\n")
    nev += 1
out.close()

h = hashlib.md5(open(outfn, "rb").read()).hexdigest()
print("RESULT file=%s" % fn)
print("RESULT events=%d strip_clusters=%d empty=%d out_of_range=%d ampsum=%d pixel_clusters=%d md5=%s"
      % (nev, nstrip, nempty, nbad, ampsum_all, npix, h))
