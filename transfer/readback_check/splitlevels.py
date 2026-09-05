"""Print the ROOT split level of the cluster / rechit / TrackExtra branches of an
EDM file.  ROOT #19773 (SiStripCluster v11->v14 schema evolution not applied in
split mode) bites only at split level 99; MiniAOD writes these at split level 1
and is therefore immune, RECO/ALCARECO writes them at 99 and is not.

  python3 splitlevels.py <file-or-root://url> [more files ...]
"""
import sys, ROOT
ROOT.gROOT.SetBatch(True)
pats = ("SiStripCluster","SiPixelCluster","TrackingRecHit","TrackExtra")
for fn in sys.argv[1:]:
    f = ROOT.TFile.Open(fn)
    if not f or f.IsZombie():
        print("CANNOT OPEN", fn); continue
    t = f.Get("Events")
    print("="*100); print("FILE:", fn); print("  entries:", t.GetEntries())
    for b in t.GetListOfBranches():
        n = b.GetName()
        if any(p in n for p in pats):
            print("  split=%-3d nsub=%-4d  %s" % (b.GetSplitLevel(), b.GetListOfBranches().GetEntries(), n))
            for sb in b.GetListOfBranches():
                print("      sub split=%-3d %s" % (sb.GetSplitLevel(), sb.GetName()))
    f.Close()
