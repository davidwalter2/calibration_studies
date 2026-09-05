import ROOT, sys
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
paths=[l.strip() for l in open(sys.argv[1]) if l.strip()]
out=open(sys.argv[2],'w')
for i,p in enumerate(paths):
    try:
        f=ROOT.TFile.Open(p); t=f.Get("Events"); n=int(t.GetEntries()); f.Close()
    except Exception as e:
        n=-1
    out.write(f"{p} {n}\n"); out.flush()
    if i%50==0: print(i, n, flush=True)
out.close()
print("DONE")
