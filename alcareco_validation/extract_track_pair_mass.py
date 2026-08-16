"""
Extract raw track-pair invariant mass from ALCARECO output.

Loops over all opposite-sign track pairs in the stored AlignmentTrackSelector
output and computes the invariant mass with the daughter mass hypothesis.
For Lambda, both (proton+pion) and (pion+proton) assignments are tried and the
one closer to nominal Lambda0 mass is kept.

Output: one invariant mass per line, in MeV.
"""
import sys
sys.argv.append('-b')
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.AutoLibraryLoader.enable()
from DataFormats.FWLite import Events, Handle
import math

PI_MASS = 0.13957
P_MASS  = 0.93827
LAM_MASS = 1.115683

def inv_mass(p1, p2, m1, m2):
    px1, py1, pz1 = p1
    px2, py2, pz2 = p2
    e1 = math.sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1)
    e2 = math.sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2)
    e_tot = e1 + e2
    px_tot, py_tot, pz_tot = px1 + px2, py1 + py2, pz1 + pz2
    m2_tot = e_tot*e_tot - (px_tot*px_tot + py_tot*py_tot + pz_tot*pz_tot)
    return math.sqrt(m2_tot) if m2_tot > 0 else -1.0

def process(infile, track_module, outfile, mode):
    """mode = 'ks' (pion+pion) or 'lam' (proton+pion, both assignments tried)."""
    events = Events(infile)
    handle = Handle('std::vector<reco::Track>')
    fout = open(outfile, 'w')
    n_events = 0
    n_pairs  = 0
    for ev in events:
        ev.getByLabel(track_module, handle)
        tracks = list(handle.product())
        n_events += 1
        for i in range(len(tracks)):
            for j in range(i+1, len(tracks)):
                t1, t2 = tracks[i], tracks[j]
                if t1.charge() * t2.charge() >= 0:
                    continue
                p1 = (t1.px(), t1.py(), t1.pz())
                p2 = (t2.px(), t2.py(), t2.pz())
                if mode == 'ks':
                    m = inv_mass(p1, p2, PI_MASS, PI_MASS)
                else:
                    ma = inv_mass(p1, p2, P_MASS,  PI_MASS)
                    mb = inv_mass(p1, p2, PI_MASS, P_MASS )
                    m  = ma if abs(ma - LAM_MASS) < abs(mb - LAM_MASS) else mb
                if m > 0:
                    fout.write('%f\n' % (m * 1000.0))
                    n_pairs += 1
    fout.close()
    print("Processed %d events, %d pairs -> %s" % (n_events, n_pairs, outfile))

# nominal pT cut (0.35 GeV)
process('/tmp/TkAlKsToPiPi_1000.root',          'ALCARECOTkAlKsToPiPi',
        '/tmp/ks_pair_mass.txt',          'ks')
process('/tmp/TkAlLambdaToProtonPi_1000.root',  'ALCARECOTkAlLambdaToProtonPi',
        '/tmp/lam_pair_mass.txt',         'lam')

# lowered pT cut (0.10 GeV)
process('/tmp/TkAlKsToPiPi_1000_pt0p1.root',         'ALCARECOTkAlKsToPiPi',
        '/tmp/ks_pair_mass_pt0p1.txt',         'ks')
process('/tmp/TkAlLambdaToProtonPi_1000_pt0p1.root', 'ALCARECOTkAlLambdaToProtonPi',
        '/tmp/lam_pair_mass_pt0p1.txt',        'lam')

print("Done.")
