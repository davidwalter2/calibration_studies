"""
FWLite script (python2, el7) to extract V0 kinematics from ALCARECO files.

Columns (per-candidate files):
  mass(MeV) pt eta phi vx(cm) vy(cm) vz(cm) Lxy(cm) L3d(cm)
  d0_pt d0_eta d0_phi d0_charge
  d1_pt d1_eta d1_phi d1_charge
"""
import sys
sys.argv.append('-b')
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.AutoLibraryLoader.enable()
from DataFormats.FWLite import Events, Handle
import math

def process(infile, collection_module, collection_label, outfile_cands, outfile_events):
    events = Events(infile)
    handle = Handle('std::vector<reco::VertexCompositeCandidate>')
    pvhandle = Handle('std::vector<reco::Vertex>')

    fout = open(outfile_cands, 'w')
    fevt = open(outfile_events, 'w')

    n_events = 0
    for ev in events:
        ev.getByLabel((collection_module, collection_label), handle)
        ev.getByLabel('offlinePrimaryVertices', pvhandle)

        cands = handle.product()
        pvs   = pvhandle.product()

        # use best PV (first valid one), fall back to origin
        pvx, pvy, pvz = 0.0, 0.0, 0.0
        for pv in pvs:
            if not pv.isFake() and pv.ndof() > 4:
                pvx, pvy, pvz = pv.x(), pv.y(), pv.z()
                break

        n_events += 1
        fevt.write('%d\n' % cands.size())

        for c in cands:
            if c.numberOfDaughters() < 2:
                continue
            mass = c.mass() * 1000.0
            pt   = c.pt()
            eta  = c.eta()
            phi  = c.phi()
            vx, vy, vz = c.vx(), c.vy(), c.vz()
            dx, dy, dz = vx - pvx, vy - pvy, vz - pvz
            Lxy = math.sqrt(dx*dx + dy*dy)
            L3d = math.sqrt(dx*dx + dy*dy + dz*dz)

            d0 = c.daughter(0)
            d1 = c.daughter(1)

            fout.write('%f %f %f %f %f %f %f %f %f  %f %f %f %d  %f %f %f %d\n' % (
                mass, pt, eta, phi, vx, vy, vz, Lxy, L3d,
                d0.pt(), d0.eta(), d0.phi(), int(d0.charge()),
                d1.pt(), d1.eta(), d1.phi(), int(d1.charge()),
            ))

    fout.close()
    fevt.close()
    print("Processed %d events -> %s" % (n_events, outfile_cands))

process('/tmp/TkAlKsToPiPi_500.root',
        'generalV0Candidates', 'Kshort',
        '/tmp/ks_kinematics.txt', '/tmp/ks_per_event.txt')

process('/tmp/TkAlLambdaToProtonPi_500.root',
        'generalV0Candidates', 'Lambda',
        '/tmp/lam_kinematics.txt', '/tmp/lam_per_event.txt')

print("Done.")
