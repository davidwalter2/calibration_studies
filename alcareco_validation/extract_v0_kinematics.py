"""
FWLite script (python2, el7) to extract V0 kinematics from ALCARECO files.

Columns (per-candidate files):
  mass(MeV) pt eta phi vx(cm) vy(cm) vz(cm) Lxy(cm) L3d(cm) cosTheta_XY
  d0_pt d0_eta d0_phi d0_charge
  d1_pt d1_eta d1_phi d1_charge
  sig_Lxy
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
    bshandle = Handle('reco::BeamSpot')

    fout = open(outfile_cands, 'w')
    fevt = open(outfile_events, 'w')

    n_events = 0
    for ev in events:
        ev.getByLabel((collection_module, collection_label), handle)
        ev.getByLabel('offlineBeamSpot', bshandle)

        cands = handle.product()
        bs    = bshandle.product()
        bsx, bsy, bsz = bs.x0(), bs.y0(), bs.z0()

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
            dx, dy, dz = vx - bsx, vy - bsy, vz - bsz
            Lxy = math.sqrt(dx*dx + dy*dy)
            L3d = math.sqrt(dx*dx + dy*dy + dz*dz)

            # pointing angle in XY: cos(angle between flight vector and momentum)
            # uses actual beamspot position, consistent with V0Producer's cut
            px, py = c.px(), c.py()
            pt_mag = math.sqrt(px*px + py*py)
            if Lxy > 0 and pt_mag > 0:
                cosTheta_XY = (dx * px + dy * py) / (Lxy * pt_mag)
            else:
                cosTheta_XY = -2.0

            # flight significance Lxy/sigma(Lxy) from vertex covariance matrix
            covXX = c.vertexCovariance(0, 0)
            covXY = c.vertexCovariance(0, 1)
            covYY = c.vertexCovariance(1, 1)
            if Lxy > 0:
                lx, ly = dx / Lxy, dy / Lxy
                var_Lxy = lx*lx*covXX + 2*lx*ly*covXY + ly*ly*covYY
                sig_Lxy = Lxy / math.sqrt(var_Lxy) if var_Lxy > 0 else -1.0
            else:
                sig_Lxy = -1.0

            d0 = c.daughter(0)
            d1 = c.daughter(1)

            fout.write('%f %f %f %f %f %f %f %f %f %f  %f %f %f %d  %f %f %f %d  %f\n' % (
                mass, pt, eta, phi, vx, vy, vz, Lxy, L3d, cosTheta_XY,
                d0.pt(), d0.eta(), d0.phi(), int(d0.charge()),
                d1.pt(), d1.eta(), d1.phi(), int(d1.charge()),
                sig_Lxy,
            ))

    fout.close()
    fevt.close()
    print("Processed %d events -> %s" % (n_events, outfile_cands))

process('/tmp/TkAlKsToPiPi_1000.root',
        'generalV0Candidates', 'Kshort',
        '/tmp/ks_kinematics.txt', '/tmp/ks_per_event.txt')

process('/tmp/TkAlLambdaToProtonPi_1000.root',
        'generalV0Candidates', 'Lambda',
        '/tmp/lam_kinematics.txt', '/tmp/lam_per_event.txt')

print("Done.")
