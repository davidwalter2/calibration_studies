"""Dump the structure of the B0 -> J/psi K_S candidates in the v3 ALCARECO."""
import sys
from DataFormats.FWLite import Events, Handle


def main(paths, nmax=100000):
    events = Events(paths)
    h = Handle('std::vector<reco::VertexCompositeCandidate>')
    nshown = 0
    nev = 0
    for ev in events:
        nev += 1
        ev.getByLabel(('ALCARECOTkAlJpsiXB0KsResonances', '', 'RECO'), h)
        prod = h.product()
        if prod.size() == 0:
            continue
        for ic in range(prod.size()):
            c = prod[ic]
            print(f'cand {ic}: pdgId={c.pdgId()} mass={c.mass():.5f} pt={c.pt():.3f} '
                  f'ndau={c.numberOfDaughters()} vtx=({c.vx():.4f},{c.vy():.4f},{c.vz():.4f}) '
                  f'vtxChi2={c.vertexChi2():.3f} ndof={c.vertexNdof():.1f}')
            for jd in range(c.numberOfDaughters()):
                d = c.daughter(jd)
                print(f'   dau {jd}: pdgId={d.pdgId()} mass={d.mass():.6f} pt={d.pt():.3f} '
                      f'ndau={d.numberOfDaughters()} vtx=({d.vx():.4f},{d.vy():.4f},{d.vz():.4f})')
                for kd in range(d.numberOfDaughters()):
                    dd = d.daughter(kd)
                    tag = ''
                    print(f'      gdau {kd}: pdgId={dd.pdgId()} mass={dd.mass():.6f} '
                          f'pt={dd.pt():.3f} charge={dd.charge()} ndau={dd.numberOfDaughters()}')
            nshown += 1
        if nshown >= 5:
            break
    print(f'scanned {nev} events')


if __name__ == '__main__':
    main(sys.argv[1:])
