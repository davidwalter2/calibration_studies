"""Sample inspection for the K_S closure test.

Reads B->J/psi+X v3 MC ALCARECO files with FWLite and reports:
  - track / candidate collection sizes
  - how K_S decays are represented (Pythia gen decay vs Geant4 SimVertex)
  - photon multiplicity in the K_S decay
  - the sim-level pi+pi- invariant mass
Run with: ./zchannel/fwlite.sh resolution/ksclosure/inspect_sample.py [files...]
"""
import sys
import math
import collections

from DataFormats.FWLite import Events, Handle

M_KS = 0.497611
M_PI = 0.13957039


def main(paths, nmax=2000):
    events = Events(paths)

    h_gen = Handle('std::vector<reco::GenParticle>')
    h_simtrk = Handle('std::vector<SimTrack>')
    h_simvtx = Handle('std::vector<SimVertex>')
    h_trk = Handle('std::vector<reco::Track>')
    h_b0ks = Handle('std::vector<reco::VertexCompositeCandidate>')
    h_jpsi = Handle('std::vector<reco::VertexCompositeCandidate>')

    counts = collections.Counter()
    ks_daughters = collections.Counter()
    masses = []
    ntrk = []
    nb0ks = []
    njpsi = []

    for iev, ev in enumerate(events):
        if iev >= nmax:
            break
        counts['events'] += 1
        ev.getByLabel(('genParticles', '', 'DIGI2RAW'), h_gen)
        gen = h_gen.product()
        ev.getByLabel(('ALCARECOTkAlJpsiX', '', 'RECO'), h_trk)
        ntrk.append(h_trk.product().size())
        ev.getByLabel(('ALCARECOTkAlJpsiXB0KsResonances', '', 'RECO'), h_b0ks)
        nb0ks.append(h_b0ks.product().size())
        ev.getByLabel(('ALCARECOTkAlJpsiXJpsiOnlyResonances', '', 'RECO'), h_jpsi)
        njpsi.append(h_jpsi.product().size())

        if iev == 0:
            print('--- first event candidate structure ---')
            for name, h in (('B0Ks', h_b0ks), ('JpsiOnly', h_jpsi)):
                prod = h.product()
                print(f'{name}: n={prod.size()}')
                for ic in range(min(3, prod.size())):
                    c = prod[ic]
                    print(f'   cand {ic}: pdgId={c.pdgId()} mass={c.mass():.5f} '
                          f'pt={c.pt():.3f} ndau={c.numberOfDaughters()} '
                          f'vtx=({c.vx():.4f},{c.vy():.4f},{c.vz():.4f})')
                    for jd in range(c.numberOfDaughters()):
                        d = c.daughter(jd)
                        print(f'      dau {jd}: pdgId={d.pdgId()} mass={d.mass():.5f} '
                              f'pt={d.pt():.3f} ndau={d.numberOfDaughters()}')
                        for kd in range(d.numberOfDaughters()):
                            dd = d.daughter(kd)
                            print(f'         gdau {kd}: pdgId={dd.pdgId()} '
                                  f'mass={dd.mass():.5f} pt={dd.pt():.3f}')

        # gen-level K_S
        for gp in gen:
            if abs(gp.pdgId()) != 310:
                continue
            counts['gen_ks'] += 1
            key = tuple(sorted(gp.daughter(i).pdgId() for i in range(gp.numberOfDaughters())))
            ks_daughters[key] += 1
            counts[f'gen_ks_status{gp.status()}'] += 1

        # sim-level: K_S SimTrack -> decay SimVertex with two pion SimTracks
        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simtrk)
        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simvtx)
        simtrk = h_simtrk.product()
        simvtx = h_simvtx.product()
        # map g4 trackId -> SimTrack
        by_id = {}
        for st in simtrk:
            by_id[st.trackId()] = st
        # children by parent vertex
        children = collections.defaultdict(list)
        for st in simtrk:
            if st.vertIndex() >= 0:
                children[st.vertIndex()].append(st)
        for st in simtrk:
            if abs(st.type()) != 310:
                continue
            counts['sim_ks'] += 1
            # find its decay vertex: a vertex whose parentIndex == st.trackId()
            for iv, sv in enumerate(simvtx):
                if sv.parentIndex() != st.trackId():
                    continue
                kids = children.get(iv, [])
                ptypes = tuple(sorted(k.type() for k in kids))
                counts['sim_ks_decay'] += 1
                ks_daughters[('SIM',) + ptypes] += 1
                if ptypes == (-211, 211):
                    px = py = pz = e = 0.0
                    for k in kids:
                        m = k.momentum()
                        px += m.px(); py += m.py(); pz += m.pz()
                        e += math.sqrt(m.px()**2 + m.py()**2 + m.pz()**2 + M_PI**2)
                    mm = math.sqrt(max(e*e - px*px - py*py - pz*pz, 0.0))
                    masses.append((mm, math.sqrt(sv.position().x()**2 + sv.position().y()**2),
                                   sv.position().z(), sv.processType()))
                break

    print('--- counts ---')
    for k, v in sorted(counts.items()):
        print(f'{k:24s} {v}')
    print('--- K_S daughter patterns ---')
    for k, v in ks_daughters.most_common(20):
        print(f'{str(k):40s} {v}')
    if ntrk:
        import statistics
        print(f'tracks/event mean {statistics.mean(ntrk):.2f}  max {max(ntrk)}')
        print(f'B0Ks cands/event mean {statistics.mean(nb0ks):.4f}  max {max(nb0ks)} '
              f'nonzero {sum(1 for x in nb0ks if x)}')
        print(f'JpsiOnly cands/event mean {statistics.mean(njpsi):.3f}')
    if masses:
        ms = [m[0] for m in masses]
        ms.sort()
        n = len(ms)
        print(f'sim pi+pi- mass: n={n} median={ms[n//2]:.6f} '
              f'mean={sum(ms)/n:.6f} min={ms[0]:.6f} max={ms[-1]:.6f}')
        dev = [abs(m - M_KS) for m in ms]
        dev.sort()
        print(f'  |m - M_KS| median={dev[n//2]:.3e} p99={dev[int(0.99*n)]:.3e}')


if __name__ == '__main__':
    args = sys.argv[1:]
    nmax = 2000
    files = []
    for a in args:
        if a.startswith('--nmax='):
            nmax = int(a.split('=')[1])
        else:
            files.append(a)
    main(files, nmax)
