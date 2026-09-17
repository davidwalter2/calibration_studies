"""Generator/simulation decay bookkeeping for the B -> J/psi + X v3 MC.

Answers, on the sample itself:
  * which decayer handles K_S, D0 -> K pi, D*+ -> D0 pi  (Pythia8 / EvtGen /
    Geant4), by looking at whether the gen record contains the daughters;
  * whether those decays carry FSR photons in the gen record;
  * the gen-level invariant mass of the charged daughter pair, which is what a
    delta kernel would be placed at;
  * whether the SimTrack/SimVertex container holds anything other than the
    signal bunch crossing (premixed pileup keeps no pileup sim truth).
"""
import sys
import math
import collections

from DataFormats.FWLite import Events, Handle

MASS = {211: 0.13957039, 321: 0.493677, 310: 0.497611, 421: 1.86484,
        413: 2.01026, 13: 0.1056583745}


def main(paths, nmax=-1):
    events = Events(paths)
    h_gen = Handle('std::vector<reco::GenParticle>')
    h_simtrk = Handle('std::vector<SimTrack>')
    h_simvtx = Handle('std::vector<SimVertex>')

    c = collections.Counter()
    modes = collections.defaultdict(collections.Counter)
    masses = collections.defaultdict(list)
    simev = collections.Counter()
    nev = 0

    for ev in events:
        if nmax > 0 and nev >= nmax:
            break
        nev += 1
        ev.getByLabel(('genParticles', '', 'DIGI2RAW'), h_gen)
        gen = h_gen.product()

        for gp in gen:
            pid = gp.pdgId()
            apid = abs(pid)
            if apid not in (310, 421, 413, 511, 443):
                continue
            nd = gp.numberOfDaughters()
            daus = [gp.daughter(i).pdgId() for i in range(nd)]
            key = tuple(sorted(daus))
            name = {310: 'K_S', 421: 'D0', 413: 'Dstar', 511: 'B0', 443: 'Jpsi'}[apid]
            # only the LAST copy (no same-pdgId daughter) is a real decay
            if pid in daus or -pid in daus:
                c[f'{name}_copy'] += 1
                continue
            c[f'{name}_decays'] += 1
            modes[name][key] += 1
            ngam = sum(1 for d in daus if d == 22)
            c[f'{name}_withphoton'] += 1 if ngam else 0
            c[f'{name}_nphoton_{ngam}'] += 1
            # charged-pair invariant mass for the signature modes
            chg = [gp.daughter(i) for i in range(nd) if abs(gp.daughter(i).pdgId()) in (211, 321)]
            if name == 'D0' and len(chg) == 2 and {abs(x.pdgId()) for x in chg} == {211, 321}:
                e = sum(math.sqrt(x.p()**2 + MASS[abs(x.pdgId())]**2) for x in chg)
                px = sum(x.px() for x in chg); py = sum(x.py() for x in chg); pz = sum(x.pz() for x in chg)
                masses['D0_Kpi'].append(math.sqrt(max(e*e - px*px - py*py - pz*pz, 0.)))
                c['D0_Kpi_mode'] += 1
                if ngam:
                    c['D0_Kpi_withphoton'] += 1
            if name == 'Dstar' and len(daus) >= 2:
                d0 = [gp.daughter(i) for i in range(nd) if abs(gp.daughter(i).pdgId()) == 421]
                pi = [gp.daughter(i) for i in range(nd) if abs(gp.daughter(i).pdgId()) == 211]
                if len(d0) == 1 and len(pi) == 1:
                    c['Dstar_D0pi_mode'] += 1
                    if ngam:
                        c['Dstar_D0pi_withphoton'] += 1
            if name in ('D0', 'Dstar'):
                # strict mode bookkeeping: the two-body decay and the same
                # decay with one extra photon, which is what an FSR generator
                # (PHOTOS via EvtGen, or Pythia's ParticleDecays photon
                # radiation) would produce
                nogam = tuple(sorted(d for d in daus if d != 22))
                sig = {'D0': ({211, 321},), 'Dstar': ({211, 421},)}[name][0]
                if len(nogam) == 2 and {abs(d) for d in nogam} == sig:
                    c[f'{name}_2body'] += 1
                    c[f'{name}_2body_ngam{ngam}'] += 1
                    if name == 'D0':
                        ch = [gp.daughter(i) for i in range(nd)
                              if abs(gp.daughter(i).pdgId()) in (211, 321)]
                        e = sum(math.sqrt(x.p()**2 + MASS[abs(x.pdgId())]**2) for x in ch)
                        px = sum(x.px() for x in ch); py = sum(x.py() for x in ch)
                        pz = sum(x.pz() for x in ch)
                        masses['D0_Kpi_2body'].append(
                            math.sqrt(max(e*e - px*px - py*py - pz*pz, 0.)))
            if name == 'B0':
                if 443 in daus and 310 in daus:
                    c['B0_to_JpsiKs'] += 1
                if 443 in daus:
                    c['B0_to_Jpsi_X'] += 1

        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simtrk)
        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simvtx)
        for sv in h_simvtx.product():
            eid = sv.eventId()
            simev[(eid.bunchCrossing(), eid.event())] += 1

    print(f'=== events {nev} ===')
    for k, v in sorted(c.items()):
        print(f'  {k:28s} {v}  ({v/max(nev,1):.4f}/evt)')
    for name in ('K_S', 'D0', 'Dstar', 'B0'):
        print(f'--- {name} decay modes (top 10) ---')
        for k, v in modes[name].most_common(10):
            print(f'   {str(k):50s} {v}')
    for k, v in masses.items():
        v.sort()
        n = len(v)
        if n:
            print(f'{k}: n={n} median={v[n//2]:.6f} mean={sum(v)/n:.6f} '
                  f'min={v[0]:.6f} max={v[-1]:.6f}')
    print('--- SimVertex EncodedEventId (bx, event) populations ---')
    for k, v in simev.most_common(8):
        print(f'   bx={k[0]} event={k[1]}: {v} vertices')


if __name__ == '__main__':
    args = sys.argv[1:]
    nmax = -1
    files = []
    for a in args:
        if a.startswith('--nmax='):
            nmax = int(a.split('=', 1)[1])
        elif a.startswith('--filelist='):
            with open(a.split('=', 1)[1]) as f:
                files += [l.strip() for l in f if l.strip()]
        else:
            files.append(a)
    main(files, nmax)
