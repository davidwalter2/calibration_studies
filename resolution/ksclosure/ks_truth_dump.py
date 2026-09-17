"""Dump the Geant4 K_S -> pi+ pi- truth of the B -> J/psi + X v3 MC.

One row per simulated K_S -> pi+ pi- decay whose two pions are inside the
tracker acceptance, keyed by (run, lumi, event) -- unique in this campaign, so
the rows join offline onto the CVH two-track refit output.  The maker's own gen
matching is muon-specific, which is why the truth comes from a separate pass.

Premixed pileup keeps no pileup simulation truth (verified: every SimVertex in
this sample carries EncodedEventId bx=0 event=0), so every row here is from the
signal interaction.

Run: ./zchannel/fwlite.sh resolution/ksclosure/ks_truth_dump.py \
        --out=truth.npz --filelist=chunk.txt
"""
import sys
import math

import numpy as np
from DataFormats.FWLite import Events, Handle

B_HADRONS = {511, 521, 531, 541, 5122, 5132, 5232, 5332, 5142, 5242}
M_PI = 0.13957039
PT_MIN = 0.15
ETA_MAX = 2.6


def main(paths, out, nmax=-1,
         candsrc=('ALCARECOTkAlJpsiXB0KsResonances', '', 'RECO')):
    h_gen = Handle('std::vector<reco::GenParticle>')
    h_bc = Handle('std::vector<int>')
    h_simtrk = Handle('std::vector<SimTrack>')
    h_simvtx = Handle('std::vector<SimVertex>')
    h_cand = Handle('std::vector<reco::VertexCompositeCandidate>')

    cols = {k: [] for k in (
        'run', 'lumi', 'event', 'vx', 'vy', 'vz', 'kspx', 'kspy', 'kspz',
        'pxp', 'pyp', 'pzp', 'pxm', 'pym', 'pzm', 'fromb', 'motherpdg',
        'prodr', 'prodz')}
    nev = 0
    nbad = 0
    nskip = 0
    # One Events object per file: the production has a tail of zero-length and
    # truncated files (cmsRun handles them with skipBadFiles, FWLite has no
    # such option) and a single bad file must not take the chunk down.
    for path in paths:
        try:
            events = Events([path])
            niter = 0
            for ev in events:
                niter += 1
                if nmax > 0 and nev >= nmax:
                    break
                nev += 1
                # ONLY events that carry a K_S candidate can contribute a row
                # to the join, and reading the candidate collection is far
                # cheaper than the SimTrack/SimVertex containers (7.8 M
                # vertices per hundred events). 83 % of events are skipped
                # here, which is what makes the truth pass cheaper than the
                # refit it has to keep up with.
                if candsrc:
                    ev.getByLabel(candsrc, h_cand)
                    if h_cand.product().size() == 0:
                        nskip += 1
                        continue
                _fill(ev, cols, h_gen, h_bc, h_simtrk, h_simvtx)
        except Exception as e:
            nbad += 1
            print(f'SKIPPING {path}: {type(e).__name__}')
            continue
        if nmax > 0 and nev >= nmax:
            break
    if nbad:
        print(f'{nbad} input files skipped (unreadable)')
    if candsrc:
        print(f'{nskip} of {nev} events had no {candsrc[0]} candidate and '
              f'were not unpacked')

    arr = {k: np.array(v, dtype=(np.int64 if k in ('run', 'lumi', 'event', 'fromb', 'motherpdg')
                                 else np.float64)) for k, v in cols.items()}
    arr['nevents'] = np.array([nev], dtype=np.int64)
    arr['nbadfiles'] = np.array([nbad], dtype=np.int64)
    arr['nskipped'] = np.array([nskip], dtype=np.int64)
    np.savez_compressed(out, **arr)
    print(f'{nev} events, {len(cols["run"])} K_S -> pipi rows -> {out}')


def _fill(ev, cols, h_gen, h_bc, h_simtrk, h_simvtx):
        aux = ev.eventAuxiliary()
        run, lumi, evt = aux.run(), aux.luminosityBlock(), aux.event()

        ev.getByLabel(('genParticles', '', 'DIGI2RAW'), h_gen)
        gen = h_gen.product()
        ev.getByLabel(('genParticles', '', 'DIGI2RAW'), h_bc)
        bc2idx = {b: i for i, b in enumerate(h_bc.product())}
        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simtrk)
        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simvtx)
        simtrk = h_simtrk.product()
        simvtx = h_simvtx.product()

        children = {}
        for it, st in enumerate(simtrk):
            iv = st.vertIndex()
            if iv >= 0:
                children.setdefault(iv, []).append(it)
        vtx_of_parent = {}
        for iv, sv in enumerate(simvtx):
            pi = sv.parentIndex()
            if pi >= 0 and pi not in vtx_of_parent:
                vtx_of_parent[pi] = iv

        for st in simtrk:
            if st.type() != 310:
                continue
            iv = vtx_of_parent.get(st.trackId())
            if iv is None:
                continue
            kids = children.get(iv, [])
            if len(kids) != 2:
                continue
            types = sorted(simtrk[k].type() for k in kids)
            if types != [-211, 211]:
                continue
            mom = {}
            ok = True
            for k in kids:
                m = simtrk[k].momentum()
                pt = math.hypot(m.px(), m.py())
                if pt < PT_MIN:
                    ok = False
                    break
                if abs(math.asinh(m.pz() / pt)) > ETA_MAX:
                    ok = False
                    break
                mom[1 if simtrk[k].type() == 211 else -1] = (m.px(), m.py(), m.pz())
            if not ok or len(mom) != 2:
                continue
            sv = simvtx[iv]
            pos = sv.position()
            # production point of the K_S (its own origin vertex)
            pv = simvtx[st.vertIndex()].position() if st.vertIndex() >= 0 else pos
            fromb, mpdg = 0, 0
            gi = st.genpartIndex()
            if gi in bc2idx:
                node, seen = gen[bc2idx[gi]], 0
                if node.numberOfMothers() > 0:
                    mpdg = node.mother(0).pdgId()
                while node.numberOfMothers() > 0 and seen < 30:
                    node = node.mother(0)
                    seen += 1
                    if abs(node.pdgId()) in B_HADRONS:
                        fromb = abs(node.pdgId())
                        break
            ksm = st.momentum()
            cols['run'].append(run); cols['lumi'].append(lumi); cols['event'].append(evt)
            cols['vx'].append(pos.x()); cols['vy'].append(pos.y()); cols['vz'].append(pos.z())
            cols['kspx'].append(ksm.px()); cols['kspy'].append(ksm.py()); cols['kspz'].append(ksm.pz())
            cols['pxp'].append(mom[1][0]); cols['pyp'].append(mom[1][1]); cols['pzp'].append(mom[1][2])
            cols['pxm'].append(mom[-1][0]); cols['pym'].append(mom[-1][1]); cols['pzm'].append(mom[-1][2])
            cols['fromb'].append(fromb); cols['motherpdg'].append(mpdg)
            cols['prodr'].append(math.hypot(pv.x(), pv.y())); cols['prodz'].append(pv.z())

if __name__ == '__main__':
    out, nmax, files, allev = None, -1, [], False
    for a in sys.argv[1:]:
        if a.startswith('--out='):
            out = a.split('=', 1)[1]
        elif a.startswith('--nmax='):
            nmax = int(a.split('=', 1)[1])
        elif a.startswith('--filelist='):
            with open(a.split('=', 1)[1]) as f:
                files += [l.strip() for l in f if l.strip()]
        elif a == '--all-events':
            allev = True
        elif a.startswith('--input='):
            files += [p for p in a.split('=', 1)[1].split(',') if p]
        else:
            files.append(a)
    assert out, 'need --out='
    main(files, out, nmax,
         candsrc=None if allev else ('ALCARECOTkAlJpsiXB0KsResonances', '', 'RECO'))
