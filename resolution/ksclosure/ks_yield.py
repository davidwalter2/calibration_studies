"""K_S yield, purity and truth study on the B -> J/psi + X v3 MC ALCARECO.

Truth matching of a charged daughter uses quantities a solenoid conserves:
the DIP ANGLE lambda (exactly conserved by a z-field) and |p| (changed only by
energy loss).  The azimuth of the momentum rotates between the decay vertex and
the innermost hit, so it is only used as a loose consistency cut -- matching on
deltaR alone fails for soft displaced daughters (a 0.3 GeV pion turns ~60 mrad
over 1.5 cm).

Reports
  * gen/sim K_S census and decay modes
  * sim K_S -> pi+pi- with both daughters present in the ALCARECO track
    collection (the reconstructable ceiling of this ALCARECO)
  * the reconstructed B0 -> J/psi K_S candidate collection: yield, PURITY
    (both daughters of the candidate matched to the two pions of ONE sim K_S),
    decay radius, daughter momenta
  * the B-daughter subset

Run: ./zchannel/fwlite.sh resolution/ksclosure/ks_yield.py --filelist=... --out=X.npz
"""
import sys
import math
import collections

import numpy as np
from DataFormats.FWLite import Events, Handle

M_PI = 0.13957039
M_KS_PDG = 0.497611
B_HADRONS = {511, 521, 531, 541, 5122, 5132, 5232, 5332, 5142, 5242}


def p3(v):
    return np.array([v.px(), v.py(), v.pz()], dtype=float)


def lam_phi_p(v):
    pt = math.hypot(v[0], v[1])
    p = math.sqrt(v.dot(v))
    return (math.atan2(v[2], pt) if pt > 0 else math.copysign(math.pi / 2, v[2]),
            math.atan2(v[1], v[0]), p)


def dphi(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def main(paths, out=None, nmax=-1):
    events = Events(paths)

    h_gen = Handle('std::vector<reco::GenParticle>')
    h_bc = Handle('std::vector<int>')
    h_simtrk = Handle('std::vector<SimTrack>')
    h_simvtx = Handle('std::vector<SimVertex>')
    h_trk = Handle('std::vector<reco::Track>')
    h_b0ks = Handle('std::vector<reco::VertexCompositeCandidate>')

    c = collections.Counter()
    modes = collections.Counter()
    rows_sim, rows_reco = [], []
    nev = 0

    for ev in events:
        if nmax > 0 and nev >= nmax:
            break
        nev += 1
        c['events'] += 1

        ev.getByLabel(('genParticles', '', 'DIGI2RAW'), h_gen)
        gen = h_gen.product()
        ev.getByLabel(('genParticles', '', 'DIGI2RAW'), h_bc)
        bc2idx = {b: i for i, b in enumerate(h_bc.product())}

        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simtrk)
        ev.getByLabel(('g4SimHits', '', 'DIGI2RAW'), h_simvtx)
        simtrk = h_simtrk.product()
        simvtx = h_simvtx.product()

        ev.getByLabel(('ALCARECOTkAlJpsiX', '', 'RECO'), h_trk)
        tracks = h_trk.product()
        c['tracks'] += tracks.size()

        ev.getByLabel(('ALCARECOTkAlJpsiXB0KsResonances', '', 'RECO'), h_b0ks)
        b0ks = h_b0ks.product()
        c['b0ks_cands'] += b0ks.size()

        for gp in gen:
            if gp.pdgId() == 310:
                c['gen_ks'] += 1

        # reco track summary, indexed
        trk = []
        for it, t in enumerate(tracks):
            v = p3(t)
            try:
                iv = t.innerMomentum()
                vin = np.array([iv.x(), iv.y(), iv.z()])
            except Exception:
                vin = v
            lam, phi, p = lam_phi_p(vin)
            trk.append(dict(idx=it, q=t.charge(), lam=lam, phi=phi, p=p,
                            pref=v, nvalid=t.numberOfValidHits(),
                            pt=t.pt(), eta=t.eta()))

        def match_track(pv, q):
            """best ALCARECO track for a sim daughter momentum pv of charge q."""
            lam, phi, p = lam_phi_p(pv)
            best, bestd = None, 1e9
            for t in trk:
                if t['q'] != q:
                    continue
                dl = abs(t['lam'] - lam)
                if dl > 0.06:
                    continue
                dp = abs(t['p'] - p) / p
                if dp > 0.15:
                    continue
                if dphi(t['phi'], phi) > 0.5:
                    continue
                d = dl / 0.06 + dp / 0.15
                if d < bestd:
                    best, bestd = t, d
            return best

        children = collections.defaultdict(list)
        for it, st in enumerate(simtrk):
            if st.vertIndex() >= 0:
                children[st.vertIndex()].append(it)
        vtx_of_parent = {}
        for iv, sv in enumerate(simvtx):
            if sv.parentIndex() >= 0:
                vtx_of_parent.setdefault(sv.parentIndex(), iv)

        simks = []    # reconstructable ones (both daughters on ALCARECO tracks)
        simpipi = []  # every sim K_S -> pi+ pi- of the event
        for st in simtrk:
            if st.type() != 310:
                continue
            c['sim_ks'] += 1
            iv = vtx_of_parent.get(st.trackId())
            if iv is None:
                continue
            kids = children.get(iv, [])
            ptypes = tuple(sorted(simtrk[k].type() for k in kids))
            modes[ptypes] += 1
            if ptypes != (-211, 211):
                continue
            c['sim_ks_pipi'] += 1
            sv = simvtx[iv]
            vx, vy, vz = sv.position().x(), sv.position().y(), sv.position().z()
            dmoms = [p3(simtrk[k].momentum()) for k in kids]
            dch = [1 if simtrk[k].type() == 211 else -1 for k in kids]
            ksp = p3(st.momentum())
            fromb = 0
            gi = st.genpartIndex()
            if gi in bc2idx:
                node, seen = gen[bc2idx[gi]], 0
                while node.numberOfMothers() > 0 and seen < 30:
                    node = node.mother(0)
                    seen += 1
                    if abs(node.pdgId()) in B_HADRONS:
                        fromb = abs(node.pdgId())
                        break
            mt = [match_track(m, q) for m, q in zip(dmoms, dch)]
            idxs = {t['idx'] for t in mt if t is not None}
            nm = len(idxs)
            rec = dict(
                rdec=math.hypot(vx, vy), zdec=vz, vx=vx, vy=vy, vz=vz,
                kspt=math.hypot(ksp[0], ksp[1]), ksp=math.sqrt(ksp.dot(ksp)),
                kseta=math.asinh(ksp[2] / max(math.hypot(ksp[0], ksp[1]), 1e-9)),
                pt1=math.hypot(dmoms[0][0], dmoms[0][1]),
                pt2=math.hypot(dmoms[1][0], dmoms[1][1]),
                p1=math.sqrt(dmoms[0].dot(dmoms[0])),
                p2=math.sqrt(dmoms[1].dot(dmoms[1])),
                eta1=math.asinh(dmoms[0][2] / max(math.hypot(dmoms[0][0], dmoms[0][1]), 1e-9)),
                eta2=math.asinh(dmoms[1][2] / max(math.hypot(dmoms[1][0], dmoms[1][1]), 1e-9)),
                fromb=fromb, nmatched=nm,
                minhits=min(t['nvalid'] for t in mt) if nm == 2 else -1,
            )
            rec['dmoms'] = dmoms
            rec['dch'] = dch
            simpipi.append(rec)
            rows_sim.append({k: v for k, v in rec.items()
                             if k not in ('dmoms', 'dch')})
            if nm == 2:
                c['sim_ks_pipi_bothtrk'] += 1
                simks.append((idxs, rec))

        # reconstructed K_S candidates (sub-candidate 1 of the B0Ks pairing)
        for ic in range(b0ks.size()):
            cand = b0ks[ic]
            if cand.numberOfDaughters() < 2:
                continue
            ks = cand.daughter(1)
            if ks.numberOfDaughters() < 2:
                continue
            # Truth match directly on the DAUGHTER MOMENTA. The V0-style
            # producer refits the daughters at the decay vertex, so their
            # directions are the sim pion directions there -- no bending
            # correction is needed and no TrackRef resolution (a PyROOT
            # dynamic_cast to RecoChargedCandidate) is required.
            dcand = []
            for jd in range(2):
                d = ks.daughter(jd)
                dcand.append((np.array([d.px(), d.py(), d.pz()]), d.charge()))
            kv = np.array([ks.vx(), ks.vy(), ks.vz()])
            truth = None
            for rec in simpipi:
                ok = 0
                for pv, q in dcand:
                    lam, phi, p = lam_phi_p(pv)
                    for sm, sq in zip(rec['dmoms'], rec['dch']):
                        if sq != q:
                            continue
                        sl, sp, spp = lam_phi_p(sm)
                        if (abs(sl - lam) < 0.05 and dphi(sp, phi) < 0.1
                                and abs(spp - p) / p < 0.2):
                            ok += 1
                            break
                if ok == 2:
                    truth = rec
                    break
            rows_reco.append(dict(
                mass=ks.mass(), bmass=cand.mass(), kspt=ks.pt(), kseta=ks.eta(),
                rdec=math.hypot(kv[0], kv[1]), zdec=kv[2],
                dvtx=(math.dist(kv, (truth['vx'], truth['vy'], truth['vz']))
                      if truth else -1.0),
                simboth=truth['nmatched'] if truth else -1,
                matched=1 if truth else 0,
                fromb=truth['fromb'] if truth else -1,
                simr=truth['rdec'] if truth else -1.0,
                dpt1=ks.daughter(0).pt(), dpt2=ks.daughter(1).pt(),
                resolved=1,
            ))
            c['reco_ks'] += 1
            if truth:
                c['reco_ks_truth'] += 1
                if truth['fromb']:
                    c['reco_ks_fromb'] += 1
                if truth['fromb'] == 511:
                    c['reco_ks_fromB0'] += 1

    print(f'=== events {nev} ===')
    for k, v in sorted(c.items()):
        print(f'  {k:28s} {v}  ({v/max(nev,1):.4f}/evt)')
    print('=== sim K_S decay modes (top 8) ===')
    for k, v in modes.most_common(8):
        print(f'  {str(k):40s} {v}')

    if rows_sim:
        a = {k: np.array([r[k] for r in rows_sim]) for k in rows_sim[0]}
        n = len(rows_sim)
        both = a['nmatched'] == 2
        print(f'=== sim K_S -> pipi: {n}; both daughters in the ALCARECO: '
              f'{int(both.sum())} ({100*both.mean():.3f} %) = {both.sum()/nev:.4f}/evt ===')
        print(f'  from a B hadron {100*(a["fromb"]>0).mean():.2f} % (all), '
              f'{100*(a["fromb"][both]>0).mean():.1f} % (reconstructable)')
        for nm, sel in (('all', np.ones(n, bool)), ('both-trk', both)):
            if sel.sum() == 0:
                continue
            print(f'  [{nm}] r: q10 {np.quantile(a["rdec"][sel],.1):.2f} med '
                  f'{np.median(a["rdec"][sel]):.2f} q90 {np.quantile(a["rdec"][sel],.9):.2f} cm | '
                  f'|z| med {np.median(abs(a["zdec"][sel])):.2f} | K_S pt med '
                  f'{np.median(a["kspt"][sel]):.2f} | pi pt med '
                  f'{np.median(np.minimum(a["pt1"],a["pt2"])[sel]):.3f}/'
                  f'{np.median(np.maximum(a["pt1"],a["pt2"])[sel]):.3f} | '
                  f'pi p med {np.median(np.minimum(a["p1"],a["p2"])[sel]):.3f}/'
                  f'{np.median(np.maximum(a["p1"],a["p2"])[sel]):.3f}')
        if both.sum():
            print(f'  reconstructable, r<4/20/60 cm: '
                  f'{100*(a["rdec"][both]<4).mean():.1f}/'
                  f'{100*(a["rdec"][both]<20).mean():.1f}/'
                  f'{100*(a["rdec"][both]<60).mean():.1f} %; '
                  f'min nvalid>=8 {100*(a["minhits"][both]>=8).mean():.1f} %')
    if rows_reco:
        b = {k: np.array([r[k] for r in rows_reco]) for k in rows_reco[0]}
        print(f'=== reco K_S candidates: {len(rows_reco)} '
              f'({len(rows_reco)/nev:.4f}/evt); daughter tracks resolved '
              f'{100*b["resolved"].mean():.1f} % ===')
        print(f'  PURITY (both daughters = one sim K_S): {100*b["matched"].mean():.1f} % '
              f'-> {b["matched"].sum()/nev:.4f} true K_S/evt')
        m = b['matched'] > 0
        if m.sum():
            print(f'  of the true ones: from a B {100*(b["fromb"][m]>0).mean():.1f} %, '
                  f'B0 {100*(b["fromb"][m]==511).mean():.1f} %; '
                  f'reco-sim vertex distance med {np.median(b["dvtx"][m])*1e4:.0f} um, '
                  f'q90 {np.quantile(b["dvtx"][m],.9)*1e4:.0f} um')
            print(f'  true K_S: pre-refit mass q05/med/q95 '
                  f'{np.quantile(b["mass"][m],.05):.4f}/{np.median(b["mass"][m]):.4f}/'
                  f'{np.quantile(b["mass"][m],.95):.4f}; r med '
                  f'{np.median(b["rdec"][m]):.2f} cm; daughter pt med '
                  f'{np.median(np.minimum(b["dpt1"],b["dpt2"])[m]):.3f}')
            print(f'  fakes: mass q05/med/q95 '
                  f'{np.quantile(b["mass"][~m],.05):.4f}/{np.median(b["mass"][~m]):.4f}/'
                  f'{np.quantile(b["mass"][~m],.95):.4f}')
    if out:
        np.savez_compressed(
            out,
            **{f'sim_{k}': np.array([r[k] for r in rows_sim]) for k in (rows_sim[0] if rows_sim else [])},
            **{f'reco_{k}': np.array([r[k] for r in rows_reco]) for k in (rows_reco[0] if rows_reco else [])},
            nevents=np.array([nev]))
        print('wrote', out)


if __name__ == '__main__':
    args = sys.argv[1:]
    out, nmax, files = None, -1, []
    for a in args:
        if a.startswith('--out='):
            out = a.split('=', 1)[1]
        elif a.startswith('--nmax='):
            nmax = int(a.split('=', 1)[1])
        elif a.startswith('--filelist='):
            with open(a.split('=', 1)[1]) as f:
                files += [l.strip() for l in f if l.strip()]
        else:
            files.append(a)
    main(files, out, nmax)
