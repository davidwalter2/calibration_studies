#!/usr/bin/env python3
"""Dump the pre-FSR -> post-FSR muon pair **matched leg by leg, by charge**.

`dump_gen_fsr.py` sorts the pre-FSR and the post-FSR muons by ``pT``
*independently*, which loses the per-leg correspondence whenever the softer
muon radiates past the harder one (6.3 % of the radiating events).  The per-leg
factorised FSR kernel needs the correspondence itself, so this dumper writes the
two legs labelled by **charge**, which is unambiguous: Photos keeps a
``status == 746`` copy of each muon it touched, with the same ``pdgId``.

Per leg ``q = +, -``:  ``ptq``, ``etaq``, ``phiq`` (status 1, post-FSR) and
``ptq_pre``, ``etaq_pre``, ``phiq_pre`` (status 746, or the status-1 muon itself
when Photos did not radiate).  ``x_q = E'_q/E_q`` in the **Z rest frame** is the
quantity the collinear factorisation calls ``x``; it is recomputed offline from
the stored four-vectors, so ``eq``/``eq_pre`` (lab energies) and the Z
four-momentum are written too.

Run under CMSSW 15_0 FWLite (``./fwlite.sh``) on submit50/51/52.
"""

import argparse
import math
import os
import time

import numpy as np

DR_DRESS = 0.1

COLS = (
    "run", "lumi", "event", "weight",
    "m_pre", "m_post", "npre", "nph",
    "pt_pre", "y_pre", "pz_pre", "e_pre",
    "ptp", "etap", "phip", "ep", "ptm", "etam", "phim", "em",
    "ptp_pre", "etap_pre", "phip_pre", "ep_pre",
    "ptm_pre", "etam_pre", "phim_pre", "em_pre",
)


def m4(ps):
    e = sum(p.energy() for p in ps)
    x = sum(p.px() for p in ps)
    y = sum(p.py() for p in ps)
    z = sum(p.pz() for p in ps)
    v = e * e - x * x - y * y - z * z
    return math.sqrt(v) if v > 0.0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filelist")
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--nfiles", type=int, default=1)
    p.add_argument("--skip-files", type=int, default=0)
    p.add_argument("--nmax", type=int, default=0)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--progress", type=int, default=0)
    args = p.parse_args()

    from DataFormats.FWLite import Events, Handle

    files = list(args.files)
    if args.filelist:
        with open(args.filelist) as fh:
            allf = [l.strip() for l in fh if l.strip()]
        allf = [f[5:] if f.startswith("file:") else f for f in allf]
        files += allf[args.skip_files: args.skip_files + args.nfiles]
    if not files:
        raise SystemExit("no input files")

    h_pruned = Handle("std::vector<reco::GenParticle>")
    h_gen = Handle("GenEventInfoProduct")
    cols = {k: [] for k in COLS}
    nseen = nbad = 0
    t0 = time.time()

    for path in files:
        nfile = 0
        for e in Events(path):
            if args.nmax and nfile >= args.nmax:
                break
            nfile += 1
            nseen += 1
            if args.progress and nseen % args.progress == 0:
                print(f"  {nseen} events, {time.time()-t0:.0f} s", flush=True)

            e.getByLabel("prunedGenParticles", h_pruned)
            zb, post, pre746, phot = [], [], [], []
            for g in h_pruned.product():
                pid = abs(g.pdgId())
                if pid == 13:
                    st = g.status()
                    if st == 1 and g.statusFlags().isHardProcess():
                        post.append(g)
                    elif st == 746:
                        pre746.append(g)
                elif pid == 23 and g.status() == 62:
                    zb.append(g)
                elif pid == 22 and g.status() == 1 and g.statusFlags().isPrompt():
                    phot.append(g)

            if len(zb) != 1 or len(post) != 2:
                nbad += 1
                continue
            # pdgId 13 = mu-, -13 = mu+
            po = {g.pdgId(): g for g in post}
            if len(po) != 2:
                nbad += 1
                continue
            src = pre746 if len(pre746) == 2 else post
            pr = {g.pdgId(): g for g in src}
            if len(pr) != 2:
                nbad += 1
                continue

            nph = 0
            for g in phot:
                for q in post:
                    de = g.eta() - q.eta()
                    dp = abs(g.phi() - q.phi())
                    if dp > math.pi:
                        dp = 2.0 * math.pi - dp
                    if de * de + dp * dp < DR_DRESS ** 2:
                        nph += 1
                        break

            ppz = sum(q.pz() for q in src)
            pen = sum(q.energy() for q in src)
            ppx = sum(q.px() for q in src)
            ppy = sum(q.py() for q in src)

            e.getByLabel("generator", h_gen)
            aux = e.eventAuxiliary()
            mp, mm = po[-13], po[13]
            rp, rm = pr[-13], pr[13]
            v = dict(
                run=aux.run(), lumi=aux.luminosityBlock(), event=aux.event(),
                weight=h_gen.product().weight(),
                m_pre=zb[0].mass(), m_post=m4(post),
                npre=len(pre746), nph=nph,
                pt_pre=math.hypot(ppx, ppy),
                y_pre=(0.5 * math.log((pen + ppz) / (pen - ppz))
                       if abs(ppz) < pen else 0.0),
                pz_pre=ppz, e_pre=pen,
                ptp=mp.pt(), etap=mp.eta(), phip=mp.phi(), ep=mp.energy(),
                ptm=mm.pt(), etam=mm.eta(), phim=mm.phi(), em=mm.energy(),
                ptp_pre=rp.pt(), etap_pre=rp.eta(), phip_pre=rp.phi(),
                ep_pre=rp.energy(),
                ptm_pre=rm.pt(), etam_pre=rm.eta(), phim_pre=rm.phi(),
                em_pre=rm.energy(),
            )
            for k in COLS:
                cols[k].append(v[k])

    out = {k: np.asarray(cols[k]) for k in COLS}
    for k in ("run", "lumi", "event", "nph", "npre"):
        out[k] = out[k].astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"[dump_gen_perleg] {nseen} read, {len(out['m_pre'])} kept, {nbad} bad, "
          f"{time.time()-t0:.0f} s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
