#!/usr/bin/env python3
"""Dump pre-FSR / post-FSR / dressed generator dimuon masses from DY MiniAOD.

The Z channel of the unbinned CVH mass likelihood models the *pre-FSR* (Born)
dimuon mass with ``rabbit.lineshapes.ZGammaLineshape``; what the tracker
measures is the *post-FSR bare* muon pair.  The two are related by the QED
final-state-radiation kernel, which this script samples event by event:

    dm      = m_postFSR - m_preFSR          (the additive FSR kernel sample)
    dlogm   = log(m_postFSR / m_preFSR)     (the multiplicative variant)

Generator record of ``DYJetsToMuMu_..._powhegMiNNLO-pythia8-photos`` MiniAOD
(verified on RunIISummer20UL16MiniAODv2, 4000 events):

``m_preFSR``  the mass of the hard-process Z.  ``prunedGenParticles`` holds it
              at ``status == 22`` (first copy) and ``status == 62`` (last copy,
              flagged ``fromHardProcessBeforeFSR``); the two masses are
              *bit-identical* in 4000/4000 events, and both equal the mass of
              the ``status == 746`` muon pair -- the pre-Photos muons, which
              CMS' Photos interface keeps alongside the radiated ones -- to an
              RMS of 4e-6 GeV.  The 746 pair is present only in the 58.6 % of
              events where Photos actually radiated, so the Z mass is the
              definition used here; ``m_pre746`` is written out as a check.
``m_postFSR`` the two ``status == 1`` prompt hard-process muons.  This is the
              bare pair the CVH refit sees and what the maker stores as
              ``Jpsigen_mass``.
``m_dressed`` the bare pair with every prompt final-state photon inside
              ``dR < 0.1`` of either muon added back (reference only; the
              likelihood uses bare muons).

Caveat: ``prunedGenParticles`` in MiniAOD is stored at reduced float precision,
which puts a ~0.1 MeV floor on ``dm`` (the median |m_Z62 - m_mumu| in events
with no Photos radiation is 0.13 MeV).  That is far below the kernel's own
width and irrelevant for the fit.

Run under CMSSW 15_0 FWLite (``./fwlite.sh``) on a node that has both
``/ceph/submit`` mounted and AVX2 (submit50/51/52 -- *not* submit60).

    ./fwlite.sh dump_gen_fsr.py --filelist <paths.txt> --nfiles 8 \
        --nmax 20000 -o data/gen_fsr.npz
"""

import argparse
import math
import os
import time

import numpy as np

DR_DRESS = 0.1


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--filelist", help="text file with one MiniAOD path per line")
    p.add_argument("--files", nargs="*", default=[], help="explicit MiniAOD paths")
    p.add_argument("--nfiles", type=int, default=1, help="how many from --filelist")
    p.add_argument("--skip-files", type=int, default=0, help="skip the first N files")
    p.add_argument("--nmax", type=int, default=0, help="max events per file (0 = all)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--dr-dress", type=float, default=DR_DRESS)
    p.add_argument("--progress", type=int, default=50000)
    return p.parse_args()


def dr2(a, b):
    de = a.eta() - b.eta()
    dp = abs(a.phi() - b.phi())
    if dp > math.pi:
        dp = 2.0 * math.pi - dp
    return de * de + dp * dp


def m4(ps, extra=(0.0, 0.0, 0.0, 0.0)):
    e = sum(p.energy() for p in ps) + extra[3]
    x = sum(p.px() for p in ps) + extra[0]
    y = sum(p.py() for p in ps) + extra[1]
    z = sum(p.pz() for p in ps) + extra[2]
    v = e * e - x * x - y * y - z * z
    return math.sqrt(v) if v > 0.0 else 0.0


COLS = (
    "run", "lumi", "event",
    "m_pre", "m_pre746", "m_post", "m_dress",
    "pt_pre", "y_pre", "nph", "eph",
    "pt1", "eta1", "pt2", "eta2",
    "pt1_pre", "eta1_pre", "pt2_pre", "eta2_pre",
)


def main():
    args = parse_args()
    from DataFormats.FWLite import Events, Handle

    files = list(args.files)
    if args.filelist:
        with open(args.filelist) as fh:
            allf = [l.strip() for l in fh if l.strip()]
        allf = [f[5:] if f.startswith("file:") else f for f in allf]
        files += allf[args.skip_files: args.skip_files + args.nfiles]
    if not files:
        raise SystemExit("no input files")
    print(f"[dump_gen_fsr] {len(files)} file(s), nmax/file = {args.nmax or 'all'}",
          flush=True)

    h_pruned = Handle("std::vector<reco::GenParticle>")
    cols = {k: [] for k in COLS}
    nseen = nbad = 0
    t0 = time.time()
    dr2max = args.dr_dress ** 2

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
                    sf = g.statusFlags()
                    if st == 1 and sf.isHardProcess():
                        post.append(g)
                    elif st == 746:
                        pre746.append(g)
                elif pid == 23:
                    if g.status() == 62:
                        zb.append(g)
                elif pid == 22 and g.status() == 1 and g.statusFlags().isPrompt():
                    phot.append(g)

            if len(zb) != 1 or len(post) != 2:
                nbad += 1
                continue

            ex = ey = ez = ee = 0.0
            nph = 0
            for g in phot:
                if min(dr2(g, post[0]), dr2(g, post[1])) < dr2max:
                    ex += g.px(); ey += g.py(); ez += g.pz(); ee += g.energy()
                    nph += 1

            src = pre746 if len(pre746) == 2 else post
            ppx = sum(p.px() for p in src); ppy = sum(p.py() for p in src)
            ppz = sum(p.pz() for p in src); pen = sum(p.energy() for p in src)
            y_pre = 0.5 * math.log((pen + ppz) / (pen - ppz)) if abs(ppz) < pen else 0.0

            aux = e.eventAuxiliary()
            o = sorted(post, key=lambda g: -g.pt())
            op = sorted(src, key=lambda g: -g.pt())
            v = dict(
                run=aux.run(), lumi=aux.luminosityBlock(), event=aux.event(),
                m_pre=zb[0].mass(),
                m_pre746=m4(pre746) if len(pre746) == 2 else np.nan,
                m_post=m4(post),
                m_dress=m4(post, (ex, ey, ez, ee)),
                pt_pre=math.hypot(ppx, ppy), y_pre=y_pre, nph=nph, eph=ee,
                pt1=o[0].pt(), eta1=o[0].eta(), pt2=o[1].pt(), eta2=o[1].eta(),
                pt1_pre=op[0].pt(), eta1_pre=op[0].eta(),
                pt2_pre=op[1].pt(), eta2_pre=op[1].eta(),
            )
            for k in COLS:
                cols[k].append(v[k])

    out = {k: np.asarray(cols[k]) for k in COLS}
    for k in ("run", "lumi", "event", "nph"):
        out[k] = out[k].astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"[dump_gen_fsr] {nseen} events read, {len(out['m_pre'])} kept, "
          f"{nbad} without a Z(62) + 2 hard-process status-1 muons, in "
          f"{time.time()-t0:.0f} s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
