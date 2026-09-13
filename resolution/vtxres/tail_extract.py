#!/usr/bin/env python3
"""Compact per-candidate cache for the TAIL study (STATE.md section 16).

The productions live on ceph and are only readable from a submit node; the
analysis and the figures are not.  This script is the bridge: it reads the
`globalcor_*.root` trees of one production and writes ONE small npz with
every column the tail hypotheses need -- the two beam pulls, the vertex
residual, the beam-spot RECORD the maker used, the GEN production vertex, the
gen provenance `genbkg.classify` wants, the per-leg hit counts and kinematics,
and the per-hit-class variance shares of both functionals.

Nothing is cut and nothing is fitted here.

usage:
  python3 tail_extract.py --files '<dir>/task_*/globalcor_*.root' --out <npz>
"""
import argparse, glob, os, sys
import numpy as np
import uproot

NCLS = 18   # pix_x_q{0..3}, pix_y_q{0..3}, str_N{1..5}_{lo,hi}

# flat scalars, (branch -> output name)
SCALARS = {
    "run": "run", "lumi": "lumi", "event": "event",
    "chisqval": "chisq", "ndof": "ndof", "edmval": "edmval",
    "Jpsi_mass": "mass", "Jpsi_pt": "pt", "Jpsi_eta": "eta", "Jpsi_phi": "phi",
    "Jpsi_sigmamass": "sigmamass",
    "Jpsi_d": "dca", "Jpsi_x": "vx", "Jpsi_y": "vy", "Jpsi_z": "vz",
    "Jpsi_vtxz": "vtxz", "Jpsi_vtxsig": "vtxsig", "Jpsi_vtxres": "vtxres",
    "Jpsi_vtxok": "vtxok", "Jpsi_vtxvhit": "vtxvhit", "Jpsi_vtxvms": "vtxvms",
    "Jpsi_vtxvgf": "vtxvgf", "Jpsi_vtxvbs": "vtxvbs",
    "Jpsi_vtxvbsx": "vtxvbsx", "Jpsi_vtxvbsy": "vtxvbsy",
    "Jpsi_bschi2": "bschi2", "Jpsi_bschi2fit": "bschi2fit",
    "Jpsi_bschi20": "bschi20", "Jpsi_bsok": "bsok", "Jpsi_bsmeig": "bsmeig",
    "Jpsi_bsvchk": "bsvchk", "Jpsi_vtxvchk": "vtxvchk",
    "cfmass_ok": "cfmass_ok", "Jpsi_vtxbfree": "vtxbfree",
    "Jpsi_vtxb6": "vtxb6", "Jpsi_vtxdchi2": "vtxdchi2",
    "Jpsi_covmassvtx": "covmassvtx", "niter": "niter",
    "Jpsi_bsfree": "bsfree",
    "Jpsi_fang": "fang", "Jpsi_rhomom": "rhomom",
    "Jpsi_sigmarelplus": "sigrel_plus", "Jpsi_sigmarelminus": "sigrel_minus",
    "Jpsigen_x": "genvx", "Jpsigen_y": "genvy", "Jpsigen_z": "genvz",
    "Jpsigen_pt": "genpt", "Jpsigen_eta": "geneta", "Jpsigen_phi": "genphi",
    "Jpsigen_mass": "genmass", "Jpsigen_sameDecay": "gensamedecay",
    "Pileup_nPU": "npu", "Pileup_nTrueInt": "ntrueint",
    "genweight": "genweight", "genl3d": "genl3d",
}
for s, tag in (("plus", "plus"), ("minus", "minus")):
    SCALARS.update({
        f"Mu{s}_pt": f"pt_{tag}", f"Mu{s}_eta": f"eta_{tag}",
        f"Mu{s}_phi": f"phi_{tag}",
        f"Mu{s}_nvalid": f"nvalid_{tag}",
        f"Mu{s}_nvalidpixel": f"nvalidpixel_{tag}",
        f"Mu{s}_nhits": f"nhits_{tag}",
        f"Mu{s}_nvalidFinal": f"nvalidfinal_{tag}",
        f"Mu{s}_nvalidpixelFinal": f"nvalidpixelfinal_{tag}",
        f"Mu{s}_nmatchedvalid": f"nmatched_{tag}",
        f"Mu{s}_npixDemoted": f"npixdemoted_{tag}",
        f"Mu{s}_highpurity": f"highpurity_{tag}",
        f"Mu{s}_charge": f"charge_{tag}",
        f"Mu{s}_muonMedium": f"muonmedium_{tag}",
        f"Mu{s}gen_pt": f"genpt_{tag}", f"Mu{s}gen_eta": f"geneta_{tag}",
        f"Mu{s}gen_phi": f"genphi_{tag}", f"Mu{s}gen_dr": f"gendr_{tag}",
        f"Mu{s}gen_pdgId": f"genpdg_{tag}", f"Mu{s}gen_idx": f"genidx_{tag}",
        f"Mu{s}gen_motherPdgId": f"genmoth_{tag}",
        f"Mu{s}gen_motherIdx": f"genmothidx_{tag}",
        f"Mu{s}gen_isPrompt": f"genprompt_{tag}",
        f"Mu{s}gen_fromHardProcess": f"genhard_{tag}",
    })

# fixed-length arrays, (branch -> (output name, length))
ARRAYS = {
    "Jpsi_bsz": ("bsz", 2), "Jpsi_bsres": ("bsres", 2),
    "Jpsi_bscov": ("bscov", 3), "Jpsi_bsvtx": ("bsvtx", 3),
    "Jpsi_bsspot": ("bsspot", 3), "Jpsi_bsslope": ("bsslope", 2),
    "Jpsi_bswidth": ("bswidth", 3), "Jpsi_bslinv": ("bslinv", 3),
    "Jpsi_bscovlo": ("bscovlo", 6), "Jpsi_covvtx": ("covvtx", 6),
    "Jpsi_bsvbs": ("bsvbs", 2), "Jpsi_bsvhit": ("bsvhit", 2),
    "Jpsi_bsvms": ("bsvms", 2), "Jpsi_bsvioni": ("bsvioni", 2),
    "Jpsi_bsmeanbs": ("bsmeanbs", 6),
    "Jpsi_bswidtherr": ("bswidtherr", 2),
    "Muplus_refParms": ("refparms_plus", 3),
    "Muminus_refParms": ("refparms_minus", 3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(a.files))
    if not files:
        sys.exit(f"no files matched {a.files}")
    print(f"# {len(files)} files")

    have = set(uproot.open(files[0] + ":tree").keys())
    sc = {b: o for b, o in SCALARS.items() if b in have}
    ar = {b: v for b, v in ARRAYS.items() if b in have}
    jag = [b for b in ("cfvtx_hitcls", "cfvtx_hitv", "cfbs_hitcls",
                       "cfbs_hitv", "cfbs_hitcomp", "cfmass_hitcls",
                       "cfmass_hitv", "Muplus_pixClass", "Muminus_pixClass")
           if b in have]
    missing = sorted((set(SCALARS) | set(ARRAYS)) - have)
    if missing:
        print(f"# absent in this production: {', '.join(missing)}")

    out = {o: [] for o in sc.values()}
    for b, (o, k) in ar.items():
        out[o] = []
    for o in ("vtx_hitv", "bsx_hitv", "bsy_hitv", "mass_hitv"):
        out[o] = []
    out["pixclass_plus"] = []
    out["pixclass_minus"] = []

    ntot = 0
    for fn in files:
        t = uproot.open(fn + ":tree")
        d = t.arrays(list(sc) + list(ar) + jag, library="np")
        n = len(d["run"])
        for b, o in sc.items():
            out[o].append(np.asarray(d[b]))
        for b, (o, k) in ar.items():
            v = d[b]
            out[o].append(np.stack([np.asarray(x, float) for x in v])
                          if v.dtype == object else np.asarray(v, float))
        # per-hit-class variance shares, summed into a fixed NCLS vector
        for src, dst in (("cfvtx", "vtx_hitv"), ("cfmass", "mass_hitv")):
            if f"{src}_hitcls" not in d:
                out[dst].append(np.zeros((n, NCLS)))
                continue
            M = np.zeros((n, NCLS))
            for i in range(n):
                c = np.asarray(d[f"{src}_hitcls"][i], np.int64)
                v = np.asarray(d[f"{src}_hitv"][i], float)
                ok = (c >= 0) & (c < NCLS)
                if ok.any():
                    np.add.at(M[i], c[ok], v[ok])
            out[dst].append(M)
        if "cfbs_hitcls" in d:
            Mx = np.zeros((n, NCLS)); My = np.zeros((n, NCLS))
            for i in range(n):
                c = np.asarray(d["cfbs_hitcls"][i], np.int64)
                v = np.asarray(d["cfbs_hitv"][i], float)
                comp = (np.asarray(d["cfbs_hitcomp"][i], np.int64)
                        if "cfbs_hitcomp" in d else np.zeros_like(c))
                ok = (c >= 0) & (c < NCLS)
                if ok.any():
                    np.add.at(Mx[i], c[ok & (comp == 0)], v[ok & (comp == 0)])
                    np.add.at(My[i], c[ok & (comp == 1)], v[ok & (comp == 1)])
            out["bsx_hitv"].append(Mx); out["bsy_hitv"].append(My)
        else:
            out["bsx_hitv"].append(np.zeros((n, NCLS)))
            out["bsy_hitv"].append(np.zeros((n, NCLS)))
        # pixel hit classes per leg: the counts of each class value (0..7)
        for s, tag in (("Muplus_pixClass", "pixclass_plus"),
                       ("Muminus_pixClass", "pixclass_minus")):
            if s in d:
                M = np.zeros((n, 8), np.int32)
                for i in range(n):
                    c = np.asarray(d[s][i], np.int64)
                    ok = (c >= 0) & (c < 8)
                    if ok.any():
                        np.add.at(M[i], c[ok], 1)
                out[tag].append(M)
            else:
                out[tag].append(np.zeros((n, 8), np.int32))
        ntot += n
        print(f"#   {os.path.basename(os.path.dirname(fn))}: {n}  (total {ntot})")
        if a.max and ntot >= a.max:
            break

    res = {k: np.concatenate(v, axis=0) for k, v in out.items() if len(v)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **res)
    print(f"# wrote {a.out}: {ntot} candidates, {len(res)} columns")


if __name__ == "__main__":
    main()
