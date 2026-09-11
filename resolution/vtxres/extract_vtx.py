#!/usr/bin/env python3
"""Per-material-group CF inputs for the VERTEX and the MASS functional,
read straight from the IN-MAKER exponents.

This is `matres/extract_groups.py` with the offline step-record machinery
removed: the maker now computes the per-group exponents at fit time
(`cvhcf`, `exportCfGroupExponents=True`), so the extraction is a re-shaping of
`cfvtx_grp_*` / `cfmass_grp_*` into the CSR layout `make_material_card.py`
and `hitlik/make_hitlik_card.py::build_mass_term` consume.  Same keys, same
meaning, so every downstream consumer runs unchanged.

BOTH functionals come out of the SAME candidates in the SAME order, which is
what a JOINT (vertex + mass) card needs.

Output (npz), per functional
    tgrid (nt,) ; sigma, m0, mgen, vgf, chi2ndof, ok (n,)
    hit_ptr (n+1,), hit_cls, hit_v            per-hit-class Gaussian shares
    grp_ptr (n+1,), grp_id, S<fam> (nnz, nt)  the per-group exponents
    Gvqms, Gvqio (nnz,)                       the FIT'S OWN Q variance per
                                              group -- the `gaussq` arm
    vhit, vms, vioni (n,)                     the family split of sigma^2
    families, group_names, hit_classes, functional
    run, lumi, event                          the candidate key
    genpt_plus/minus, geneta_plus/minus       GEN kinematics (MC only)
    vtxz, vtxsig, massz                       the two pulls, for the joint

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 extract_vtx.py --files '<prod>/task_*/globalcor_*.root' \
      --functional vtx --groups .../materialGroups50.txt -j 24 \
      --max-chi2-ndof 3 -o out.npz
"""
import argparse, glob, os, sys, time
from multiprocessing import Pool
import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
_MAT = os.path.join(_RES, "matres")
for _p in (_HERE, _RES, _MAT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAMS = ("ms", "io_re", "io_im", "rad_re", "rad_im")
# branch suffix of each family, per functional prefix
SUF = {"ms": "ms", "io_re": "ioni_re", "io_im": "ioni_im",
       "rad_re": "rad_re", "rad_im": "rad_im"}

_ARGS = None


def _init(a):
    global _ARGS
    _ARGS = a


def process_file(fn):
    a = _ARGS
    pre = "cfvtx" if a.functional == "vtx" else "cfmass"
    scal = {
        "vtx": dict(sigma="Jpsi_vtxsig", m0="Jpsi_vtxres", vgf="Jpsi_vtxvgf",
                    ok="Jpsi_vtxok", vhit="Jpsi_vtxvhit", vms="Jpsi_vtxvms",
                    vioni="Jpsi_vtxvioni", z="Jpsi_vtxz"),
        "mass": dict(sigma="Jpsi_sigmamass", m0="Jpsi_mass", vgf="cfmass_vgf",
                     ok="cfmass_ok", vms="Jpsi_massvms", vioni="Jpsi_massvioni"),
    }[a.functional]
    want = list(dict.fromkeys(list(scal.values()) + ["Jpsi_vtxok", "cfmass_ok"])) + [
        f"{pre}_grp", f"{pre}_grp_vqms", f"{pre}_grp_vqio",
        f"{pre}_hitcls", f"{pre}_hitv", "chisqval", "ndof",
        "run", "lumi", "event", "Jpsi_mass", "Jpsigen_mass",
        "Jpsi_vtxz", "Jpsi_vtxsig", "Jpsi_vtxvchk", "Jpsi_vtxsgnchk",
        "Jpsi_sigmamass", "Jpsi_pt", "Jpsi_eta",
        "Muplusgen_pt", "Muminusgen_pt", "Muplusgen_eta", "Muminusgen_eta",
        "Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta",
    ] + [f"{pre}_grp_{SUF[f]}" for f in FAMS]
    try:
        fh = uproot.open(fn)
    except Exception as e:
        print(f"# skip {fn}: {e}", file=sys.stderr)
        return None
    with fh:
        tg = np.asarray(fh["runtree"]["cftau"].array(library="np")[0], np.float64)
        t = fh["tree"]
        keys = set(k.split(";")[0] for k in t.keys())
        got = [b for b in want if b in keys]
        d = t.arrays(got, library="np")
    n0 = len(d[scal["sigma"]])
    sig = np.asarray(d[scal["sigma"]], np.float64)
    ndof = np.asarray(d["ndof"], np.float64)
    chi2n = np.where(ndof > 0, np.asarray(d["chisqval"], np.float64) / np.maximum(ndof, 1), 1e9)
    # THE SELECTION IS THE SAME FOR BOTH FUNCTIONALS, deliberately: a joint
    # (vertex + mass) card pairs the two npz ROW BY ROW, so a candidate kept
    # by one and dropped by the other would silently pair the wrong events.
    keep = np.isfinite(sig) & (sig > 0)
    keep &= np.asarray(d["Jpsi_vtxok"], bool) if "Jpsi_vtxok" in d else True
    keep &= np.asarray(d["cfmass_ok"], bool) if "cfmass_ok" in d else True
    keep &= np.asarray(d["Jpsi_vtxsig"], np.float64) > 0
    keep &= np.asarray(d["Jpsi_sigmamass"], np.float64) > 0
    if a.max_chi2_ndof > 0:
        keep &= chi2n < a.max_chi2_ndof
    if "Jpsi_vtxvchk" in d and a.max_vchk > 0:
        keep &= np.abs(np.asarray(d["Jpsi_vtxvchk"], np.float64)) < a.max_vchk
    idx = np.flatnonzero(keep)
    if a.max_cands:
        idx = idx[: a.max_cands]
    n = len(idx)
    if n == 0:
        return None

    nt = len(tg)
    gid_l, ptr = [], np.zeros(n + 1, np.int64)
    fam_l = {f: [] for f in FAMS}
    vqms_l, vqio_l = [], []
    grp = d[f"{pre}_grp"]
    fams_raw = {f: d[f"{pre}_grp_{SUF[f]}"] for f in FAMS}
    vqm, vqi = d.get(f"{pre}_grp_vqms"), d.get(f"{pre}_grp_vqio")
    for k, i in enumerate(idx):
        g = np.asarray(grp[i], np.int64)
        ptr[k + 1] = ptr[k] + g.size
        gid_l.append(g)
        for f in FAMS:
            fam_l[f].append(np.asarray(fams_raw[f][i], np.float32).reshape(-1, nt))
        if vqm is not None:
            vqms_l.append(np.asarray(vqm[i], np.float32))
            vqio_l.append(np.asarray(vqi[i], np.float32))
    hptr = np.zeros(n + 1, np.int64)
    hc_l, hv_l = [], []
    hc, hv = d[f"{pre}_hitcls"], d[f"{pre}_hitv"]
    for k, i in enumerate(idx):
        c = np.asarray(hc[i], np.int16)
        hptr[k + 1] = hptr[k] + c.size
        hc_l.append(c)
        hv_l.append(np.asarray(hv[i], np.float32))

    res = {"tgrid": tg, "grp_ptr": ptr, "hit_ptr": hptr,
           "grp_id": (np.concatenate(gid_l) if gid_l else np.zeros(0, np.int64)),
           "hit_cls": (np.concatenate(hc_l) if hc_l else np.zeros(0, np.int16)),
           "hit_v": (np.concatenate(hv_l) if hv_l else np.zeros(0, np.float32))}
    for f in FAMS:
        res["S" + f] = (np.concatenate(fam_l[f]) if fam_l[f]
                        else np.zeros((0, nt), np.float32))
    if vqms_l:
        res["Gvqms"] = np.concatenate(vqms_l)
        res["Gvqio"] = np.concatenate(vqio_l)
    res["sigma"] = sig[idx]
    res["m0"] = np.asarray(d[scal["m0"]], np.float64)[idx]
    res["vgf"] = np.asarray(d[scal["vgf"]], np.float64)[idx]
    res["chi2ndof"] = chi2n[idx]
    for nm in ("vhit", "vms", "vioni"):
        if nm in scal and scal[nm] in d:
            res[nm] = np.asarray(d[scal[nm]], np.float64)[idx]
    res["mgen"] = (np.asarray(d["Jpsigen_mass"], np.float64)[idx]
                   if "Jpsigen_mass" in d else np.zeros(n))
    for nm in ("run", "lumi", "event"):
        if nm in d:
            res[nm] = np.asarray(d[nm], np.int64)[idx]
    for nm, br in (("vtxz", "Jpsi_vtxz"), ("vtxsig", "Jpsi_vtxsig"),
                   ("jpsipt", "Jpsi_pt"), ("jpsieta", "Jpsi_eta"),
                   ("genpt_plus", "Muplusgen_pt"), ("genpt_minus", "Muminusgen_pt"),
                   ("geneta_plus", "Muplusgen_eta"), ("geneta_minus", "Muminusgen_eta"),
                   ("pt_plus", "Muplus_pt"), ("pt_minus", "Muminus_pt"),
                   ("sigmamass", "Jpsi_sigmamass"), ("mass", "Jpsi_mass")):
        if br in d:
            res[nm] = np.asarray(d[br], np.float64)[idx]
    return fn, res, {"n0": n0, "nsel": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--functional", choices=["vtx", "mass"], default="vtx")
    ap.add_argument("--groups", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--max-chi2-ndof", type=float, default=3.0)
    ap.add_argument("--max-vchk", type=float, default=1e-4,
                    help="cut on the maker's own closure figure -- a cut on "
                         "the FIT'S arithmetic, not on the residual")
    ap.add_argument("--max-cands", type=int, default=0, help="per file")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    import groups as G
    files = sorted(glob.glob(a.files))
    if a.require_complete:
        files = [f for f in files
                 if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    if a.max_files:
        files = files[: a.max_files]
    print(f"# {len(files)} files, functional {a.functional}")
    import hitres_classes
    t0 = time.time()
    parts, stats = [], []
    with Pool(a.jobs, initializer=_init, initargs=(a,)) as pool:
        # ORDERED, not `imap_unordered`: the vertex and the mass extraction
        # must emit the SAME candidates in the SAME order, because a joint
        # card pairs the two npz row by row.  With `imap_unordered` the two
        # runs assemble their `parts` in whatever order the workers finish,
        # and the joint card's (run, lumi, event) gate catches it -- which is
        # how this was found.
        for i, r in enumerate(pool.imap(process_file, files)):
            if r is None:
                continue
            fn, res, st = r
            parts.append(res)
            stats.append(st)
            if (i + 1) % 10 == 0 or i + 1 == len(files):
                print(f"[{i+1}/{len(files)}] cumulative "
                      f"{sum(s['nsel'] for s in stats)} "
                      f"({time.time()-t0:.0f} s)", flush=True)
    if not parts:
        sys.exit("nothing extracted")

    # the group catalog: names only, from the tier file
    gmap, _gp = G.read_groups(a.groups)
    ngroups = (max(gmap) + 1) if gmap else 0
    gnames, _ = G.group_param_names(ngroups, a.groups)
    out = {"tgrid": parts[0]["tgrid"],
           "families": np.array(FAMS),
           "group_names": np.array(gnames),
           "hit_classes": np.array(hitres_classes.CLASSES),
           "functional": np.array(a.functional),
           "amount_convention": np.array("exp(k_g) per group, weights frozen")}
    scalars = [k for k in parts[0]
               if k not in ("tgrid", "grp_ptr", "hit_ptr", "grp_id", "hit_cls",
                            "hit_v", "Gvqms", "Gvqio")
               and not k.startswith("S")]
    for k in scalars:
        out[k] = np.concatenate([p[k] for p in parts if k in p])
    for pref in ("grp", "hit"):
        ptrs, off = [], 0
        for j, p in enumerate(parts):
            q = p[f"{pref}_ptr"]
            ptrs.append((q[1:] if j else q) + (off if j else 0))
            off += int(q[-1])
        out[f"{pref}_ptr"] = np.concatenate(ptrs)
    out["grp_id"] = np.concatenate([p["grp_id"] for p in parts])
    out["hit_cls"] = np.concatenate([p["hit_cls"] for p in parts])
    out["hit_v"] = np.concatenate([p["hit_v"] for p in parts])
    for f in FAMS:
        out["S" + f] = np.concatenate([p["S" + f] for p in parts])
    if "Gvqms" in parts[0]:
        out["Gvqms"] = np.concatenate([p["Gvqms"] for p in parts])
        out["Gvqio"] = np.concatenate([p["Gvqio"] for p in parts])
    n = len(out["sigma"])
    np.savez_compressed(a.output, **out)
    nnz = out["grp_id"].size
    print(f"\n=== {a.functional}: {n} candidates, {nnz} group rows "
          f"({nnz/max(n,1):.2f}/cand), {out['hit_cls'].size/max(n,1):.2f} hit rows/cand")
    print(f"    exponent storage {nnz*len(out['tgrid'])*4*len(FAMS)/max(n,1)/1024:.2f} kB/cand")
    print(f"    sum_g (vqms+vqio) + vgf closure: ", end="")
    if "Gvqms" in out:
        p = out["grp_ptr"]
        s = np.add.reduceat(out["Gvqms"] + out["Gvqio"], p[:-1]) * (np.diff(p) > 0)
        c = s + out["vgf"]
        print(f"median {np.median(c):.6f}  p1 {np.percentile(c,1):.6f}  "
              f"p99 {np.percentile(c,99):.6f}")
    else:
        print("(no vq arrays in the tree)")
    print(f"    -> {a.output} ({os.path.getsize(a.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
