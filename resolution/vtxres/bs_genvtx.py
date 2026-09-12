#!/usr/bin/env python3
"""Does the SIMULATED luminous region match the beam-spot record the maker
reads?  If not, the beam rows are mis-specified and the study must say so.

Reads the GEN production vertex (`Jpsigen_x/y/z`, cm) of every written
candidate and compares its mean / width / tilt with the `offlineBeamSpot`
record the fit used (`Jpsi_bsspot`, `Jpsi_bswidth`, `Jpsi_bsslope`, present
only in a rows-ON production).

usage: bs_genvtx.py <dir-with-globalcor_*.root> [more dirs ...]
"""
import sys, glob, numpy as np, uproot

def load(paths, names):
    out = {n: [] for n in names}
    for p in paths:
        try:
            t = uproot.open(p)["tree"]
        except Exception as e:
            print(f"  [skip] {p}: {e}")
            continue
        have = set(t.keys())
        for n in names:
            if n in have:
                out[n].append(t[n].array(library="np"))
    return {n: (np.concatenate(v) if v else None) for n, v in out.items()}

def main():
    dirs = sys.argv[1:]
    paths = []
    for d in dirs:
        paths += sorted(glob.glob(f"{d}/**/globalcor_*.root", recursive=True))
    print(f"{len(paths)} files")
    names = ["Jpsigen_x", "Jpsigen_y", "Jpsigen_z",
             "Jpsi_bsspot", "Jpsi_bswidth", "Jpsi_bsslope",
             "Jpsi_bsvtx", "Muplusgen_pt", "Muminusgen_pt"]
    a = load(paths, names)
    gx, gy, gz = a["Jpsigen_x"], a["Jpsigen_y"], a["Jpsigen_z"]
    # -99 is the NO-GEN-MATCH sentinel, written on all three at once; it is
    # ~4 % of candidates and it is not a vertex.
    ok = (np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gz)
          & (np.abs(gx) < 5.) & (np.abs(gy) < 5.) & (np.abs(gz) < 60.))
    print(f"  gen-matched {ok.mean()*100:.2f} % ({ok.sum()} of {ok.size})")
    gx, gy, gz = gx[ok], gy[ok], gz[ok]
    n = gx.size
    print(f"\nGEN production vertex, {n} candidates (cm)")
    for nm, v in (("x", gx), ("y", gy), ("z", gz)):
        print(f"  {nm}: mean {v.mean():+.6f}  rms {v.std():.6f}"
              f"  err(mean) {v.std()/np.sqrt(n):.6f}")
    # the tilt: slope of <x> and <y> against z
    for nm, v in (("dx/dz", gx), ("dy/dz", gy)):
        s, c = np.polyfit(gz, v, 1)
        # error on the slope
        res = v - (s*gz + c)
        se = res.std()/ (gz.std()*np.sqrt(n))
        print(f"  {nm}: {s:+.3e} +- {se:.1e}")
    # the transverse widths AFTER removing the tilt
    sx = np.polyfit(gz, gx, 1)[0]
    sy = np.polyfit(gz, gy, 1)[0]
    rx = gx - sx*gz
    ry = gy - sy*gz
    print(f"  width x|z: {rx.std()*1e4:.2f} um   width y|z: {ry.std()*1e4:.2f} um")
    print(f"  corr(x,y)|z: {np.corrcoef(rx, ry)[0,1]:+.4f}")
    if a["Jpsi_bsspot"] is not None:
        sp = np.asarray(a["Jpsi_bsspot"]).reshape(-1, 3)
        wd = np.asarray(a["Jpsi_bswidth"]).reshape(-1, 3)
        sl = np.asarray(a["Jpsi_bsslope"]).reshape(-1, 2)
        print("\nBEAM-SPOT RECORD the maker read (unique rows)")
        u = np.unique(np.round(np.hstack([sp, wd, sl]), 9), axis=0)
        for r in u[:5]:
            print(f"  x0 {r[0]:+.6f}  y0 {r[1]:+.6f}  z0 {r[2]:+.6f} cm")
            print(f"  sigx {r[3]*1e4:.2f} um  sigy {r[4]*1e4:.2f} um  sigz {r[5]:.4f} cm")
            print(f"  dxdz {r[6]:+.3e}  dydz {r[7]:+.3e}")
        if u.shape[0] > 5:
            print(f"  ... {u.shape[0]} distinct rows")
        print("\nRECORD vs SIMULATION")
        r = u[0]
        print(f"  x0:   record {r[0]*1e4:+9.2f} um   gen mean {gx.mean()*1e4:+9.2f} um"
              f"   diff {(gx.mean()-r[0])*1e4:+.2f} um")
        print(f"  y0:   record {r[1]*1e4:+9.2f} um   gen mean {gy.mean()*1e4:+9.2f} um"
              f"   diff {(gy.mean()-r[1])*1e4:+.2f} um")
        print(f"  z0:   record {r[2]:+9.4f} cm   gen mean {gz.mean():+9.4f} cm"
              f"   diff {gz.mean()-r[2]:+.4f} cm")
        print(f"  sigx: record {r[3]*1e4:9.2f} um   gen {rx.std()*1e4:9.2f} um"
              f"   ratio {rx.std()/r[3]:.3f}")
        print(f"  sigy: record {r[4]*1e4:9.2f} um   gen {ry.std()*1e4:9.2f} um"
              f"   ratio {ry.std()/r[4]:.3f}")
        print(f"  sigz: record {r[5]:9.4f} cm   gen {gz.std():9.4f} cm"
              f"   ratio {gz.std()/r[5]:.3f}")
        print(f"  dxdz: record {r[6]:+.3e}   gen {sx:+.3e}")
        print(f"  dydz: record {r[7]:+.3e}   gen {sy:+.3e}")

if __name__ == "__main__":
    main()
