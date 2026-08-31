#!/usr/bin/env python3
"""Inventory of the private gun samples, read from the configs they were made with.

Every simprod task directory keeps the exact `step1_task.py` (GEN-SIM) and
`step2_task.py` (DIGI-RECO) it ran, so the sample's provenance does not have
to be reconstructed from shell scripts or memory -- it is parsed back out of
the job's own configuration here.

Fields that matter for whether two samples are comparable:
  species / pT / eta   the generator
  GT                   the fit MUST use the same one: it re-evaluates hit
                       positions with its own CPEs, so a mismatched
                       SiPixelLorentzAngle or template payload shifts local-x
                       per module and fakes a hit-quality bias
  geometry             Ideal vs Extended (aligned)
  stepper              DeltaOneStepTracker etc -- the CMSSW defaults are
                       ~100x looser than the official SIM and shift widths
                       by 12 % (NOTES.md)
  G4Commands           the physics ablations (MS / brems / pair off)
  PSimHits             whether the tracker sim hits were kept

usage: python simprod/inventory.py [--write]
"""
import argparse
import glob
import os
import re

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"


def grab(txt, pat, default="-"):
    """LAST match, not the first.

    run_simprod_mugun.sh builds each job's config by CONCATENATING the shared
    step1_gensim.py template with per-sample override lines, so the template's
    J/psi-gun defaults (ParticleID 443, pT 5-30) appear FIRST and the sample's
    real values LAST. Taking the first match reports every sample as the J/psi
    gun -- which is exactly what the first version of this script did.
    """
    m = re.findall(pat, txt)
    return m[-1].strip() if m else default


def census_verdict(tl):
    """Did the deactivation actually take?

    The watcher prints, at EndOfRun, the number of steps each process DEFINED,
    split into the primary and all tracks. A process that was switched off for
    the primary must be ABSENT from the primary column; anything else means
    the sample is silently un-ablated.
    """
    req = re.findall(r"requested INACTIVE '([^']+)'", tl)
    if not req:
        return "-"
    seen = dict((m[0], int(m[1]))
                for m in re.findall(r"\[procact\]\s+(\S+)\s+primary (\d+)\s+all \d+", tl))
    if not seen:
        return "NO CENSUS"
    bad = [r for r in req if seen.get(r, 0) > 0]
    return "OK" if not bad else "FAILED:" + ",".join(bad)


def read_sample(d):
    tasks = sorted(glob.glob(os.path.join(d, "task_*")))
    if not tasks:
        return None
    s1 = os.path.join(tasks[0], "step1_task.py")
    s2 = os.path.join(tasks[0], "step2_task.py")
    if not os.path.exists(s1):
        return None
    t1 = open(s1).read()
    # The ablation is read from step1.log, not from the config. The config
    # only carries `cms.untracked.vstring(*_inact)` -- the list is built from
    # the environment at cmsRun time, so the file does not record what was
    # actually switched off. The log carries both the resolved request and,
    # better, the [procact] STEP CENSUS: what Geant4 actually ran. That is the
    # evidence, the flag is not.
    lg = os.path.join(tasks[0], "step1.log")
    tl = open(lg, errors="ignore").read() if os.path.exists(lg) else ""
    t2 = open(s2).read() if os.path.exists(s2) else ""
    nstep2 = sum(1 for t in tasks if os.path.exists(os.path.join(t, "step2.root")))
    info = dict(
        name=os.path.basename(d),
        ntask=len(tasks), nstep2=nstep2,
        nev=grab(t1, r"input\s*=\s*cms\.untracked\.int32\((\d+)\)"),
        pdg=grab(t1, r"ParticleID\s*=\s*cms\.vint32\(([^)]*)\)"),
        ptmin=grab(t1, r"MinPt\s*=\s*cms\.double\(([^)]*)\)"),
        ptmax=grab(t1, r"MaxPt\s*=\s*cms\.double\(([^)]*)\)"),
        etamax=grab(t1, r"MaxEta\s*=\s*cms\.double\(([^)]*)\)"),
        anti=grab(t1, r"AddAntiParticle\s*=\s*cms\.bool\((\w+)\)"),
        gt=grab(t1, r"SIMPROD_GT'\s*,\s*'([\w]+)'"),
        gt2=grab(t2, r"GlobalTag\(process\.GlobalTag,\s*'([\w]+)'"),
        geom=("Ideal" if "XMLFromDBSource.label=\"Ideal\"" in t1
              or 'XMLFromDBSource.label="Ideal"' in t1 else "Extended?"),
        step=grab(t1, r"DeltaOneStepTracker\s*=\s*([\w.e-]+)"),
        # The ABLATION is carried by process.g4SimHits.Watchers, NOT by
        # G4Commands. G4Commands is applied while Geant4 is still PreInit and
        # is a silent no-op (measured: two runs with and without came out
        # bit-identical). An inventory keyed on G4Commands therefore reports
        # every sample as unablated, including ones that are not -- which is
        # what the first version of this file did.
        g4cmd=grab(t1, r"G4Commands\s*=\s*cms\.vstring\(([^)]*)\)", ""),
        inact=",".join(re.findall(r"requested INACTIVE '([^']+)'", tl)),
        inactfor=",".join(re.findall(r"restricted to particle '([^']+)'", tl)),
        census=census_verdict(tl),
        psim="TrackerHits" in t2 or "RAWSIM" in t1,
    )
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true",
                   help="also write simprod/SAMPLES.md")
    args = p.parse_args()
    rows = []
    for d in sorted(glob.glob(f"{CEPH}/resolution_simprod_*")):
        r = read_sample(d)
        if r:
            rows.append(r)
    hdr = (f"{'sample':<30}{'files':>8}{'evt/f':>7}  {'pdg':<12}{'pT':<8}"
           f"{'GT':<26}{'stepTrk':<9}{'ABLATION (inactivated, for)':<46}")
    print(hdr); print("-" * len(hdr))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        gt = r["gt2"] if r["gt2"] != "-" else r["gt"]
        ab = (r["inact"].replace("'", "").replace('"', "") or "").strip()
        pa = (r["inactfor"].replace("'", "").replace('"', "") or "").strip()
        abl = "none" if not ab else (
            ab + (f" [{pa}]" if pa else " [all]") + f"  census={r['census']}")
        if r["g4cmd"]:
            abl += "  +G4Commands(INERT)"
        s = (f"{r['name'].replace('resolution_simprod_',''):<30}"
             f"{r['nstep2']:>4}/{r['ntask']:<3}{r['nev']:>7}  "
             f"{r['pdg']:<12}{r['ptmin']+'-'+r['ptmax']:<8}"
             f"{gt:<26}{r['step']:<9}{abl:<46}")
        print(s); lines.append(s)
    if args.write:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SAMPLES.md")
        with open(out, "w") as fh:
            fh.write("# Private gun samples\n\n"
                     "Auto-generated by `simprod/inventory.py`, parsed from each "
                     "sample's OWN `step1_task.py` / `step2_task.py`.\n\n```\n"
                     + "\n".join(lines) + "\n```\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
