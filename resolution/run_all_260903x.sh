#!/bin/bash
# Full re-production of the track-resolution refits on the 2026-09-03 build
# that carries the two NEW per-block exports:
#
#   ioniqscaleidx / ioniqscalev  -- the factor by which the CGF substitution
#       scaled each leg's ionization noise block, [sc, nstep] per leg. It is
#       exactly 1.0 for CgfQoPMode=0 and for every two-track fit; on the mode-1
#       arms it is ~2e-5, which is precisely the factor by which the offline
#       `--ioni-norm var` normalisation was wrong on those files.
#   radstepidx / radstepv / radstepspecv (+ radvgrid, radstepstride, radstepnv)
#       -- the per-step radiative (brems + pair) records and their two dN/dv
#       shapes, for the radiative CF term (cf_brems_exact.py's layout exactly).
#
# Every pre-existing branch is bit-identical to the pre-change build (verified
# on 30-event smokes in all three configurations), so these samples are the
# 2026-08-30 / 2026-09-02 productions PLUS the two exports. File size grows
# ~1.9x (the per-step spectra), ~340 MB/task.
#
# NEW TAGS with the _260903x suffix throughout: nothing existing is
# overwritten, and `.complete` sentinels make every stage resumable.
#
# ORDER. The two CgfQoPMode=1 arms are ~50 min/task against ~5 min for the
# legacy-Q arms, so they go first and get the whole 80-slot budget; everything
# else follows. NPAR is capped at 80 concurrent cmsRun for the whole script
# (the machine-wide rule) and STAGGER=3 spreads the starts -- firing 80 jobs
# at one instant wedges them all in the NSS/sssd user lookup.
#
# NO EXTRACTIONS RUN HERE. extract_parallel.sh takes one BLAS-pinned python
# per shard and the pair has previously put this user over `ulimit -u`, which
# kills running cmsRun jobs with what looks like a physics segfault (see
# finish_muon_closure_260902.sh). Extract after this script reports ALL DONE.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=${LOG:-$RES/runs/prod260903x}
mkdir -p "$LOG"
cd "$RES"

export CFG=${CFG:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhResClosure.py}
CFG_ST=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhResClosure.py
CFG_TT=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
export STAGGER=3
NPAR=${NPAR:-80}

# Common single-track muon-gun configuration, exactly the 2026-08-30 arms.
E_MUGUN="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1"
# J/psi gun ditrack, exactly run_ditrack_jpsigun_260902.sh.
E_JPSIGUN="trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False applyHltFilter=False useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0"
# B->J/psi+X v3 ditrack, exactly run_ditrack_btojpsix_v3_260902.sh (ladder rung
# B: SIM-matched grid field + aligned geometry + the production GT).
E_BTOJPSIX="trackSrc=ALCARECOTkAlJpsiX useLegacyPairLoop=False doTrigger=True applyHltFilter=False useIdealGeometry=False useOpera3D=True globalTag=106X_mcRun2_asymptotic_v17 CgfQoPMode=0"

ndone() { ls -d "$CEPH/resolution_trackres_$1"/task_*/.complete 2>/dev/null | wc -l; }

guard() {
  local used lim
  used=$(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l); lim=$(ulimit -u)
  echo "    [guard] threads $used / $lim"
  if [ "$used" -gt $(( lim * 6 / 10 )) ]; then echo "    [guard] ABORT"; exit 1; fi
}

# sample <tag> <cfg> <filelist> <ntask> <extra...>
sample() {
  local tag=$1 cfg=$2 flist=$3 ntask=$4; shift 4
  local extra="$*"
  local have; have=$(ndone "$tag")
  if [ "$have" -eq "$ntask" ]; then echo "=== skip $tag ($have/$ntask already complete)"; return 0; fi
  echo "=== $tag  ($have/$ntask done, $ntask tasks, NPAR=$NPAR)  $(date +%H:%M:%S)"
  guard
  local blk end
  for (( blk=0; blk<ntask; blk+=NPAR )); do
    end=$(( blk + NPAR - 1 )); [ "$end" -ge "$ntask" ] && end=$(( ntask - 1 ))
    echo "    block $blk-$end  $(date +%H:%M:%S)"
    CFG=$cfg FILELIST=$RES/$flist OUTTAG=$tag EXTRA="$extra" \
      ./run_local_trackres.sh "$NPAR" "$blk" "$end" >> "$LOG/refit_$tag.log" 2>&1
  done
  echo "    [$tag] $(ndone "$tag")/$ntask complete  $(date +%H:%M:%S)"
}

echo "######## START $(date) ########"

# ---- the long ones first: CgfQoPMode=1 single-track (~50 min/task)
sample mugun_ul16_260903x   "$CFG_ST" simprod/filelist_mugun_ul16.txt   160 "$E_MUGUN CgfQoPMode=1"
sample mugun_lowpt_260903x  "$CFG_ST" simprod/filelist_mugun_lowpt.txt  160 "$E_MUGUN CgfQoPMode=1"

# ---- legacy-Q single-track arms (~5 min/task)
sample mugun_ul16_260903x_m0          "$CFG_ST" simprod/filelist_mugun_ul16.txt           160 "$E_MUGUN CgfQoPMode=0"
sample mugun_lowpt_260903x_m0         "$CFG_ST" simprod/filelist_mugun_lowpt.txt          160 "$E_MUGUN CgfQoPMode=0"
sample mugun_lowpt_noms_260903x_m0    "$CFG_ST" simprod/filelist_mugun_lowpt_noms.txt      80 "$E_MUGUN CgfQoPMode=0"
sample mugun_lowpt_nomsrad_260903x_m0 "$CFG_ST" simprod/filelist_mugun_lowpt_nomsrad.txt   80 "$E_MUGUN CgfQoPMode=0"

# ---- ditrack (~12-17 min/task)
sample jpsigun_ul16_260903x_m0  "$CFG_TT" simprod/filelist_jpsigun_ul16.txt      160 "$E_JPSIGUN"
sample btojpsix_v3_260903x_m0   "$CFG_TT" simprod/filelist_btojpsix_v3_chunk50.txt 48 "$E_BTOJPSIX"

echo "######## SUMMARY $(date) ########"
for t in mugun_ul16_260903x mugun_lowpt_260903x \
         mugun_ul16_260903x_m0 mugun_lowpt_260903x_m0 \
         mugun_lowpt_noms_260903x_m0 mugun_lowpt_nomsrad_260903x_m0 \
         jpsigun_ul16_260903x_m0 btojpsix_v3_260903x_m0; do
  printf '  %-34s %s complete\n' "$t" "$(ndone $t)"
done
echo "######## ALL DONE $(date) ########"
