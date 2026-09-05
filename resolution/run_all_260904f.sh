#!/bin/bash
# Re-production of the three refits affected by the Gauss-Newton
# MOMENTUM-FLOOR CLAMP FIX of 2026-09-04.
#
# WHAT CHANGED. The clamp that keeps a Gauss-Newton step from driving the
# state into the propagator's refusal region used a HARD-CODED 2.0 GeV floor
# in the single-track maker and a `clampMomentumFloor` default of 2.0 GeV in
# the two-track maker. That value was chosen when Geant4ePropagator's
# PropagationPtotLimit was 1.0 GeV. The drivers lowered the propagation limit
# to 0.2 GeV long ago; the clamp floor was never lowered with it, so every
# track whose TRUE momentum is below 2 GeV was pinned at 2 GeV -- and where
# the reference momentum was ALREADY below the floor the rescale is negative,
# max(s,0) makes it a hard zero, and the whole coupled step freezes at the
# seed. On the flat-pT J/psi gun that is 12 % of candidates (chi2/ndof > 10,
# 99.7-100 % of them with a daughter below 2 GeV) and it carries the entire
# +0.21e-3 mass-scale offset (NOTES.md 2026-09-04 censoring entry).
#
# The floor is now a `clampMomentumFloor` cfi parameter on BOTH makers and the
# drivers derive it from the propagation limit as 1.25*plimit = 0.25 GeV.
#
# NEW TAGS `_260904f_m0`; nothing existing is overwritten and `.complete`
# sentinels make every stage resumable. Configurations are otherwise
# CHARACTER-FOR-CHARACTER the _260903x ones (run_all_260903x.sh), so the two
# productions differ in the clamp floor and nothing else.
#
# NO EXTRACTIONS RUN HERE (see run_all_260903x.sh: extract_parallel.sh + a
# running refit has put this user over `ulimit -u` before, which kills cmsRun
# with what looks like a physics segfault).
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=${LOG:-$RES/runs/prod260904f}
mkdir -p "$LOG"
cd "$RES"

CFG_ST=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhResClosure.py
CFG_TT=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
export STAGGER=3
NPAR=${NPAR:-80}

E_MUGUN="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1"
E_JPSIGUN="trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False applyHltFilter=False useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0"
E_BTOJPSIX="trackSrc=ALCARECOTkAlJpsiX useLegacyPairLoop=False doTrigger=True applyHltFilter=False useIdealGeometry=False useOpera3D=True globalTag=106X_mcRun2_asymptotic_v17 CgfQoPMode=0"

ndone() { ls -d "$CEPH/resolution_trackres_$1"/task_*/.complete 2>/dev/null | wc -l; }

guard() {
  local used lim
  used=$(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l); lim=$(ulimit -u)
  echo "    [guard] threads $used / $lim"
  if [ "$used" -gt $(( lim * 6 / 10 )) ]; then echo "    [guard] ABORT"; exit 1; fi
}

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
sample jpsigun_ul16_260904f_m0  "$CFG_TT" simprod/filelist_jpsigun_ul16.txt        160 "$E_JPSIGUN"
sample mugun_lowpt_260904f_m0   "$CFG_ST" simprod/filelist_mugun_lowpt.txt         160 "$E_MUGUN CgfQoPMode=0"
sample btojpsix_v3_260904f_m0   "$CFG_TT" simprod/filelist_btojpsix_v3_chunk50.txt  48 "$E_BTOJPSIX"
echo "######## SUMMARY $(date) ########"
for t in jpsigun_ul16_260904f_m0 mugun_lowpt_260904f_m0 btojpsix_v3_260904f_m0; do
  printf '  %-34s %s complete\n' "$t" "$(ndone $t)"
done
echo "######## ALL DONE $(date) ########"
