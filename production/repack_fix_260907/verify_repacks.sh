#!/bin/bash
# Gate on the four repacked inputs before they are put in front of a production:
#   1. ROOT split level of the cluster / rechit / TrackExtra branches must be 1
#      (the ROOT #19773 condition -- 99 is what silently corrupts strip hits);
#   2. edmProvDump must READ (that is the corrupt-provenance failure, exit 91);
#   3. the Events entry count must equal what the chunk list tiles.
# Prints the size too, which has to go into the chunk list's 4th field.
set -uo pipefail
OUTDIR=${1:-/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack}
BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run"
declare -A WANTEV=(
  [BDA060EF-B8F8-7349-9277-363C3AB7EA76]=51231
  [0909778B-8728-764F-B41B-1C9DCE5C849E]=56267
  [4B9D2D77-92ED-8440-8697-DD4AC560E61C]=48621
  [03249796-312B-514A-990A-00EC873E70E2]=49516 )
source /cvmfs/cms.cern.ch/cmsset_default.sh > /dev/null
( cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src && eval "$(scramv1 runtime -sh)" ) 2>/dev/null
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src && eval "$(scramv1 runtime -sh)"
cd /tmp
for f in "${!WANTEV[@]}"; do
  p=$OUTDIR/$f.root
  echo "==================== $f"
  [[ -s $p ]] || { echo "  MISSING"; continue; }
  echo "  size    : $(stat -c %s "$p")"
  python3 /work/submit/david_w/ZMass/calibration_studies/transfer/readback_check/splitlevels.py "$p" 2>/dev/null \
    | grep -E '^  entries|^  split'
  echo "  want entries: ${WANTEV[$f]}"
done
echo
echo "==================== edmProvDump (el7 / CMSSW_10_6_26_dev)"
for f in "${!WANTEV[@]}"; do
  p=$OUTDIR/$f.root
  [[ -s $p ]] || continue
  out=$(APPTAINER_BIND="$BIND" cmssw-el7 --command-to-run bash -c \
        "cd /work/submit/david_w/ZMass/CMSSW_10_6_26_dev/src && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval \$(scramv1 runtime -sh) && edmProvDump $p" 2>&1)
  if grep -qiE 'exception|EntryError|FormatIncompatibility' <<< "$out"; then
    echo "  $f : PROVDUMP FAILS"; grep -iE 'exception|EntryError|FormatIncompatibility' <<< "$out" | head -3
  else
    echo "  $f : provdump OK ($(wc -l <<< "$out") lines)"
  fi
done
