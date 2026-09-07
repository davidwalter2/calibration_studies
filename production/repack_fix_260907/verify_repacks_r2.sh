#!/bin/bash
# verify_repacks.sh for the round-2 files (different expected entry counts:
# these are read from the CENTRAL originals, which is the point -- FDB8C946's
# local copy claimed 19 797 and the original holds 51 478).
set -uo pipefail
OUTDIR=/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack
BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run"
declare -A WANTEV=( [FDB8C946-2D24-0849-AFBB-4CF6EB0AD8E9]=51478
                    [0B395A0D-B510-814C-ADFB-E201F55CC6AB]=56789
                    [290E1F42-3F99-7B4F-8793-9EFB57F8B51E]=46077 )
source /cvmfs/cms.cern.ch/cmsset_default.sh > /dev/null
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src && eval "$(scramv1 runtime -sh)"
cd /tmp
for f in "${!WANTEV[@]}"; do
  p=$OUTDIR/$f.root
  echo "==================== $f  (want ${WANTEV[$f]} events)"
  [[ -s $p ]] || { echo "  MISSING"; continue; }
  echo "  size: $(stat -c %s "$p")"
  python3 /work/submit/david_w/ZMass/calibration_studies/transfer/readback_check/splitlevels.py "$p" 2>/dev/null \
    | grep -E '^  entries|^  split'
  out=$(APPTAINER_BIND="$BIND" cmssw-el7 --command-to-run bash -c \
        "cd /work/submit/david_w/ZMass/CMSSW_10_6_26_dev/src && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval \$(scramv1 runtime -sh) && edmProvDump $p" 2>&1)
  grep -qiE 'exception|EntryError|FormatIncompatibility' <<< "$out" \
    && { echo "  provdump FAILS"; grep -iE 'exception|EntryError' <<< "$out" | head -2; } \
    || echo "  provdump OK ($(wc -l <<< "$out") lines)"
done
echo; echo "==================== NUL scan"
# in a CLEAN env: the venv python cannot start under the CMSSW one sourced above
# (PYTHONHOME/sys.prefix point at the release's python3 and `encodings` is lost)
env -i HOME="$HOME" PATH=/usr/bin:/bin bash -lc \
  'source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
   python3 /work/submit/david_w/ZMass/calibration_studies/production/repack_fix_260907/scan_pset_nulls.py '"$OUTDIR"'/FDB8C946*.root '"$OUTDIR"'/0B395A0D*.root '"$OUTDIR"'/290E1F42*.root'
