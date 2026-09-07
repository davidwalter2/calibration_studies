#!/bin/bash
# Round 2 of the input repair: the three files the sample-wide scans found that
# the first pass did not cover.
#
#   2830000/FDB8C946-...  "unopenable": ROOT opens it and reports 19 797 events,
#       but it has NO StreamerInfo, so CMSSW's PoolSource drops it and
#       skipBadFiles turns task_1313 into a silently EMPTY 19 797-event chunk.
#       Its local repack is 724 MB against a 1942 MB original, i.e. it is also
#       TRUNCATED -- so its true event count has to be re-read after repacking
#       and the chunk line re-tiled if it differs.
#   60000/0B395A0D-...   NUL runs in ParameterSets (8 entries, 428 B)
#   60000/290E1F42-...   NUL runs in ParameterSets (8 entries, 413 B)
#       These two RAN FINE (their 7 tasks sit at 0.000-0.020 % failures), i.e.
#       the zeroed psets happen to be ones cmsRun never parses.  They are
#       repacked anyway so their outputs can be compared candidate by candidate
#       against the existing ones -- the point is to PROVE the NULs were inert,
#       not to assume it.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRATCH=/ceph/submit/data/user/d/david_w/ZMass/scratch_repack_260906
OUTDIR=/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack
LFNBASE=/store/mc/RunIISummer20UL16RECO/JPsiToMuMu_Pt8toInf-pythia8/ALCARECO/TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13-v2
BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run"
export X509_USER_PROXY=/home/submit/david_w/x509up_u$(id -u)
mkdir -p "$SCRATCH/orig" "$SCRATCH/logs" "$OUTDIR"

one () {   # <subdir> <basename> <central size>
  local sub=$1 f=$2 want=$3 dst=$SCRATCH/orig/$2.root
  if [[ ! -s $dst || $(stat -c %s "$dst") -ne $want ]]; then
    for try in 1 2 3; do
      rm -f "$dst.part"
      echo "[$f] xrdcp attempt $try $(date -Is)"
      xrdcp -f "root://cms-xrd-global.cern.ch/$LFNBASE/$sub/$f.root" "$dst.part"
      local got; got=$(stat -c %s "$dst.part" 2>/dev/null || echo 0)
      echo "[$f] got=$got want=$want"
      [[ $got -eq $want ]] && { mv "$dst.part" "$dst"; break; }   # trust SIZE, not rc
    done
  fi
  [[ -s $dst && $(stat -c %s "$dst") -eq $want ]] || { echo "[$f] FETCH FAILED"; return 1; }
  echo "[$f] fetched OK $(date -Is)"
  APPTAINER_BIND="$BIND" cmssw-el7 --command-to-run \
      "$HERE/repack_worker.sh" "$dst" "$OUTDIR/$f.root" > "$SCRATCH/logs/repack_$f.log" 2>&1
  echo "[$f] repack rc=$? $(date -Is)"
}

one 2830000 FDB8C946-2D24-0849-AFBB-4CF6EB0AD8E9 1942543999 > "$SCRATCH/logs/r2_FDB8C946.log" 2>&1 &
one 60000   0B395A0D-B510-814C-ADFB-E201F55CC6AB 2145380665 > "$SCRATCH/logs/r2_0B395A0D.log" 2>&1 &
one 60000   290E1F42-3F99-7B4F-8793-9EFB57F8B51E 1879194716 > "$SCRATCH/logs/r2_290E1F42.log" 2>&1 &
wait
echo "ROUND2 DONE $(date -Is)"
