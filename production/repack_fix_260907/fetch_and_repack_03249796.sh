#!/bin/bash
# The fourth file only: fetch the central original (its group-store repack is
# split-1 but has the corrupt provenance blob) and repack it to split-1.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRATCH=/ceph/submit/data/user/d/david_w/ZMass/scratch_repack_260906
OUTDIR=/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack
LFN=/store/mc/RunIISummer20UL16RECO/JPsiToMuMu_Pt8toInf-pythia8/ALCARECO/TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13-v2/70000/03249796-312B-514A-990A-00EC873E70E2.root
BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run"
f=03249796-312B-514A-990A-00EC873E70E2
want=1986290360
dst=$SCRATCH/orig/$f.root
export X509_USER_PROXY=/home/submit/david_w/x509up_u$(id -u)
mkdir -p "$SCRATCH/orig" "$OUTDIR"
if [[ ! -s $dst || $(stat -c %s "$dst") -ne $want ]]; then
  for try in 1 2 3; do
    rm -f "$dst.part"
    echo "[$f] xrdcp attempt $try $(date -Is)"
    xrdcp -f "root://cms-xrd-global.cern.ch/$LFN" "$dst.part"
    got=$(stat -c %s "$dst.part" 2>/dev/null || echo 0)
    echo "[$f] got=$got want=$want"
    # xrdcp has returned 0 on a truncated copy on this cluster: trust the SIZE.
    [[ $got -eq $want ]] && { mv "$dst.part" "$dst"; break; }
  done
fi
[[ -s $dst && $(stat -c %s "$dst") -eq $want ]] || { echo "[$f] FETCH FAILED"; exit 1; }
echo "[$f] fetched OK $(stat -c %s "$dst") B $(date -Is)"
APPTAINER_BIND="$BIND" cmssw-el7 --command-to-run \
    "$HERE/repack_worker.sh" "$dst" "$OUTDIR/$f.root" > "$SCRATCH/logs/repack_$f.log" 2>&1
echo "[$f] repack rc=$? $(date -Is)"
echo "FOURTH DONE"
