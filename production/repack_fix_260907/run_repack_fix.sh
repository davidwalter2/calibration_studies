#!/bin/bash
# Repair the four bad J/psi MC ALCARECO inputs of jpsimc_20M_260906_v2.
#
#   * 2830000/BDA060EF..., 60000/0909778B..., 60000/4B9D2D77...
#       the /ceph GROUP-STORE repacks have a corrupt provenance blob (exit 91).
#       On 2026-09-06 they were "recovered" by xrdcp-ing the file from
#       cms-xrd-global into .../restaged/jpsimc_20M_260905/ -- but the CENTRAL
#       copy is the UN-REPACKED split-99 original, so the recovery re-introduced
#       ROOT #19773 and the 12 chunks made from it are unusable.  Those staged
#       files are byte-size-identical to what the redirector serves today, so
#       they ARE the central originals and can be repacked in place, with no
#       second 6 GB download.
#   * 70000/03249796...
#       the group-store copy IS correctly split-1 (verified) but also has the
#       corrupt provenance blob, so it must be re-made from the central
#       original, which has to be fetched first.
#
# Output: /ceph/.../restaged/jpsimc_20M_260906_repack/  (never the group store).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRATCH=/ceph/submit/data/user/d/david_w/ZMass/scratch_repack_260906
OUTDIR=/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack
STAGED=/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260905
LFNBASE=/store/mc/RunIISummer20UL16RECO/JPsiToMuMu_Pt8toInf-pythia8/ALCARECO/TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13-v2
BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run"
mkdir -p "$SCRATCH/orig" "$SCRATCH/logs" "$OUTDIR"
export X509_USER_PROXY=/home/submit/david_w/x509up_u$(id -u)

run_repack () {   # <in> <out> <tag>
  APPTAINER_BIND="$BIND" cmssw-el7 --command-to-run \
      "$HERE/repack_worker.sh" "$1" "$2" > "$SCRATCH/logs/repack_$3.log" 2>&1
  echo "[$3] repack rc=$? -> $2"
}

# --- the three staged split-99 originals: repack straight from POSIX ---------
for f in BDA060EF-B8F8-7349-9277-363C3AB7EA76 \
         0909778B-8728-764F-B41B-1C9DCE5C849E \
         4B9D2D77-92ED-8440-8697-DD4AC560E61C ; do
  run_repack "$STAGED/$f.root" "$OUTDIR/$f.root" "$f" &
done

# --- the fourth: fetch the central original, verify its size, then repack ----
(
  f=03249796-312B-514A-990A-00EC873E70E2
  sub=70000            # the LFN subdirectory -- NOT optional, the base alone 404s
  want=1986290360
  dst=$SCRATCH/orig/$f.root
  if [[ ! -s $dst || $(stat -c %s "$dst") -ne $want ]]; then
    for try in 1 2 3; do
      rm -f "$dst.part"
      echo "[$f] xrdcp attempt $try $(date -Is)"
      xrdcp -f -s "root://cms-xrd-global.cern.ch/$LFNBASE/$sub/$f.root" "$dst.part"
      got=$(stat -c %s "$dst.part" 2>/dev/null || echo 0)
      echo "[$f] got=$got want=$want"
      # xrdcp has returned 0 on a truncated copy here before: trust the SIZE.
      [[ $got -eq $want ]] && { mv "$dst.part" "$dst"; break; }
    done
  fi
  [[ -s $dst && $(stat -c %s "$dst") -eq $want ]] || { echo "[$f] FETCH FAILED"; exit 1; }
  echo "[$f] fetched OK $(stat -c %s "$dst") B"
  run_repack "$dst" "$OUTDIR/$f.root" "$f"
) > "$SCRATCH/logs/fetch_03249796.log" 2>&1 &

wait
echo "ALL DONE $(date -Is)"
ls -la "$OUTDIR"
