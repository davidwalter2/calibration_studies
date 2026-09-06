#!/bin/bash
# One HTCondor job = one CHUNK of one UL16 DY MiniAODv2 file -> one CVH
# two-track (Z -> mumu) refit output directory on /ceph.
#
# WHY THIS IS A GRID JOB AND NOT array_dymc_dev2.sbatch WITH A DIFFERENT HEADER
# ------------------------------------------------------------------------
# The submit HTCondor pool has exactly ONE local execute slot -- 1 CPU on
# submit06, whose START expression demands `Submit_LocalTest` -- and everything
# else is glideins flocked to the CMS global pool via t3serv009. Measured
# 2026-09-06 on real slots:
#   * mit_tier3 (t3btch001)      : el7, NO /ceph, NO /work, NO /home, no singularity
#   * global pool (DESY, IIHE)   : el9 under cms:rhel9-x86_64, NO /ceph, cvmfs OK,
#                                  xrootd read of the input OK, /srv scratch 3-7 TB
# So NOTHING that runs condor here can see the shared filesystem. The three
# things a slurm task takes for granted therefore have to be carried:
#   1. the CMSSW area  -> base release from cvmfs + a 64 MB overlay tarball
#      (`scram project` then untar; the area is NOT relocatable as a whole --
#      .SCRAM/RuntimeCache.json and MakeData/*.mk bake in the absolute path,
#      so an untar-in-place run reports the ORIGINAL CMSSW_BASE and silently
#      uses the wrong libraries)
#   2. the input       -> addressed BY DOOR (submit50-55), never by LFN through
#      a redirector: these are our own repacked copies and the same LFN exists
#      centrally with different content. See the chunk block below.
#   3. the output      -> xrdcp back to root://submit50.mit.edu/, whose namespace
#      root is /ceph/submit
#
# THE STAGE-OUT IS VERIFIED BY SIZE, NEVER BY EXIT CODE. `xrdcp` has truncated
# outputs on this cluster before (two condor attempts of the BtoJpsiX
# production were destroyed that way) and returned 0 while doing it. The
# `.complete` sentinel is copied LAST, after every payload file has been read
# back and its size compared, so a partial stage-out can never look finished.
#
# usage: job_jpsimc_v2.sh <idx>
# env expected: OUTBASE OUTHOST OUTROOT CFGREL PAYLOAD CHUNKS INITBASE NTHREADS
#               EXTRA INDOORS
set -uo pipefail

IDX=${1:?usage: job_jpsimc_v2.sh <idx>}
: "${OUTBASE:?}"                       # POSIX /ceph path, for the log line only
OUTHOST=${OUTHOST:-root://submit50.mit.edu/}
OUTROOT=${OUTROOT:?}                   # xrootd-side path, e.g. /data/user/d/david_w/...
INDOORS=${INDOORS:-submit50.mit.edu}   # see the chunk block below
NTHREADS=${NTHREADS:-1}
SCRAM_ARCH_USE=${SCRAM_ARCH_USE:-el9_amd64_gcc12}
RELEASE=${RELEASE:-CMSSW_15_0_19_patch2}
CFGREL=${CFGREL:-src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
PAYLOAD=${PAYLOAD:-overlay_dev2.tgz}
CHUNKS=${CHUNKS:-chunks.txt}           # transferred in with the job
INITBASE=${INITBASE:-}                 # scalar-potential coefficients, transferred in

T0=$(date +%s)
TASK=task_$(printf '%04d' "$IDX")
SCRATCH=${_CONDOR_SCRATCH_DIR:-$(pwd)}
cd "$SCRATCH"
echo ">>> $TASK host=$(hostname -f 2>/dev/null || hostname) site=${GLIDEIN_Site:-?} $(date -Is)"
echo ">>> scratch=$SCRATCH threads=$NTHREADS"

fail() { echo "[FATAL] $*" >&2; exit "${2:-1}"; }

# ---- cvmfs guard: a PARTIAL mount passes a shallow -d test -----------------
for p in /cvmfs/cms.cern.ch/cmsset_default.sh \
         /cvmfs/cms.cern.ch/share/etc/default-scramv1-version \
         "/cvmfs/cms.cern.ch/$SCRAM_ARCH_USE/cms/cmssw-patch/$RELEASE"; do
  [[ -e "$p" ]] || fail "cvmfs incomplete on $(hostname): missing $p" 10
done

# ---- the chunk ------------------------------------------------------------
# THE INPUT IS ADDRESSED BY DOOR, NOT BY LFN -- and that is not a style choice.
# These files are OUR OWN repacked split-1 copies living under the CMS /store
# namespace in the submit group store. The same LFN EXISTS CENTRALLY with
# DIFFERENT CONTENT (the un-repacked split-99 original): measured on the first
# file of the list, submit50 serves 1 783 846 343 B and
# cms-xrd-global.cern.ch serves 1 840 069 145 B. Going through a redirector
# would therefore silently feed the production the wrong sample -- one whose
# split level makes the CVH refit fail on nearly every candidate -- and the
# outputs would still look perfectly valid.
#
# So the mapping is a pure prefix strip against a NAMED door whose xrootd
# namespace root IS /ceph/submit. It is exact, it has no redirector in the
# path, and it works uniformly for the group-store files and for the 12 chunk
# lines repointed at the re-staged copies in the user store.
#
# The size of what we opened is checked against the size recorded in the chunk
# list at submit time (4th field), so a door that ever served a different
# replica is caught before a single event is processed.
LINE=$(sed -n "$((IDX + 1))p" "$CHUNKS")
[[ -n "$LINE" ]] || fail "empty chunk line $((IDX+1)) of $CHUNKS" 2
read -r CEPHPATH SKIP NEV WANTSIZE _rest <<< "$LINE"
[[ -n "$CEPHPATH" && -n "$SKIP" && -n "$NEV" ]] || fail "malformed chunk line: $LINE" 2
XPATH=${CEPHPATH#/ceph/submit}
[[ "$XPATH" != "$CEPHPATH" ]] || fail "chunk path is not under /ceph/submit: $CEPHPATH" 2

export X509_USER_PROXY=${X509_USER_PROXY:-$PWD/x509up}
# Spread 1642 jobs over the doors rather than reading ~740 GB through one
# login node; on failure fall through to the next door rather than dying.
read -r -a DOORS <<< "${INDOORS:-submit50.mit.edu}"
NDOOR=${#DOORS[@]}
INURL=""
for k in $(seq 0 $((NDOOR - 1))); do
  D=${DOORS[$(( (IDX + k) % NDOOR ))]}
  GOT=$(timeout 180 xrdfs "$D" stat "$XPATH" 2>/dev/null | awk '/^Size:/{print $2}')
  [[ -z "$GOT" ]] && { echo "    door $D: no answer for $XPATH" >&2; continue; }
  if [[ -n "$WANTSIZE" && "$GOT" != "$WANTSIZE" ]]; then
    fail "door $D served $GOT B for $XPATH, chunk list says $WANTSIZE -- WRONG REPLICA" 8
  fi
  INURL="root://$D/$XPATH"; INDOOR=$D; break
done
[[ -n "$INURL" ]] || fail "no door of [${INDOORS}] served $XPATH" 3
echo ">>> door=$INDOOR path=$XPATH size=${GOT:-?} skip=$SKIP nev=$NEV"

# ---- CMSSW: cvmfs base release + the overlay ------------------------------
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=$SCRAM_ARCH_USE
scram project CMSSW "$RELEASE" > scram_project.log 2>&1 || fail "scram project failed" 11
[[ -f "$PAYLOAD" ]] || fail "payload $PAYLOAD not transferred" 12
tar xzf "$PAYLOAD" -C "$RELEASE" || fail "payload untar failed" 12
cd "$RELEASE"
eval "$(scramv1 runtime -sh)" || fail "scram runtime failed" 11
# The relocation trap: if CMSSW_BASE is not the area we just built, the job is
# about to run the WRONG libraries -- silently, because cvmfs provides a
# complete release under the old path's release base.
[[ "$CMSSW_BASE" == "$SCRATCH/$RELEASE" ]] \
  || fail "CMSSW_BASE=$CMSSW_BASE is not $SCRATCH/$RELEASE (payload not relocated)" 11
[[ -s "$CMSSW_BASE/lib/$SCRAM_ARCH/pluginAnalysisHitAnalyzerAuto.so" ]] \
  || fail "CVH plugin missing from the overlay" 12
cd "$SCRATCH"

# ---- run ------------------------------------------------------------------
# `scalarPot3DInitFile` is the ONE option in EXTRA whose value is an absolute
# path outside the release (it comes from mfs/, on /work). Everything else the
# driver needs is either inside the payload -- `materialGroupsFile` defaults to
# $CMSSW_BASE/src/Analysis/HitAnalyzer/data/materialGroups50.txt -- or is a
# conditions payload from the global tag over frontier. Rewrite it to the copy
# condor transferred in, and ASSERT it: without the file the maker throws
# "ScalarPot3DEval: cannot open ..." while constructing, which costs the whole
# job 130 s in.
if [[ -n "$INITBASE" ]]; then
  [[ -s "$SCRATCH/$INITBASE" ]] || fail "coefficient file $INITBASE not transferred" 13
  NEWEXTRA=""
  for tok in ${EXTRA:-}; do
    case "$tok" in
      scalarPot3DInitFile=*) tok="scalarPot3DInitFile=$SCRATCH/$INITBASE";;
    esac
    NEWEXTRA="$NEWEXTRA $tok"
  done
  EXTRA=$NEWEXTRA
fi
# Nothing else may point outside the sandbox.
for tok in ${EXTRA:-}; do
  case "$tok" in
    *=/work/*|*=/ceph/*|*=/home/*) fail "option points off the worker: $tok" 14;;
  esac
done

mkdir -p rundir && cd rundir
# shellcheck disable=SC2086
cmsRun "$CMSSW_BASE/$CFGREL" input="$INURL" skipEvents="$SKIP" nEvents="$NEV" \
    numberOfThreads="$NTHREADS" ${EXTRA:-} > local.log 2>&1
rc=$?
T1=$(date +%s)
echo ">>> cmsRun rc=$rc after $((T1-T0)) s"
if [[ $rc -ne 0 ]]; then tail -60 local.log >&2; fail "cmsRun rc=$rc" "$rc"; fi

shopt -s nullglob
out=(globalcor_*.root)
[[ ${#out[@]} -eq $NTHREADS ]] \
  || fail "expected $NTHREADS stream file(s), found ${#out[@]}" 4
for f in "${out[@]}"; do [[ -s "$f" ]] || fail "empty output $f" 4; done
# `skipBadFiles=True` makes an unreadable input SILENT: the job writes a valid,
# EMPTY output. The maker always prints a fit summary, so attempted=0 catches it.
grep -q "fit summary  attempted=0 " local.log && fail "attempted=0 -- input yielded nothing" 5

# ---- stage out, verified by size -----------------------------------------
DESTDIR="${OUTROOT%/}/$TASK"                 # server-side path
DEST="${OUTHOST%/}/$DESTDIR"                 # full URL
XHOST=${OUTHOST#root://}; XHOST=${XHOST%/}   # bare host for xrdfs
echo ">>> staging out to $DEST"
# submit_jpsimc_v2.sh pre-creates every task dir over POSIX ceph, but a resumed
# or hand-driven job must not depend on that: -p makes xrdcp create the path.
timeout 300 xrdfs "$XHOST" mkdir -p "$DESTDIR" >/dev/null 2>&1
copy_verified() {
  local src=$1 dst=$2 want got
  want=$(stat -c %s "$src")
  for attempt in 1 2 3; do
    timeout 3600 xrdcp -f -p -N "$src" "$dst" >/dev/null 2>&1
    got=$(timeout 300 xrdfs "$XHOST" stat "${dst#${OUTHOST%/}/}" 2>/dev/null \
          | awk '/^Size:/{print $2}')
    [[ "$got" == "$want" ]] && { echo "    ok $(basename "$src") $want B"; return 0; }
    echo "    retry $attempt: $(basename "$src") wanted $want got ${got:-none}" >&2
  done
  return 1
}
for f in "${out[@]}" ; do copy_verified "$f" "$DEST/$f" || fail "stage-out of $f could not be verified" 20; done
copy_verified local.log "$DEST/local.log" || echo "[warn] log stage-out unverified" >&2

# The sentinel goes LAST and only once every payload file has been read back.
# It carries content rather than being empty: a zero-byte xrdcp is the one
# transfer whose "success" is indistinguishable from a door that silently
# dropped it, and the provenance is free.
{ echo "$TASK"; echo "host=$(hostname -f 2>/dev/null || hostname) site=${GLIDEIN_Site:-?}"
  echo "finished=$(date -Is) seconds=$(( $(date +%s) - T0 ))"
  echo "streams=$NTHREADS"; } > .complete
copy_verified .complete "$DEST/.complete" || fail "sentinel stage-out failed" 21
echo "[done] $TASK in $(( $(date +%s) - T0 )) s"
