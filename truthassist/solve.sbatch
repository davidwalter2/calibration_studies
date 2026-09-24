#!/bin/bash
# Truth-assisted solve: aggregate the COMPLETE derive tasks and solve for the
# free parameters (alignment 0-5 + pixel classes 16-21; Bz 6 and material 7
# frozen). One dense ~65 GB matrix, accumulated in place.
#SBATCH --job-name=cvhtruthsolve
#SBATCH --partition=submit
#SBATCH --cpus-per-task=45
#SBATCH --mem=90G
#SBATCH --time=06:00:00
set -uo pipefail
B=${BASE:-/ceph/submit/data/user/d/david_w/ZMass/cvh/truthassist_260924}
OUT=${OUT:-$B/solve/truthcor}
mkdir -p "$(dirname "$OUT")"
export SCRAM_ARCH=el9_amd64_gcc12
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2/src && eval "$(scramv1 runtime -sh)" || exit 1
export OPENBLAS_NUM_THREADS=45 OMP_NUM_THREADS=45 MKL_NUM_THREADS=45
FILES=()
for c in "$B"/jpsi_derive/task_*/.complete "$B"/z_derive/task_*/.complete; do
    [[ -f $c ]] && FILES+=("$(dirname "$c")"/globalcor_truth_*.root)
done
echo "host $(hostname): ${#FILES[@]} files from complete derive tasks"
/usr/bin/time -v python3 /work/submit/david_w/ZMass/calibration_studies/truthassist/solve_truth.py \
    -i "${FILES[@]}" -o "$OUT" --threads 40 ${SOLVE_ARGS:-}
