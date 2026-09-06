#!/bin/bash
# Rebuild the full-scale card on the COMPLETE 380-task DY cache and put every
# phase-1 variant on the GPU in one job. Chained so that nothing waits on a
# human: it starts the moment `append_pairs.py` finishes writing.
set -uo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
CACHE=$FS/runs/zpairs_dyv2_full.npz
CARD=$FS/cards/z_full380.hdf5
for i in $(seq 1 240); do
  [ -f "$CACHE" ] && python3 -c "import numpy,sys; numpy.load('$CACHE')" 2>/dev/null && break
  sleep 30
done
python3 -c "import numpy; numpy.load('$CACHE')" || { echo "cache never became readable"; exit 1; }
echo "[full380] cache ready $(date +%H:%M:%S)"
THREADS=32 $FS/run_tf.sh python3 -u $FS/make_card.py --pairs "$CACHE" \
    --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
    --shape 5 -o "$CARD" 2>&1 | tee $FS/logs/card_full380.log
[ -f "$CARD" ] || { echo "card not built"; exit 1; }
echo "[full380] card built $(date +%H:%M:%S)"
$FS/stage_eng.sh card "$CARD"
eng "bash -lc 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -G h200:1 -p mit_normal_gpu --export=ALL,CARD=\$HOME/orcd/pool/zmass/cards/$(basename $CARD),TAG=f380,CH=32768 fullscale_variants.sbatch'"
echo "[full380] submitted $(date +%H:%M:%S)"
