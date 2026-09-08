#!/bin/bash
# Stage the extended K(m) ladder cards and queue their fits.
set -uo pipefail
cd "$(dirname "$0")"
ROWS=""
for N in "$@"; do
  for c in z_full380_fl_s$N z_V_s$N; do
    [ -f cards/$c.hdf5 ] || { echo "[skip] cards/$c.hdf5 not built"; continue; }
    sz=$(stat -c%s cards/$c.hdf5)
    [ "$sz" -lt 1000000000 ] && { echo "[skip] cards/$c.hdf5 only $sz bytes (still writing)"; continue; }
    echo "[stage] $c ($((sz/1024/1024)) MB)"
    timeout 1800 rsync -a --info=progress2 cards/$c.hdf5 engaging:orcd/pool/zmass/cards/ >/dev/null || continue
    case $c in
      z_full380_fl_s*) ROWS="$ROWS Ss$N" ;;
      z_V_s*)          ROWS="$ROWS SVs$N" ;;
    esac
  done
done
[ -z "$ROWS" ] && { echo "nothing to submit"; exit 0; }
echo "[submit] rows:$ROWS"
timeout 300 eng "bash -lc 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable -G h200:1 --export=ALL,ROWS=\"$ROWS\",FRESH=1 rabbit_vmass_batch.sbatch'"
