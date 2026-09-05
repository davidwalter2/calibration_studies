#!/bin/bash
#SBATCH -J gpu_check
#SBATCH -A mit_general
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 00:15:00
#SBATCH -o /home/david_w/orcd/pool/zmass/logs/%x_%j.out
#SBATCH -e /home/david_w/orcd/pool/zmass/logs/%x_%j.err
# Does TF actually see the L40S, and is float64 matmul on the device?
set -u
source $HOME/orcd/pool/zmass/engaging/setup_env_engaging.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import time, numpy as np, tensorflow as tf
print("TF", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
assert gpus, "no GPU visible to TensorFlow"
with tf.device('/GPU:0'):
    a = tf.random.uniform((4096, 4096), dtype=tf.float64)
    b = tf.linalg.matmul(a, a); _ = b.numpy()          # warm up
    t0 = time.time(); 
    for _ in range(5): b = tf.linalg.matmul(a, a)
    _ = b.numpy(); tg = (time.time()-t0)/5
with tf.device('/CPU:0'):
    a2 = tf.identity(a)
    b2 = tf.linalg.matmul(a2, a2); _ = b2.numpy()
    t0 = time.time(); b2 = tf.linalg.matmul(a2, a2); _ = b2.numpy(); tc = time.time()-t0
print(f"float64 4096^3 matmul: GPU {tg*1e3:.1f} ms   CPU {tc*1e3:.1f} ms   speedup {tc/tg:.1f}x")
print("device of result:", b.device)
PY
