#!/bin/bash
# ---------------------------------------------------------------------------
# Build (or re-enter) the TensorFlow/rabbit environment on MIT Engaging (ORCD).
#
#   source setup_env_engaging.sh          # just activate an existing env
#   ./setup_env_engaging.sh --install     # create it from scratch (~10 min)
#
# The env lives in POOL, not $HOME: the CUDA wheels alone are ~5 GB and $HOME
# is quota'd at 200 GB with snapshots.  Pool is 1 TB, unbacked-up, which is the
# right trade for a rebuildable environment.
#
# Versions are pinned to the wmassdev CVMFS image used on submit
# (/cvmfs/unpacked.cern.ch/.../cmswmassdocker/wmassdevrolling:latest) so the
# fits are numerically identical on both sites:
#     python 3.13   numpy 2.4.6   scipy 1.18.0   TF 2.21.0
# ---------------------------------------------------------------------------

# This file is meant to be SOURCED from job scripts, so it must not leak shell
# options or misread the caller's positional parameters: `source foo.sh` inside
# a script that was invoked as `masslik_fit_gpu.sbatch gun --rabbit` sees
# $1 == "gun".  Hence the explicit "am I being executed?" test below rather
# than a bare [ "$1" = --install ].
_ENG_SOURCED=1
(return 0 2>/dev/null) || _ENG_SOURCED=0

ENGAGING_ENV=${ENGAGING_ENV:-$HOME/orcd/pool/env/tf}
ZMASS=${ZMASS:-$HOME/orcd/pool/zmass}

# conda/mamba: the only module on Engaging that ships a modern python.
# (system python3 on the login node is 3.6.8; the `python/3.10.8-x86_64`
#  module is under `deprecated-modules`.)
module load miniforge/25.11.0-0 2>/dev/null
export PATH=/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin:$PATH

# keep every cache out of $HOME
export CONDA_PKGS_DIRS=$HOME/orcd/pool/tmp/conda-pkgs
export PIP_CACHE_DIR=$HOME/orcd/pool/tmp/pip-cache
export TMPDIR=$HOME/orcd/pool/tmp

if [ "$_ENG_SOURCED" = 0 ] && [ "${1:-}" = "--install" ]; then
  set -eu
  mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$ENGAGING_ENV")"

  echo "=== [1/5] conda env: python 3.13 -> $ENGAGING_ENV"
  conda create -y -p "$ENGAGING_ENV" python=3.13 pip

  PY="$ENGAGING_ENV/bin/python"

  echo "=== [2/5] TensorFlow (GPU) + scientific stack"
  # tensorflow[and-cuda] pulls the nvidia-* CUDA 12 wheels; no `module load cuda`
  # is needed (and mixing the module with the wheels breaks the cuDNN lookup).
  "$PY" -m pip install --no-input \
      "tensorflow[and-cuda]==2.21.0" \
      "tensorflow-probability" \
      "numpy==2.4.6" "scipy==1.18.0" "h5py==3.14.0" \
      "matplotlib==3.11.0" "mplhep==1.3.0" "uproot==5.7.4" \
      "hist" "boost-histogram" "pandas" "seaborn" "narf-ioutils" \
      "tf-keras" "lz4" "hdf5plugin" "plotly" "kaleido" 2>&1 | tail -25
  # tf-keras: tensorflow_probability >= TF 2.16 imports it in
  #   _validate_tf_environment and dies with ModuleNotFoundError otherwise --
  #   rabbit imports tfp, so rabbit_fit.py cannot start without it.
  # lz4 + hdf5plugin: wums[pickling] extras; wums/output_tools.py imports
  #   lz4.frame at module scope and rabbit_fit.py imports output_tools.
  # (jax is only touched by wums/fitutilsjax.py, which nothing here imports.)

  echo "=== [3/5] rabbit (editable, no deps -- they are pinned above)"
  # NB: the `rabbit` package on PyPI is an unrelated CLI proxy.  The WMass
  # rabbit is only on GitHub; see stage_engaging.sh for how it gets here.
  if [ -d "$ZMASS/rabbit" ]; then
    "$PY" -m pip install --no-input --no-deps -e "$ZMASS/rabbit"
  else
    echo "  !! $ZMASS/rabbit missing -- run stage_engaging.sh first"
  fi

  echo "=== [4/5] wums"
  # wums 0.2.0 from PyPI, NOT the 0.1.12 tree in submit's mfs venv: rabbit's
  # tensorwriter imports `wums.sparse_hist`, which does not exist in 0.1.12, so
  # the older tree cannot run rabbit at all.  0.2.0 carries plot_tools,
  # output_tools and sparse_hist together and is what rabbit is developed
  # against.  (The submit-side `cf_*.py` scripts still run against 0.1.12 via
  # calibration_studies/env_tf/pypath -- if a plot_tools call ever diverges
  # between the two, that is where it will show up, not in the fit numbers.)
  "$PY" -m pip install --no-input "wums[pickling,plotting]"

  echo "=== [5/5] verify"
  "$PY" - <<'PYEOF'
import numpy, scipy, h5py, matplotlib, mplhep, uproot, sys
import tensorflow as tf
print("python", sys.version.split()[0], "numpy", numpy.__version__,
      "scipy", scipy.__version__, "TF", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
import rabbit, wums; print("rabbit", rabbit.__file__); print("wums", wums.__file__)
PYEOF
  echo "=== install complete: $ENGAGING_ENV"
fi

# ------------------------------------------------------------------ activate
export PATH="$ENGAGING_ENV/bin:$PATH"

# TF 2.21 installed as `tensorflow[and-cuda]` ships CUDA as the nvidia-*-cu12
# wheels, but does NOT put their lib/ dirs on the loader path by itself: on a
# GPU node it finds libcuda.so.1 (the driver, in /usr/lib64) and then fails
# `dlopen libcudart.so.12 / libcudnn.so.9`, logs "Cannot dlopen some GPU
# libraries ... Skipping registering GPU devices", and reports
# list_physical_devices('GPU') == [] with no error.  Wiring them in here is
# what makes the GPU visible.  ptxas comes from the cuda_nvcc wheel and XLA
# needs it on PATH.
_SP="$ENGAGING_ENV/lib/python3.13/site-packages"
# NOGLOB-SAFE, and it has to be.  A job script that expands an unquoted flag
# carrying a regex (`PRECOND=--preconditionParams .*`) protects itself with
# `set -f` and then sources THIS file: with globbing off the pattern below does
# not expand, no lib dir is found, LD_LIBRARY_PATH is left without the CUDA
# wheels, and TF then "Skipping registering GPU devices" at a log level
# TF_CPP_MIN_LOG_LEVEL=2 suppresses -- the fit runs on the CPU at ~1/100 the
# speed with no error anywhere (measured: jobs 22679406/22679407, two 4 h H200
# allocations that never touched the GPU, 2026-09-13).  So turn globbing back
# on for the loop and restore the caller's setting.
case $- in
  *f*) _NOGLOB=1; set +f ;;
  *)   _NOGLOB=0 ;;
esac
if [ -d "$_SP/nvidia" ]; then
  for _d in "$_SP"/nvidia/*/lib; do
    [ -d "$_d" ] && LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
  export LD_LIBRARY_PATH
  [ -d "$_SP/nvidia/cuda_nvcc/bin" ] && export PATH="$_SP/nvidia/cuda_nvcc/bin:$PATH"
fi
if [ "$_NOGLOB" = 1 ]; then set -f; fi
unset _SP _d _NOGLOB
export PYTHONPATH="$ZMASS/resolution:$ZMASS/rabbit${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2
# the CVH offline CF caches were produced with Kokoulin off; keep it that way
export CVH_IONI_KOKOULIN=0
unset _ENG_SOURCED
