# Environment for calibration_studies (lineshape / LHAPDF).
#
# Provides: lhapdf 6.5.4 + NNPDF31_nnlo_as_0118 (from CVMFS) and the scientific
# stack (scipy/h5py/matplotlib/mplhep/hist) from the LCG view, plus pip-installed
# wums, hdf5plugin and numpy>=2 in the venv.
#
# Usage:  source setup_env.sh   (from any directory)
#
# Ordering note: the LCG view's setup.sh prepends its own site-packages to
# PYTHONPATH, which would otherwise shadow venv packages (e.g. its numpy 1.26
# over the venv's numpy 2 that is required to unpickle the histmaker hdf5
# files). So after sourcing the view and activating the venv we put the venv
# site-packages first on PYTHONPATH. lhapdf (only in the view) still resolves
# from the trailing view path, and its shared libs come from the view's
# LD_LIBRARY_PATH.

_LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_106a/x86_64-el9-gcc11-opt
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env"

source "${_LCG_VIEW}/setup.sh"
source "${_ENV_DIR}/bin/activate"

_VENV_SP="$(${_ENV_DIR}/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="${_VENV_SP}${PYTHONPATH:+:${PYTHONPATH}}"

echo "calibration_studies env ready: $(python --version 2>&1), numpy $(python -c 'import numpy;print(numpy.__version__)' 2>/dev/null), lhapdf $(python -c 'import lhapdf;print(lhapdf.version())' 2>/dev/null)"
