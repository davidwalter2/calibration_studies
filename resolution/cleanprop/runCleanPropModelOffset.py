# runCleanPropModelOffset.py -- the CMSSW area's runCleanPropModel.py with the
# start POSITION offset from the origin (the reference paths of
# cleanprop/realtrk_path_average.py).
#
# The configuration is runCleanPropModel.py executed unchanged (same options,
# conditions and switches); only G4ePropagationExport's start position is
# replaced afterwards by CLEANPROP_X0 / CLEANPROP_Y0 / CLEANPROP_Z0 (cm, global
# frame; 0 when unset).  The start direction and momentum are the ordinary
# pt / eta / phi options.
import os

_BASE = os.path.join(os.environ["CMSSW_BASE"], "src", "Analysis", "HitAnalyzer",
                     "test", "runCleanPropModel.py")
exec(compile(open(_BASE).read(), _BASE, "exec"))

_x0 = [float(os.environ.get(f"CLEANPROP_{c}0", "0")) for c in "XYZ"]
process.propExport.initialPosition = cms.vdouble(*_x0)  # noqa: F821 (from _BASE)
print(f"[pathavg] start position = {_x0} cm")
