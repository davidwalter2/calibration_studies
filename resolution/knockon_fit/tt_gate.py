"""build_pairs_tt on K_S -> pi pi two-track step-record files (the mass
window moved to m_KS): switch-off bit identity against a reference module and
the switched-on families.  usage: tt_gate.py MODULE CACHE FILES..."""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mod = importlib.import_module(sys.argv[1])
# the reco-only K_S files carry Jpsigen_mass = -99: centre the window there
# (z is then meaningless; the gate compares the exponents only)
mod.MJPSI = -99.0
ns = argparse.Namespace(files=sys.argv[3:], ntasks=None,
                        pairs_cache=sys.argv[2])
mod.build_pairs_tt(ns, None)
