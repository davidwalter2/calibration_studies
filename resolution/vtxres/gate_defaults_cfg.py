#!/usr/bin/env python3
"""EXACT gate on the per-channel `bsConstraint` DEFAULTS (task 2).

The defaults are a PYTHON change, so the decisive test is a python one: the
producer PSet that dev2's NEW DEFAULTS expand to must be character-identical
to the one dev3 expands to when the flag is given EXPLICITLY.  Anything else
that moved in the PSet shows up here too, which is the point.

Run inside the CMSSW environment of each area (the driver imports cms):

    ./gate_defaults_cfg.py dump  <cfi-or-driver spec>   # prints a PSet dump
    ./gate_defaults_cfg.py check                        # runs both + diffs

`check` shells out to the two areas itself.
"""
import subprocess
import sys

DEV2 = '/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2'
DEV3 = '/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev3'

# (label, module import, extra kwargs applied on TOP -- the dev3 leg supplies
# the flag explicitly, the dev2 leg relies on the new default)
CFIS = [
    ('ZMuMu',      'ResidualGlobalCorrectionMakerTwoTrackZMuMuG4e_cfi',      'globalCorZ',        True),
    ('Upsilon',    'ResidualGlobalCorrectionMakerTwoTrackUpsilonMuMuG4e_cfi', 'globalCorUpsilon', True),
    ('DiMuon',     'ResidualGlobalCorrectionMakerDiMuonG4e_cfi',              'ResidualGlobalCorrectionMakerDiMuonG4e', True),
    ('JpsiMuMu',   'ResidualGlobalCorrectionMakerTwoTrackJpsiMuMuG4e_cfi',    'globalCorJpsi',    False),
    ('JpsiKMuMu',  'ResidualGlobalCorrectionMakerTwoTrackJpsiKMuMuG4e_cfi',   None,               False),
    ('PiPi',       'ResidualGlobalCorrectionMakerTwoTrackPiPiG4e_cfi',        None,               False),
    ('KPi',        'ResidualGlobalCorrectionMakerTwoTrackKPiG4e_cfi',         None,               False),
    ('ProtonPi',   'ResidualGlobalCorrectionMakerTwoTrackProtonPiG4e_cfi',    None,               False),
]

DUMP = r'''
import importlib, sys
import FWCore.ParameterSet.Config as cms
mod = importlib.import_module('Analysis.HitAnalyzer.%s')
obj = None
for n in dir(mod):
    o = getattr(mod, n)
    if isinstance(o, cms.EDProducer):
        obj = o
        break
assert obj is not None, 'no EDProducer in %s'
print('bsConstraint=%%s' %% obj.bsConstraint.value())
'''


def run(area, modname):
    script = DUMP % (modname, modname)
    cmd = ('cd {a}/src && source /cvmfs/cms.cern.ch/cmsset_default.sh '
           "&& eval $(scramv1 runtime -sh) && python3 -".format(a=area))
    r = subprocess.run(['bash', '-lc', cmd], input=script,
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith('bsConstraint='):
            return line.split('=', 1)[1]
    raise SystemExit('FAILED %s %s:\n%s\n%s' % (area, modname, r.stdout, r.stderr))


def main():
    print('%-12s %-8s %-8s %s' % ('channel', 'dev3', 'dev2', 'wanted'))
    bad = 0
    for label, mod, _var, want in CFIS:
        old = run(DEV3, mod)
        new = run(DEV2, mod)
        ok = (new == str(want))
        bad += (not ok)
        print('%-12s %-8s %-8s %-8s %s' % (label, old, new, want, 'OK' if ok else '*** WRONG ***'))
    print('\n%d channel(s) wrong' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
