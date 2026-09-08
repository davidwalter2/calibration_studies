#!/bin/bash
# Stage the phase-3 inputs to Engaging so the ~36 GB card can be BUILT there.
# submit's /work quota cannot hold it (two builds died with Errno 122 partway
# through the HDF5 write); Engaging's pool has 869 GB free and the fit runs
# there anyway, so the card never crosses the wire.
set -uo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
D=engaging:orcd/pool/zmass
eng 'mkdir -p ~/orcd/pool/zmass/runs ~/orcd/pool/zmass/data ~/orcd/pool/zmass/zchannel/data'
rsync -a --info=progress2 "$GRP" $D/data/ &
rsync -a $Z/data/kern_loose_band3.3e-4.npz $Z/data/acc_loose_d8.json $D/zchannel/data/ &
rsync -a $FS/runs/quad_jpsiv2_ok.npz $FS/runs/quad_dyv2.npz $D/runs/ &
wait
# the two big ones in parallel: a single stream is per-stream limited near
# 50-90 MB/s (engaging/README.md sec. 3)
rsync -a $FS/runs/gpairs_v2_n50.npz   $D/runs/ &
rsync -a $FS/runs/gzpairs_dyv2_n50.npz $D/runs/ &
wait
eng 'ls -la ~/orcd/pool/zmass/runs ~/orcd/pool/zmass/data ~/orcd/pool/zmass/zchannel/data'
