#!/bin/bash
echo "NODE=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"
source $HOME/orcd/pool/zmass/engaging/setup_env_engaging.sh
python $HOME/orcd/pool/zmass/engaging/probe_cuda.py 2>&1 | grep -viE "oneDNN|absl::InitializeLog|cpu_feature_guard|To enable"
