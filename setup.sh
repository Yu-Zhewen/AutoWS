#!/bin/bash

# setup conda environment
conda create -n fpgaconvnet-autows python=3.10
conda activate fpgaconvnet-autows
pip install -r requirements.txt

# checkout fpgaconvnet repo
git clone https://github.com/Yu-Zhewen/fpgaconvnet-torch
cd fpgaconvnet-torch
git checkout 33a97cf38179058231a80b1f2865f263f8468a2a
cd ..

git clone https://github.com/AlexMontgomerie/fpgaconvnet-optimiser
cd fpgaconvnet-optimiser
git checkout 17307e9b27b873ff80512e906082a5d5132596e5
git submodule update --init --recursive
cd ../

WORK_DIR=$(pwd)
export FPGACONVNET_TORCH=${WORK_DIR}/fpgaconvnet-torch
export FPGACONVNET_OPTIMISER=${WORK_DIR}/fpgaconvnet-optimiser
export FPGACONVNET_MODEL=${WORK_DIR}/fpgaconvnet-optimiser/fpgaconvnet-model
export PYTHONPATH=$PYTHONPATH:$FPGACONVNET_TORCH:$FPGACONVNET_OPTIMISER:$FPGACONVNET_MODEL