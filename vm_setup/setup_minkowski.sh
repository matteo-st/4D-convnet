#!/bin/bash

# Run this script from the project root directory
#cd ~/4D-convnet

# Create and activate conda environment
#conda create -n py3-mink-conda-test python=3.8 -y
#source activate py3-mink-conda-test

# Install dependencies
#conda install openblas-devel -c anaconda -y
#conda install pytorch==2.1.1 torchvision==0.16.1 cudatoolkit=11.8 -c pytorch -c conda-forge -y
cd code
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --force_cuda
