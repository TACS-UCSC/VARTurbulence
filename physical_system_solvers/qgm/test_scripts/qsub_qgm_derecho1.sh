#!/bin/bash

#PBS -N qgm_derecho1
#PBS -A UCSC0009
#PBS -l walltime=10:00:00
#PBS -q main
#PBS -o /glade/derecho/scratch/llupinji/scripts/numerical_simulations/GFD-2023-lupin/test_scripts/qsub_qgm_derecho1.out
#PBS -e /glade/derecho/scratch/llupinji/scripts/numerical_simulations/GFD-2023-lupin/test_scripts/qsub_qgm_derecho1.err
#PBS -l select=1:ncpus=4:mem=120GB

# Load necessary modules (adjust for your system)
module purge
mamba init
source ~/.bashrc
mamba activate tacs2

# Set working directory
cd /glade/derecho/scratch/llupinji/scripts/numerical_simulations/GFD-2023-lupin/

# Run the original kdv_sims.py script
python ./moist_QG_channel_3d_derecho1.py -s $SEED -l cold -n none -dd /glade/derecho/scratch/llupinji/data/qgm_sim -nf 0.2 -tt 20000 --name test2 --dt .025