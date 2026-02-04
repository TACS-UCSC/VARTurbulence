#!/bin/bash

# PBS Job Submission Script for DDPM Turbulence Reconstruction
# This script runs all permutations of setup files and configuration files

#PBS -N latentReg1
#PBS -A UCSC0009
#PBS -l walltime=10:00:00
#PBS -q main
#PBS -o /glade/derecho/scratch/llupinji/scripts/VARTurbulence/job_scripts/latentReg1.out
#PBS -e /glade/derecho/scratch/llupinji/scripts/VARTurbulence/job_scripts/latentReg1.err
#PBS -l select=1:ncpus=4:ngpus=1:mem=120GB

# Load necessary modules (adjust for your system)
module purge ## remove any default loaded modules 
mamba init ## initialize mamba, to load the environments
source ~/.bashrc ## loading my .bashrc file
mamba activate tacs2_latReg

# Set working directory
cd /glade/derecho/scratch/llupinji/scripts/VARTurbulence

python train_vqvae_well.py \
    --data_loc "/glade/derecho/scratch/llupinji/data/the_well_datasets/datasets/rayleigh_benard/data/test/rayleigh_benard_Rayleigh_1e6_Prandtl_1e-1.hdf5" \
    --epochs 10 \
    --batch_size 16 \