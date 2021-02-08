#!/usr/bin/env bash
#$ -V
#$ -cwd
#$ -j y
#$ -l mem_free=32G

siffile=/home/ingn/sifs/micron-interactive.sif

hostname

TZ=America/Los_Angeles date

echo $@

# Make sure the host we're on has AVX and AVX2 available
# TensorFlow binaries are compiled with AVX on and segfault out if the CPU doesn't have those instructions
if [[ -z `lscpu | grep avx` ]]
then 
  echo "avx instructions not found"
  echo $@ >> /home/ingn/devel/micron2/scripts/avx_related_fails.txt

  exit 1
fi

module load singularity/3.6.0

echo "Starting singularity"
singularity exec -B /common/ingn:/common $siffile bash ./set_up_micron_create_dataset.sh $@

