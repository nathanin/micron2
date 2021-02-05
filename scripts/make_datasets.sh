#!/usr/bin/env bash
#$ -V
#$ -cwd
#$ -j y
#$ -l mem_free=32G,h=csclprd3-c049.local

siffile=/home/ingn/sifs/micron-interactive.sif

hostname

TZ=America/Los_Angeles date

echo $@

if [[ -z `lscpu | grep avx` ]]
then 
  echo "avx instructions not found"
  echo $@ >> /home/ingn/devel/micron2/scripts/avx_related_fails.txt

  exit 1
fi

module load singularity/3.6.0

echo "Starting singularity"
singularity exec -B /common/ingn:/common $siffile bash ./set_up_micron_create_dataset.sh $@

