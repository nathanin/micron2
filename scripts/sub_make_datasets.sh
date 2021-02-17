#!/usr/bin/env bash

# Usage
#
# Read from a `datasets.txt` file with space-delimited arguments:
# datahome sample region
# the sample and region fields are used to generate the folder name
# underneath the datahome/ directory to use for input/output.
#
# Two ways to avoid over-writing or repeating jobs, marked below.

while read d s r; do 

  jobname=B_${s}_${r}

  # 1. if it looks like there's a log for the sample in the current directory, skip submitting
  if [[ -z `ls . | grep $jobname` ]]
  then
    qsub -N $jobname ./make_datasets.sh $d $s $r
  else
    echo Found job $jobname log. Skip submitting.
  fi

# 2. skips commented lines
done < <(grep -v ^\# datasets.txt)

