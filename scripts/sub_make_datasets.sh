#!/usr/bin/env bash

# Dataseet, sample, region, hostname

while read d s r; do 
# Add parsing ? like check for commented lines

  jobname=B_${s}_${r}

  if [[ -z `ls . | grep $jobname` ]]
  then
    echo "qsub -N $jobname ./make_datasets.sh $d $s $r"
    qsub -N $jobname ./make_datasets.sh $d $s $r
  else
    echo Found job $jobname log. Skip submitting.
  fi

done < <(grep -v ^\# datasets.txt)

