#!/usr/bin/env bash

# Dataseet, sample, region, hostname

while read d s r h; do 
# Add parsing ? like check for commented lines

  echo "qsub -l h=${h} -N B_${s}_${r} ./make_datasets.sh $d $s $r"
  qsub -l h=${h} -N B_${s}_${r} ./make_datasets.sh $d $s $r

done < <(grep -v ^\# datasets.txt)

