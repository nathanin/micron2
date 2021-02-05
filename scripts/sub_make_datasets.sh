#!/usr/bin/env bash

# Dataseet, sample, region, hostname

while read d s r h; do qsub -N B_${s}_${r} ./make_datasets.sh $d $s $r $h; done < datasets.txt

