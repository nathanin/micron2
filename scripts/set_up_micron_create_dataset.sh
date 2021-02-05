#!/usr/bin/env bash

# write this as if we're inside a singularity image

source /usr/local/miniconda3/bin/activate micron2

which python
# cd /home/ingn/devel/micron2
# pip install -e .

cd /home/ingn/devel/micron2/scripts

echo ""
python ./script_create_dataset.py --help

echo ""
echo "running:"
echo "python ./script_create_dataset.py -d $1 -s $2 -r $3"
python ./script_create_dataset.py -d $1 -s $2 -r $3

