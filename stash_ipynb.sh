#!/usr/bin/env bash

output=notebook_stash/$( date --iso-8601='minutes' ).tar.gz
echo $output

# git ls-tree -r master | grep ipynb | cut -f 2

tar -zcvf $output `git ls-tree -r master | grep ipynb | cut -f 2`

