#!/bin/bash

aseg=$1  # segments from audioseg
seg=$2   # segments in our format

uttid=`basename $aseg .seg`

cat $aseg | grep -v sil | awk -v var=$uttid '{print var " " $2 " " $3 " unk"}' > $seg
