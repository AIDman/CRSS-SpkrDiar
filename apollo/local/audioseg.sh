#!/bin/bash

lambda=20
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

data=$1
dir=$2

mkdir -p $dir/log $dir/segments
rm -f $dir/segments/segments.scp

for i in `awk '{print $4}' $data/wav.scp`
do
   bname=`basename $i .wav`
   sfbcep --mel --cms --normalize --delta --acceleration $i $dir/log/${bname}.mfcc
   ssad --min-length=1 --threshold=0.5 --all $i $dir/log/${bname}.sad
   sbic --segmentation=$dir/log/${bname}.sad --label=speech --lambda=$lambda $dir/log/${bname}.mfcc $dir/log/${bname}.seg #the lower the lamda, the more number of segment
   local/audioseg2seg.sh $dir/log/${bname}.seg $dir/segments/${bname}.seg
   echo $dir/segments/${bname}.seg >> $dir/segments/segments.scp
   echo $dir/segments/${bname}.seg "Num Segments: " `wc -l $dir/segments/${bname}.seg`
done	
