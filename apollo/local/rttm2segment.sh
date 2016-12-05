#! /bin/bash

rttm_dir=$1 # kaldi format data directory
segment_dir=$2

mkdir -p $segment_dir; rm -f $segment_dir/segments.scp

while read -r rttm; do
    name=`basename $rttm .rttm`
    awk '{print $2 " " $4 " " $4+$5 " " $8}' $rttm > $segment_dir/${name}.seg
    echo $segment_dir/${name}.seg >> $segment_dir/segments.scp
done < $rttm_dir/rttms.scp

