#!/bin/bash 

# Copyright    2013  Daniel Povey
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Compute energy based VAD output 
# We do this in just one job; it's fast.
#

cmd=run.pl

sanity_check=false # compute DER while give all segments a single label

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir> <path-to-vad-dir>";
   exit 1;
fi

ref_dir=$1
match_dir=$2
result_dir=$3


for f in $ref_dir/rttms.scp $match_dir/rttms.scp; do
  if [ ! -f $f ]; then
    echo "Generate rttms.scp file for either reference or matching does not exist"
    exit 1;
  fi
done

mkdir -p $result_dir/log; rm -rf $result_dir/*

while read ref <&3 && read match<&4; do
	#$sanity_check && cat $result_dir/match.rttm | awk '{$8=1;print $0}' > $result_dir/fake.rttm	
	#$sanity_check && perl local/md-eval-v21.pl -r $result_dir/ref.rttm -s $result_dir/fake.rttm 2>&1 | tee $result_dir/diar_err.fake  	

	name=`basename $ref .rttm`

	perl local/md-eval-v21.pl -r $ref -s $match > $result_dir/${name}.der 
 
done 3<$ref_dir/rttms.scp 4<$match_dir/rttms.scp
