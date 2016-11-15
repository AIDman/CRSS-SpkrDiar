#!/bin/bash

# Copyright     2013  Daniel Povey
#               2014  David Snyder
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

# Begin configuration section.
nj=1
cmd="run.pl"
use_segment_label=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Iput arguments number mismatch"
  exit 1;
fi

segdir=$1
dir=$2

# Set various variables.
mkdir -p $dir/log $dir/rttms $dir/sols; rm -f $dir/rttms/rttms.scp $dir/sols/sols.scp 

# Perform ILP clustering using GLPK tool
while read -r ilp_problem; do
     name=`basename $ilp_problem .ilp`	
     glpsol --tmlim 180 --lp $ilp_problem -o $dir/sols/${name}.sol > $dir/log/${name}.ilp.sol.log
     echo "$dir/sols/${name}.sol" >> $dir/sols/sols.scp	
done < $dir/ilps/ilp.scp

# Interperate ILP clustering result in GLPK format into rttm for compute DER infuture
glpk_to_rttm --use-segment-label=$use_segment_label $dir/sols/sols.scp $segdir/segments.scp $dir/rttms

