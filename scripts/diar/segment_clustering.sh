#!/bin/bash

# Copyright     2013  Daniel Povey
#               2014  David Snyder
# Apache 2.0.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0

min_update_segment=0
lambda=5.0
dist_type="KL2"
target_cluster_num=2

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

segdir=$1
data=$2
dir=$3

# Set various variables.
mkdir -p $dir/log $dir/segments $dir/rttms; rm -f $dir/segments/segments.scp $dir/rttms/rttms.scp
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

## Set up features.
feats="ark,s,cs:copy-feats scp:$data/feats.scp ark:- | apply-cmvn --norm-vars=true scp:$data/cmvn.scp ark:- ark:- | add-deltas --delta-order=1 ark:- ark:-|"

if [ $stage -le 0 ]; then

  $cmd JOB=1:$nj $dir/log/segment_clustering.JOB.log \
    segmentClustering --min-update-segment=$min_update_segment --target-cluster-num=$target_cluster_num \
		--lambda=$lambda --dist-type=$dist_type $segdir/segments.scp "$feats" $dir/segments $dir/rttms || exit 1;

fi


