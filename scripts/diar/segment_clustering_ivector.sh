#!/bin/bash

# Copyright     2013  Daniel Povey
#               2014  David Snyder
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 <extractor-dir> <data> <ivector-dir>"
  echo " e.g.: $0 exp/extractor_2048_male data/train_male exp/ivectors_male"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters|10>                          # Number of iterations of E-M"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --num-threads <n|8>                              # Number of threads for each process"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  exit 1;
fi

segdir=$1
extractor_dir=$2
data=$3
dir=$4


for f in $extractor_dir/final.ie $extractor_dir/final.ubm $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log $dir/segments $dir/rttms; rm -f $dir/segments/* $dir/rttms/*
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

delta_opts=`cat $extractor_dir/delta_opts 2>/dev/null`

## Set up features.
feats="ark,s,cs:copy-feats scp:$data/feats.scp ark:- | apply-cmvn --norm-vars=true scp:$data/cmvn.scp ark:- ark:- | add-deltas $delta_opts scp:$data/feats.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  dubm="fgmm-global-to-gmm $extractor_dir/final.ubm -|"

  $cmd JOB=1:$nj $dir/log/extract_posterior.JOB.log \
    gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
    fgmm-global-gselect-to-post --min-post=$min_post $extractor_dir/final.ubm "$feats" \
	ark,s,cs:- ark:- \| scale-post ark:- $posterior_scale ark,t:$dir/posterior.JOB || exit 1;

  $cmd JOB=1:$nj $dir/log/segment_clustering_ivector.JOB.log \
    segmentClusteringIvector --lambda=5 --lambda-ivec=15 $segdir/segments.scp "$feats" ark,s,cs:$dir/posterior.JOB $extractor_dir/final.ie \
	$dir/segments $dir/rttms|| exit 1;

fi


