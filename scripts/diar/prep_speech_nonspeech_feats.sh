#!/bin/bash 

# Copyright    2013  Daniel Povey
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Compute energy based VAD output 
# We do this in just one job; it's fast.
#

nj=1
cmd=run.pl
vad_energy_mean_scale_high=1.1
vad_energy_mean_scale_low=0.9

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

data=$1
dir=$2

mkdir -p $dir/speech $dir/nonspeech

echo "--vad-energy-mean-scale=$vad_energy_mean_scale_high" > $dir/vad_high.conf
echo "--vad-energy-mean-scale=$vad_energy_mean_scale_low" > $dir/vad_low.conf

compute-vad --config=$dir/vad_high.conf scp:$data/feats.scp ark,t,scp:$dir/vad_high.ark,$dir/vad_high.scp || exit 1;
compute-vad --config=$dir/vad_low.conf scp:$data/feats.scp ark,t,scp:$dir/vad_low.ark,$dir/vad_low.scp || exit 1;

copy-feats scp:$data/feats.scp ark:- | select-voiced-frames-v2 ark:- scp:$dir/vad_high.scp \
					ark,t,scp:$dir/speech/feats.ark,$dir/speech/feats.scp || exit 1;

copy-feats scp:$data/feats.scp ark:- | select-voiced-frames-v2 --select-unvoiced-frames=true ark:- scp:$dir/vad_low.scp \
					ark,t,scp:$dir/nonspeech/feats.ark,$dir/nonspeech/feats.scp || exit 1;

cp -f $data/wav.scp $data/utt2spk $data/spk2utt $dir/speech/
cp -f $data/wav.scp $data/utt2spk $data/spk2utt $dir/nonspeech/

