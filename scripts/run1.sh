#!/bin/bash

log_start(){
  echo "#####################################################################"
  echo "Spawning *** $1 *** on" `date` `hostname`
  echo ---------------------------------------------------------------------
}

log_end(){
  echo ---------------------------------------------------------------------
  echo "Done *** $1 *** on" `date` `hostname`
  echo "#####################################################################"
}

. cmd.sh
. path.sh

set -e # exit on error


ami_annotated_segment=/home/chengzhu/work/SpeechCorpus/ami_dir/segments
session="demo"
#session="is_sessions_file_1"

run_mfcc(){
    log_start "Extract MFCC features"

    mfccdir=mfcc
    for x in $session; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/$session exp/make_mfcc/$x $mfccdir || exit 1;
    done

    log_end "Extract MFCC features"
}
#run_mfcc


make_ref(){
    log_start "Generate Reference Segments/Labels/RTTM files"

    local/make_ami_ref.sh data/$session $ami_annotated_segment exp/ref/$session

    log_end "Generate Reference Segments/Labels/RTTM files"
}
#make_ref 


bottom_up_clustering(){
    log_start "Bottom Up Clustering"

    diar/segment_clustering.sh --nj 1 --lambda 15 exp/ref/$session/segments data/$session exp/clustering/$session
    
    log_end "Bottom Up Clustering"
}
bottom_up_clustering

compute_der(){
	
    diar/compute_DER.sh --sanity_check false exp/ref/$session/rttms exp/clustering/$session/rttms exp/result_DER/$session	
    grep OVERALL exp/result_DER/$session/*.der && grep OVERALL exp/result_DER/$session/*.der | awk '{ sum += $7; n++ } END { if (n > 0) print "Avergage: " sum / n; }'
}
compute_der


