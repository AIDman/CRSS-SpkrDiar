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

eval_data="12_meeting"

run_mfcc(){
    log_start "Extract MFCC features"

    mfccdir=mfcc
    for x in $eval_data; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/$eval_data exp/make_mfcc/$x $mfccdir || exit 1;
    done

    log_end "Extract MFCC features"
}
run_mfcc


make_ref(){
    log_start "Generate Reference Segments/Labels/RTTM files"

    local/make_ami_ref.sh data/$eval_data $ami_annotated_segment exp/ref/$eval_data

    log_end "Generate Reference Segments/Labels/RTTM files"
}
make_ref 


bottom_up_clustering(){
    log_start "Bottom Up Clustering"

    diar/segment_clustering.sh --nj 1 --lambda 15 exp/ref/$eval_data/segments data/$eval_data exp/clustering/$eval_data
    
    log_end "Bottom Up Clustering"
}
bottom_up_clustering

compute_der(){
	
    diar/compute_DER.sh --sanity_check false exp/ref/$eval_data/rttms exp/clustering/$eval_data/rttms exp/result_DER/$eval_data	
    grep OVERALL exp/result_DER/$eval_data/*.der && grep OVERALL exp/result_DER/$eval_data/*.der | awk '{ sum += $7; n++ } END { if (n > 0) print "Avergage: " sum / n; }'
}
compute_der


