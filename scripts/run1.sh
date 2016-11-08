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

data="demo"
#data="is_sessions_file_1"
run_mfcc(){
    log_start "Extract MFCC features"

    mfccdir=mfcc
    datadir=$1
    for x in $datadir; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    done

    log_end "Extract MFCC features"
}
run_mfcc $data


make_ref(){
    log_start "Generate Reference Segments/Labels/RTTM files"

    ami_annotated_segment=/home/chengzhu/work/SpeechCorpus/ami_dir/segments

    x=$1
    local/make_ami_ref.sh data/$x $ami_annotated_segment exp/ref/$x

    log_end "Generate Reference Segments/Labels/RTTM files"
}
make_ref $data 


bottom_up_clustering(){
    log_start "Bottom Up Clustering"

    x=$1	

    mkdir -p exp/clustering/$x/segments exp/clustering/$x/rttms; rm -f exp/clustering/$x/segments/*; rm -f exp/clustering/$x/rttms/*
    feats="ark,s,cs:copy-feats scp:data/$x/feats.scp ark:- | apply-cmvn --norm-vars=true scp:data/$x/cmvn.scp ark:- ark:- | add-deltas --delta-order=1 ark:- ark:-|"	
    #segmentClustering --target_cluster_num=0 --lambda=15 --dist_type=KL2 exp/ref/$x/segments/segments.scp "$feats" exp/clustering/$x/segments exp/clustering/$x/rttms 2>&1 | tee log
    segmentClustering  --lambda=15 --dist_type=KL2 exp/ref/$x/segments/segments.scp "$feats" exp/clustering/$x/segments exp/clustering/$x/rttms 2>&1 | tee log
    
    log_end "Bottom Up Clustering"
}
bottom_up_clustering $data

bottom_up_clustering_der(){
     		
    x=$1
	
    diar/compute_DER.sh --sanity_check false exp/ref/$x/rttms exp/clustering/$x/rttms exp/result_DER/$x	
    grep OVERALL exp/result_DER/$x/*.der		

}
bottom_up_clustering_der $data


