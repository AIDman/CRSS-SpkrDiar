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

#data="demo"
data="is_sessions_file_1"
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

    diar/segment_clustering.sh --nj 1  exp/ref/$x/segments data/$x exp/clustering/$x
    
    log_end "Bottom Up Clustering"
}
bottom_up_clustering $data

bottom_up_clustering_der(){
     		
    x=$1
	
    diar/compute_DER.sh --sanity_check false exp/ref/$x/rttms exp/clustering/$x/rttms exp/result_DER/$x	
    grep OVERALL exp/result_DER/$x/*.der		

}
bottom_up_clustering_der $data


