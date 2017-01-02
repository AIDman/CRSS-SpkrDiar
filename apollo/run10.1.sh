#!/bin/bash

# Baseline i-vector Bottom-UP Clustering (with oracle segmentation)

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

apollo_corpus=/home/chengzhu/work/NASA/Apollo11_Diar_Corpus_Longer_Each
eval_data="GNC" # data for diarization

prep_data(){
   local/prep_apollo_data.sh $apollo_corpus
}
prep_data

run_mfcc(){
    log_start "Extract MFCC features"

    mfccdir=mfcc
    for x in $eval_data; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    done

    log_end "Extract MFCC features"
}
run_mfcc 

make_ref(){
    log_start "Generate Reference Segments/Labels/RTTM files"

    local/rttm2segment.sh data/$eval_data exp/ref/$eval_data/segments

    log_end "Generate Reference Segments/Labels/RTTM files"
}
make_ref 

bottom_up_clustering(){
    log_start "Bottom Up Clustering With Ivector"

    diar/segment_clustering_ivector.sh --nj 1 --apply-cmvn-utterance false --apply-cmvn-sliding false \
       --ivector-dist-stop 0.7 exp/ref/$eval_data/segments exp/extractor_256 data/$eval_data exp/clustering_ivector/$eval_data

    log_end "Bottom Up Clustering With Ivector"
}
bottom_up_clustering


compute_der(){
	
    diar/compute_DER.sh --sanity_check false --column_forgive 0.25 data/$eval_data exp/clustering_ivector/$eval_data/rttms exp/result_DER/$eval_data	
    grep OVERALL exp/result_DER/$eval_data/*.der && grep OVERALL exp/result_DER/$eval_data/*.der | awk '{ sum += $7; n++ } END { if (n > 0) print "Avergage: " sum / n; }'
}
compute_der

