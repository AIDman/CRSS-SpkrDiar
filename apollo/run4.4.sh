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

apollo_corpus=/home/chengzhu/work/NASA/Apollo11_Diar_Corpus
eval_data="EECOM" # data for diarization
dev_data="all_apollo"  # dev_data is for UBM, TV matrix training for i-vector

prep_data(){
   local/prep_apollo_data.sh $apollo_corpus
}
#prep_data

run_mfcc(){
    log_start "Extract MFCC features"

    mfccdir=mfcc
    for x in $eval_data; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    done

    log_end "Extract MFCC features"
}
#run_mfcc 


run_vad(){
    log_start "Doing VAD"
    for x in $eval_data $dev_data; do
     vaddir=exp/vad/$x
     diar/compute_vad_decision.sh --nj 1 data/$x $vaddir/log $vaddir
    done
    log_end "Finish VAD"
}
#run_vad


make_ref(){
    log_start "Generate Reference Segments/Labels/RTTM files"

    local/rttm2segment.sh data/$eval_data exp/ref/$eval_data/segments

    log_end "Generate Reference Segments/Labels/RTTM files"
}
#make_ref 

train_extractor(){
    ubmdim=256
    ivdim=32

    sid/train_diag_ubm.sh --parallel-opts "" --nj 1 --apply-cmvn-utterance false --apply-cmvn-sliding false \
    	 		--cmd "$train_cmd" data/$dev_data ${ubmdim} exp/diag_ubm_${ubmdim} || exit 1;

    sid/train_full_ubm.sh --nj 1 --apply-cmvn-utterance false --apply-cmvn-sliding false \
			--cmd "$train_cmd" data/$dev_data exp/diag_ubm_${ubmdim} exp/full_ubm_${ubmdim} || exit 1;

    sid/train_ivector_extractor.sh --nj 1 --apply-cmvn-utterance false --apply-cmvn-sliding false \
      			--cmd "$train_cmd" --num-gselect 15 --ivector-dim $ivdim --num-iters 5 exp/full_ubm_${ubmdim}/final.ubm data/$dev_data \
      			exp/extractor_$ubmdim || exit 1;
}
#train_extractor

bottom_up_clustering(){
    log_start "Bottom Up Clustering With Ivector"

    diar/segment_clustering_ivector.sh --nj 1 --use-segment-label false --min-update-segment 60 --apply-cmvn-utterance false --apply-cmvn-sliding false \
       --ivector-dist-stop 0.5 exp/clustering/$eval_data/segments exp/extractor_256 data/$eval_data exp/clustering_ivector_1/$eval_data
    diar/segment_clustering_ivector.sh --nj 1 --use-segment-label true --apply-cmvn-utterance false --apply-cmvn-sliding false \
       --ivector-dist-stop 0.7 exp/clustering_ivector_1/$eval_data/segments exp/extractor_256 data/$eval_data exp/clustering_ivector_2/$eval_data

    log_end "Bottom Up Clustering With Ivector"
}
bottom_up_clustering

compute_der(){
	
    diar/compute_DER.sh --sanity_check false data/$eval_data exp/clustering_ivector_2/$eval_data/rttms exp/result_DER/$eval_data	
    grep OVERALL exp/result_DER/$eval_data/*.der && grep OVERALL exp/result_DER/$eval_data/*.der | awk '{ sum += $7; n++ } END { if (n > 0) print "Avergage: " sum / n; }'
}
compute_der
