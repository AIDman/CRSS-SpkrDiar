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

eval_data="test1" # data for diarization
dev_data="ami_full"  # dev_data is for UBM, TV matrix training for i-vector

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
    for x in $dev_data; do
     vaddir=exp/vad/$x
     diar/compute_vad_decision.sh --nj 1 data/$x $vaddir/log $vaddir
    done
    log_end "Finish VAD"
}
#run_vad


make_ref(){
    log_start "Generate Reference Segments/Labels/RTTM files"

    ami_annotated_segment=/home/chengzhu/work/SpeechCorpus/ami_dir/segments

    local/make_ami_ref.sh data/$eval_data $ami_annotated_segment exp/ref/$eval_data

    log_end "Generate Reference Segments/Labels/RTTM files"
}
#make_ref 

train_extractor(){
    ubmdim=256
    ivdim=32

    sid/train_diag_ubm.sh --parallel-opts "" --nj 1 --cmd "$train_cmd" data/$dev_data ${ubmdim} \
    exp/diag_ubm_${ubmdim} || exit 1;

    sid/train_full_ubm.sh --nj 1 --cmd "$train_cmd" data/$dev_data \
       exp/diag_ubm_${ubmdim} exp/full_ubm_${ubmdim} || exit 1;

    sid/train_ivector_extractor.sh --nj 1 --cmd "$train_cmd" --num-gselect 15 \
      --ivector-dim $ivdim --num-iters 5 exp/full_ubm_${ubmdim}/final.ubm data/$dev_data \
      exp/extractor_$ubmdim || exit 1;
}
#train_extractor

test_ivectors(){

    diar/test_ivector_score.sh --nj 1 --apply-cmvn-utterance false --apply-cmvn-utterance false \
	exp/extractor_256 data/$eval_data exp/ref/$eval_data/labels exp/temp/test_ivectors 
}
test_ivectors

