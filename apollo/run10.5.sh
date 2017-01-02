#!/bin/bash

# explore => FFQS with preclustere seed
# consolidate => yes
# constraint mearge => yes
# post correctiom ==> yes, expected speaker error diffence


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
eval_data="GNC" # data for diarization
dev_data="all_apollo"  # dev_data is for UBM, TV matrix training for i-vector

post_correction(){
    log_start "Post Correction"

    diar/segment_clustering_ivector_post_correction.sh --nj 1 --apply-cmvn-utterance false --apply-cmvn-sliding false \
	--mode "uncertainty" --prob-type "Gaussian" --max-check-pair-percentage 0.5 --nbest 5 --cluster-samples 10 exp/clustering_ivector_constraint/$eval_data/segments exp/ref/$eval_data/segments exp/extractor_256 data/$eval_data exp/clustering_ivector_post_correction/$eval_data	

    log_end "Post Correction"	
}
post_correction

compute_der(){
	
    diar/compute_DER.sh --sanity_check false data/$eval_data exp/clustering_ivector_post_correction/$eval_data/rttms exp/result_DER/$eval_data	
    grep OVERALL exp/result_DER/$eval_data/*.der && grep OVERALL exp/result_DER/$eval_data/*.der | awk '{ sum += $7; n++ } END { if (n > 0) print "Avergage: " sum / n; }'
}
compute_der

