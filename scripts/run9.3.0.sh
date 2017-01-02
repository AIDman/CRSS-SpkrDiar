#!/bin/bash

# explore => FFQS with preclustere seed
# consolidate => yes
# constraint mearge => yes
# post correctiom ==> yes, random selection


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

eval_data="12_meeting" # data for diarization

post_correction(){
    log_start "Post Correction"

    diar/segment_clustering_ivector_post_correction.sh --nj 1 --apply-cmvn-utterance false --apply-cmvn-sliding false \
	--mode "random" --prob-type "Gaussian" --max-check-pair-percentage 0.5 --nbest 3 --cluster-samples 10 exp/clustering_ivector_constraint/$eval_data/segments exp/ref/$eval_data/segments exp/extractor_256 data/$eval_data exp/clustering_ivector_post_correction/$eval_data	

    log_end "Post Correction"	
}
post_correction

compute_der(){
	
    diar/compute_DER.sh --sanity_check false exp/ref//$eval_data/rttms exp/clustering_ivector_post_correction/$eval_data/rttms exp/result_DER/$eval_data	
    grep OVERALL exp/result_DER/$eval_data/*.der && grep OVERALL exp/result_DER/$eval_data/*.der | awk '{ sum += $7; n++ } END { if (n > 0) print "Avergage: " sum / n; }'
}
compute_der

