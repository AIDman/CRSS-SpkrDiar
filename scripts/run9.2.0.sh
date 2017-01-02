#!/bin/bash

# explore => Random Search
# consolidate => no
# constraint merge => yes
# post correctiom ==> no

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

bottom_up_clustering(){
    log_start "Bottom Up Clustering With Ivector"

    diar/segment_clustering_ivector_constraint.sh --nj 1 --apply-cmvn-utterance false --apply-cmvn-sliding false \
       --explore-mode "Random" --max-explore-pair-percentage 0.05 --merge-constraint true \
	--do-consolidate false --ivector-dist-stop 5.7 \
	exp/ref/$eval_data/segments exp/extractor_256 data/$eval_data exp/clustering_ivector_constraint/$eval_data

    log_end "Bottom Up Clustering With Ivector"
}
bottom_up_clustering

compute_der(){
	
    diar/compute_DER.sh --sanity_check false exp/ref/$eval_data/rttms exp/clustering_ivector_constraint/$eval_data/rttms exp/result_DER/$eval_data	
    grep OVERALL exp/result_DER/$eval_data/*.der && grep OVERALL exp/result_DER/$eval_data/*.der | awk '{ sum += $7; n++ } END { if (n > 0) print "Avergage: " sum / n; }'
}
compute_der

