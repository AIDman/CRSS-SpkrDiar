#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e 
set -o pipefail

# Begin configuration section.
cmd=run.pl
nj=4
speech_num_gauss=4
sil_num_gauss=96
num_iters=10
impr_thres=0.002
stage=-10
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

data=$1
data_speech=$2
data_silence=$3
dir=$4

function build_0gram {
wordlist=$1; lm=$2
echo "=== Building zerogram $lm from ${wordlist}. ..."
awk '{print $1}' $wordlist | sort -u > $lm
python -c """
import math
with open('$lm', 'r+') as f:
 lines = f.readlines()
 p = math.log10(1/float(len(lines)));
 lines = ['%f\\t%s'%(p,l) for l in lines]
 f.seek(0); f.write('\\n\\\\data\\\\\\nngram  1=       %d\\n\\n\\\\1-grams:\\n' % len(lines))
 f.write(''.join(lines) + '\\\\end\\\\')
"""
}

for f in $data/feats.scp $data/vad.scp; do
  [ ! -s $f ] && echo "$0: could not find $f or $f is empty" && exit 1
done 

#feat_dim=`feat-to-dim "scp:head -n 1 $data/feats.scp |" ark,t:- | awk '{print $2}'` || exit 1
feat_dim=39

# Prepare a lang directory
if [ $stage -le -2 ]; then
  mkdir -p $dir/local
  mkdir -p $dir/local/dict
  mkdir -p $dir/local/lm

  echo "1" > $dir/local/dict/silence_phones.txt
  echo "1" > $dir/local/dict/optional_silence.txt
  echo "2" > $dir/local/dict/nonsilence_phones.txt
  echo -e "1 1\n2 2" > $dir/local/dict/lexicon.txt
  echo -e "1\n2\n1 2" > $dir/local/dict/extra_questions.txt

  mkdir -p $dir/lang
  diar/prepare_vad_lang.sh --num-sil-states 1 --num-nonsil-states 1 \
    $dir/local/dict $dir/local/lang $dir/lang || exit 1
  fstisstochastic $dir/lang/G.fst  || echo "[info]: G not stochastic."
  diar/prepare_vad_lang.sh --num-sil-states 30 --num-nonsil-states 75 \
    $dir/local/dict $dir/local/lang $dir/lang_test || exit 1
fi

if [ $stage -le -1 ]; then 
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang_test/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans_test.mdl || exit 1
  
  diar/make_vad_graph.sh --iter trans $dir/lang $dir $dir/graph || exit 1
  diar/make_vad_graph.sh --iter trans_test $dir/lang_test $dir $dir/graph_test || exit 1
fi

utils/split_data.sh $data $nj || exit 1
#feats="ark:copy-feats scp:$data/feats.scp ark:- | "
#feats_speech="ark:copy-feats scp:$data_speech/feats.scp ark:- |"
#feats_silence="ark:copy-feats scp:$data_silence/feats.scp ark:- |"

feats="ark,s,cs:add-deltas --delta-window=3 --delta-order=2 scp:$data/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
feats_speech="ark,s,cs:add-deltas --delta-window=3 --delta-order=2 scp:$data_speech/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
feats_silence="ark,s,cs:add-deltas --delta-window=3 --delta-order=2 scp:$data_silence/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"

if [ $stage -le 0 ]; then

    $cmd $dir/log/init_gmm_speech.log \
      gmm-global-init-from-feats --num-gauss=$speech_num_gauss --num-iters=10 \
      "$feats_speech" \
      $dir/speech.0.mdl || exit 1
    $cmd $dir/log/init_gmm_silence.log \
      gmm-global-init-from-feats --num-gauss=$sil_num_gauss --num-iters=6 \
      "$feats_silence" \
      $dir/silence.0.mdl || exit 1

  {
    cat $dir/trans.mdl
    echo "<DIMENSION> $feat_dim <NUMPDFS> 2"
    gmm-global-copy --binary=false $dir/silence.0.mdl -
    gmm-global-copy --binary=false $dir/speech.0.mdl -
  } > $dir/0.mdl || exit 1

  x=0
  while [ $x -lt $num_iters ]; do
    $cmd $dir/log/decode.$x.log \
      gmm-decode-simple \
      --allow-partial=true --word-symbol-table=$dir/graph/words.txt \
      $dir/$x.mdl $dir/graph/HCLG.fst \
      "$feats" ark:/dev/null ark:$dir/$x.ali || exit 1

    $cmd $dir/log/update.$x.log \
      gmm-acc-stats-ali \
      $dir/$x.mdl "$feats" \
      ark:$dir/$x.ali - \| \
      gmm-est $dir/$x.mdl - $dir/$[x+1].mdl || exit 1

    objf_impr=$(cat $dir/log/update.$x.log | grep "GMM update: Overall .* objective function" | perl -pe 's/.*GMM update: Overall (\S+) objective function .*/\$1/')

    if [ "$(perl -e "if ($objf_impr < $impr_thres) { print true; }")" == true ]; then
      break;
    fi

    x=$[x+1]
  done

  rm -f $dir/final.mdl 2>/dev/null || true
  
  gmm-copy --binary=false $dir/${x}.mdl $dir/${x}.txt.mdl 
  copy-transition-model --binary=false $dir/trans_test.mdl $dir/final.mdl
  grep -A 100000 "DIMENSION" $dir/${x}.txt.mdl >> $dir/final.mdl   

  $cmd $dir/log/decode.final.log \
    gmm-decode-simple \
    --acoustic-scale=0.1 --allow-partial=true --word-symbol-table=$dir/graph_test/words.txt \
    $dir/final.mdl $dir/graph_test/HCLG.fst \
    "$feats" ark:/dev/null ark:$dir/final.ali || exit 1
fi

if $cleanup; then
  for x in `seq $[num_iters - 1]`; do
    if [ $[x % 10] -ne 0 ]; then
      rm $dir/$x.mdl
    fi
  done
fi

gzip $dir/final.ali

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

