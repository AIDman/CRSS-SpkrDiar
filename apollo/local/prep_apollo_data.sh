#!/bin/bash

corpus=$1

subdirs=`find $corpus -maxdepth 1 -mindepth 1 -type d`

mkdir -p data/all_apollo
rm -f data/all_apollo/wav.scp data/all_apollo/spk2utt data/all_apollo/utt2spk data/all_apollo/rttms.scp
for i in $subdirs;do
    channel_name=`basename $i`	
    mkdir -p data/$channel_name
    ls $i/*.wav > data/$channel_name/wav.list
    cat data/$channel_name/wav.list | awk -F'/' '{print $NF}' | sed s/.wav//g | awk '{print $1 " " $1}' > data/$channel_name/spk2utt	   	
    cat data/$channel_name/wav.list | awk -F'/' '{print $NF}' | sed s/.wav//g | awk '{print $1 " " $1}' > data/$channel_name/utt2spk
    cat data/$channel_name/wav.list | awk -F'/' '{print $NF}' | sed s/.wav//g > data/$channel_name/uttid
    cat data/$channel_name/wav.list | awk '{print "sox --ignore-length " $0 " -t wav - |"}' > data/$channel_name/wav.sox 	
    paste -d " " data/$channel_name/uttid data/$channel_name/wav.sox > data/$channel_name/wav.scp 	   	
    ls $i/*.rttm > data/$channel_name/rttms.scp
    rm data/$channel_name/wav.list data/$channel_name/uttid data/$channel_name/wav.sox

    cat data/$channel_name/wav.scp >> data/all_apollo/wav.scp	
    cat data/$channel_name/spk2utt >> data/all_apollo/spk2utt	
    cat data/$channel_name/utt2spk >> data/all_apollo/utt2spk	
done

mv data/all_apollo/wav.scp data/all_apollo/wav.scp.tmp
mv data/all_apollo/spk2utt data/all_apollo/spk2utt.tmp
mv data/all_apollo/utt2spk data/all_apollo/utt2spk.tmp

cat data/all_apollo/wav.scp.tmp | sort > data/all_apollo/wav.scp
cat data/all_apollo/spk2utt.tmp | sort > data/all_apollo/spk2utt
cat data/all_apollo/utt2spk.tmp | sort > data/all_apollo/utt2spk

