# CRSS Speaker Diarization Toolkit (CRSS-SpkrDiar)
## About
CRSS-SpkDiar is a C++ based speaker diarization toolkit, built on top of famous open source speech recognition platform of [Kaldi](http://kaldi.sourceforge.net/). The main objectives of this toolkit are:

  - Simple integration with Kaldi ASR, 
  - Simple intergration of i-vector modules within Kaldi for Diarization,
  - Simple intergration of DNN modules within Kaldi for Diarization,
  - Perform speaker diarization unsupervised/supervised/semi-supervised fashion,
  - Benchmark on open database (AMI meeting corpus, and Apollo-MCC corpus).

Authors: [Chengzhu Yu](https://sites.google.com/site/chengzhuyu0/home) and [Navid Shokouhi](https://scholar.google.com/citations?user=DHxzPt8AAAAJ&hl=en).


## Current Stage of Development
##### _Completed:_
 - VAD (GMM based)
 - BIC segmentation (optional)
 - Bottom-Up Clustering
   - BIC distance 
   - KL divergence
   - i-vector cosine distance
   - i-vector Mahalanobis 
   - i-vector PLDA (optional)
 - Bottom-Up Clustering Using i-vector cosine distance score (CDS)
 - Interger linear programming (ILP) Clustering

##### _To Be Completed:_
 - VAD based segmentation (viterbi)
 - Resementation
 - Evaluations
 
##### _Furthur Extensions:_
 - DNN speaker embedding features 
 - Interface with Kaldi ASR

## Dependencies
  - Kaldi
  - GLPK (if only you want to try ILP)
  
## Benchmark Performance
We evaluate our performance on [AMI meeting corpus](http://groups.inf.ed.ac.uk/ami/download/) and compare the numbers with those reported in [Pycasp](http://multimedia.icsi.berkeley.edu/scalable-big-data-analysis/pycasp/) from ICSI. Note: To evaluate only the clustering module, the numbers on CRSS-SpkDiar is on top of oracle segmentation. We're currently working to include segmentation.

| Session       |      Pycasp   |   CRSS-SpkDiar (run2.sh) |
| ------------- | ------------- | -------------   | 
IS1000a.Mix-Headset | 25.38 | 12.07|
IS1001a.Mix-Headset | 32.34 | 43.64|
IS1001b.Mix-Headset | 10.57 | 12.16 |
IS1001c.Mix-Headset | 28.40 | 6.17|
IS1003b.Mix-Headset | 34.30 | 10.56|
IS1003d.Mix-Headset | 50.75 | 24.67|
IS1006b.Mix-Headset | 16.57 | 7.34 |
IS1006d.Mix-Headset | 53.05 | 21.56 |
IS1008a.Mix-Headset | 1.65  | 4.07 |
IS1008b.Mix-Headset | 8.58  | 3.60  |
IS1008c.Mix-Headset | 9.30  | 6.36 | 
IS1008d.Mix-Headset | 26.27 | 5.99 | 
Average             | 24.76% |**13.19%** | 
