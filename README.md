# CRSS Speaker Diarization Toolkit (CRSS-SpkrDiar)
### About
CRSS-SpkDiar is a C++ based speaker diarization toolkit, built on top of famous open source speech recognition platform of [Kaldi](http://kaldi.sourceforge.net/). The main objective of this toolkit is as follow:

  - Simple integration with Kaldi ASR, 
  - Simple intergration of i-vector training and extraction within Kaldi for Diarization
  - Simple intergration of DNN within Kaldi for Diarization 
  - Perform speaker diarization unsupervised/supervised/semisupervised fasion
  - Bentchmark on open database

Authors: [Chengzhu Yu](https://sites.google.com/site/chengzhuyu0/home) and [Navid Shokouhi](https://scholar.google.com/citations?user=DHxzPt8AAAAJ&hl=en) .

### Current Stage of Development
##### Completed:
 - Bottom-Up Clustering Using BIC, Symmetric KL divergence
 - Bottom-Up Clustering Using i-vector cosine distance score (CDS).
 - ILP Clustering with i-vector CDS as distance.

##### To Be Completed:
 - Test and incorporate PLDA, Conditional Bayes, Mahalanobis distance for clustering    
 - Sementation
 - Supervised/Semeisupervised Diarization
 - Generate flexible interface with Kaldi ASR

### Dependencies
  - Kaldi
  - GLPK (if only you want to try ILP)
  
### Benchmark Pefformance
We evaluate our performance on [AMI meeting corpus](http://groups.inf.ed.ac.uk/ami/download/) and comparing the numbers with those reported in [Pycasp](http://multimedia.icsi.berkeley.edu/scalable-big-data-analysis/pycasp/).
  
exp/result_DER/demo/IS1000a.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 10.03 percent `(ALL)
exp/result_DER/demo/IS1001a.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 29.16 percent `(ALL)
exp/result_DER/demo/IS1001b.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 6.12 percent `(ALL)
exp/result_DER/demo/IS1001c.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 10.88 percent `(ALL)
exp/result_DER/demo/IS1003b.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 28.81 percent `(ALL)
exp/result_DER/demo/IS1003d.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 40.32 percent `(ALL)
exp/result_DER/demo/IS1006b.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 6.57 percent `(ALL)
exp/result_DER/demo/IS1006d.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 48.16 percent `(ALL)
exp/result_DER/demo/IS1008a.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 21.68 percent `(ALL)
exp/result_DER/demo/IS1008b.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 2.47 percent `(ALL)
exp/result_DER/demo/IS1008c.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 37.64 percent `(ALL)
exp/result_DER/demo/IS1008d.Mix-Headset.der: OVERALL SPEAKER DIARIZATION ERROR = 32.27 percent `(ALL)
Avergage: 22.8425
