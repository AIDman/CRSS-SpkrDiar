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
------------ | -------------
IS1001a.Mix-Headset.der | 29.16 
IS1001b.Mix-Headset.der | 6.12  
IS1001c.Mix-Headset.der 10.88
IS1003b.Mix-Headset.der 28.81
IS1003d.Mix-Headset.der 40.32
IS1006b.Mix-Headset.der 6.57
IS1006d.Mix-Headset.der 48.16
IS1008a.Mix-Headset.der 21.68
IS1008b.Mix-Headset.der 2.47
IS1008c.Mix-Headset.der 37.64
IS1008d.Mix-Headset.der 32.27
Average: 22.84
