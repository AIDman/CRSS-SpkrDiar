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

##### To Do:
 - Test and incorporate PLDA, Conditional Bayes, Mahalanobis distance for clustering    
 - Sementation
 - Supervised/Semeisupervised Diarization
 - Generate flexible interface with Kaldi ASR

### Dependencies
  - Kaldi
  - GLPK
  - Python
  
### Installation
