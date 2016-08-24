Speaker diarization project built using Kaldi libraries. 
The system is primarily based on integer linear programming clustering. 
Authors: Navid Shokouhi, Chengzhu Yu

Uses integer linear programming to cluster i-vectors extracted on 
audio segments. Segments are obtained from Bayesian information 
criterion (BIC) segmentation.


Software prerequisites:
GLP toolkit: https://www.gnu.org/software/glpk/
kaldi: https://sourceforge.net/projects/kaldi/
python numpy



Instructions: 
modify KALDI_ROOT in all Makefiles:
    src/diar/Makefile
    src/diarbin/Makefile

in src directory run:
    make

in scripts/path.sh set KALDI_ROOT variable


# Demo:
To run deme script cd to scripts/
Experiments run on the AMI meeting corpus:
$ bash test.sh


# Content:
.
├── readme.txt
├── scripts
│   ├── test.sh (main bash script)
│   ├── cmd.sh
│   ├── path.sh (Kaldi path file)
│   ├── diar/ (diarization scripts: compute error rates, generate optimization scripts, etc.)
│   ├── local/ (local bash and python scripts used to run demo)
│   ├── conf/ (Kaldi config files)
│   ├── data
│   │   ├── dev/ (data files for background data (only utt2spk is used here))
│   │   └── toy/ (data for demo experiment)
│   ├── exp/
│   │   ├── dev.iv (background i-vectors used to train PLDA and background covariances)
│   │   └── extractor_1024 (i-vector extraction models)
│   ├── sid/ (speaker id scripts used for i-vector extraction)
│   ├── steps/ (Kaldi scripts)
│   └── utils/ (Kaldi scripts)
└── src/ (C++ diarization source code)
    ├── diar/
    │   ├── bic.cc
    │   ├── bic.h
    │   ├── diar-utils.cc
    │   ├── diar-utils.h
    │   ├── ilp.cc
    │   ├── ilp.h
    │   └── Makefile
    ├── diarbin/
    │   ├── changeDetectBIC.cc
    │   ├── glpkToRTTM.cc
    │   ├── ivectorTest.cc
    │   ├── labelToRTTM.cc
    │   ├── labelToSegment.cc
    │   ├── Makefile
    │   ├── segIvectorExtract.cc
    │   └── writeTemplateILP.cc
    └── Makefile
