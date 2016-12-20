
all:

#KALDI_DIR = /home/nxs113020/kaldi-trunk
KALDI_DIR = /home/chengzhu/work/Toolhouse/kaldi-trunk

EXTRA_CXXFLAGS = -Wno-sign-compare -I${KALDI_DIR}/src -std=c++11
include ${KALDI_DIR}/src/kaldi.mk

BINFILES = label-to-segment glpk-to-rttm label-to-rttm segment-clustering segment-clustering-ivector ivector-test construct-ilp-problem \
           select-top-frames select-voiced-frames-v2 int-vector-to-segment compute-viterbi-vad-from-frame-likes segment-clustering-ivector-constraint \
           segment-clustering-post-correction-ivector segment-clustering-ivector-constraint-cluster-seed

OBJFILES = 

TESTFILES = 

ADDLIBS = ../diar/kaldi-diar.a ${KALDI_DIR}/src/ivector/kaldi-ivector.a  ${KALDI_DIR}/src/hmm/kaldi-hmm.a ${KALDI_DIR}/src/gmm/kaldi-gmm.a \
		  ${KALDI_DIR}/src/tree/kaldi-tree.a ${KALDI_DIR}/src/matrix/kaldi-matrix.a ${KALDI_DIR}/src/thread/kaldi-thread.a \
		  ${KALDI_DIR}/src/util/kaldi-util.a  ${KALDI_DIR}/src/base/kaldi-base.a


include ${KALDI_DIR}/src/makefiles/default_rules.mk
