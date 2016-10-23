#ifndef KALDI_CLUSTERING_H_
#define KALDI_CLUSTERING_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "hmm/posterior.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "ivector/plda.h"
#include "diar-utils.h"


namespace kaldi{

class Cluster {
public:
	vector<Segment> list_;
	string name_;
	int32 frames_;
};

class ClusterCollections {
public:
	vector<Cluster> list_;


};


#endif
