#ifndef KALDI_SRC_DIAR_CLUSTERING_H_
#define KALDI_SRC_DIAR_CLUSTERING_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "diar-utils.h"


namespace kaldi{

class Cluster {
public:
	static int cluster_id_generator;
	vector<Segment> list_;
	string name_;
	int32 frames_;

	Cluster(Segment&);
};

int Cluster::cluster_id_generator = 1;

class ClusterCollection {
public:
	ClusterCollection();
	InitFromNonLabeledSegments(const SegmentCollection& NonClusteredSegmemts);
	InitFromLabeledSegments(SegmentCollection);
	BottomUpClustering(const Matrix<BaseFloat> &feats);
	DistanceOfTwoClusters(Cluster& cluster1, Cluster& cluster2);
	WriteToSegments(const std::string& segment_dir);

public:
	vector<Cluster> list_;
	Cluster& head_cluster_;
};


void ClusterCollection::InitFromNonLabeledSegments(const SegmentCollection& NonClusteredSegmemts) {
	 std::int32 num_segments = NonClusteredSegments.Size();
	 if(num_segments < 1) KALDI_ERR << "Clusters could not be initialized from empty segments";
	 Cluster& head_cluster_ = new Cluster(Segment&);	
}


#endif
