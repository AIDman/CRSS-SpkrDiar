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
	Cluster(Segment& one_segment);
	void AddSegment(const Segment& new_segment);
	std::vector<Segment> AllSegments();
	std::string Label();
	int32 NumFrames();
	Cluster* prev;
	Cluster* next;

	static int cluster_id_generator;
	static string cluster_prefix;

private:
	std::vector<Segment> list_;
	std::string label_;
	int32 frames_;
};

int Cluster::id_generator = 1;
std::string Cluster::prefix = "C";


class ClusterCollection {
public:
	ClusterCollection();
	void InitFromNonLabeledSegments(const SegmentCollection& non_clustered_segmemts);
	//InitFromLabeledSegments(SegmentCollection);
	void BottomUpClustering(const Matrix<BaseFloat> &feats, int32 target_cluster_num);
	void FindMinDistClusters(const vector<Cluster*> &min_dist_clusters);
	static void MergeClusters(const Matrix<BaseFloat> &feats, Cluster* clust1, Cluster* clust2);
	BaseFloat DistanceOfTwoClusters(const Matrix<BaseFloat> &feats, Cluster* cluster1, Cluster* cluster2);
	void WriteToSegments(const std::string& segment_dir);
	Cluster* Head();

private:
	int32 num_clusters_;
	Cluster* head_cluster_;
};





#endif
