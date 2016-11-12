#ifndef KALDI_SRC_DIAR_CLUSTERING_H_
#define KALDI_SRC_DIAR_CLUSTERING_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "diar-utils.h"
#include "segment.h"


namespace kaldi{

#define GLR_DISTANCE 0
#define KL2_DISTANCE 1
#define IVECTOR_DISTANCE 2

class IvectorInfo {
public:
	Matrix<BaseFloat>* feats_;
	Posterior* posteriors_;
	IvectorExtractor* extractor_;
	IvectorInfo(Matrix<BaseFloat>* feats, Posterior* posterior, IvectorExtractor* extractor);
};

class Cluster {
public:
	Cluster(Segment one_segment);
	void AddSegment(Segment new_segment);
	std::vector<Segment> AllSegments() const;
	std::string Label() const;
	int32 NumFrames() const;
	int32 NumFramesAfterMask() const;
	int32 NumSegments() const;
	Segment KthSegment(int32 k) const;
	BaseFloat LogDet(const Matrix<BaseFloat> &feats) const;
	void ComputeMean(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& feats_mean) const; 
	void ComputeCovDiag(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& feats_cov_diag) const;
	void ComputeSum(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& featus_sum) const; 
	void ComputeVarSum(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& var_sum) const;
	void CollectFeatures(const Matrix<BaseFloat>& feats, Matrix<BaseFloat>& feats_collect) const;
	void CollectPosteriors(const Posterior& posterior, Posterior& postprobs_collect) const;
	Vector<double> Ivector();
	void SetIvector(Vector<double>& ivec);
	void SetIvector(Vector<double>& ivec, SpMatrix<double>& ivec_covar);
	void SetIvector(IvectorInfo& ivec_info);

	Cluster* prev;
	Cluster* next;

	static int id_generator;
	static string prefix;

private:
	std::vector<Segment> all_segments_;
	std::string label_;
	int32 frames_;
	int32 frames_after_mask_;
	Vector<double> ivector_;
	SpMatrix<double> ivector_covar_;
};


class ClusterCollection {
public:
	ClusterCollection();
	string UttID();
	int32 NumFrames();
	int32 NumFramesAfterMask();
	void InitFromNonLabeledSegments(SegmentCollection non_clustered_segmemts);
	//InitFromLabeledSegments(SegmentCollection);
	void BottomUpClustering(const Matrix<BaseFloat> &feats, const BaseFloat& lambda = 5.0, int32 target_cluster_num = 0, const int32& dist_type = 0, const int32& min_update_len  = 0);
	void BottomUpClusteringIvector(IvectorInfo& ivec_info, 
									const BaseFloat& lambda = 5.0, 
									int32 target_cluster_num = 0, 
									const int32& min_update_len = 0);
	void SetIvector(IvectorInfo& ivec_info);

	void FindMinDistClusters(const Matrix<BaseFloat> &feats, 
							std::vector<std::vector<BaseFloat> >& dist_matrix, 
							std::vector<bool>& to_be_updated, 
							std::unordered_map<Cluster*, int32>& cluster_idx_map, 
							std::vector<Cluster*> &min_dist_clusters);

	void FindMinDistClustersIvector(std::vector<std::vector<BaseFloat> >& dist_matrix, 
									std::vector<bool>& to_be_updated, 
									std::unordered_map<Cluster*, int32>& cluster_idx_map, 
									std::vector<Cluster*> &min_dist_clusters);

	static void MergeClusters(Cluster* clust1, Cluster* clust2);
	BaseFloat DistanceOfTwoClustersGLR(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2);
	BaseFloat DistanceOfTwoClustersKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2); 
	BaseFloat DistanceOfTwoClustersIvectorKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2,
															const Posterior& posterior, const IvectorExtractor& extractor);
	void Write(const std::string& segment_dir);
	void WriteToRttm(const std::string& rttm_outputdir);
	Cluster* Head();

private:
	string uttid_;
	int32 num_clusters_;
	Cluster* head_cluster_;
	int32 dist_type_;
};



template <typename T>
std::string ToString(T val)
{
    std::stringstream stream;
    stream << val;
    return stream.str();
}


}

#endif
