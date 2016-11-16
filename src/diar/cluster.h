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

struct IvectorInfo {
	Matrix<BaseFloat>* feats;
	Posterior* posteriors;
	IvectorExtractor* extractor;
	IvectorInfo(Matrix<BaseFloat>* _feats, Posterior* _posteriors, IvectorExtractor* _extractor) {
		feats = _feats;
		posteriors = _posteriors;
		extractor = _extractor;
		return;
	}
};

class Cluster {
public:
	Cluster(Segment one_segment);
	void AddSegment(Segment new_segment);
	std::vector<Segment> AllSegments() const;
	std::string Label() const;
	void SetLabel(const std::string& label);
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
	SpMatrix<double> IvectorCovar();
	void SetIvector(Vector<double>& ivec);
	void SetIvector(Vector<double>& ivec, SpMatrix<double>& ivec_covar);
	void SetIvector(IvectorInfo& ivec_info);
	void NormalizeIvector(Vector<double>& ivectors_average);

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
	void InitFromNonLabeledSegments(SegmentCollection& non_clustered_segmemts);
	void InitFromLabeledSegments(SegmentCollection& pre_clustered_segments);
	void BottomUpClustering(const Matrix<BaseFloat> &feats, const DiarConfig& config);
	void BottomUpClusteringIvector(IvectorInfo& ivec_info, const DiarConfig& config);
	void SetIvector(IvectorInfo& ivec_info);
	void ComputeIvectorMean(Vector<double>& ivectors_average);
	void NormalizeIvectors(Vector<double>& ivectors_average);

	void FindMinDistClusters(const Matrix<BaseFloat> &feats,
							const DiarConfig& config, 
							std::vector<std::vector<BaseFloat> >& dist_matrix, 
							std::vector<bool>& to_be_updated, 
							std::unordered_map<Cluster*, int32>& cluster_idx_map, 
							std::vector<Cluster*> &min_dist_clusters);

	BaseFloat FindMinDistClustersIvector(const DiarConfig& config,
									std::vector<std::vector<BaseFloat> >& dist_matrix, 
									std::vector<bool>& to_be_updated, 
									std::unordered_map<Cluster*, int32>& cluster_idx_map, 
									std::vector<Cluster*> &min_dist_clusters);

	static void MergeClusters(Cluster* clust1, Cluster* clust2);
	BaseFloat DistanceOfTwoClustersGLR(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2);
	BaseFloat DistanceOfTwoClustersKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2); 
	//BaseFloat DistanceOfTwoClustersIvectorKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2,
	//														const Posterior& posterior, const IvectorExtractor& extractor);
	void Write(const std::string& segment_dir);
	void WriteToRttm(const std::string& rttm_outputdir);
	Cluster* Head();

private:
	string uttid_;
	int32 num_clusters_;
	Cluster* head_cluster_;
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
