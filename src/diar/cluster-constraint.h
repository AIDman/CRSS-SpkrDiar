#ifndef KALDI_SRC_DIAR_CLUSTER_CONSTRAINT_H_
#define KALDI_SRC_DIAR_CLUSTER_CONSTRAINT_H_

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
#include "cluster.h"

namespace kaldi{

class ClusterCollectionConstraint : public ClusterCollection {
public:
	ClusterCollectionConstraint();
	ClusterCollectionConstraint(SegmentCollection* segments);

	void IvectorHacExploreFarthestFirstSearch(IvectorInfo& ivec_info, const DiarConfig& config, const int32& max_query);
	void IvectorHacConsolidate(IvectorInfo& ivec_info, const DiarConfig& config, const int32& max_query_per_cluster);
	void InitClustersWithExploredClusters();
	void ConstraintBottomUpClusteringIvector(IvectorInfo& ivec_info, const DiarConfig& config);
	BaseFloat FindMinDistClustersIvectorConstraint(const DiarConfig& config, 
													std::vector<std::vector<BaseFloat>>& dist_matrix, 
													std::vector<bool>& to_be_updated, 
													std::unordered_map<Cluster*, int32>& cluster_idx_map, 
													std::vector<Cluster*> &min_dist_clusters);
	

private:
	SegmentCollection* segments_;
	std::vector<std::vector<int32> > explored_clusters_;
	std::vector<std::vector<BaseFloat>> dist_mat_;
};

template <typename T>
std::vector<size_t> ordered(std::vector<T> const& values) {
    std::vector<size_t> indices(values.size());
    std::iota(begin(indices), end(indices), static_cast<size_t>(0));

    std::sort(
        begin(indices), end(indices),
        [&](size_t a, size_t b) { return values[a] < values[b]; }
    );
    return indices;
}

}

#endif