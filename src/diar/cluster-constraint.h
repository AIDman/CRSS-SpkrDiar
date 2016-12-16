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

class ClusterCollectionConstrained : public ClusterCollection {
public:
	ClusterCollectionConstrained();
	ClusterCollectionConstrained(SegmentCollection* segments);

	void IvectorHacExploreFarthestFirstSearch(IvectorInfo& ivec_info, const DiarConfig& config, const int32& max_query);
	//void IvectorHacExploreFromClusterMedians();
	void InitClustersWithExploredClusters();
	void ConstrainedBottomUpClusteringIvector(IvectorInfo& ivec_info, const DiarConfig& config);

private:
	SegmentCollection* segments_;
	std::vector<std::vector<int32> > explored_clusters_;
};

}

#endif