#include <iomanip>
#include <cfloat>
#include "cluster.h"
#include "segment.h"
#include "cluster-constraint.h"

namespace kaldi {

ClusterCollectionConstraint::ClusterCollectionConstraint(SegmentCollection* segments) {
	ClusterCollection();
	this->segments_ = segments;
	this->uttid_ = segments->UttID();
}

void ClusterCollectionConstraint::IvectorHacExploreFarthestFirstSearch(IvectorInfo& ivec_info, const DiarConfig& config, const int32& max_query) {
	int32 nsegs = segments_->Size();

	// Compute and assign i-vectors for each segments
	std::vector< Vector<double> > ivector_collect;
    for (size_t k = 0; k < nsegs; k++) {
	    Segment* kth_segment = this->segments_->KthSegment(k);
    	kth_segment->SetIvector(*ivec_info.feats, 
                            *ivec_info.posteriors, 
                            *ivec_info.extractor);

    	ivector_collect.push_back(kth_segment->Ivector());
    }

    // compute mean of ivectors
    Vector<double> total_mean;
    ComputeMean(ivector_collect, total_mean);

    // Apply mean & length normalization to i-vector of segments
    for (size_t k = 0; k<nsegs; k++) {
        // mean normalization
        Segment* kth_segment = this->segments_->KthSegment(k);
        Vector<double> kth_ivector = kth_segment->Ivector();
        SpMatrix<double> kth_ivector_covar = kth_segment->IvectorCovar();
        kth_ivector.AddVec(-1, total_mean);

        // length normalization
        BaseFloat norm = kth_ivector.Norm(2.0);
        BaseFloat ratio = norm / sqrt(kth_ivector.Dim()); // how much larger it is
        kth_ivector.Scale(1.0 / ratio);
        kth_ivector_covar.Scale(1.0 / ratio);

        kth_segment->SetIvector(kth_ivector);
    }

    // Compute distance matrix for segments 
	std::vector<std::vector<BaseFloat> > dist_mat(nsegs, std::vector<BaseFloat>(nsegs, -10000.0));
	for(size_t i = 0; i < nsegs; i++) {
		 Segment* ith_segment = this->segments_->KthSegment(i);
         Vector<double> i_ivector = ith_segment->Ivector();
		for(size_t j = i + 1; j < nsegs; j++) {
			Segment* jth_segment = this->segments_->KthSegment(j);
         	Vector<double> j_ivector = jth_segment->Ivector();
			
			dist_mat[i][j] =  1 - CosineDistance(i_ivector, j_ivector);
            dist_mat[j][i] = dist_mat[i][j];
		} 
	}

    // Exploring Stage: 
    
    srand(time(NULL));
    int32 seed_seg_id = rand() % nsegs;
    std::vector<int32> seed_cluster;
    seed_cluster.push_back(seed_seg_id);
    this->explored_clusters_.push_back(seed_cluster);

    int32 query_count=0, farthest_seg_id=0;
    while(query_count <= max_query) {
    	std::string farthest_seg_label;
        BaseFloat farthest_dist = -10000.0;
        for(size_t i = 0; i < nsegs; i++) {
            BaseFloat dist = 0.0;
            for(size_t c = 0; c < this->explored_clusters_.size(); c++) {
                for(size_t s = 0; s < this->explored_clusters_[c].size(); s++) {
                    if(i == this->explored_clusters_[c][s]){
                        continue;
                    }
                    dist += dist_mat[i][this->explored_clusters_[c][s]];
                }
            }

            if(dist > farthest_dist) {
                farthest_dist = dist;
                farthest_seg_id = i;
            }
            farthest_seg_label = this->segments_->KthSegment(farthest_seg_id)->Label();
        }

        bool match_existing_cluster = false;
        for(size_t c = 0; c < this->explored_clusters_.size(); c++) {
            query_count++;
            std::string cth_cluster_label = this->segments_->KthSegment(this->explored_clusters_[c][0])->Label();                
            if(farthest_seg_label == cth_cluster_label) {
            	KALDI_LOG << "Belongs To Existing Cluster : " << farthest_seg_label << "Query Count " << query_count;
                this->explored_clusters_[c].push_back(farthest_seg_id);
                match_existing_cluster = true;
                break;
            }
        }

        if(!match_existing_cluster) {
        	KALDI_LOG << "Belongs To New Cluster : " << farthest_seg_label  << "Query Count " << query_count;
            std::vector<int32> new_cluster;
            new_cluster.push_back(farthest_seg_id);
            this->explored_clusters_.push_back(new_cluster);
        }        
    }

    KALDI_LOG << "Finished Farthest First Search Exploring.";
    KALDI_LOG << "A Total of " << this->explored_clusters_.size() << " clusters are explored:";
    for(int k = 0; k < this->explored_clusters_.size(); k++) {
    	KALDI_LOG << "Cluster" << k << "found " << this->explored_clusters_[k].size() << " segments";
    }

    return;
}

void ClusterCollectionConstraint::IvectorHacConsolidate(IvectorInfo& ivec_info, const DiarConfig& config, const int32& max_query_per_cluster) {
	
}

void ClusterCollectionConstraint::InitClustersWithExploredClusters() {
 	KALDI_ASSERT(explored_clusters_.size() > 0);
 	std::unordered_map<int32, int32> clustered_segments;
	Cluster* prev_cluster = NULL;
	Cluster* new_cluster = NULL; 

 	for(size_t i = 0; i < this->explored_clusters_.size(); i++) {
 		for(size_t j = 0; j < this->explored_clusters_[i].size(); j++) {
 			int32 segment_idx = this->explored_clusters_[i][j];
 			clustered_segments[segment_idx] = 1;
 			if(i == 0 && j == 0) {
 				Cluster* head_cluster = new Cluster(*(this->segments_->KthSegment(segment_idx)));
 				head_cluster->SetMergeable(false);
	 			head_cluster->prev = NULL;
		 		prev_cluster = head_cluster;
		 		this->num_clusters_++;
 				this->head_cluster_ = head_cluster;
 				continue;
 			}

 			if(j == 0) {
 				new_cluster = new Cluster(*(this->segments_->KthSegment(segment_idx)));
		 		this->num_clusters_++;
 				new_cluster->SetMergeable(false);
 				prev_cluster->next = new_cluster;
 				new_cluster->prev = prev_cluster;
 				prev_cluster = new_cluster;
 			} else{
 				prev_cluster->AddSegment(*(this->segments_->KthSegment(segment_idx)));
 			}
 		}
 	}

 	for(size_t i = 0; i<this->segments_->Size(); i++) {
 		if(clustered_segments.find(i) == clustered_segments.end()) {
 			new_cluster = new Cluster(*(this->segments_->KthSegment(i)));
		 	this->num_clusters_++;
 			prev_cluster->next = new_cluster;
 			new_cluster->prev = prev_cluster;
 			prev_cluster = new_cluster;
 		} 
 	}

 	new_cluster->next=NULL;

 	Cluster* curr = this->head_cluster_;
 	int32 ct = 0;
 	KALDI_LOG << "Total Cluster Number: " << this->num_clusters_;
 	while(curr) {
 		KALDI_LOG<< "**Cluster " << ct;
 		ct++;
 		curr = curr->next;
 	}

 	return;
}

void ClusterCollectionConstraint::ConstraintBottomUpClusteringIvector(IvectorInfo& ivec_info, const DiarConfig& config) {
	// Compute i-vector for each cluster
	this->SetIvector(ivec_info);

	// Compute averge of i-vector for normalization
    Vector<double> ivectors_average;
    this->ComputeIvectorMean(ivectors_average);

    // Normalize ivectors
    this->NormalizeIvectors(ivectors_average);

	// Assign each cluster an idx number, to easier manipulation
	std::unordered_map<Cluster*, int32> cluster_idx_map;
	Cluster* curr = this->head_cluster_;
	int32 cluster_idx = 0;
	while(curr){
		cluster_idx_map[curr]=cluster_idx++;
		curr = curr->next;
	}

	// Initiallize distante matrix, cluster active status, which will be updated during bottom up clustering
	std::vector<std::vector<BaseFloat>> dist_matrix(this->num_clusters_, std::vector<BaseFloat>(100000.0, num_clusters_));
	std::vector<bool> to_be_updated(this->num_clusters_, true);

	// Cluster only Segments/clusters contain larger than specified number of frame
	Cluster* itr = this->head_cluster_;
	int32 idx = 0, non_update_size = 0;
	std::vector<BaseFloat> flt_max_vec(this->num_clusters_, 100000.0);
	while(config.min_update_segment != 0 && itr) {
		if(itr->NumFramesAfterMask() < config.min_update_segment) {
			dist_matrix[idx] = flt_max_vec;
			to_be_updated[idx] = false;
			non_update_size++;
			idx++;
		}
		itr = itr->next;
	}

	// Remember the original cluster number before clustering, for logging purpose
	int32 init_cluster_num = this->num_clusters_;

	// Start HAC clustering
	while(this->num_clusters_ > non_update_size + 2) {

		std::vector<Cluster*> min_dist_clusters(2);
		BaseFloat min_dist_ivector = FindMinDistClustersIvector(config, 
																dist_matrix, 
																to_be_updated, 
																cluster_idx_map, 
																min_dist_clusters);

		KALDI_LOG << "Min ivector distance: " << min_dist_ivector;

		if (min_dist_clusters[0]==NULL || min_dist_clusters[1]==NULL) 
				KALDI_ERR << "NULL CLUSTER!";

		if(min_dist_ivector > config.ivector_dist_stop || this->num_clusters_ <= config.target_cluster_num) {
			KALDI_LOG << "Cluster Number Reduced From: " << init_cluster_num << " To "<< num_clusters_;
			return;
		}	

		MergeClusters(min_dist_clusters[0], min_dist_clusters[1]);

		for(size_t i=0; i<to_be_updated.size();i++) {
			if(i == cluster_idx_map[min_dist_clusters[0]]) {
				to_be_updated[i] = true;
				min_dist_clusters[0]->SetIvector(ivec_info);
				min_dist_clusters[0]->NormalizeIvector(ivectors_average);
			}else{
				to_be_updated[i] = false;
			}
		}
		this->num_clusters_--;
	}

	KALDI_LOG << "Cluster Number Reduced From: " << init_cluster_num << " To "<< num_clusters_;
	return;
}

}