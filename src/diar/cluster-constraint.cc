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
	for(size_t i = 0; i < segments->Size(); i++) {
		this->dist_mat_.push_back(std::vector<BaseFloat>(segments->Size()));
	}
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
	for(size_t i = 0; i < nsegs; i++) {
		Segment* ith_segment = this->segments_->KthSegment(i);
        Vector<double> i_ivector = ith_segment->Ivector();
		for(size_t j = i + 1; j < nsegs; j++) {
			Segment* jth_segment = this->segments_->KthSegment(j);
         	Vector<double> j_ivector = jth_segment->Ivector();
			
			this->dist_mat_[i][j] =  1 - CosineDistance(i_ivector, j_ivector);
            this->dist_mat_[j][i] = this->dist_mat_[i][j];
		} 
	}

    // Exploring Stage: 
    srand(time(NULL));
    int32 seed_seg_id = rand() % nsegs;
    std::vector<int32> seed_cluster;
    seed_cluster.push_back(seed_seg_id);
    this->explored_clusters_.push_back(seed_cluster);
    std::unordered_map<int32,int32> clustered_seg;
    clustered_seg[seed_seg_id] = 1;

    int32 query_count=0, farthest_seg_id=0;
    while(query_count <= max_query) {
    	std::string farthest_seg_label;
        BaseFloat farthest_dist = -10000.0;
        for(size_t i = 0; i < nsegs; i++) {
        	if(clustered_seg.find(i) != clustered_seg.end()) continue;
            BaseFloat dist = 0.0;
            for(size_t c = 0; c < this->explored_clusters_.size(); c++) {
                for(size_t s = 0; s < this->explored_clusters_[c].size(); s++) {
                    if(i == this->explored_clusters_[c][s]){
                        continue;
                    }
                    dist += this->dist_mat_[i][this->explored_clusters_[c][s]];
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
            	KALDI_LOG << "Belongs To Existing Cluster : " << farthest_seg_label << " Query Count " << query_count;
                this->explored_clusters_[c].push_back(farthest_seg_id);
    			clustered_seg[farthest_seg_id] = 1;
                match_existing_cluster = true;
                break;
            }
        }

        if(!match_existing_cluster) {
        	KALDI_LOG << "Belongs To New Cluster : " << farthest_seg_label  << " Query Count " << query_count;
            std::vector<int32> new_cluster;
            new_cluster.push_back(farthest_seg_id);
            this->explored_clusters_.push_back(new_cluster);
    		clustered_seg[farthest_seg_id] = 1;
        }        
    }

    KALDI_LOG << "Finished Farthest First Search Exploring.";
    KALDI_LOG << "A Total of " << this->explored_clusters_.size() << " clusters are explored:";
    for(int k = 0; k < this->explored_clusters_.size(); k++) {
    	KALDI_LOG << "Cluster" << k << "found " << this->explored_clusters_[k].size() << " segments";
    }

    return;
}


void ClusterCollectionConstraint::IvectorHacExploreFarthestFirstSearch(IvectorInfo& ivec_info, const DiarConfig& config, 
																		const int32& max_query, const std::vector<bool>& is_seed_candidate) {
	
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
	for(size_t i = 0; i < nsegs; i++) {
		Segment* ith_segment = this->segments_->KthSegment(i);
        Vector<double> i_ivector = ith_segment->Ivector();
		for(size_t j = i + 1; j < nsegs; j++) {
			Segment* jth_segment = this->segments_->KthSegment(j);
         	Vector<double> j_ivector = jth_segment->Ivector();
			
			this->dist_mat_[i][j] =  1 - CosineDistance(i_ivector, j_ivector);
            this->dist_mat_[j][i] = this->dist_mat_[i][j];
		} 
	}

    // Exploring Stage: 
    std::unordered_map<int32,int32> clustered_seg;
    int32 query_count=0, farthest_seg_id=0;
    for(int32 i = 0; i < nsegs; i++) {
    	if(is_seed_candidate[i]) {
    		std::string ith_label = this->segments_->KthSegment(i)->Label();
	        bool match_existing_cluster = false;
	        for(size_t c = 0; c < this->explored_clusters_.size(); c++) {
	            query_count++;
	            std::string cth_cluster_label = this->segments_->KthSegment(this->explored_clusters_[c][0])->Label();                
	            if(ith_label == cth_cluster_label) {
	                this->explored_clusters_[c].push_back(i);
	    			clustered_seg[i] = 1;
	                match_existing_cluster = true;
	                break;
	            }
	        }

	        if(!match_existing_cluster) {
	            std::vector<int32> new_cluster;
	            new_cluster.push_back(i);
	            this->explored_clusters_.push_back(new_cluster);
	    		clustered_seg[i] = 1;
	        }            		
    	}
    }

    while(query_count <= max_query) {
    	std::string farthest_seg_label;
        BaseFloat farthest_dist = -10000.0;
        for(size_t i = 0; i < nsegs; i++) {
        	if(clustered_seg.find(i) != clustered_seg.end()) continue;
            BaseFloat dist = 0.0;
            for(size_t c = 0; c < this->explored_clusters_.size(); c++) {
                for(size_t s = 0; s < this->explored_clusters_[c].size(); s++) {
                    if(i == this->explored_clusters_[c][s]){
                        continue;
                    }
                    dist += this->dist_mat_[i][this->explored_clusters_[c][s]];
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
            	KALDI_LOG << "Belongs To Existing Cluster : " << farthest_seg_label << " Query Count " << query_count;
                this->explored_clusters_[c].push_back(farthest_seg_id);
    			clustered_seg[farthest_seg_id] = 1;
                match_existing_cluster = true;
                break;
            }
        }

        if(!match_existing_cluster) {
        	KALDI_LOG << "Belongs To New Cluster : " << farthest_seg_label  << " Query Count " << query_count;
            std::vector<int32> new_cluster;
            new_cluster.push_back(farthest_seg_id);
            this->explored_clusters_.push_back(new_cluster);
    		clustered_seg[farthest_seg_id] = 1;
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
	std::unordered_map<int32, int32> clustered_segments;
	for(size_t i = 0; i < this->explored_clusters_.size();i++) {
		for(size_t j = 1; j < explored_clusters_[i].size(); j++) {
			clustered_segments[explored_clusters_[i][j]] = 1;
		}
	}

	for(size_t i = 0; i < this->explored_clusters_.size();i++) {
		std::vector<BaseFloat> dist = this->dist_mat_[this->explored_clusters_[i][0]];
		for(size_t j = 1; j < explored_clusters_[i].size(); j++) {
			std::vector<BaseFloat> to_add = this->dist_mat_[explored_clusters_[i][j]];
			for(size_t k = 0; k < dist.size(); k++) {
				dist[k] += to_add[k];
			}
		}

		std::vector<long unsigned int> ranks = ordered(dist);
		for(size_t k = 0; k< max_query_per_cluster; k++) {
			if(ranks[k] != i && clustered_segments.find(ranks[k]) == clustered_segments.end()) {
				string rank_k_label = this->segments_->KthSegment(ranks[k])->Label();
				string cluster_i_label = this->segments_->KthSegment(this->explored_clusters_[i][0])->Label();
				if(rank_k_label == cluster_i_label) {
					this->explored_clusters_[i].push_back(ranks[k]);
					clustered_segments[ranks[k]] = 1;
					KALDI_LOG << "Total Segments For Cluster " << cluster_i_label << " is " << this->explored_clusters_[i].size();
				}	
			}
		}
	}
	return;
}


void ClusterCollectionConstraint::InitClustersWithExploredClusters() {
 	KALDI_ASSERT(explored_clusters_.size() > 0);
 	std::unordered_map<int32, int32> clustered_segments;
	Cluster* prev_cluster = NULL;
	Cluster* new_cluster = NULL; 

 	for(size_t i = 0; i < this->explored_clusters_.size(); i++) {
 		for(size_t j = 0; j < this->explored_clusters_[i].size(); j++) {
 			int32 segment_idx = this->explored_clusters_[i][j];
 			if(clustered_segments.find(segment_idx) != clustered_segments.end()) 
 					continue;
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

 	/*
 	Cluster* curr = this->head_cluster_;
 	int32 ct = 0;
 	KALDI_LOG << "Total Cluster Number: " << this->num_clusters_;
 	while(curr) {
 		KALDI_LOG<< "**Cluster " << ct;
 		ct++;
 		curr = curr->next;
 	}
 	*/

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
	int32 idx = 0, non_update_size = this->explored_clusters_.size();
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
		BaseFloat min_dist_ivector = FindMinDistClustersIvectorConstraint(config, 
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

		Cluster* updated;
		if(!min_dist_clusters[0]->IsMergeable()) {
			MergeClusters(min_dist_clusters[0], min_dist_clusters[1]);
			updated = min_dist_clusters[0];
		} else{
			MergeClusters(min_dist_clusters[1], min_dist_clusters[0]);
			updated = min_dist_clusters[1];
		}

		for(size_t i=0; i<to_be_updated.size();i++) {
			if(i == cluster_idx_map[updated]) {
				to_be_updated[i] = true;
				updated->SetIvector(ivec_info);
				updated->NormalizeIvector(ivectors_average);
			}else{
				to_be_updated[i] = false;
			}
		}
		this->num_clusters_--;
	}

	KALDI_LOG << "Cluster Number Reduced From: " << init_cluster_num << " To "<< num_clusters_;
	return;
}

BaseFloat ClusterCollectionConstraint::FindMinDistClustersIvectorConstraint(const DiarConfig& config, 
														std::vector<std::vector<BaseFloat>>& dist_matrix, 
														std::vector<bool>& to_be_updated, 
														std::unordered_map<Cluster*, int32>& cluster_idx_map, 
														std::vector<Cluster*> &min_dist_clusters) {

	KALDI_ASSERT(num_clusters_>=2);
	Cluster* p1 = this->head_cluster_;
	BaseFloat min_dist = FLT_MAX; 
	BaseFloat dist;
	while(p1){
		int32 p1_idx = cluster_idx_map[p1];
		Cluster* p2 = p1->next;
		while(p2) {

			int32 p2_idx = cluster_idx_map[p2];

			if(!to_be_updated[p1_idx]) {
				dist = dist_matrix[p1_idx][p2_idx]; // been updated before, avoid recalculation
			}else{
				if (config.ivector_dist_type == "CosineDistance") {
					dist = 1 - CosineDistance(p1->Ivector(), p2->Ivector());
				} else if(config.ivector_dist_type == "IvectorKL2") {
					dist = SymetricKlDistanceDiag(p1->Ivector(), p2->Ivector(), p1->IvectorCovar(), p2->IvectorCovar());
				}
				dist_matrix[p1_idx][p2_idx] = dist;
			}

	    	if(dist<min_dist && (p1->IsMergeable() || p2->IsMergeable())) {
				min_dist = dist;
				min_dist_clusters[0] = p1;
				min_dist_clusters[1] = p2;
			}
			p2 = p2->next;
		}
		p1 = p1->next;
	}
	return min_dist;
}



}