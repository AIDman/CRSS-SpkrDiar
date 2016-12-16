#include <iomanip>
#include <cfloat>
#include "cluster.h"

namespace kaldi {

Cluster::Cluster(Segment one_segment){
	this->label_ = Cluster::prefix + ToString(Cluster::id_generator++);
	this->all_segments_.push_back(one_segment);
	this->frames_ =  one_segment.Size();
	this->frames_after_mask_ = one_segment.SizeAfterMask();
	this->mergeable_ = true;
}

int Cluster::id_generator = 1;
std::string Cluster::prefix = "C";

void Cluster::AddSegment(Segment new_segment) {
	this->all_segments_.push_back(new_segment);
	this->frames_ = this->frames_ + new_segment.Size();
	this->frames_after_mask_ += new_segment.SizeAfterMask();
}


std::vector<Segment> Cluster::AllSegments() const {
	return this->all_segments_;
}


Segment Cluster::KthSegment(int32 k) const {
	return this->all_segments_[k];
}


std::string Cluster::Label() const {
	return this->label_;
}

void Cluster::SetLabel(const std::string& label) {
	this->label_ = label;
	return;
}

void Cluster::SetMergeable(bool mergeable) {
	this->mergeable_ = mergeable;
	return;
}


int32 Cluster::NumFrames() const {
	return this->frames_;
}


int32 Cluster::NumFramesAfterMask() const {
	return this->frames_after_mask_;
}


int32 Cluster::NumSegments() const {
	return this->all_segments_.size();
}


void Cluster::ComputeMean(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& mean_vec) const {
	int32 tot_frames = 0;
	for(int32 i=0; i<this->all_segments_.size();i++) {
		Segment seg = this->all_segments_[i];
		Vector<BaseFloat> mask = seg.Mask();
		for(int32 j = seg.StartIdx(), k = 0; j<=seg.EndIdx(); j++, k++) {
			if(mask(k) == 1.0) {
				mean_vec.AddVec(1.0, feats.Row(j));
				tot_frames++;
			}
		}
	}
	mean_vec.Scale(1.0/tot_frames);
	return;
}


void Cluster::ComputeCovDiag(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& cov_vec) const {
	int32 dim = feats.NumCols();
	int32 tot_frames = 0;
	for(int32 i=0; i<this->all_segments_.size();i++) {
		Segment seg = this->all_segments_[i];
		Vector<BaseFloat> mask = seg.Mask();
		for(int32 j = seg.StartIdx(), k = 0; j<=seg.EndIdx(); j++, k++) {
			if(mask(k) == 1.0) {
				cov_vec.AddVec2(1.0, feats.Row(j));
				tot_frames++;
			}
		}
	}
	cov_vec.Scale(1.0/tot_frames);

	Vector<BaseFloat> mean_vec(dim);
	this->ComputeMean(feats, mean_vec);

	cov_vec.AddVec2(-1.0, mean_vec);
	return;
}


// To Repeat the GLR code in Audioseg
void Cluster::ComputeSum(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& feats_sum) const {
	int32 tot_frames = 0;
	for(int32 i=0; i<this->all_segments_.size();i++) {
		Segment seg = this->all_segments_[i];
		Vector<BaseFloat> mask = seg.Mask();
		for(int32 j = seg.StartIdx(), k = 0; j<=seg.EndIdx(); j++, k++) {
			if(mask(k) == 1.0) {
				feats_sum.AddVec(1.0, feats.Row(j));
				tot_frames++;
			}
		}
	}
	return;
}


// To Repeat the GLR code in Audioseg
void Cluster::ComputeVarSum(const Matrix<BaseFloat>& feats, Vector<BaseFloat>& var_sum) const {
	int32 tot_frames = 0;
	for(int32 i=0; i<this->all_segments_.size();i++) {
		Segment seg = this->all_segments_[i];
		Vector<BaseFloat> mask = seg.Mask();
		for(int32 j = seg.StartIdx(), k = 0; j<=seg.EndIdx(); j++, k++) {
			if(mask(k) == 1.0) {
				var_sum.AddVec2(1.0, feats.Row(j));
				tot_frames++;
			}
		}
	}
	return;
}


// Followed implementation in Audioseg.
BaseFloat Cluster::LogDet(const Matrix<BaseFloat> &feats) const {
	int32 dim = feats.NumCols();
	Vector<BaseFloat> m(dim); 
	this->ComputeSum(feats, m);
	Vector<BaseFloat> v(dim);
	this->ComputeVarSum(feats, v);
	BaseFloat z = 1.0 / this->NumFramesAfterMask();
	BaseFloat zz = z * z;
	BaseFloat s = 0.0;
	for (int i=0; i<feats.NumCols();i++) {
		s += log(v(i) * z - m(i) * m(i) *zz);
	}
	return s;
}


void Cluster::CollectFeatures(const Matrix<BaseFloat>& feats, Matrix<BaseFloat>& cluster_feats) const {
	int insert_pos = 0;
	for(int32 i=0; i<this->all_segments_.size(); i++) {
		Segment seg = this->all_segments_[i];
		Vector<BaseFloat> mask = seg.Mask();
		for(int32 idx = seg.StartIdx(), k = 0; idx <= seg.EndIdx(); idx++, k++) {
			if(mask(k) == 1.0) {
				cluster_feats.Row(insert_pos).CopyFromVec(feats.Row(idx));
				insert_pos++;
			}
		}
	}
	return;
}


void Cluster::CollectPosteriors(const Posterior& posterior, Posterior& cluster_posteriors) const {
	int32 insert_pos = 0;

	for(int32 i=0; i<this->all_segments_.size(); i++) {
		Segment seg = this->all_segments_[i];
		Vector<BaseFloat> mask = seg.Mask();
		for(int32 idx = seg.StartIdx(), k = 0; idx <= seg.EndIdx(); idx++, k++) {
			if(mask(k) == 1.0) {
				cluster_posteriors[insert_pos] = posterior[idx];
				insert_pos++;
			}
		}
	}
	return;
}


Vector<double> Cluster::Ivector() {
	return this->ivector_;
}

SpMatrix<double> Cluster::IvectorCovar() {
	return this->ivector_covar_;
}


void Cluster::SetIvector(Vector<double>& ivect) {
	this->ivector_ = ivect;
	return;
}


void Cluster::SetIvector(Vector<double>& ivect, SpMatrix<double>& ivect_covar) {
	this->ivector_ = ivect;
	this->ivector_covar_ = ivect_covar;
	return;
}


void Cluster::SetIvector(IvectorInfo& ivec_info) {
	// post probs of cluster
	int32 clst_frms = this->NumFramesAfterMask();
	Posterior cluster_posteriors(clst_frms);
	this->CollectPosteriors(*ivec_info.posteriors, cluster_posteriors);

	// features of cluster
	int32 dim = (*ivec_info.feats).NumCols();
	Matrix<BaseFloat> cluster_feats(clst_frms,dim);
	this->CollectFeatures(*ivec_info.feats, cluster_feats);

	Vector<double> ivector_mean;
	SpMatrix<double> ivector_covar;
	ComputeIvector(cluster_feats, cluster_posteriors, *ivec_info.extractor, ivector_mean, ivector_covar);

	// length nomalization
	BaseFloat norm = ivector_mean.Norm(2.0);
    BaseFloat ratio = norm / sqrt(ivector_mean.Dim()); // how much larger it is
    ivector_mean.Scale(1.0 / ratio);

	this->ivector_ = ivector_mean;
	this->ivector_covar_ = ivector_covar; //TO DO: ivector_covar should changes after length normalization

    return;
}

void Cluster::NormalizeIvector(Vector<double>& ivectors_average) {
	this->ivector_.AddVec(-1.0, ivectors_average);
	return;
}

ClusterCollection::ClusterCollection() {
	num_clusters_ = 0;
	head_cluster_ = NULL;
}


string ClusterCollection::UttID() {
	return this->uttid_;
}


int32 ClusterCollection::NumFrames() {
	Cluster* curr = this->head_cluster_;
	int32 tot_frames = 0;
	while(curr) {
		tot_frames += curr->NumFrames();
		curr = curr->next;
	}
	return tot_frames;
}

int32 ClusterCollection::NumFramesAfterMask() {
	Cluster* curr = this->head_cluster_;
	int32 tot_frames = 0;
	while(curr) {
		tot_frames += curr->NumFramesAfterMask();
		curr = curr->next;
	}
	return tot_frames;
}

void ClusterCollection::InitFromNonLabeledSegments(SegmentCollection& non_clustered_segments) {
	 int32 num_segments = non_clustered_segments.Size();

	 if(num_segments < 1) KALDI_ERR << "Clusters could not be initialized from empty segments";

	 Segment* head_segment = non_clustered_segments.KthSegment(0);	 
	 Cluster* head_cluster = new Cluster(*head_segment);

	 Cluster* prev_cluster = NULL; 
	 for(int32 i=0; i<num_segments;i++){
	 	if(i==0) {
	 		head_cluster->prev = NULL;
	 		prev_cluster = head_cluster;
	 		continue;
	 	} 

	 	Cluster* new_cluster = new Cluster(*(non_clustered_segments.KthSegment(i)));
	 	prev_cluster->next = new_cluster;
	 	new_cluster->prev = prev_cluster;
	 	prev_cluster = new_cluster;
	 	if(i==num_segments-1) new_cluster->next=NULL; // last cluster's next point to NULL
	 }

	 this->num_clusters_ = num_segments;
	 this->uttid_ = non_clustered_segments.UttID();
	 this->head_cluster_ = head_cluster;
	 return;
}


void ClusterCollection::InitFromLabeledSegments(SegmentCollection& pre_clustered_segments) {
	 int32 num_segments = pre_clustered_segments.Size();

	 if(num_segments < 1) KALDI_ERR << "Clusters could not be initialized from empty segments";

	 // need to put all segments with same label into one class
	 std::unordered_map<std::string, Cluster*> cluster_label_map;

	 Cluster* head_cluster = new Cluster(*(pre_clustered_segments.KthSegment(0)));
	 head_cluster->SetLabel(pre_clustered_segments.KthSegment(0)->Label());
	 cluster_label_map[head_cluster->Label()] = head_cluster;

	 Cluster* prev_cluster = NULL;
	 this->num_clusters_ = 0; 
	 Cluster* last_cluster = head_cluster;
	 for(int32 i=0; i<num_segments;i++){
	 	if(i==0) {
	 		head_cluster->prev = NULL;
	 		prev_cluster = head_cluster;
	 		this->num_clusters_++;
	 		continue;
	 	} 

	 	Segment* curr_segment = pre_clustered_segments.KthSegment(i);
	 	std::string curr_spk_label = curr_segment->Label();

	 	if(cluster_label_map.find(curr_spk_label) == cluster_label_map.end()) {
		 	Cluster* new_cluster = new Cluster(*curr_segment);
		 	new_cluster->SetLabel(curr_spk_label);
		 	prev_cluster->next = new_cluster;
		 	new_cluster->prev = prev_cluster;
		 	prev_cluster = new_cluster;
		 	last_cluster = new_cluster;
		 	cluster_label_map[curr_spk_label] = new_cluster;
	 		this->num_clusters_++;
		 } else{
		 	cluster_label_map[curr_spk_label]->AddSegment(*curr_segment);
		 }
	 }
	 last_cluster->next=NULL; // make sure the last cluster's next point to null
	 this->uttid_ = pre_clustered_segments.UttID();
	 this->head_cluster_ = head_cluster;
	 return;
}


Cluster* ClusterCollection::Head() {
	return this->head_cluster_;
}


void ClusterCollection::SetIvector(IvectorInfo& ivec_info) {
	Cluster* curr = this->head_cluster_;
	while(curr) {
		curr->SetIvector(ivec_info);
		curr = curr->next;
	}
	return;
}


void ClusterCollection::ComputeIvectorMean(Vector<double>& ivectors_average) {
	Cluster* curr = this->head_cluster_;
	ivectors_average.Resize(curr->Ivector().Dim());
	while(curr) {
		Vector<double> curr_ivector = curr->Ivector();
		ivectors_average.AddVec(1.0/this->num_clusters_, curr_ivector);
		curr = curr->next;
	}
	return;
}


void ClusterCollection::NormalizeIvectors(Vector<double>& ivectors_average) {
	Cluster* curr = this->head_cluster_;
	while(curr) {
		Vector<double> curr_ivector = curr->Ivector();
		curr_ivector.AddVec(-1.0, ivectors_average);
		curr->SetIvector(curr_ivector);
		curr = curr->next;
	}
}


/* Bottom Up Clustering Using GLR or KL2 Distance. The penalty factor lambda is set to FLX_MAT by default, 
 * this will cause the clustering continues untill one cluster left, or meets other critera. 
 * The target_cluster_num is also set to 0 following the same logic. 
 */
void ClusterCollection::BottomUpClustering(const Matrix<BaseFloat> &feats, const DiarConfig& config) {

	// Assign each cluster and idx number, to easier manipulation
	std::unordered_map<Cluster*, int32> cluster_idx_map;
	Cluster* curr = this->head_cluster_;
	int32 cluster_idx = 0;
	while(curr){
		cluster_idx_map[curr]=cluster_idx++;
		curr = curr->next;
	}

	// Initiallize distante matrix and cluster active status, to be updated during bottom up clustering
	std::vector<std::vector<BaseFloat>> dist_matrix(this->num_clusters_, std::vector<BaseFloat>(100000.0, num_clusters_));
	std::vector<bool> to_be_updated(this->num_clusters_, true);

	// Cluster only Segments/clusters contain larger than specified number of frame
	Cluster* itr = this->head_cluster_;
	int32 idx = 0, non_updateable = 0;
	std::vector<BaseFloat> flt_max_vec(this->num_clusters_, 1000000.0);
	while(config.min_update_segment != 0 && itr) {
		if(itr->NumFramesAfterMask() < config.min_update_segment) {
			dist_matrix[idx] = flt_max_vec;
			to_be_updated[idx] = false;
			non_updateable++;
			idx++;
		}
		itr = itr->next;
	}

	// Compute the penalty BIC critera
	int32 dim = feats.NumCols();
	int32 tot_frames = this->NumFramesAfterMask(); 
	BaseFloat penalty = 0.5 * (dim + dim) * log(tot_frames);

	// Start HAC clustering
	int32 init_cluster_num = this->num_clusters_;
	while(this->num_clusters_ > non_updateable + 2) {

		std::vector<Cluster*> min_dist_clusters(2);
		FindMinDistClusters(feats, 
							config, 
							dist_matrix, 
							to_be_updated, 
							cluster_idx_map, 
							min_dist_clusters);

		if (min_dist_clusters[0]==NULL || min_dist_clusters[1]==NULL) 
				KALDI_ERR << "NULL CLUSTER!";

		BaseFloat dist_glr = DistanceOfTwoClustersGLR(feats, min_dist_clusters[0], min_dist_clusters[1]);

		KALDI_LOG << "Min GLR Distance: " << dist_glr << " Current Cluster Num:" << num_clusters_;

		BaseFloat delta_bic = dist_glr - config.lambda * penalty;

		if(delta_bic > 0 || this->num_clusters_ <= config.target_cluster_num) {
			KALDI_LOG << "Cluster Number Reduced From: " << init_cluster_num << " To "<< num_clusters_;
			return;
		} 	

		MergeClusters(min_dist_clusters[0], min_dist_clusters[1]);

		for(size_t i=0; i<to_be_updated.size();i++) {
			if(i == cluster_idx_map[min_dist_clusters[0]]) {
				to_be_updated[i] = true;
			}else{
				to_be_updated[i] = false;
			}
		}

		this->num_clusters_--;
	}

	KALDI_LOG << "Cluster Number Reduced From: " << init_cluster_num << " To "<< num_clusters_;

	return;
}


void ClusterCollection::BottomUpClusteringIvector(IvectorInfo& ivec_info, const DiarConfig& config) {
	// Compute i-vector for each cluster
	this->SetIvector(ivec_info);

	// Compute averge of i-vector for normalization
    Vector<double> ivectors_average;
    this->ComputeIvectorMean(ivectors_average);

    // Normalize ivectors
    this->NormalizeIvectors(ivectors_average);

	// Assign each cluster and idx number, to easier manipulation
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


void ClusterCollection::FindMinDistClusters(const Matrix<BaseFloat> &feats, const DiarConfig& config, std::vector<std::vector<BaseFloat>>& dist_matrix, std::vector<bool>& to_be_updated, 
	std::unordered_map<Cluster*, int32>& cluster_idx_map, std::vector<Cluster*> &min_dist_clusters) {

	if(num_clusters_<2) KALDI_ERR << "Less than two clusters, could not find min dist clusters";
	Cluster* p1 = this->head_cluster_;
	BaseFloat min_dist = FLT_MAX; // set a random large number
	BaseFloat dist;
	while(p1){
		int32 p1_idx = cluster_idx_map[p1];
		Cluster* p2 = p1->next;
		while(p2) {
			int32 p2_idx = cluster_idx_map[p2];
			if(!to_be_updated[p1_idx]) {
				// been updated before, avoid recalculation
				dist = dist_matrix[p1_idx][p2_idx];

			} else{
				// to be updated
				if(config.dist_type== "GLR") {
					dist = DistanceOfTwoClustersGLR(feats, p1, p2);
				}else if (config.dist_type == "KL2") {
					dist = DistanceOfTwoClustersKL2(feats, p1, p2);
				}

				dist_matrix[p1_idx][p2_idx] = dist;
			}

	    	if(dist<min_dist) {
				min_dist = dist;
				min_dist_clusters[0] = p1;
				min_dist_clusters[1] = p2;
			}
			p2 = p2->next;
		}
		p1 = p1->next;
	}
	return;
}


BaseFloat ClusterCollection::FindMinDistClustersIvector(const DiarConfig& config, 
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

	    	if(dist<min_dist) {
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


BaseFloat ClusterCollection::DistanceOfTwoClustersGLR(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2) {

	if(!cluster1 || !cluster2) KALDI_ERR << "CLUSTER COULD NOT BE NULL!";
	Cluster* cluster1_copy = new Cluster(*cluster1);
	Cluster* cluster2_copy = new Cluster(*cluster2);
	// Avoid the original clusters connection is corrupte during merging for logdet computation
	cluster2_copy->next = NULL;
	cluster2_copy->prev = NULL;

	ClusterCollection::MergeClusters(cluster1_copy,cluster2_copy);
	Cluster* cluster12 = cluster1_copy;
	BaseFloat log_det12 = cluster12->LogDet(feats);
	BaseFloat log_det1 = cluster1->LogDet(feats);
	BaseFloat log_det2 = cluster2->LogDet(feats);
	BaseFloat dist = log_det12 * cluster12->NumFramesAfterMask() - log_det1 * cluster1->NumFramesAfterMask() - log_det2 * cluster2->NumFramesAfterMask();

	delete cluster12;

	return 0.5*dist;
}


BaseFloat ClusterCollection::DistanceOfTwoClustersKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2) {
	int32 dim = feats.NumCols();

	Vector<BaseFloat> mean_vec_1(dim);
	cluster1->ComputeMean(feats, mean_vec_1);

	Vector<BaseFloat> mean_vec_2(dim);
	cluster2->ComputeMean(feats, mean_vec_2);

	Vector<BaseFloat> cov_vec_1(dim);
	cluster1->ComputeCovDiag(feats, cov_vec_1);

	Vector<BaseFloat> cov_vec_2(dim);
	cluster2->ComputeCovDiag(feats, cov_vec_2);

	return SymetricKlDistanceDiag(mean_vec_1, mean_vec_2, cov_vec_1, cov_vec_2);	
}


/*
BaseFloat ClusterCollection::DistanceOfTwoClustersIvectorKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2,
															const Posterior& posterior, const IvectorExtractor& extractor) {

	int32 dim = feats.NumCols();
	int32 nframes = this->NumFrames();
	Matrix<BaseFloat> feats_clust1(nframes,dim), feats_clust2(nframes,dim); 
	cluster1->CollectFeatures(feats, feats_clust1);	
	cluster2->CollectFeatures(feats, feats_clust2);

	Posterior postprob_clust1(nframes), postprob_clust2(nframes);
	cluster1->CollectPosteriors(posterior, postprob_clust1);
	cluster2->CollectPosteriors(posterior, postprob_clust2);

	Vector<double> ivector_clust1_mean, ivector_clust2_mean;
	SpMatrix<double> ivector_clust1_covar, ivector_clust2_covar;
	ComputeIvector(feats_clust1, postprob_clust1, extractor, ivector_clust1_mean, ivector_clust1_covar);
	ComputeIvector(feats_clust1, postprob_clust2, extractor, ivector_clust2_mean, ivector_clust2_covar);

	//return  SymetricKlDistance(ivector_clust1_mean, ivector_clust2_mean, ivector_clust1_covar, ivector_clust2_covar);
	return  CosineDistance(ivector_clust1_mean, ivector_clust2_mean);
}
*/

void ClusterCollection::MergeClusters(Cluster* clust1, Cluster* clust2) {
	std::vector<Segment> clust2_segments = clust2->AllSegments();
	for(int i=0; i<clust2_segments.size();i++) {
		clust1->AddSegment(clust2_segments[i]);
	}

	if(clust2->prev) clust2->prev->next = clust2->next;
	if(clust2->next) clust2->next->prev = clust2->prev; 

	delete clust2;
	return;
}


void ClusterCollection::Write(const std::string& segments_dirname) {
	std::string segments_wxfilename = segments_dirname + "/" + this->uttid_ + ".seg";
	std::string segments_scpfilename = segments_dirname + "/" + "segments.scp";
	std::ofstream fout;
	std::ofstream fscp;
	fout.open(segments_wxfilename.c_str());
	fscp.open(segments_scpfilename.c_str(), std::ios::app);
	Cluster* curr = this->head_cluster_;
	while(curr){
		string clst_label = curr->Label();
		for (size_t i =0; i < curr->NumSegments(); i++){
			BaseFloat seg_start = FrameIndexToSeconds(curr->KthSegment(i).StartIdx());
			BaseFloat seg_end = FrameIndexToSeconds(curr->KthSegment(i).EndIdx());
			fout << this->uttid_ << " ";
			fout << std::fixed << std::setprecision(3);
			fout << seg_start << " ";
			fout << seg_end << " ";
			fout << clst_label << "\n";
		}
		curr=curr->next;
	}

	fscp << segments_wxfilename << "\n";
	fscp.close();
	fout.close();
}


void ClusterCollection::WriteToRttm(const std::string& rttm_outputdir) {
	std::string rttmName = rttm_outputdir + "/" + this->uttid_ +".rttm";
	std::string rttm_scpfilename = rttm_outputdir + "/" + "rttms.scp";
	std::ofstream fout;
	std::ofstream fscp;
	fout.open(rttmName.c_str());
	fscp.open(rttm_scpfilename.c_str(), std::ios::app);
	Cluster* curr = this->head_cluster_;
	while(curr){
		std::vector<Segment> cluster_segments = curr->AllSegments();
		string cluster_label = curr->Label();
		for (size_t i =0; i < cluster_segments.size(); i++){
			BaseFloat segStart = FrameIndexToSeconds(cluster_segments[i].StartIdx());
			BaseFloat segLength = FrameIndexToSeconds(cluster_segments[i].EndIdx()) - segStart;
			fout << "SPEAKER ";
			fout << this->uttid_ << " ";
			fout << 1 << " ";
			fout << std::fixed << std::setprecision(3);
			fout << segStart << " ";
			fout << segLength << " ";
			fout << "<NA> <NA> ";
			fout << cluster_label << " ";
			fout << "<NA>\n";
		}
		curr=curr->next;
	}
	fscp << rttmName << "\n";
	fscp.close();
	fout.close();
}

}