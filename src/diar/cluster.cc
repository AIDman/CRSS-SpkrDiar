#include <iomanip>
#include <cfloat>
#include "cluster.h"

namespace kaldi {


Cluster::Cluster(Segment one_segment){
	this->label_ = Cluster::prefix + ToString(Cluster::id_generator++);
	this->list_.push_back(one_segment);
	this->frames_ =  one_segment.Size();
}

int Cluster::id_generator = 1;
std::string Cluster::prefix = "C";

void Cluster::AddSegment(Segment new_segment) {
	this->list_.push_back(new_segment);
	this->frames_ = this->frames_ + new_segment.Size();
}


std::vector<Segment> Cluster::AllSegments() const {
	return this->list_;
}


Segment Cluster::KthSegment(int32 k) const {
	return this->list_[k];
}


std::string Cluster::Label() const {
	return this->label_;
}


int32 Cluster::NumFrames() const {
	return this->frames_;
}


int32 Cluster::NumSegments() const {
	return this->list_.size();
}


Vector<BaseFloat> Cluster::ComputeMean(const Matrix<BaseFloat>& feats, const Cluster* clust) {
	std::vector<Segment> all_segments = clust->AllSegments();
	int32 dim = feats.NumCols();
	Vector<BaseFloat> mean_vec(dim);
	int32 tot_frames = 0;
	for(int i=0; i<all_segments.size();i++) {
		Segment seg = all_segments[i];
		for(int j = seg.StartIdx(); j<=seg.EndIdx(); j++) {
			mean_vec.AddVec(1.0, feats.Row(j));
			tot_frames++;
		}
	}
	mean_vec.Scale(1.0/tot_frames);
	return mean_vec;
}


Vector<BaseFloat> Cluster::ComputeCovDiag(const Matrix<BaseFloat>& feats, const Cluster* clust) {
	std::vector<Segment> all_segments = clust->AllSegments();
	int32 dim = feats.NumCols();
	Vector<BaseFloat> cov_vec(dim);
	int32 tot_frames = 0;
	for(int i=0; i<all_segments.size();i++) {
		Segment seg = all_segments[i];
		for(int j = seg.StartIdx(); j<=seg.EndIdx(); j++) {
			cov_vec.AddVec2(1.0, feats.Row(j));
			tot_frames++;
		}
	}
	cov_vec.Scale(1.0/tot_frames);

	Vector<BaseFloat> mean_vec = Cluster::ComputeMean(feats, clust);
	cov_vec.AddVec2(-1.0, mean_vec);
	return cov_vec;
}


// To Repeat the GLR code in Audioseg
Vector<BaseFloat> Cluster::ComputeSum(const Matrix<BaseFloat>& feats, const Cluster* clust) {
	std::vector<Segment> all_segments = clust->AllSegments();
	int32 dim = feats.NumCols();
	Vector<BaseFloat> mean_vec(dim);
	int32 tot_frames = 0;
	for(int i=0; i<all_segments.size();i++) {
		Segment seg = all_segments[i];
		for(int j = seg.StartIdx(); j<=seg.EndIdx(); j++) {
			mean_vec.AddVec(1.0, feats.Row(j));
			tot_frames++;
		}
	}
	return mean_vec;
}


// To Repeat the GLR code in Audioseg
Vector<BaseFloat> Cluster::ComputeVarSum(const Matrix<BaseFloat>& feats, const Cluster* clust) {
	std::vector<Segment> all_segments = clust->AllSegments();
	int32 dim = feats.NumCols();
	Vector<BaseFloat> cov_vec(dim);
	int32 tot_frames = 0;
	for(int i=0; i<all_segments.size();i++) {
		Segment seg = all_segments[i];
		for(int j = seg.StartIdx(); j<=seg.EndIdx(); j++) {
			cov_vec.AddVec2(1.0, feats.Row(j));
			tot_frames++;
		}
	}
	return cov_vec;
}


// Followed implementation in Audioseg.
BaseFloat Cluster::LogDet(const Matrix<BaseFloat> &feats) const {
	Vector<BaseFloat> m = Cluster::ComputeSum(feats, this);
	Vector<BaseFloat> v = Cluster::ComputeVarSum(feats, this);
	BaseFloat z = 1.0 / this->frames_;
	BaseFloat zz = z * z;
	BaseFloat s = 0.0;
	for (int i=0; i<feats.NumCols();i++) {
		s += log(v(i) * z - m(i) * m(i) *zz);
	}
	return s;
}


void Cluster::CollectFeatures(const Matrix<BaseFloat>& feats, const Cluster* clust, Matrix<BaseFloat>& feats_collect) {
	int32 dim = feats.NumCols();
	int32 nframes = clust->NumFrames();
	feats_collect.Resize(nframes, dim);
	std::vector<Segment> all_segments = clust->AllSegments();
	int insert_pos = 0;
	for(int32 i=0; i<all_segments.size(); i++) {
		Segment seg = all_segments[i];
		for(int32 idx = seg.StartIdx(); idx <= seg.EndIdx(); idx++) {
			feats_collect.Row(insert_pos).CopyFromVec(feats.Row(idx));
			insert_pos++;
		}
	}
	return;
}

void Cluster::CollectPosteriors(const Posterior& posterior, const Cluster* clust, Posterior& posteriors_collect) {
	int32 nframes = clust->NumFrames();
	std::vector<Segment> all_segments = clust->AllSegments();
	posteriors_collect.resize(nframes);
	int32 insert_pos = 0;

	for(int32 i=0; i<all_segments.size(); i++) {
		Segment seg = all_segments[i];
		for(int32 idx = seg.StartIdx(); idx <= seg.EndIdx(); idx++) {
			posteriors_collect[insert_pos] = posterior[idx];
			insert_pos++;
		}
	}
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

void ClusterCollection::InitFromNonLabeledSegments(SegmentCollection non_clustered_segments) {
	 int32 num_segments = non_clustered_segments.Size();
	 if(num_segments < 1) KALDI_ERR << "Clusters could not be initialized from empty segments";
	 Cluster* head_cluster = new Cluster(non_clustered_segments.KthSegment(0));
	 Cluster* prev_cluster = NULL; 
	 for(int32 i=0; i<num_segments;i++){
	 	if(i==0) {
	 		head_cluster->prev = NULL;
	 		prev_cluster = head_cluster;
	 		continue;
	 	} 

	 	Cluster* new_cluster = new Cluster(non_clustered_segments.KthSegment(i));
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


Cluster* ClusterCollection::Head() {
	return this->head_cluster_;
}


/* Bottom Up Clustering Using GLR or KL2 Distance. The penalty factor lambda is set to FLX_MAT by default, 
 * this will cause the clustering continues untill one cluster left, or meets other critera. 
 * The target_cluster_num is also set to 0 following the same logic. 
 */
void ClusterCollection::BottomUpClustering(const Matrix<BaseFloat> &feats, const BaseFloat& lambda, int32 target_cluster_num, const int32& dist_type) {
	this->dist_type_ = dist_type;

	// Give each initial cluster and idx number, to easy manipulation
	std::unordered_map<Cluster*, int32> cluster_idx_map;
	Cluster* curr = this->head_cluster_;
	int32 cluster_idx = 0;
	while(curr){
		cluster_idx_map[curr]=cluster_idx;
		cluster_idx++;
		curr = curr->next;
	}

	// Initiallize distante matrix and cluster active status, to be updated during bottom up clustering
	std::vector<std::vector<BaseFloat>> dist_matrix(this->num_clusters_, std::vector<BaseFloat>(100000.0, num_clusters_));
	std::vector<bool> to_be_updated(this->num_clusters_, true);

	// Compute the penalty BIC critera
	int32 dim = feats.NumCols();
	int32 tot_frames = this->NumFrames();
	int32 init_cluster_num = this->num_clusters_;
	BaseFloat penalty = 0.5 * (dim + dim) * log(tot_frames);

	// Start HAC clustering
	while(this->num_clusters_ > 1) {

		std::vector<Cluster*> min_dist_clusters(2);
		FindMinDistClusters(feats, dist_matrix, to_be_updated, cluster_idx_map, min_dist_clusters);

		if (min_dist_clusters[0]==NULL || min_dist_clusters[1]==NULL) 
				KALDI_ERR << "NULL CLUSTER!";

		BaseFloat dist_glr = DistanceOfTwoClustersGLR(feats, min_dist_clusters[0], min_dist_clusters[1]);
		BaseFloat delta_bic = dist_glr - lambda * penalty;

		if(delta_bic > 0 || this->num_clusters_ <= target_cluster_num) {
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
	return;
}


void ClusterCollection::BottomUpClusteringIvector(const Matrix<BaseFloat> &feats, const Posterior& posterior, const IvectorExtractor& extractor, 
												const BaseFloat& lambda, int32 target_cluster_num, const int32& dist_type ) {
	this->dist_type_ = dist_type;

	// Give each initial cluster and idx number, to easy manipulation
	std::unordered_map<Cluster*, int32> cluster_idx_map;
	Cluster* curr = this->head_cluster_;
	int32 cluster_idx = 0;
	while(curr){
		cluster_idx_map[curr]=cluster_idx;
		cluster_idx++;
		curr = curr->next;
	}

	// Initiallize distante matrix and cluster active status, to be updated during bottom up clustering
	std::vector<std::vector<BaseFloat>> dist_matrix(this->num_clusters_, std::vector<BaseFloat>(100000.0, num_clusters_));
	std::vector<bool> to_be_updated(this->num_clusters_, true);

	// Compute the penalty BIC critera
	int32 dim = feats.NumCols();
	int32 tot_frames = this->NumFrames();
	int32 init_cluster_num = this->num_clusters_;
	BaseFloat penalty = 0.5 * (dim + dim) * log(tot_frames);

	// Start HAC clustering
	while(this->num_clusters_ > 1) {

		std::vector<Cluster*> min_dist_clusters(2);
		FindMinDistClustersIvector(feats,  posterior, extractor, dist_matrix, to_be_updated, cluster_idx_map, min_dist_clusters);

		if (min_dist_clusters[0]==NULL || min_dist_clusters[1]==NULL) 
				KALDI_ERR << "NULL CLUSTER!";

		BaseFloat dist_glr = DistanceOfTwoClustersGLR(feats, min_dist_clusters[0], min_dist_clusters[1]);
		BaseFloat delta_bic = dist_glr - lambda * penalty;

		if(delta_bic > 0 || this->num_clusters_ <= target_cluster_num) {
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
	return;
}


void ClusterCollection::FindMinDistClusters(const Matrix<BaseFloat> &feats, std::vector<std::vector<BaseFloat>>& dist_matrix, std::vector<bool>& to_be_updated, 
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

			}else{
				// to be updated

				switch(this->dist_type_) {
				
				case GLR_DISTANCE:
					dist = DistanceOfTwoClustersGLR(feats, p1, p2);
					break;
				
				case KL2_DISTANCE:
					dist = DistanceOfTwoClustersKL2(feats, p1, p2);
					break;

				default:;
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


void ClusterCollection::FindMinDistClustersIvector(const Matrix<BaseFloat> &feats, const Posterior& posterior, const IvectorExtractor& extractor, 
	std::vector<std::vector<BaseFloat>>& dist_matrix, std::vector<bool>& to_be_updated, 
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

			}else{
				dist = DistanceOfTwoClustersIvectorKL2(feats, p1, p2, posterior, extractor);
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
	BaseFloat dist = log_det12 * cluster12->NumFrames() - log_det1 * cluster1->NumFrames() - log_det2 * cluster2->NumFrames();

	delete cluster12;

	return 0.5*dist;
}


BaseFloat ClusterCollection::DistanceOfTwoClustersKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2) {
	Vector<BaseFloat> mean_vec_1 = Cluster::ComputeMean(feats, cluster1);
	Vector<BaseFloat> mean_vec_2 = Cluster::ComputeMean(feats, cluster2);
	Vector<BaseFloat> cov_vec_1 = Cluster::ComputeCovDiag(feats, cluster1);
	Vector<BaseFloat> cov_vec_2 = Cluster::ComputeCovDiag(feats, cluster2);
	return SymetricKlDistance(mean_vec_1, mean_vec_2, cov_vec_1, cov_vec_2);	
}


BaseFloat ClusterCollection::DistanceOfTwoClustersIvectorKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2,
															const Posterior& posterior, const IvectorExtractor& extractor) {

	Matrix<BaseFloat> feats_clust1, feats_clust2; 
	Cluster::CollectFeatures(feats, cluster1, feats_clust1);	
	Cluster::CollectFeatures(feats, cluster2, feats_clust2);

	Posterior postprob_clust1, postprob_clust2;
	Cluster::CollectPosteriors(posterior, cluster1, postprob_clust1);
	Cluster::CollectPosteriors(posterior, cluster2, postprob_clust2);

	Vector<double> ivector_clust1_mean, ivector_clust2_mean;
	SpMatrix<double> ivector_clust1_covar, ivector_clust2_covar;
	ComputeIvector(feats_clust1, postprob_clust1, extractor, ivector_clust1_mean, ivector_clust1_covar);
	ComputeIvector(feats_clust1, postprob_clust2, extractor, ivector_clust2_mean, ivector_clust2_covar);

	//return  SymetricKlDistance(ivector_clust1_mean, ivector_clust2_mean, ivector_clust1_covar, ivector_clust2_covar);
	return  cosineDistance(ivector_clust1_mean, ivector_clust2_mean);
}


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