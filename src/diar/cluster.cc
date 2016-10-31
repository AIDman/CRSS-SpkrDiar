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


// To Repeat the code in Audioseg
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


// To Repeat the code in Audioseg
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


BaseFloat Cluster::LogDet(const Matrix<BaseFloat> &feats) const {
	Matrix<BaseFloat> feats_collect;
	int32 featdim = feats.NumCols();
	int32 tot_frames = 0;
	int32 insert_frame = 0;
	for(int i=0;i<this->list_.size();i++) {
		Segment seg = this->list_[i];
		int32 seg_size = seg.Size();
		tot_frames += seg_size;
		feats_collect.Resize(tot_frames, featdim);
		feats_collect.Range(insert_frame, seg_size, 0, featdim).CopyFromMat(feats.Range(seg.StartIdx(),seg_size,0,featdim));
		insert_frame += seg_size;
	}

	return logDetCovariance(feats_collect);
}


ClusterCollection::ClusterCollection() {
	num_clusters_ = 0;
	head_cluster_ = NULL;
}


string ClusterCollection::UttID() {
	return this->uttid_;
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


void ClusterCollection::BottomUpClustering(const Matrix<BaseFloat> &feats, int32 target_cluster_num) {
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

	// Start HAC clustering
	while(this->num_clusters_ > target_cluster_num) {
		std::vector<Cluster*> min_dist_clusters(2);
		FindMinDistClusters(feats, dist_matrix, to_be_updated, cluster_idx_map, min_dist_clusters);
		if (min_dist_clusters[0]==NULL || min_dist_clusters[1]==NULL) 
				KALDI_ERR << "NULL CLUSTER!";
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
				dist = dist_matrix[p1_idx][p2_idx];
			}else{
				dist = DistanceOfTwoClustersGLR(feats, p1, p2);
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
	KALDI_LOG << min_dist << " " << min_dist_clusters[0]->NumFrames() << " " << min_dist_clusters[0]->LogDet(feats) << " " << min_dist_clusters[1]->NumFrames() << " " << min_dist_clusters[1]->LogDet(feats);
	return;
}

/*
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
	//KALDI_LOG << dist << " " << cluster12->NumFrames() << " " << log_det12 << " " << cluster1->NumFrames() << " " << log_det1 << " " << cluster2->NumFrames() << " " << log_det2;
	//KALDI_LOG <<  log_det12 << " " << cluster12->NumFrames() ;
	//KALDI_LOG <<  log_det1 << " " << log_det2 << " " << cluster2->NumFrames() << " IDX: " << cluster2->AllSegments()[0].StartIdx() << " " << cluster2->AllSegments()[0].EndIdx();
	//KALDI_LOG << feats.Range(cluster2->AllSegments()[0].StartIdx(), cluster2->NumFrames(), 0, feats.NumCols());

	delete cluster12;

	return 0.5*dist;
}
*/

// Repeat GLR distance calculation in Audioseg. Does not seem to correct mathematically, but performs better..
BaseFloat ClusterCollection::DistanceOfTwoClustersGLR(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2) {
	BaseFloat log_det1 = cluster1->LogDet(feats);
	BaseFloat log_det2 = cluster2->LogDet(feats);

	Vector<BaseFloat> m1 = Cluster::ComputeSum(feats, cluster1);
	Vector<BaseFloat> m2 = Cluster::ComputeSum(feats, cluster2);
	Vector<BaseFloat> v1 = Cluster::ComputeVarSum(feats, cluster1);
	Vector<BaseFloat> v2 = Cluster::ComputeVarSum(feats, cluster2);

	int32 n = cluster1->NumFrames() + cluster2->NumFrames();
	BaseFloat z = 1.0 / n;
	BaseFloat zz = z * z;

	Vector<BaseFloat> mean_sum(feats.NumCols());
	mean_sum.AddVec(1.0, m1);
	mean_sum.AddVec(1.0, m2);

	Vector<BaseFloat> cov_sum(feats.NumCols());
	cov_sum.AddVec(z, v1);
	cov_sum.AddVec(z, v2);
	cov_sum.AddVec2(-1.0*zz, mean_sum);

	BaseFloat s = 0.0;
	for(int i = 0; i<feats.NumCols(); i++) {
		s += log(cov_sum(i));
	}

	BaseFloat dist = n*s - log_det1 * cluster1->NumFrames() - log_det2 * cluster2->NumFrames();

	return 0.5*dist;
}

BaseFloat ClusterCollection::DistanceOfTwoClustersKL2(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2) {

	Vector<BaseFloat> mean_vec_1 = Cluster::ComputeMean(feats, cluster1);
	Vector<BaseFloat> mean_vec_2 = Cluster::ComputeMean(feats, cluster2);
	Vector<BaseFloat> cov_vec_1 = Cluster::ComputeCovDiag(feats, cluster1);
	Vector<BaseFloat> cov_vec_2 = Cluster::ComputeCovDiag(feats, cluster2);
	Vector<BaseFloat> diff_mean(feats.NumCols());
	diff_mean.AddVec(1.0, mean_vec_1);
	diff_mean.AddVec(-1.0, mean_vec_2);
	Vector<BaseFloat> inv_cov_1(cov_vec_1), inv_cov_2(cov_vec_2);
	inv_cov_1.InvertElements();
	inv_cov_2.InvertElements();

	BaseFloat dist = 0.0;
	for (size_t i = 0; i < feats.NumCols(); i++) {
		dist += 0.5 * ((cov_vec_1(i) * inv_cov_2(i) + cov_vec_2(i) * inv_cov_1(i)) + diff_mean(i) * diff_mean(i) * (inv_cov_1(i) + inv_cov_2(i)));
	}

	return dist;	
}

/*
BaseFloat ClusterCollection::DistanceOfTwoClustersGLR(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2) {

	BaseFloat log_det1 = cluster1->LogDet(feats);
	BaseFloat log_det2 = cluster2->LogDet(feats);

	Matrix<BaseFloat> feats_collect;
	std::vector<Segment> cluster1_segments = cluster1->AllSegments();
	std::vector<Segment> cluster2_segments = cluster2->AllSegments();
	int32 featdim = feats.NumCols();
	int32 tot_frames = 0;
	int32 insert_frame = 0;
	for(int i=0;i<cluster1_segments.size();i++) {
		Segment seg = cluster1_segments[i];
		int32 seg_size = seg.Size();
		tot_frames += seg_size;
		feats_collect.Resize(tot_frames, featdim);
		feats_collect.Range(insert_frame, seg_size, 0, featdim).CopyFromMat(feats.Range(seg.StartIdx(),seg_size,0,featdim));
		insert_frame += seg_size;
	}
	for(int i=0;i<cluster2_segments.size();i++) {
		Segment seg = cluster2_segments[i];
		int32 seg_size = seg.Size();
		tot_frames += seg_size;
		feats_collect.Resize(tot_frames, featdim);
		feats_collect.Range(insert_frame, seg_size, 0, featdim).CopyFromMat(feats.Range(seg.StartIdx(),seg_size,0,featdim));
		insert_frame += seg_size;
	}
	

	BaseFloat log_det12 = logDetCovariance(feats_collect);
	BaseFloat dist = log_det12 * tot_frames - log_det1 * cluster1->NumFrames() - log_det2 * cluster2->NumFrames();
	//KALDI_LOG << dist << " " << tot_frames << " " << log_det12 << " " << cluster1->NumFrames() << " " << log_det1 << " " << cluster2->NumFrames() << " " << log_det2;
	return dist*0.5;
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