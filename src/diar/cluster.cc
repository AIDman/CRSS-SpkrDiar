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

BaseFloat Cluster::LogDet(const Matrix<BaseFloat> &feats) const {
	Matrix<BaseFloat> feats_collect;
	int32 featdim = feats.NumCols();
	int32 tot_frames = 0;
	int32 insert_frame = 0;
	for(int i=0;i<list_.size();i++) {
		Segment seg = list_[i];
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
	while(this->num_clusters_ > target_cluster_num) {
		std::vector<Cluster*> min_dist_clusters(2);
		FindMinDistClusters(feats, min_dist_clusters);
		if (min_dist_clusters[0]==NULL || min_dist_clusters[1]==NULL) 
				KALDI_ERR << "NULL CLUSTER!";
		MergeClusters(min_dist_clusters[0], min_dist_clusters[1]);
		this->num_clusters_--;
	}
}


void ClusterCollection::FindMinDistClusters(const Matrix<BaseFloat> &feats, std::vector<Cluster*> &min_dist_clusters) {
	if(num_clusters_<2) KALDI_ERR << "Less than two clusters, could not find min dist clusters";
	Cluster* p1 = this->head_cluster_;
	BaseFloat min_dist = FLT_MAX; // set a random large number
	while(p1){
		Cluster* p2 = p1->next;
		while(p2) {
			BaseFloat dist = DistanceOfTwoClusters(feats, p1, p2);
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


BaseFloat ClusterCollection::DistanceOfTwoClusters(const Matrix<BaseFloat> &feats, const Cluster* cluster1, const Cluster* cluster2) {

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

}