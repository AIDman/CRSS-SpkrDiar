#include "cluster.h"

namespace kaldi {


Cluster::Cluster(Segment& one_segment){
	this->label_ = Cluster::prefix + std::to_string(Cluster::id_generator++);
	this->list_.push_back(one_segment);
	this->frames_ =  one_segment.Size();
}


void Cluster::AddSegment(const Segment& new_segment) {
	this->list_.push_back(new_segment);
	this->frames_ += new_segment.Size();
}


std::vector<Segment> Cluster::AllSegments(){
	return this->list_;
}


std::string Cluster::Label() {
	return this->label_;
}


int32 Cluster::NumFrames() {
	return this->_frames;
}


BaseFloat Cluster::LogDet(const Matrix<BaseFloat> &feats) {
	Matrix<BaseFloat> feats_collect;
	int32 featdim = feats.NumCols();
	int32 tot_frames = 0;
	int32 insert_frame = 0;
	for(int i=0;i<list_.size();i++) {
		Segment seg = list_[i];
		seg_size = seg.Size();
		tot_frame += seg_size;
		feats_collect.Resize(tot_frame. featdim);
		feats_collect(insert_frame, seg_size, 0, dim).CopyFromMat(feats.Range(seg.Startidx(),seg_size,0,dim));
		insert_frame += seg_size;
	}

	return logDetCovariance(feats_collect);
}


ClusterCollection::ClusterCollection() {
	num_clusters_ = 0;
	head_cluster_ = NULL;
}


void ClusterCollection::InitFromNonLabeledSegments(const SegmentCollection& non_clustered_segments) {
	 std::int32 num_segments = NonClusteredSegments.Size();
	 if(num_segments < 1) KALDI_ERR << "Clusters could not be initialized from empty segments";
	 Cluster* prev_cluster = NULL; 
	 for(int32 i=0; i<num_segments;i++){
	 	if(i==0) {
	 		Cluster* head_cluster_ = new Cluster(non_clustered_segments[i]);
	 		head_cluster_->prev = NULL;
	 		prev_cluster = head_cluster_;
	 		continue;
	 	} 

	 	Cluster* new_cluster = new Cluster(non_clustered_segments[i]);
	 	prev_cluster->next = new_cluster;
	 	new_cluster->prev = prev_cluster;
	 	if(i==num_segments-1) new_cluster=NULL;
	 }

	 this->num_clusters_ = num_segments;
	 return;
}


Cluster* ClusterCollection::Head() {
	return this->head_cluster_;
}


void ClusterCollection::BottomUpClustering(const Matrix<BaseFloat> &feats, int32 target_cluster_num) {
	while(num_clusters_ > target_cluster_num) {
		vector<Cluster*> min_dist_clusters(2);
		FindMinDistClusters(min_dist_clusters);
		MergeClusters(min_dist_clusters[0], min_dist_clusters[1]);
	}
}


void ClusterCollection::FindMinDistClusters(const Matrix<BaseFloat> &feats, const vector<Cluster*> &min_dist_clusters) {
	if(num_clusters_<2) KALDI_ERR << "Less than two clusters, could not find min dist clusters";
	Cluster* p1 = this->head_cluster_;
	Cluster* p2 = p1->next;
	BaseFloat min_dist = 100000.0; // set a random large number
	while(p1){
		if(p1->next==NULL) break;
		p2 = p1->next;
		while(p2) {
			BaseFloat dist = DistanceOfTwoClusters(p1, p2);
			if(dist<min_dist) {
				mid_dist = dist;
				mid_dist_clusters[0] = p1;
				min_dist_clusters[1] = p2;
			}
		}
	}
	return;
}


BaseFloat ClusterCollection::DistanceOfTwoClusters(const Matrix<BaseFloat> &feats, Cluster* cluster1, Cluster* cluster2) {

	Cluster cluster1_copy(*cluster1);
	Cluster cluster2_copy(*cluster2);
	ClusterCollection::MergeClusters(&cluster1_copy,&cluster2_copy);
	Cluster* cluster12 = &cluster1_copy;

	BaseFloat log_det12 = cluster12->LogDet(feats);
	BaseFloat log_det1 = cluster1->LogDet(feats);
	BaseFloat log_det2 = cluster2->LogDet(feats);

	BaseFloat dist = log_det12 * cluster12->NumFrames() - log_det1 * cluster1->NumFrames() - log_det2 * cluster2->NumFrames();

	delete [] &cluster1_copy, &cluster2_copy, cluster_1_and_2;

	return 0.5*dist;
}


static void ClusterCollection::MergeClusters(Cluster* clust1, Cluster* clust2) {
	vector<Segment> clust2_segments = clust2.AllSegments();
	for(int i=0; i<clust2_segments.size();i++) {
		clust1.AddSegment(clust2_segments[i]);
	}

	if(clust2->prev) clust2->prev->next = clust2->next;
	if(clust2->next) clust2->next->prev = clust2->prev; 

	delete [] clust2;
}


}