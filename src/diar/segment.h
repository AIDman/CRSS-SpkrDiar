#ifndef KALDI_SRC_DIAR_SEGMENTS_H_
#define KALDI_SRC_DIAR_SEGMENTS_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "hmm/posterior.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "ivector/plda.h"
#include "diar-utils.h"


namespace kaldi{

class Segment {
public:
	Segment();
	Segment(const int32 start, const int32 end);
	Segment(const std::string label, const int32 start, const int32 end);
	std::string Label(); // return cluster label
	int32 StartIdx();  // return start frame index of segment
	int32 EndIdx();	// return last frame index of segment
	int32 Size(); // return the frame length of the segment
	Vector<double> Ivector(); // return ith i-vector
	void SetLabel(std::string label);
	void SetIvector(Vector<double>& ivec);	
	void SetIvector(const Matrix<BaseFloat>& feats, 
						   const Posterior& posteriors, 
						   const IvectorExtractor& extractor);	

private:
	std::string label_;
	int32 start_;
	int32 end_;
	Vector<double> ivector_;
};

// Segments are collection of segment, and the operations on those segments.
class SegmentCollection {
public:
	SegmentCollection();
	SegmentCollection(const std::string uttid);
	SegmentCollection(const Vector<BaseFloat>& frame_labels, const std::string uttid);

	int32 Size() const;
	std::string UttID();
	//void ToLabels(Vector<BaseFloat>&);
	void ToRTTM(const std::string& uttid, const std::string& rttmName);
	Segment* KthSegment(int32 k);
	SegmentCollection GetSpeechSegments();
	SegmentCollection GetLargeSegments(int32 min_seg_len);
	/*
	void ExtractIvectors(const Matrix<BaseFloat>& feats,
								 const Posterior& posterior,
								 const IvectorExtractor& extractor);
	void GetSegmentIvector(const Matrix<BaseFloat>& segFeats, 
						   const Posterior& segPosterior, 
						   const IvectorExtractor& extractor,
						   Segment& seg);

	void NormalizeIvectors();
	*/
	void Append(Segment* seg);
	void Read(const std::string& segments_rxfilename);
	void Write(const std::string& segments_dirname);
	/*
	void ReadIvectors(const std::string& ivector_rxfilename); 
	void WriteIvectors(const std::string& ivector_wxfilename); 
	*/

private:
	std::vector<Segment*> segment_list_;
	std::string uttid_;
	std::vector< Vector<double> > ivector_list_; 
}; 


}
#endif
