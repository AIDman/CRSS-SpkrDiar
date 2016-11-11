#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include "segment.h"


namespace kaldi {

// Segment Class Implementations
Segment::Segment(){}

Segment::Segment(const int32 start, const int32 end) {
	this->label_ = "";
	this->start_ = start;
	this->end_ = end;
	this->size_ = end - start + 1;
	mask_.Resize(this->size_);
	mask_.Set(1.0);
}

Segment::Segment(const std::string label, const int32 start, const int32 end) {
	this->label_ = label;
	this->start_ = start;
	this->end_ = end;
	this->size_ = end - start + 1;
	mask_.Resize(this->size_);
	mask_.Set(1.0);
}

std::string Segment::Label() {
	return this->label_;
}

int32 Segment::StartIdx() {
	return this->start_;
}

int32 Segment::EndIdx() {
	return this->end_;
} 

int32 Segment::Size() {
	return this->size_;
}

void Segment::SetLabel(std::string label) {
	this->label_ = label;
}

void Segment::SetIvector(Vector<double>& ivec) {
	this->ivector_ = ivec;
}


void Segment::SetIvector(const Matrix<BaseFloat>& feats, 
						   const Posterior& posterior, 
						   const IvectorExtractor& extractor) {
	// post probs of segment
 	Posterior::const_iterator start_iter = posterior.begin() + this->start_;
	Posterior::const_iterator end_iter = posterior.begin() + this->end_ + 1;
	Posterior seg_posterior(start_iter, end_iter);

	// feature of segment
	int32 dim = feats.NumCols();
	int32 nframes = this->end_ - this->start_ + 1;
	Matrix<BaseFloat> seg_feats(nframes, dim);
	int insert_pos = 0;
	for(int32 idx = this->start_; idx <= this->end_; idx++) {
		seg_feats.Row(insert_pos).CopyFromVec(feats.Row(idx));
		insert_pos++;
	}

	// ivector extraction
    bool need_2nd_order_stats = false;
    IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
                                             extractor.FeatDim(),
                                             need_2nd_order_stats);
    utt_stats.AccStats(seg_feats, seg_posterior);
    this->ivector_.Resize(extractor.IvectorDim());
    this->ivector_(0) = extractor.PriorOffset();
    extractor.GetIvectorDistribution(utt_stats, &ivector_, NULL);
    return;
}


Vector<BaseFloat> Segment::Mask() {
	return this->mask_;
}

Vector<double> Segment::Ivector() {
	return this->ivector_;
}


// SegmentCollection Class Implementations
SegmentCollection::SegmentCollection(){}

SegmentCollection::SegmentCollection(const std::string uttid){
	this->uttid_ = uttid;
}


SegmentCollection::SegmentCollection(const Vector<BaseFloat>& frame_labels, const std::string uttid) {
	// NOTE: The rules of input label is as follow
	// -1 -> overlap
	// 0 -> nonspeech
	// 1,2,..,n -> speaker1, speaker2, speakern   
	// Other conditions may be added in the future.
	this->uttid_ = uttid;
	int32 state;
	int32 prevState;
	std::vector<int32> segmentStartEnd;
	int32 startSeg = 0;
	int32 endSeg = 0;
	segmentStartEnd.push_back(startSeg);
	segmentStartEnd.push_back(endSeg);
	for (size_t i=1; i<frame_labels.Dim(); i++) {
		state = frame_labels(i);
		prevState = frame_labels(i-1);
		if (state != prevState || i==frame_labels.Dim()-1) {
			if (i==frame_labels.Dim()-1) {
				i++;
			}
			if (prevState == -1) {
				segmentStartEnd[1] = i-1;
				Segment* new_seg = new Segment("overlap",segmentStartEnd[0],segmentStartEnd[1]);
				this->segment_list_.push_back(new_seg);
				segmentStartEnd[0] = i;
			}else if (prevState == 0) {
				segmentStartEnd[1] = i-1;
				Segment* new_seg = new Segment("nonspeech",segmentStartEnd[0],segmentStartEnd[1]);
				this->segment_list_.push_back(new_seg);
				segmentStartEnd[0] = i;				
			}else if (prevState > 0) {
				std:: string stateStr = numberToString(prevState);
				segmentStartEnd[1] = i-1;
				Segment* new_seg = new Segment(stateStr,segmentStartEnd[0],segmentStartEnd[1]);
				this->segment_list_.push_back(new_seg);
				segmentStartEnd[0] = i;
			} 
		}
	}
}


std::string SegmentCollection::UttID() {
	return this->uttid_;
}


int32 SegmentCollection::Size() const {
	return int32(this->segment_list_.size());
}

/*
Segment SegmentCollection::GetSeg(int32 index) const {
	return this->segment_list_[index];
}
*/


/*
void Segments::ToLabels(Vector<BaseFloat>& labels){
	// NOTE: The rules of input label is as follow
	// -1 -> overlap
	// 0 -> nonspeech
	// 1,2,..,n -> speaker1, speaker2, speakern   
	// Other conditions may be added in the future. 
	std::vector<int32> lastSegment = this->segment_list_.back().second;
	labels.Resize(lastSegment[1]+1);
	for (size_t i=0; i< this->segment_list_.size(); i++) {
		std::vector<int32> segmentStartEnd = this->segment_list_[i].second;
		int32 segLen = segmentStartEnd[1] - segmentStartEnd[0] + 1;
		Vector<BaseFloat> segLabels(segLen);		
		if (this->segment_list_[i].first == "nonspeech") { 
			segLabels.Set(0.0);
		}else if (this->segment_list_[i].first == "overlap") {
			segLabels.Set(-1.0);
		}else {
			int32 stateInt = std::atoi(this->segment_list_[i].first.c_str());
			segLabels.Set(stateInt);
		}
		labels.Range(segmentStartEnd[0],segLen).CopyFromVec(segLabels);
	}
}
*/


void SegmentCollection::ToRTTM(const std::string& uttid, const std::string& rttmName) {
	std::ofstream fout;
	fout.open(rttmName.c_str());
	for (size_t i =0; i < this->segment_list_.size(); i++){
		std::string spkrID = this->segment_list_[i]->Label();
		BaseFloat segStart = FrameIndexToSeconds(this->segment_list_[i]->StartIdx());
		BaseFloat segLength = FrameIndexToSeconds(this->segment_list_[i]->EndIdx()) - segStart;
		fout << "SPEAKER ";
		fout << uttid << " ";
		fout << 1 << " ";
		fout << std::fixed << std::setprecision(3);
		fout << segStart << " ";
		fout << segLength << " ";
		fout << "<NA> <NA> ";
		fout << spkrID << " ";
		fout << "<NA>\n";
	}
	fout.close();
}


SegmentCollection SegmentCollection::GetSpeechSegments() {
	SegmentCollection speechSegments(this->uttid_);
	for (size_t i = 0; i < this->segment_list_.size(); i++) {
		if (this->segment_list_[i]->Label() != "nonspeech" && this->segment_list_[i]->Label() != "overlap") {
			speechSegments.Append(this->segment_list_[i]);
		}
	}
	return speechSegments;
}

SegmentCollection SegmentCollection::GetLargeSegments(int32 min_seg_len) {
	SegmentCollection largeSegments(uttid_);
	for (size_t i = 0; i < this->segment_list_.size(); i++) {
		Segment* curr_seg = this->segment_list_[i];
		int32 curr_seg_size = curr_seg->EndIdx() - curr_seg->StartIdx() + 1;
		if (curr_seg_size >= min_seg_len) {
			largeSegments.Append(this->segment_list_[i]);
		}
	}
	return largeSegments;
}


Segment* SegmentCollection::KthSegment(int32 k) {
	return segment_list_[k];
}


/*
void SegmentCollection::ExtractIvectors(const Matrix<BaseFloat>& feats,
							   const Posterior& posterior,
							   const IvectorExtractor& extractor) {
	int32 dim = extractor.FeatDim();
	size_t num_segs = this->segment_list_->size();
	for (size_t i=0; i < num_segs; i++){
		std::string segment_label = this->segment_list_[i]->Label();
		Matrix<BaseFloat> seg_feats(this->segment_list_[i]->EndIdx() - this->segment_list_[i]->StartIdx() +1, dim);
		seg_feats.CopyFromMat(feats.Range(this->segment_list_[i]->StartIdx(), this->segment_list_[i]->EndIdx() - this->segment_list_[i]->StartIdx() + 1, 0, dim));
		Posterior::const_iterator start_iter = posterior.begin() + this->segment_list_[i]->StartIdx();
		Posterior::const_iterator end_iter = posterior.begin() + this->segment_list_[i]->EndIdx() + 1;
		Posterior seg_posterior(start_iter, end_iter);
		GetSegmentIvector(seg_feats, seg_posterior, extractor, this->segment_list_[i]);
	}
	return;
}


void SegmentCollection::GetSegmentIvector(const Matrix<BaseFloat>& segFeats, 
							     const Posterior& segPosterior, 
							     const IvectorExtractor& extractor, Segment& seg) {
	Vector<double> ivector;
    bool need_2nd_order_stats = false;
    IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
                                             extractor.FeatDim(),
                                             need_2nd_order_stats);
    utt_stats.AccStats(segFeats, segPosterior);
    ivector.Resize(extractor.IvectorDim());
    ivector(0) = extractor.PriorOffset();
    extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);
    seg.SetIvector(ivector);
    ivector_list_.push_back(ivector);
    return;
}


void SegmentCollection::NormalizeIvectors() {
	// NOTE: Add variance normalization to this function. 
	Vector<double> ivectorMean;
	computeMean(this->ivector_list_, ivectorMean);
	//SpMatrix<double> ivectorCovariance = computeCovariance(this->ivector_list_, ivectorMean);
	for (size_t i = 0; i < this->ivector_list_.size(); i++) {
		this->ivector_list_[i].AddVec(-1, ivectorMean);
	}
}
*/

void SegmentCollection::Append(Segment* seg) {
	this->segment_list_.push_back(seg);
}


void SegmentCollection::Read(const std::string& segments_rxfilename) {
	// segments_rxfilename contains only segments information from single audio stream.
    Input ki(segments_rxfilename);  // no binary argment: never binary.
    std::string line;
    /* read each line from segments file */
    while (std::getline(ki.Stream(), line)) {
		std::vector<std::string> split_line;
		// Split the line by space or tab and check the number of fields in each
		// line. There must be 3 fields--reacording wav file name,
		// start time, end time; 4th field (speaker ID info) is optional.
		SplitStringToVector(line, " \t\r", true, &split_line);
		if (split_line.size() != 3 && split_line.size() != 4) {
			KALDI_WARN << "Invalid line in segments file: " << line;
			continue;
		}

		std::string recording = split_line[0],
		start_str = split_line[1],
		end_str = split_line[2];

		if (this->uttid_.empty()) {
			this->uttid_ = recording;
		}else {
			if (this->uttid_ != recording) {
				KALDI_ERR << "Only one audio stream is permitted per segment file.";
			}
		}
		  
		// Convert the start time and endtime to real from string. Segment is
		// ignored if start or end time cannot be converted to real.
		double start, end;
		if (!ConvertStringToReal(start_str, &start)) {
			KALDI_WARN << "Invalid line in segments file [bad start]: " << line;
			continue;
		}
		if (!ConvertStringToReal(end_str, &end)) {
			KALDI_WARN << "Invalid line in segments file [bad end]: " << line;
			continue;
		}
		// start time must not be negative; start time must not be greater than
		// end time, except if end time is -1
		if (start < 0 || (end != -1.0 && end <= 0) || ((start > end) && (end > 0))) {
			KALDI_WARN << "Invalid line in segments file [empty or invalid segment]: "
		           << line;
		continue;
		}
		std::string spkrLabel = "unk";  // default speaker label is unknown.
		// if each line has 4elements then 4th element must be speaker label
		if (split_line.size() == 4) {
			spkrLabel = split_line[3];
		}

		std::vector<int32> segStartEnd;
		Segment* seg = new Segment(spkrLabel, SecondsToFrameIndex(BaseFloat(start)),SecondsToFrameIndex(BaseFloat(end)));
		this->segment_list_.push_back(seg);
	}	
}


void SegmentCollection::Write(const std::string& segments_dirname) {
	std::string segments_wxfilename = segments_dirname + "/" + this->uttid_ + ".seg";
	std::string segments_scpfilename = segments_dirname + "/" + "segments.scp";
	std::ofstream fout;
	std::ofstream fscp;
	fout.open(segments_wxfilename.c_str());
	fscp.open(segments_scpfilename.c_str(), std::ios::app);
	for (size_t i =0; i < this->segment_list_.size(); i++){
		std::string spkrLabel = this->segment_list_[i]->Label();
		BaseFloat segStart = FrameIndexToSeconds(this->segment_list_[i]->StartIdx());
		BaseFloat segEnd = FrameIndexToSeconds(this->segment_list_[i]->EndIdx());
		fout << this->uttid_ << " ";
		fout << std::fixed << std::setprecision(3);
		fout << segStart << " ";
		fout << segEnd << " ";
		fout << spkrLabel << "\n";
	}
	fscp << segments_wxfilename << "\n";
	fscp.close();
	fout.close();
}

/*
void SegmentCollection::ReadIvectors(const std::string& ivector_rxfilename) {
	SequentialDoubleVectorReader ivector_reader(ivector_rxfilename);
	size_t i = 0;
    for (; !ivector_reader.Done(); ivector_reader.Next()) {
        std::string ivectorKey = ivector_reader.Key();
        if(i<this->segment_list_.size()) segment_list_[i++].SetIvector(ivector_reader.Value());   
        this->ivector_list_.push_back(ivector_reader.Value());
    }
    if (this->ivector_list_.size() != this->segment_list_.size()) {
    	KALDI_ERR << "Number of ivectors doesn't match number of segments!";
    }
}

void Segments::WriteIvectors(const std::string& ivector_wxfilename){
	DoubleVectorWriter ivectorWriter(ivector_wxfilename);
	for (size_t i =0; i < this->segment_list_.size(); i++){
		std::string segID = makeSegKey(this->segment_list_[i].second, this->uttid_);
		if (this->ivector_list_.empty()) {
			KALDI_ERR << "ivector list for " << this->uttid_ << " Segments do not exist!"; 
		}
		ivectorWriter.Write(segID, this->ivector_list_[i]);
	}
}
*/

}


