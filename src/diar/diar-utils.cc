#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include "diar-utils.h"


namespace kaldi {

// Segment Class Implementations
Segment::Segment(){}

Segment::Segment(const int32 start, const int32 end) {
	this->label_ = "";
	this->start_ = start;
	this->end_ = end;
}

Segment::Segment(const std::string label, const int32 start, const int32 end) {
	this->label_ = label;
	this->start_ = start;
	this->end_ = end;
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
	return this->end_ - this->start_ + 1;
}

void Segment::SetLabel(std::string label) {
	this->label_ = label;
}

void Segment::SetIvector(Vector<double> ivec) {
	this->ivector_ = ivec;
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
				Segment new_seg("overlap",segmentStartEnd[0],segmentStartEnd[1]);
				this->segment_list_.push_back(new_seg);
				segmentStartEnd[0] = i;
			}else if (prevState == 0) {
				segmentStartEnd[1] = i-1;
				Segment new_seg("nonspeech",segmentStartEnd[0],segmentStartEnd[1]);
				this->segment_list_.push_back(new_seg);
				segmentStartEnd[0] = i;				
			}else if (prevState > 0) {
				std:: string stateStr = numberToString(prevState);
				segmentStartEnd[1] = i-1;
				Segment new_seg(stateStr,segmentStartEnd[0],segmentStartEnd[1]);
				this->segment_list_.push_back(new_seg);
				segmentStartEnd[0] = i;
			} 
		}
	}
}


/*
Segment SegmentCollection::Begin() {
	return this->segment_list_[0];
}


segUnit SegmentCollection::End() {
	return this->segment_list_.back();
}
*/


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
		std::string spkrID = this->segment_list_[i].Label();
		BaseFloat segStart = FrameIndexToSeconds(this->segment_list_[i].StartIdx());
		BaseFloat segLength = FrameIndexToSeconds(this->segment_list_[i].EndIdx()) - segStart;
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
		if (this->segment_list_[i].Label() != "nonspeech" && this->segment_list_[i].Label() != "overlap") {
			speechSegments.Append(this->segment_list_[i]);
		}
	}
	return speechSegments;
}

SegmentCollection SegmentCollection::GetLargeSegments(int32 min_seg_len) {
	SegmentCollection largeSegments(uttid_);
	for (size_t i = 0; i < this->segment_list_.size(); i++) {
		Segment curr_seg = this->segment_list_[i];
		int32 curr_seg_size = curr_seg.EndIdx() - curr_seg.StartIdx() + 1;
		if (curr_seg_size >= min_seg_len) {
			largeSegments.Append(this->segment_list_[i]);
		}
	}
	return largeSegments;
}


void SegmentCollection::ExtractIvectors(const Matrix<BaseFloat>& feats,
							   const Posterior& posterior,
							   const IvectorExtractor& extractor) {
	int32 featsDim = extractor.FeatDim();
	size_t numSegs = this->segment_list_.size();
	for (size_t i=0; i < numSegs; i++){
		std::string segmentLabel = this->segment_list_[i].Label();
		Matrix<BaseFloat> segFeats(this->segment_list_[i].EndIdx() - this->segment_list_[i].StartIdx() +1, featsDim);
		segFeats.CopyFromMat(feats.Range(this->segment_list_[i].StartIdx(), this->segment_list_[i].EndIdx() - this->segment_list_[i].StartIdx() + 1, 0, featsDim));

		Posterior::const_iterator startIter = posterior.begin() + this->segment_list_[i].StartIdx();
		Posterior::const_iterator endIter = posterior.begin() + this->segment_list_[i].EndIdx() + 1;
		Posterior segPosterior(startIter, endIter);
		
		//KALDI_LOG << " Segment Range : segmentStartEnd[0]" << " <-> " << segmentStartEnd[1] << " The seg size is: " << segPosterior.size();
		GetSegmentIvector(segFeats, segPosterior, extractor, this->segment_list_[i]);
	}
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


void SegmentCollection::Append(Segment& seg) {
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
		// line. There must be 4 fields--segment name , reacording wav file name,
		// start time, end time; 5th field (speaker ID info) is optional.
		SplitStringToVector(line, " \t\r", true, &split_line);
		if (split_line.size() != 4 && split_line.size() != 5) {
			KALDI_WARN << "Invalid line in segments file: " << line;
			continue;
		}

		std::string segment = split_line[0],
		recording = split_line[1],
		start_str = split_line[2],
		end_str = split_line[3];

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
		// if each line has 5 elements then 5th element must be speaker label
		if (split_line.size() == 5) {
			spkrLabel = split_line[4];
		}

		std::vector<int32> segStartEnd;
		Segment seg(spkrLabel, SecondsToFrameIndex(BaseFloat(start)),SecondsToFrameIndex(BaseFloat(end)));
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
		//std::string segID = makeSegKey(this->segment_list_[i].second, this->uttid_);
		std::string segID = this->uttid_;
		std::string spkrLabel = this->segment_list_[i].Label();
		BaseFloat segStart = FrameIndexToSeconds(this->segment_list_[i].StartIdx());
		BaseFloat segEnd = FrameIndexToSeconds(this->segment_list_[i].EndIdx());
		fout << segID << " ";
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



BaseFloat FrameIndexToSeconds(int32 frame) {
	// Find corresponding start point (in seconds) of a given frame.
	return frame*FRAMESHIFT;
}

/*
std::string makeSegKey(const std::vector<int32>& segmentStartEnd, 
				const std::string uttid) { 
	// Make unique key for each segment of each utterance, by concatenating uttid with segment start and end
	// Such that the key is format of "uttid_segStartFrame_segEndFrame".
	std::string segStartEndString;
	std::stringstream tmp; 
	tmp << segmentStartEnd[0];
	tmp << "_";
	tmp << segmentStartEnd[1];
	segStartEndString = tmp.str();
	std::string	segID = uttid + "_" + segStartEndString;

	return segID;       
}
*/


std::vector<std::string>& split(const std::string& s, 
								char delim, 
								std::vector<std::string>& elems) {
	// Split string by delimiter e.g., ' ', ','. 
	// PASS output vector by reference.
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
    return elems;
}


std::vector<std::string> split(const std::string& s, char delim) {
	// Split string by delimiter, e.g. ',' 
	// CREATE output vector.
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


std::vector<std::string> returnNonEmptyFields(const std::vector<std::string>& fields) {
	// Return non empty elements of vector of strings.
	std::vector<std::string> nonEmptyFields; 
	for(size_t i = 0; i < fields.size(); i++){
		if(fields[i] != ""){
			nonEmptyFields.push_back(fields[i]);
		}
	}
	return nonEmptyFields;
}


void computeDistanceMatrix(const std::vector< Vector<double> >& vectorList, 
							Matrix<BaseFloat>& distanceMatrix,
							const std::vector< Vector<double> >& backgroundIvectors,
							const std::vector< std::string >& backgroundIvectorLabels) {
	distanceMatrix.Resize(vectorList.size(),vectorList.size());
	// Calculate total mean and covariance:
	Vector<double> vectorMean;
	computeMean(vectorList, vectorMean);
	SpMatrix<double> vectorCovariance = computeCovariance(vectorList, vectorMean);
	SpMatrix<double> withinCovariance = computeWithinCovariance(backgroundIvectors,
																backgroundIvectorLabels);

	// Compute PLDA model from background i-vectors:
	Plda plda;
	estimatePLDA(backgroundIvectors, backgroundIvectorLabels, plda);


	for (size_t i=0; i<vectorList.size();i++){
		for (size_t j=0;j<vectorList.size();j++){
			if (i == j){
				distanceMatrix(i,j) = 0;
			}else{
				// distanceMatrix(i,j) = mahalanobisDistance(vectorList[i], 
				// 										  vectorList[j], 
				// 										  vectorCovariance);
				// distanceMatrix(i,j) = conditionalBayesDistance(vectorList[i], 
				// 										 	   vectorList[j], 
				// 											   withinCovariance);
				// distanceMatrix(i,j) = 1 - cosineDistance(vectorList[i],vectorList[j]);
				distanceMatrix(i,j) = pldaScoring(vectorList[i],vectorList[j],plda);
			}
		}
	}
}


BaseFloat mahalanobisDistance(const Vector<double>& v1, const Vector<double>& v2, 
							  const SpMatrix<double>& cov) {

	Vector<double> iv1(v1.Dim());
	iv1.CopyFromVec(v1);
	Vector<double> iv2(v2.Dim());
	iv2.CopyFromVec(v2);
	SpMatrix<double> Sigma(v2.Dim());
	Sigma.CopyFromSp(cov);
	Sigma.Invert();
	iv1.AddVec(-1.,iv2);

	// Now, calculate the quadratic term: (iv1 - iv2)^T Sigma (iv1-iv2)
	Vector<double> S_iv1(iv1.Dim());
	S_iv1.SetZero();
	S_iv1.AddSpVec(1.0, Sigma, iv1, 0.0);
	return sqrt(VecVec(iv1, iv1));
}


BaseFloat cosineDistance(const Vector<double>& v1, const Vector<double>& v2) {
	 BaseFloat dotProduct = VecVec(v1, v2);
	 BaseFloat norm1 = VecVec(v1, v1) + FLT_EPSILON;
	 BaseFloat norm2 = VecVec(v2, v2) + FLT_EPSILON;

	 return dotProduct / (sqrt(norm1)*sqrt(norm2));  
}


BaseFloat conditionalBayesDistance(const Vector<double>& v1, const Vector<double>& v2, 
									const SpMatrix<double>& withinCov) {
	// Distance matrix suggested by Rouvier and Meignier (Odyssey 12). 
	// This measure is computes a mahalanobis distance while assuming 
	// a similar within-class covariance for all speakers in development data. 
	// Different from regular mahalanobis distance in the covariance. 
	return mahalanobisDistance(v1, v2, withinCov);
}

BaseFloat pldaScoring(const Vector<double>& v1, const Vector<double>& v2, 
					  Plda& plda) {
	PldaConfig plda_scoring_config;
	plda_scoring_config.normalize_length = true;
	Vector<double> *v1_transformed = new Vector<double>(v1.Dim());
	plda.TransformIvector(plda_scoring_config, v1, v1_transformed);
	Vector<double> *v2_transformed = new Vector<double>(v2.Dim());
	plda.TransformIvector(plda_scoring_config, v2, v2_transformed);

	int32 numberOfv1utt = 1;// this is for when v1 represents train examples
	// numberOfv1utt doesn't apply here for our usage of plda scoring.
	return plda.LogLikelihoodRatio(*v1_transformed, numberOfv1utt, *v2_transformed);
}


void estimatePLDA(std::vector< Vector<double> > backgroundIvectors,
				  std::vector<std::string> backgroundIvectorLabels,
				  Plda& plda) {
	std::map<std::string, std::vector<int32> > spk2utt_map;
	for (size_t i = 0; i < backgroundIvectorLabels.size(); i++) {
		std::string spk = backgroundIvectorLabels[i];
		spk2utt_map[spk].push_back(i);
	}

	PldaEstimationConfig plda_config;
	PldaStats plda_stats;
	std::map<std::string, std::vector<int32> >::iterator iter;
	for (iter = spk2utt_map.begin(); iter!=spk2utt_map.end(); iter++) {
		std::vector< Vector<double> > ivectors;
		std::string spk = iter->first;
		std::vector<int32> uttlist = iter->second;
		int32 nUtts = 0;
		for (size_t i = 0; i < uttlist.size(); i++) {
			ivectors.push_back(backgroundIvectors[uttlist[i]]);
			nUtts++;
		}

		if (nUtts==0) {
			KALDI_ERR << "No ivectors for speaker " << spk;
		}else {
			Matrix<double> ivector_mat(nUtts, ivectors[0].Dim());
			for (size_t i = 0; i < ivectors.size(); i++) {
				ivector_mat.Row(i).CopyFromVec(ivectors[i]);
			}
			double weight = 1.0;
			plda_stats.AddSamples(weight, ivector_mat);
		}
	}
	plda_stats.Sort();
    PldaEstimator plda_estimator(plda_stats);
    plda_estimator.Estimate(plda_config, &plda);
}
 
BaseFloat sigmoidRectifier(BaseFloat logLikelihoodRatio) {
	// Warps log-likelihood ratio (x) such that large values of x represent shorter 
	// distances and vice-versa. 
	return Exp(-logLikelihoodRatio)/(1. + Exp(-logLikelihoodRatio));
}


}


