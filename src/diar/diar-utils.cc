#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include "diar-utils.h"


namespace kaldi {




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


BaseFloat SymetricKlDistance(const Vector<BaseFloat>& mean_vec_1, const Vector<BaseFloat>& mean_vec_2,
												const Vector<BaseFloat>& cov_vec_1, const Vector<BaseFloat>& cov_vec_2) {

	int32 dim = mean_vec_1.Dim();
	Vector<BaseFloat> diff_mean(dim);
	diff_mean.AddVec(1.0, mean_vec_1);
	diff_mean.AddVec(-1.0, mean_vec_2);
	Vector<BaseFloat> inv_cov_1(cov_vec_1), inv_cov_2(cov_vec_2);
	inv_cov_1.InvertElements();
	inv_cov_2.InvertElements();

	BaseFloat dist = 0.0;
	for (size_t i = 0; i < dim; i++) {
		dist += 0.5 * ((cov_vec_1(i) * inv_cov_2(i) + cov_vec_2(i) * inv_cov_1(i)) + diff_mean(i) * diff_mean(i) * (inv_cov_1(i) + inv_cov_2(i)));
	}

	return dist;	
}


void ComputeIvector(const Matrix<BaseFloat>& feats, const Posterior& posteriors, 
					const IvectorExtractor& extractor, Vector<double>& ivector_mean, SpMatrix<double>& ivector_covar) {
    bool need_2nd_order_stats = false;
    IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
                                             extractor.FeatDim(),
                                             need_2nd_order_stats);
    utt_stats.AccStats(feats, posteriors);
    ivector_mean.Resize(extractor.IvectorDim());
    ivector_covar.Resize(extractor.IvectorDim());
    ivector_mean(0) = extractor.PriorOffset();
    extractor.GetIvectorDistribution(utt_stats, &ivector_mean, &ivector_covar);
}

}


