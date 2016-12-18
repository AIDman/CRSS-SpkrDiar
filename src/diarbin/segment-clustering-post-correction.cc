// diarbin/segment-clustering-ivector-constraint.cc

#include <vector>
#include <string>
#include <cfloat>
#include <unordered_map>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/diar-utils.h"
#include "diar/segment.h"
#include "diar/cluster.h"
#include "diar/cluster-constraint.h"

namespace kaldi {
BaseFloat IvectorGaussPostProb(const Vector<double>& mean, const Vector<double>& covar, const Vector<double>& ivector) {
    int32 dim = mean.Dim();
    double log_det = 0.0;
    for (size_t i = 0; i < dim; i++) {
        log_det += log(covar(i)+0.001);
    }

    double log_prob = 0.0;
    for(int32 i = 0; i < dim; i++) {
        log_prob += (ivector(i) - mean(i)) * (ivector(i) - mean(i)) / (covar(i) + 0.001);
    }

    log_prob = -0.5*log_prob;

    const double PI = 3.14159265358979;

    BaseFloat logp = log_prob - 0.5 * log_det -0.5 * dim * log(2*PI);

    return exp(logp);
}
}

int main(int argc, char *argv[]) {
    try{
        typedef kaldi::int32 int32;
        using namespace kaldi;

        const char *usage = "To to later. \n";
        int32 max_check_pair = 100;
        int32 nbest = 5;

        kaldi::ParseOptions po(usage);
        po.Register("max-check-pair", &max_check_pair, "maximal pair will be checked for correction");
        po.Register("nbest", &nbest, "maximal check pair for each segment");
        po.Read(argc, argv);

        if (po.NumArgs() != 6) {
            po.PrintUsage();
            exit(1);
        }

        std::string segments_scpfile = po.GetArg(1),
                    feature_rspecifier = po.GetArg(2),
                    posterior_rspecifier = po.GetArg(3),
                    ivector_extractor_rxfilename = po.GetArg(4),
                    segments_dirname = po.GetArg(5),
                    rttm_outputdir = po.GetArg(6);

        RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
        RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
        IvectorExtractor extractor;
        ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

        // read in segments from each file
        Input ki(segments_scpfile);  // no binary argment: never binary.
        std::string line;
        while (std::getline(ki.Stream(), line)) {
            // read segments
            SegmentCollection utt_segments;
            utt_segments.Read(line);
            SegmentCollection speech_segments = utt_segments.GetSpeechSegments();

            // read features
            Matrix<BaseFloat> feats = feature_reader.Value(utt_segments.UttID());

            // the last segment, might exceeds total featuures lengh
            speech_segments.FixNonValidSegments(feats.NumCols());
 
            // read posterior
            Posterior posteriors = posterior_reader.Value(utt_segments.UttID());

            // get ivect_info
            IvectorInfo ivec_info(&feats, &posteriors, &extractor);

            // Compute I-vector For each Segment
            std::vector< Vector<double> > ivector_collect;
            for (size_t k = 0; k<speech_segments.Size(); k++) {
                Segment* kth_segment = speech_segments.KthSegment(k);
                kth_segment->SetIvector(feats, posteriors, extractor);
                ivector_collect.push_back(kth_segment->Ivector());
            }

            // compute mean of ivectors
            Vector<double> total_mean;
            ComputeMean(ivector_collect, total_mean);

            // Mean Normalization & length Normalization
            for (size_t k = 0; k<speech_segments.Size(); k++) {
                // mean normalization
                Segment* kth_segment = speech_segments.KthSegment(k);
                Vector<double> kth_ivector = kth_segment->Ivector();
                SpMatrix<double> kth_ivector_covar = kth_segment->IvectorCovar();
                kth_ivector.AddVec(-1, total_mean);

                // length normalization
                BaseFloat norm = kth_ivector.Norm(2.0);
                BaseFloat ratio = norm / sqrt(kth_ivector.Dim()); // how much larger it is
                kth_ivector.Scale(1.0 / ratio);
                kth_ivector_covar.Scale(1.0 / ratio);

                kth_segment->SetIvector(kth_ivector, kth_ivector_covar);
            }

            // Collect i-Vectors for each clusters
            std::unordered_map<std::string, std::vector<Vector<double> > > cluster2ivectors;
            std::vector<std::string> cluster_list;
            for (size_t k = 0; k < speech_segments.Size(); k++) {
                Segment* kth_segment = speech_segments.KthSegment(k);
                std::string kth_label = kth_segment->Label();
                if(cluster2ivectors.find(kth_label) == cluster2ivectors.end()) {
                    cluster_list.push_back(kth_label);
                }
                cluster2ivectors[kth_label].push_back(kth_segment->Ivector());
            }

            // Compute i-Vector Gaussian Mean and Variance for each cluster
            std::unordered_map<std::string, Vector<double>> cluster2mean;
            std::unordered_map<std::string, Vector<double>> cluster2covar;
            for (size_t i = 0; i < cluster_list.size(); i++) {
                std::string cluster_label = cluster_list[i];
                Vector<double> cluster_mean;
                ComputeMean(cluster2ivectors[cluster_label], cluster_mean);
                Vector<double> cluster_covar;
                ComputeDiagCovar(cluster2ivectors[cluster_label], cluster_covar);
                cluster2mean[cluster_label] = cluster_mean;
                cluster2covar[cluster_label] = cluster_covar;
                KALDI_LOG << "Mean: " << cluster_mean;
                KALDI_LOG << "Cova: " << cluster_covar;
            }

            // Posterior probability of i-vector of each segment
            // against all clusters
            int32 num_seg = speech_segments.Size();
            int32 num_clst = cluster_list.size();
            std::vector<std::vector<BaseFloat> > segments_post_probs(num_seg, std::vector<BaseFloat>(num_clst, 0.0));
            for (size_t k = 0; k < num_seg; k++) {
                Segment* kth_segment = speech_segments.KthSegment(k);
                Vector<double> kth_ivector = kth_segment->Ivector();
                for(size_t c = 0; c < num_clst; c++) {
                    std::string cluster_label = cluster_list[c];
                    segments_post_probs[k][c] = IvectorGaussPostProb(cluster2mean[cluster_label], cluster2covar[cluster_label], kth_ivector);
                    KALDI_LOG << segments_post_probs[k][c] << "**";
                }
            }
            
            // Compute Expected Speaker Error Difference for each segment
            std::vector<BaseFloat> expected_spk_err_diff(num_seg);
            for(size_t i = 0; i < num_seg; i++) {
                Segment* ith_segment = speech_segments.KthSegment(i);
                std::string ith_label = ith_segment->Label();                
                BaseFloat prob_correct = 0.0;
                BaseFloat prob_incorrect = 0.0;
                BaseFloat prob_sum = 0.0;
                for(size_t j = 0; j < num_clst; j++) {
                    prob_sum += segments_post_probs[i][j];
                    if(ith_label == cluster_list[j]) {
                        prob_correct += segments_post_probs[i][j];
                    } else {
                        prob_incorrect += segments_post_probs[i][j];
                    }
                }

                //prob_correct = prob_correct / prob_sum;
                //prob_incorrect = prob_incorrect / prob_sum;
                int32 dur = ith_segment->Size();
                expected_spk_err_diff[i] = prob_incorrect * dur;

                KALDI_LOG << "Correct: " << prob_correct <<"  Incorrect: " << prob_incorrect; 
            }
  
            //segment_clusters.Write(segments_dirname);
            //qsegment_clusters.WriteToRttm(rttm_outputdir);
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}

