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
        log_det += log(covar(i)+0.00001);
    }

    double log_prob = 0.0;
    for(int32 i = 0; i < dim; i++) {
        log_prob += (ivector(i) - mean(i)) * (ivector(i) - mean(i)) / (covar(i) + 0.00001);
    }

    log_prob = -0.5*log_prob;

    const double PI = 3.14159265358979;

    BaseFloat logp = log_prob - 0.5 * log_det -0.5 * dim * log(2*PI);

    return logp;
}
}

int main(int argc, char *argv[]) {
    try{
        typedef kaldi::int32 int32;
        using namespace kaldi;

        const char *usage = "To to later. \n";
        int32 max_check_pair = 100;
        int32 cluster_samples = 20;
        int32 nbest = 5;
        std::string mode = "prob";

        kaldi::ParseOptions po(usage);
        po.Register("max-check-pair", &max_check_pair, "maximal pair will be checked for correction");
        po.Register("nbest", &nbest, "maximal check pair for each segment");
        po.Register("cluster-samples", &cluster_samples, "the number of samples to compare with in a cluster for majority voting");
        po.Register("mode", &mode, "rank by randome or probability");
        po.Read(argc, argv);

        if (po.NumArgs() != 7) {
            po.PrintUsage();
            exit(1);
        }

        std::string segments_scpfile = po.GetArg(1),
                    ref_segments_scpfile = po.GetArg(2),
                    feature_rspecifier = po.GetArg(3),
                    posterior_rspecifier = po.GetArg(4),
                    ivector_extractor_rxfilename = po.GetArg(5),
                    segments_dirname = po.GetArg(6),
                    rttm_outputdir = po.GetArg(7);

        RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
        RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
        IvectorExtractor extractor;
        ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

        // read in segments from each file
        Input ki(segments_scpfile);  // no binary argment: never binary.
        std::string line;
        Input kii(ref_segments_scpfile);
        std::string ref;
        while (std::getline(ki.Stream(), line)) {
            // Read segments
            SegmentCollection utt_segments;
            utt_segments.Read(line);
            SegmentCollection speech_segments = utt_segments.GetSpeechSegments();

            // Read ref segments
            SegmentCollection ref_segments;
            std::getline(kii.Stream(), ref);
            ref_segments.Read(ref);

            // Read features
            Matrix<BaseFloat> feats = feature_reader.Value(utt_segments.UttID());

            // The last segment, might exceeds total featuures lengh
            ref_segments.FixNonValidSegments(feats.NumCols());

            KALDI_LOG << "Ref Segments Number: " << ref_segments.Size();
            KALDI_LOG << "Input Segments Number: " << speech_segments.Size();
 
            KALDI_ASSERT(speech_segments.Size() == ref_segments.Size());

            // Read posterior
            Posterior posteriors = posterior_reader.Value(utt_segments.UttID());

            // Get ivect_info
            IvectorInfo ivec_info(&feats, &posteriors, &extractor);

            // Compute I-vector For each Segment
            std::vector< Vector<double> > ivector_collect;
            for (size_t k = 0; k<speech_segments.Size(); k++) {
                Segment* kth_segment = speech_segments.KthSegment(k);
                kth_segment->SetIvector(feats, posteriors, extractor);
                ivector_collect.push_back(kth_segment->Ivector());
            }

            // Compute mean of ivectors
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
            std::unordered_map<std::string, std::vector<int32> > cluster2segid;
            std::vector<std::string> cluster_list;
            for (size_t k = 0; k < speech_segments.Size(); k++) {
                Segment* kth_segment = speech_segments.KthSegment(k);
                std::string kth_label = kth_segment->Label();
                if(cluster2ivectors.find(kth_label) == cluster2ivectors.end()) {
                    cluster_list.push_back(kth_label);
                }
                cluster2ivectors[kth_label].push_back(kth_segment->Ivector());
                cluster2segid[kth_label].push_back(k);
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
            std::vector<std::vector<BaseFloat> > segments_post_probs_r(num_clst, std::vector<BaseFloat>(num_seg, 0.0));
            for (size_t k = 0; k < num_seg; k++) {
                Segment* kth_segment = speech_segments.KthSegment(k);
                Vector<double> kth_ivector = kth_segment->Ivector();
                for(size_t c = 0; c < num_clst; c++) {
                    std::string cluster_label = cluster_list[c];
                    segments_post_probs[k][c] = IvectorGaussPostProb(cluster2mean[cluster_label], cluster2covar[cluster_label], kth_ivector);
                    segments_post_probs_r[c][k] = segments_post_probs[k][c];
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
                    if(ith_label == cluster_list[j]) {
                        prob_correct += exp(segments_post_probs[i][j]);
                    } else {
                        prob_incorrect += exp(segments_post_probs[i][j]);
                    }
                }

                prob_incorrect /= (prob_incorrect + prob_correct);
                int32 dur = ith_segment->Size();
                expected_spk_err_diff[i] = prob_incorrect * dur;
                expected_spk_err_diff[i] = prob_incorrect;

                KALDI_LOG << "Incorrect: " << prob_incorrect; 
                KALDI_LOG << "Expeced SPK diff: " << expected_spk_err_diff[i]; 
            }

            // generate random ranking to compared with probability based ranking: 
            srand((unsigned)time(NULL));
            std::vector<int32> rand_rank(max_check_pair);
            int32 seed_seg_id;
            for(int32 i = 0; i < max_check_pair; i++) {
                seed_seg_id = rand() % (num_seg-1);
                rand_rank[i] = seed_seg_id;
                KALDI_LOG << "generated randome number " << seed_seg_id;
            }

            // To check and fix, pothentiallly incorrectly labelled segments
            std::vector<long unsigned int> ranked_seg_id = ordered_descend(expected_spk_err_diff);
            for(size_t i = 0; i < max_check_pair; i++) {
                int32 sid;
                if(mode == "random") {
                   sid = rand_rank[i];
                } else {
                   sid = ranked_seg_id[i];
                }
                KALDI_LOG << "Expeced DIFF: " << expected_spk_err_diff[sid] << " Assigned Label: " << speech_segments.KthSegment(sid)->Label();
                std::string sid_label = ref_segments.KthSegment(sid)->Label();
                std::vector<long unsigned int> ranked_cluster_candidate = ordered_descend(segments_post_probs[sid]);
                for(size_t j = 0; j < nbest; j++) {
                    int32 cid = ranked_cluster_candidate[j];
                    int32 mvc = 0;
                    int32 tot = cluster2segid[cluster_list[cid]].size();
                    std::vector<int32> sid_list;
                    std::vector<BaseFloat> sid_pb_list;
                    for(int k=0; k < tot; k++) {
                        sid_list.push_back(cluster2segid[cluster_list[cid]][k]);
                        sid_pb_list.push_back(segments_post_probs[i][sid_label[k]]);
                    }
                    std::vector<long unsigned int> ranked_sid_idx = ordered_descend(sid_pb_list);
                    for(int k = 0; k < ((tot < cluster_samples) ? tot : cluster_samples);  k++) {
                        //int32 sid_to_comp = cluster2segid[cluster_list[cid]][k];
                        int32 sid_to_comp = sid_list[ranked_sid_idx[k]];
                        std::string clb_to_comp = ref_segments.KthSegment(sid_to_comp)->Label();
                        if(sid_label == clb_to_comp) {
                            mvc++;
                        }
                    }
                    if(mvc > ((tot < cluster_samples) ? tot : cluster_samples) - mvc) {
                        speech_segments.KthSegment(sid)->SetLabel(cluster_list[cid]);
                        break;
                    }
                }
                
            }
  
            ClusterCollection segment_clusters;
            segment_clusters.InitFromLabeledSegments(speech_segments);

            segment_clusters.Write(segments_dirname);
            segment_clusters.WriteToRttm(rttm_outputdir);
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}
