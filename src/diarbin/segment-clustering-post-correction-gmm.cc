// diarbin/segment-clustering-ivector-constraint.cc

#include <vector>
#include <string>
#include <cfloat>
#include <unordered_map>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"
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
    BaseFloat logp = log_prob - 0.5 * log_det - 0.5 * dim * log(2*PI);
    return exp(logp);
}

// We initialize the GMM parameters by setting the variance to the global
// variance of the features, and the means to distinct randomly chosen frames.
void InitGmmFromRandomFrames(const Matrix<BaseFloat> &feats, DiagGmm *gmm) {
  int32 num_gauss = gmm->NumGauss(), num_frames = feats.NumRows(),
      dim = feats.NumCols();
  KALDI_ASSERT(num_frames >= 10 * num_gauss && "Too few frames to train on");
  Vector<double> mean(dim), var(dim);
  for (int32 i = 0; i < num_frames; i++) {
    mean.AddVec(1.0 / num_frames, feats.Row(i));
    var.AddVec2(1.0 / num_frames, feats.Row(i));
  }
  var.AddVec2(-1.0, mean);
  if (var.Max() <= 0.0)
    KALDI_ERR << "Features do not have positive variance " << var;
  
  DiagGmmNormal gmm_normal(*gmm);

  std::set<int32> used_frames;
  for (int32 g = 0; g < num_gauss; g++) {
    int32 random_frame = RandInt(0, num_frames - 1);
    while (used_frames.count(random_frame) != 0)
      random_frame = RandInt(0, num_frames - 1);
    used_frames.insert(random_frame);
    gmm_normal.weights_(g) = 1.0 / num_gauss;
    gmm_normal.means_.Row(g).CopyFromVec(feats.Row(random_frame));
    gmm_normal.vars_.Row(g).CopyFromVec(var);
  }
  gmm->CopyFromNormal(gmm_normal);
  gmm->ComputeGconsts();
}

void TrainOneIter(const Matrix<BaseFloat> &feats,
                  const MleDiagGmmOptions &gmm_opts,
                  int32 iter,
                  int32 num_threads,
                  DiagGmm *gmm) {
  AccumDiagGmm gmm_acc(*gmm, kGmmAll);

  Vector<BaseFloat> frame_weights(feats.NumRows(), kUndefined);
  frame_weights.Set(1.0);

  double tot_like;
  tot_like = gmm_acc.AccumulateFromDiagMultiThreaded(*gmm, feats, frame_weights,
                                                     num_threads);

  KALDI_LOG << "Likelihood per frame on iteration " << iter
            << " was " << (tot_like / feats.NumRows()) << " over "
            << feats.NumRows() << " frames.";
  
  BaseFloat objf_change, count;
  MleDiagGmmUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, &objf_change, &count);

  KALDI_LOG << "Objective-function change on iteration " << iter << " was "
            << (objf_change / count) << " over " << count << " frames.";
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
        while (std::getline(ki.Stream(), line)) {
            // read segments
            SegmentCollection utt_segments;
            utt_segments.Read(line);
            SegmentCollection speech_segments = utt_segments.GetSpeechSegments();

            // read features
            Matrix<BaseFloat> feats = feature_reader.Value(utt_segments.UttID());

            // The last segment, might exceeds total featuures lengh
            ref_segments.FixNonValidSegments(feats.NumCols());

            KALDI_LOG << "Ref Segments Number: " << ref_segments.Size();
            KALDI_LOG << "Input Segments Number: " << speech_segments.Size();
 
            KALDI_ASSERT(speech_segments.Size() == ref_segments.Size());


            // the last segment, might exceeds total featuures lengh
            speech_segments.FixNonValidSegments(feats.NumCols());
            
            // initiate clusters from labelled segments
            ClusterCollection segment_clusters;
            segment_clusters.InitFromLabeledSegments(speech_segments);

            // train GMM for each cluster
            Cluster* curr = segment_clusters.Head();
            std::vector<std::pair<std::string, DiagGmm*> > label2gmm;
            while(curr) {
                int32 num_gauss_init = 1;
                int32 num_gauss = 4;
                int32 dim = feats.NumCols();
                int32 cluster_frames = curr->NumFramesAfterMask();
                if(cluster_frames < num_gauss * 15) continue;
                Matrix<BaseFloat> cluster_feats(cluster_frames, dim);
                curr->CollectFeatures(feats, cluster_feats);
               
                MleDiagGmmOptions gmm_opts;
                DiagGmm* gmm = new DiagGmm(num_gauss_init, dim);
                InitGmmFromRandomFrames(cluster_feats, gmm);

                // we'll increase the #Gaussians by splitting,
                // till halfway through training.
                int32 cur_num_gauss = num_gauss_init;
                int32 gauss_inc = 1, num_iters = 5;                
                for (int32 iter = 0; iter < num_iters; iter++) {
                    TrainOneIter(cluster_feats, gmm_opts, iter, 1, gmm);

                    int32 next_num_gauss = std::min(num_gauss, cur_num_gauss + gauss_inc);
                    if (next_num_gauss > gmm->NumGauss()) {
                        KALDI_LOG << "Splitting to " << next_num_gauss << " Gaussians.";
                        gmm->Split(next_num_gauss, 0.1);
                        cur_num_gauss = next_num_gauss;
                    }
                }

 
                for(int32 iter = 0; iter < 10; iter++) {
                    AccumDiagGmm gmm_accs;
                    gmm_accs.Resize(*gmm, StringToGmmFlags("mvw"));

                    BaseFloat file_like = 0.0;
                    for (int32 i = 0; i < cluster_frames; i++) {
                        file_like += gmm_accs.AccumulateFromDiag(*gmm, cluster_feats.Row(i), 1.0);
                    }
                    BaseFloat objf_impr, count;
                    MleDiagGmmUpdate(gmm_opts, gmm_accs,
                                    StringToGmmFlags("mvw"),
                                    gmm, &objf_impr, &count);
                }

                label2gmm.push_back(std::make_pair(curr->Label(), gmm));
                curr = curr->next;
            }

            
            // Posterior probability of each segment against all clusters
            int32 num_seg = speech_segments.Size();
            int32 num_clst = label2gmm.size();
            std::vector<std::vector<BaseFloat> > segments_post_probs(num_seg, std::vector<BaseFloat>(num_clst, 0.0));
            for (size_t k = 0; k < num_seg; k++) {
                Segment* kth_segment = speech_segments.KthSegment(k);
                for(size_t c = 0; c < num_clst; c++) {
                    std::string cluster_label = label2gmm[c].first;
                    DiagGmm* gmm = label2gmm[c].second;
                    int32 num_frames = kth_segment->EndIdx() - kth_segment->StartIdx() + 1;
                    std::vector<BaseFloat> likes;
                    for (int32 i = kth_segment->StartIdx(); i <= kth_segment->EndIdx(); i++) {
                        BaseFloat pb = gmm->LogLikelihood(feats.Row(i));
                        likes.push_back(pb);
                    }
                    BaseFloat avg_like = std::accumulate(likes.begin(),likes.end(), 0.0) / num_frames;
                    segments_post_probs[k][c] = avg_like;
                    //KALDI_LOG << segments_post_probs[k][c] << "**";
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
                    if(ith_label == label2gmm[j].first) {
                        prob_correct += exp(segments_post_probs[i][j]);
                    } else {
                        prob_incorrect += exp(segments_post_probs[i][j]);
                    }
                }
                prob_incorrect /= prob_correct + prob_incorrect;
                int32 dur = ith_segment->Size();
                expected_spk_err_diff[i] = prob_incorrect * dur;

                KALDI_LOG << "Incorrect: " << prob_incorrect; 
                KALDI_LOG << "Expeced SER diff: " << prob_incorrect * dur; 
            }

            /*
            // To check and fix, pothentiallly incorrectly labelled segments
            std::vector<long unsigned int> ranked_seg_id = ordered_descend(expected_spk_err_diff);
            for(size_t i = 0; i < max_check_pair; i++) {
                int32 sid = ranked_seg_id[i];
                KALDI_LOG << "Expeced DIFF: " << expected_spk_err_diff[sid];
                std::string sid_label = ref_segments.KthSegment(sid)->Label();

                std::vector<long unsigned int> ranked_cluster_candidate = ordered_descend(segments_post_probs[sid]);
                for(size_t j = 0; j < nbest; j++) {
                    int32 cid = ranked_cluster_candidate[j];
                    int32 mvc = 0;
                    int32 tot = cluster2segid[cluster_list[cid]].size();
                    for(int k = 0; k < tot;  k++) {
                        int32 sid_to_comp = cluster2segid[cluster_list[cid]][k];
                        std::string clb_to_comp = ref_segments.KthSegment(sid_to_comp)->Label();
                        if(sid_label == clb_to_comp) {
                            mvc++;
                        }
                    }

                    if(mvc > tot - mvc)
                        speech_segments.KthSegment(sid)->SetLabel(cluster_list[cid]);

                }
                
            }
  
            ClusterCollection segment_clusters;
            segment_clusters.InitFromLabeledSegments(speech_segments);

            segment_clusters.Write(segments_dirname);
            segment_clusters.WriteToRttm(rttm_outputdir);
            */
  
            //segment_clusters.Write(segments_dirname);
            //qsegment_clusters.WriteToRttm(rttm_outputdir);
            
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}

