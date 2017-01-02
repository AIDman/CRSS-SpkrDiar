// diarbin/segment-clustering-ivector-constraint.cc

#include <vector>
#include <string>
#include <cfloat>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/diar-utils.h"
#include "diar/segment.h"
#include "diar/cluster.h"
#include "diar/cluster-constraint.h"

int main(int argc, char *argv[]) {
    try{
        typedef kaldi::int32 int32;
        using namespace kaldi;

        const char *usage = "Bottom Up clustering on segments based on i-vector. \n";

        bool merge_constraint = false;
        bool do_consolidate = true;
        BaseFloat max_explore_pair_percentage = 0.3;
        BaseFloat max_consolidate_pair_percentage = 0.3;
        std::string explore_mode = "FFQS";

        bool use_segment_label = false;
        int32 min_update_segment = 0;
        int32 target_cluster_num = 0;
        BaseFloat ivector_dist_stop = 1.0;
        std::string ivector_dist_type = "CosineDistance";

        kaldi::ParseOptions po(usage);
        po.Register("explore-mode", &explore_mode, "Either, FFQS or Random");
        po.Register("merge-constraint", &merge_constraint, "If 'true', explored clusters are not allowed to merge to other clusters");
        po.Register("do-consolidate", &do_consolidate, "If 'true', perform consolidate step, after explore step");
        po.Register("max-explore-pair-percentage", &max_explore_pair_percentage, "maximal constraint pair will be used");
        po.Register("max-consolidate-pair-percentage", &max_consolidate_pair_percentage, "maximal constraint pair will be used for each cluster, during consolidate stage");


        po.Register("min-update-segment", &min_update_segment, "Clustering segments having frames larger than");
        po.Register("target-cluster_num", &target_cluster_num, "Target cluster number as stopping criterion");
        po.Register("ivector-dist-stop", &ivector_dist_stop, "Ivector distance threshold af stopping crierion");
        po.Register("ivector-dist-type", &ivector_dist_type, "Ivector Distance Type For Clustering. e.g., CosineDistance");
        po.Register("use-segment-label", &use_segment_label, "If 'true', use segment labels to initialize the clusters");
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

        // Load configuration file
        DiarConfig config;
        config.min_update_segment = min_update_segment;
        config.target_cluster_num = target_cluster_num;
        config.ivector_dist_type = ivector_dist_type;
        config.ivector_dist_stop = ivector_dist_stop;

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

            KALDI_LOG << "Input Number of Segments " << speech_segments.Size();
 
            // read posterior
            Posterior posteriors = posterior_reader.Value(utt_segments.UttID());

            // get ivect_info
            IvectorInfo ivec_info(&feats, &posteriors, &extractor);

            // initiate ClusterCollectionConstrain object with speech segments
            ClusterCollectionConstraint segment_clusters(&speech_segments);
 
            // perfrom explore starge (active learning for inital cluster construction)
            int32 max_constraint_pair = speech_segments.Size() * max_explore_pair_percentage;
            if(explore_mode == "FFQS") {
                segment_clusters.IvectorHacExploreFarthestFirstSearch(ivec_info, config, max_constraint_pair);
            } else if (explore_mode == "Random") {
                segment_clusters.IvectorHacExploreRandom(ivec_info, config, max_constraint_pair);
            } else {
                KALDI_ERR << "Non existing explore_mode!";
            }
            
            int32 max_pair_per_cluster = speech_segments.Size() * max_consolidate_pair_percentage / segment_clusters.explored_clusters_.size();
            if(do_consolidate){
                segment_clusters.IvectorHacConsolidate(ivec_info, config, max_pair_per_cluster);
            }

            segment_clusters.InitClustersWithExploredClusters();

            if(merge_constraint) {
                segment_clusters.ConstraintBottomUpClusteringIvector(ivec_info, config);
            } else{
                segment_clusters.BottomUpClusteringIvector(ivec_info, config);
            }    
          
            segment_clusters.Write(segments_dirname);
            segment_clusters.WriteToRttm(rttm_outputdir);
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}
