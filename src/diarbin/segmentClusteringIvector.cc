// diarbin/segmentClustering.cc

#include <vector>
#include <string>
#include <cfloat>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/diar-utils.h"
#include "diar/segment.h"
#include "diar/cluster.h"

int main(int argc, char *argv[]) {
    try{
        typedef kaldi::int32 int32;
        using namespace kaldi;

        const char *usage = "Bottom Up clustering on segments based on i-vector. \n";

        bool use_segment_label = false;
        int32 min_update_segment = 0;
        int32 target_cluster_num = 0;
        BaseFloat ivector_dist_stop = 1.0;
        std::string ivector_dist_type = "CosineDistance";

        kaldi::ParseOptions po(usage);
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
 
            // read posterior
            Posterior posteriors = posterior_reader.Value(utt_segments.UttID());

            // get ivect_info
            IvectorInfo ivec_info(&feats, &posteriors, &extractor);

            // Initiate Clusters
            ClusterCollection segment_clusters;
            if (use_segment_label) {
                KALDI_LOG << "Clusters intitiats from previously clusterred segments"; 
                segment_clusters.InitFromLabeledSegments(speech_segments);
            } else {
                KALDI_LOG << "Clusters intitiats from non-clusterred segments"; 
                segment_clusters.InitFromNonLabeledSegments(speech_segments);
            }
 
            segment_clusters.BottomUpClusteringIvector(ivec_info, config);
            
            segment_clusters.Write(segments_dirname);

            segment_clusters.WriteToRttm(rttm_outputdir);
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}
