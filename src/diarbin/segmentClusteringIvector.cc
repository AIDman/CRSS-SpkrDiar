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

        const char *usage = "Bottom Up clustering on segments, initial clustering using GLR, BIC. \n";

        int32 target_cluster_num = 0;
        BaseFloat lambda = FLT_MAX;
        BaseFloat lambda_ivec = FLT_MAX;
        std::string dist_type = "GLR";

        kaldi::ParseOptions po(usage);
        po.Register("target-cluster_num", &target_cluster_num, "Target cluster number as stopping criterion");
        po.Register("lambda", &lambda, "Lambda for BIC computation");
        po.Register("lambda-ivec", &lambda_ivec, "Lambda for BIC computation");
        po.Register("dist-type", &dist_type, "Distance Type Used For Clustering. Currently Supports GLR, KL2");
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

            // read features
            Matrix<BaseFloat> feats = feature_reader.Value(utt_segments.UttID());
            SegmentCollection speech_segments = utt_segments.GetSpeechSegments();
            ClusterCollection segment_clusters;
 
            // read posterior
            Posterior posteriors = posterior_reader.Value(utt_segments.UttID());

            // get ivect_info
            IvectorInfo ivec_info(&feats, &posteriors, &extractor);

            segment_clusters.InitFromNonLabeledSegments(speech_segments);
 
            //segment_clusters.BottomUpClustering(feats, lambda, target_cluster_num, KL2_DISTANCE, 50);
            segment_clusters.BottomUpClusteringIvector(ivec_info, lambda_ivec, target_cluster_num, 50);
            segment_clusters.BottomUpClusteringIvector(ivec_info, lambda_ivec, target_cluster_num, 0);
            
            segment_clusters.Write(segments_dirname);

            segment_clusters.WriteToRttm(rttm_outputdir);
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}