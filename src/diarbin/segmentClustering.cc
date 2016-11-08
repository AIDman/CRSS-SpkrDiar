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
        std::string dist_type = "GLR";

        kaldi::ParseOptions po(usage);
        po.Register("target_cluster_num", &target_cluster_num, "Target cluster number as stopping criterion");
        po.Register("lambda", &lambda, "Lambda for BIC computation");
        po.Register("dist_type", &dist_type, "Distance Type Used For Clustering. Currently Supports GLR, KL2");
        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }

        std::string segments_scpfile = po.GetArg(1),
                    feature_rspecifier = po.GetArg(2),
                    segments_dirname = po.GetArg(3),
                    rttm_outputdir = po.GetArg(4);

        RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

        // read in segments from each file
        Input ki(segments_scpfile);  // no binary argment: never binary.
        std::string line;
        while (std::getline(ki.Stream(), line)) {
            // read segments
            SegmentCollection utt_segments;
            utt_segments.Read(line);

            // read features
            const Matrix<BaseFloat> &feats = feature_reader.Value(utt_segments.UttID());
            SegmentCollection speech_segments = utt_segments.GetSpeechSegments();
            ClusterCollection segment_clusters;
 
            segment_clusters.InitFromNonLabeledSegments(speech_segments);
 
            if(dist_type == "GLR") {
                segment_clusters.BottomUpClustering(feats, lambda, target_cluster_num, GLR_DISTANCE);
            }else if(dist_type == "KL2") {
                segment_clusters.BottomUpClustering(feats, lambda, target_cluster_num, KL2_DISTANCE);
            }
            
            segment_clusters.Write(segments_dirname);

            segment_clusters.WriteToRttm(rttm_outputdir);
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}