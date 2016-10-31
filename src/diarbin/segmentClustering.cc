// diarbin/segmentClustering.cc

#include <vector>
#include <string>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/diar-utils.h"
#include "diar/cluster.h"

int main(int argc, char *argv[]) {
    try{
        typedef kaldi::int32 int32;
        using namespace kaldi;

        const char *usage = "Bottom Up clustering on segments, initial clustering using GLR, BIC. \n";

        int32 target_cluster_num = 0;

        kaldi::ParseOptions po(usage);
        po.Register("target_cluster_num", &target_cluster_num, "Target cluster number as stopping criterion");
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
            segment_clusters.BottomUpClustering(feats, target_cluster_num);
            segment_clusters.Write(segments_dirname);

            //write to rttm
            segment_clusters.WriteToRttm(rttm_outputdir);
        }

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }  
}