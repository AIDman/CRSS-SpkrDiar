// diarbin/segmentClustering.cc

#include <vector>
#include <string>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/diar-utils.h"
#include "diar/cluster.h"

int main(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    using namespace kaldi;

    const char *usage = "Bottom Up clustering on segments, initial clustering using GLR, BIC. \n";

    kaldi::ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
        po.PrintUsage();
        exit(1);
    }

    std::string segments_scpfile = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                segments_dirname = po.GetArg(3);

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

    // read in segments from each file
    Input ki(segments_scpfile);  // no binary argment: never binary.
    std::string line;
    while (std::getline(ki.Stream(), line)) {
        // read features
        std::string key = feature_reader.Key(); 
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        // read segments
        SegmentCollection uttSegments;
        uttSegments.Read(line);
        // check file mismatch
        if(uttSegments.UttID() != key){
                KALDI_ERR << "Feature and Sements file UttID mismatch";
        }
        // start clustering
        SegmentCollection speechSegments = uttSegments.GetSpeechSegments();
        ClusterCollection segmentClusters();
        segmentClusters.InitFromNoLabeledSegments(SegmentCollection);
        segmentClusters.BottomUpClustering(feats);
        segmentClusters.WriteToSegments(segments_dirname);
    }  
}