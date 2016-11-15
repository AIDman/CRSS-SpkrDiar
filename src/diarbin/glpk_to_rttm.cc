#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/ilp.h"
#include "diar/diar-utils.h"
#include "diar/segment.h"
#include "diar/cluster.h"



int main(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    using namespace kaldi;

    const char *usage = "Convert gplk output into RTTM format for compute DER \n";

    bool use_segment_label = false; 

    kaldi::ParseOptions po(usage);
    po.Register("use-segment-label", &use_segment_label, "If 'true', use segment labels to initialize the clusters");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
        po.PrintUsage();
        exit(1);
    }

    std::string glpk_scpfile = po.GetArg(1),
                segments_scpfile = po.GetArg(2),
                rttm_outdir = po.GetArg(3);


    // create empty segments to store glpk ILP generated cluster label
    Input ki(segments_scpfile);  // no binary argment: never binary.
    std::string line;

    // glpk sol scp
    Input kg(glpk_scpfile);
    std::string glpk_rspecifier;

    while (std::getline(ki.Stream(), line) && std::getline(kg.Stream(), glpk_rspecifier)) {

        GlpkILP glpk_obj;
        std::vector<std::string> ilp_cluster_label = glpk_obj.ReadGlpkSolution(glpk_rspecifier);

        SegmentCollection utt_segments;
        utt_segments.Read(line);
        SegmentCollection speech_segments = utt_segments.GetSpeechSegments();

        // create clusters
        ClusterCollection segment_clusters;
        if (use_segment_label) {
                KALDI_LOG << "Clusters intitiats from previously clusterred segments"; 
                segment_clusters.InitFromLabeledSegments(speech_segments);
        } else {
                KALDI_LOG << "Clusters intitiats from non-clusterred segments"; 
                segment_clusters.InitFromNonLabeledSegments(speech_segments);
        }

        Cluster* curr = segment_clusters.Head();
        int32 ind = 0;
        while(curr) {
            curr->SetLabel(ilp_cluster_label[ind]);
            curr = curr->next;
            ind++;
        }
        segment_clusters.WriteToRttm(rttm_outdir);
    }
}

