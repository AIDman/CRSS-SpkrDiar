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



int main(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    using namespace kaldi;

    const char *usage = "Convert gplk output into RTTM format for compute DER \n";

    kaldi::ParseOptions po(usage);
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

        int32 ind = 0;
        for (size_t i = 0; i < speech_segments.Size(); i++) {
            speech_segments.KthSegment(i)->SetLabel(ilp_cluster_label[ind++]);
        }
        speech_segments.WriteToRTTM(rttm_outdir);
    }
}

