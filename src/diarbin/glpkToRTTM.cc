#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/ilp.h"
#include "diar/diar-utils.h"


int main(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    using namespace kaldi;

    const char *usage = "Convert gplk output into RTTM format for compute DER \n";

    int32 seg_min = 0;

    kaldi::ParseOptions po(usage);
    po.Register("seg_min", &seg_min, "segment with minimum of length seg_min is allowed for ILP");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
        po.PrintUsage();
        exit(1);
    }

    std::string glpk_rspecifier = po.GetArg(1),
                segments_scpfile = po.GetArg(2),
                rttm_scpfile = po.GetArg(3);

    GlpkILP glpkObj;
    std::vector<std::string> ilpClusterLabel = glpkObj.ReadGlpkSolution(glpk_rspecifier);

    // create empty segments to store glpk ILP generated cluster label
    Input ki(segments_scpfile);  // no binary argment: never binary.
    std::string line;

    Input ko(rttm_scpfile);
    std::string rttm_filename;

    int32 ind = 0;
    while (std::getline(ki.Stream(), line)) {
        SegmentCollection uttSegments;
        uttSegments.Read(line);

        if ( seg_min > 0 ){
            SegmentCollection speechSegments = uttSegments.GetSpeechSegments();
            SegmentCollection largeSpeechSegments = speechSegments.GetLargeSegments(seg_min);
            for (size_t i = 0; i < largeSpeechSegments.Size(); i++) {
                largeSpeechSegments.KthSegment(i).SetLabel(ilpClusterLabel[ind]);
                ind++;
            }
            std::getline(ko.Stream(), rttm_filename);
            largeSpeechSegments.ToRTTM(largeSpeechSegments.UttID(), rttm_filename);
        } else {
            SegmentCollection speechSegments = uttSegments.GetSpeechSegments();
            for (size_t i = 0; i < speechSegments.Size(); i++) {
                speechSegments.KthSegment(i).SetLabel(ilpClusterLabel[ind]);
                ind++;
            }
            std::getline(ko.Stream(), rttm_filename);
            speechSegments.ToRTTM(speechSegments.UttID(), rttm_filename);
        }
    }
}

