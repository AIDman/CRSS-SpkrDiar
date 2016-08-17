// diarbin/ivectorTest.cc

#include <vector>
#include <string>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "ivector/ivector-extractor.h"
#include "diar/diar-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/posterior.h"
#include "diar/ilp.h"

int main(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    using namespace kaldi;

    const char *usage = "Obtain glp ILP problem representation template \n";

    BaseFloat delta = 30.0;
    int32 seg_min = 0; 

    kaldi::ParseOptions po(usage);
    po.Register("delta", &delta, "delta parameter for ILP clustering");
    po.Register("seg_min", &seg_min, "segment with minimum of length seg_min is allowed for ILP");
    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
        po.PrintUsage();
        exit(1);
    }

    std::string segments_scpfile = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                posterior_rspecifier = po.GetArg(3),
                ivector_extractor_rxfilename = po.GetArg(4),
                background_ivectors_rspecifier = po.GetArg(5),
                utt2spk_rspecifier = po.GetArg(6),
                ilpTemplate_wspecifier = po.GetArg(7);

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialDoubleVectorReader ivector_reader(background_ivectors_rspecifier);
    RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
    IvectorExtractor extractor;
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

    std::vector< Vector<double> > ivectorCollect;

    // read in segments from each file
    Input ki(segments_scpfile);  // no binary argment: never binary.
    std::string line;
    while (std::getline(ki.Stream(), line)) {
        Segments uttSegments;
        uttSegments.Read(line);
        Segments speechSegments = uttSegments.GetSpeechSegments();
        if ( seg_min > 0 ) {
            Segments largeSpeechSegments = speechSegments.GetLargeSegments(seg_min);
            largeSpeechSegments.ExtractIvectors(feature_reader.Value(largeSpeechSegments.GetUttID()), posterior_reader.Value(largeSpeechSegments.GetUttID()), extractor);
            largeSpeechSegments.NormalizeIvectors();
            for (size_t i = 0; i<largeSpeechSegments.Size(); i++) {
                ivectorCollect.push_back(largeSpeechSegments.GetIvector(i));
            }
        } else{
            speechSegments.ExtractIvectors(feature_reader.Value(speechSegments.GetUttID()), posterior_reader.Value(speechSegments.GetUttID()), extractor);
            speechSegments.NormalizeIvectors();
            for (size_t i = 0; i<speechSegments.Size(); i++) {
                ivectorCollect.push_back(speechSegments.GetIvector(i));
            }
        }
    }  


    // read Background i-vectors:
    SequentialTokenReader utt2spk_reader(utt2spk_rspecifier);
    std::map<std::string, std::string> utt2spk_map;
    for (; !utt2spk_reader.Done(); utt2spk_reader.Next()) {
        std::string utt = utt2spk_reader.Key();
        std::string spk = utt2spk_reader.Value();
        utt2spk_map[utt] = spk;
    }

    std::vector< Vector<double> > backgroundIvectors;
    std::vector<std::string> backgroundLabels;
    for (; !ivector_reader.Done(); ivector_reader.Next()) {
         std::string utt_label = ivector_reader.Key();
         Vector<double> utt_ivector = ivector_reader.Value();
         backgroundIvectors.push_back(utt_ivector); 
         backgroundLabels.push_back(utt2spk_map[utt_label]); 
    }
 
    // generate distant matrix from i-vectors
    Matrix<BaseFloat> distMatrix;
    computeDistanceMatrix(ivectorCollect, distMatrix, backgroundIvectors, backgroundLabels);

    // Generate glpk format ILP problem representation
    GlpkILP ilpObj(distMatrix, delta);
    ilpObj.glpkIlpProblem();

    // Write glpk format ILP problem template to text file
    ilpObj.Write(ilpTemplate_wspecifier);
    
    KALDI_LOG << "Written ILP optimization problem template to " << ilpTemplate_wspecifier;
}