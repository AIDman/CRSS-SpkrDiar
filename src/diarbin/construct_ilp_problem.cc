// diarbin/construct_ivector_ilp_problem.cc

#include <vector>
#include <string>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "ivector/ivector-extractor.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/posterior.h"
#include "diar/diar-utils.h"
#include "diar/segment.h"
#include "diar/cluster.h"
#include "diar/ilp.h"

int main(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    using namespace kaldi;

    const char *usage = "Obtain glp ILP problem representation template \n";

    BaseFloat delta = 0.9;
    int32 min_update_segment = 0;
    bool use_segment_label = false; 

    kaldi::ParseOptions po(usage);
    po.Register("delta", &delta, "delta parameter for ILP clustering");
    po.Register("min-update-segment", &min_update_segment, "segment with minimum of length seg_min is allowed for ILP");
    po.Register("use-segment-label", &use_segment_label, "If 'true', use segment labels to initialize the clusters");
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
        po.PrintUsage();
        exit(1);
    }

    std::string segments_scpfile = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                posterior_rspecifier = po.GetArg(3),
                ivector_extractor_rxfilename = po.GetArg(4),
                ilp_output_dirname = po.GetArg(5);

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
        SegmentCollection speech_segments = utt_segments.GetSpeechSegments();

        // read features
        Matrix<BaseFloat> feats = feature_reader.Value(utt_segments.UttID());

        // read posteriors
        Posterior posteriors = posterior_reader.Value(utt_segments.UttID());

        // get ivector info
        IvectorInfo ivec_info(&feats, &posteriors, &extractor);

        // create clusters
        ClusterCollection segment_clusters;
        if (use_segment_label) {
                KALDI_LOG << "Clusters intitiats from previously clusterred segments"; 
                segment_clusters.InitFromLabeledSegments(speech_segments);
        } else {
                KALDI_LOG << "Clusters intitiats from non-clusterred segments"; 
                segment_clusters.InitFromNonLabeledSegments(speech_segments);
        }

        // extract and normalize i-vectors
        segment_clusters.SetIvector(ivec_info);
        Vector<double> ivectors_average;
        segment_clusters.ComputeIvectorMean(ivectors_average);
        segment_clusters.NormalizeIvectors(ivectors_average);


        // collect i-vectors
        std::vector< Vector<double> > ivector_collect;
        Cluster* curr = segment_clusters.Head();
        while(curr) {
            ivector_collect.push_back(curr->Ivector());
            curr = curr -> next;
        }
 
        // write distance matrix
        Matrix<BaseFloat> dist_matrix;
        ComputeDistanceMatrix(ivector_collect, dist_matrix);

        // Generate glpk format ILP problem representation
        GlpkILP ilp(utt_segments.UttID(), dist_matrix, delta);
        ilp.GlpkIlpProblem();

        // Write glpk format ILP problem template to text file
        ilp.Write(ilp_output_dirname);
    }    
}