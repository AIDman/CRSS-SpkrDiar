// diarbin/labelToSegment.cc

#include <vector>
#include <iostream>
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "diar/diar-utils.h"
#include "diar/segment.h"

int main(int argc, char *argv[]) {
	typedef kaldi::int32 int32;
	
	using namespace kaldi;

	const char *usage = "Convert labels to segments \n";

    kaldi::ParseOptions po(usage);
	po.Read(argc, argv);

	if (po.NumArgs() != 2) {
        po.PrintUsage();
        exit(1);
    }

    std::string label_rspecifier = po.GetArg(1),
    	        segments_dirname = po.GetArg(2);

    SequentialInt32VectorReader label_reader(label_rspecifier);

    for (; !label_reader.Done(); label_reader.Next()) {
        const std::vector<int32> label_vec = label_reader.Value();
        SegmentCollection all_segments(label_vec, label_reader.Key());
        all_segments.Write(segments_dirname);
    }
}
