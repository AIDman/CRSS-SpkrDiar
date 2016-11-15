#ifndef KALDI_SRC_DIAR_ILP_H_
#define KALDI_SRC_DIAR_ILP_H_

#include <vector>
#include <string>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "ivector/ivector-extractor.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/posterior.h"
#include "diar-utils.h"
#include "segment.h"

namespace kaldi{

// The ILP clustering approach implemented in this file uses refers to the paper
// [1] "Recent Improvements on ILP-based Clustering for Broadcast news speaker diarization",
// by Gregor Dupuy, Sylvain Meignier, Paul Deleglise, Yannic Esteve

typedef kaldi::int32 int32;

class GlpkILP {
public:
	GlpkILP() {};
	GlpkILP(std::string uttid, BaseFloat delta);
	GlpkILP(std::string uttid, Matrix<BaseFloat>& distance_matrix, BaseFloat delta);

	std::string UttID();

	// generate ILP problem description in CPLEX LP format
	void GlpkIlpProblem();

	// write objective function of ILP in glpk format, refer to equation (2) in the paper [1]
	std::string ProblemMinimize();

	// write constraint function for unique center assigment as in equation (2.3) in the paper[1]
	void ProblemConstraintsColumnSum();

	//  write constraint function as in equation (2.4) in the paper[1]
	void ProblemConstraintsCenter();

	// explicitly enforce distance upperbound (eq. 1.5) in paper [1]
	void DistanceUpperBound();

	// list all binary variables as in equation (2.2) in the paper [1]
	void ListBinaryVariables();


	// generate variable names represent ILP problem in glpk format
	std::string IndexToVarName(std::string variableName, int32 i, int32 j);

	// generate variable names represent ILP problem in glpk format
	std::vector<int32> VarNameToIndex(std::string& variable_name);

	// write template into filse
	void Write(std::string ilp_dir);

	// Read the ILP solution (written in glpk format)
	std::vector<std::string> ReadGlpkSolution(std::string glpk_solution_file);

private:
	std::string uttid_;
	BaseFloat delta_;
	std::vector<std::string> problem_;
	Matrix<BaseFloat> distance_matrix_;
};

}


#endif 