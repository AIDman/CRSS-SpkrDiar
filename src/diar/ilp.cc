#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <math.h>
#include "diar-utils.h"
#include "ilp.h"


namespace kaldi{

GlpkILP::GlpkILP(std::string uttid, BaseFloat delta) {
	this->delta_ = delta;
	this->uttid_ = uttid;
}


GlpkILP::GlpkILP(std::string uttid, Matrix<BaseFloat>& distance_matrix_, BaseFloat delta) {
	this->distance_matrix_ = distance_matrix_;
	this->delta_ = delta;
	this->uttid_ = uttid;
}

std::string GlpkILP::UttID() {
	return this->uttid_;
}

void GlpkILP::GlpkIlpProblem() {
	this->problem_.push_back("Minimize");
	this->problem_.push_back(ProblemMinimize());
	this->problem_.push_back("Subject To");
	ProblemConstraintsColumnSum();
	ProblemConstraintsCenter();
	//DistanceUpperBound();
	this->problem_.push_back("Binary");
	ListBinaryVariables();
	this->problem_.push_back("End");
}

void GlpkILP::DistanceUpperBound() {
	int32 n = this->distance_matrix_.NumRows();
	for (size_t i = 0; i < this->distance_matrix_.NumRows(); i++) {
		for (size_t j = 0; j < this->distance_matrix_.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->distance_matrix_(i, j);
				std::string constraint = "C" + \
						numberToString(n) + \
				 		": " + numberToString(d) + " " + IndexToVarName("x",i,j);
				constraint += " < " + numberToString(this->delta_);
				this->problem_.push_back(constraint);
				n++;
			}
		}
	}
}

std::string GlpkILP::ProblemMinimize() {
	std::string objective = "problem : " + IndexToVarName("x",0,0);
	for (size_t i = 1; i < this->distance_matrix_.NumRows(); i++) {
		objective += " + " + IndexToVarName("x",i,i);
	}
	for (size_t i = 0; i < this->distance_matrix_.NumRows(); i++) {
		for (size_t j = 0; j < this->distance_matrix_.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->distance_matrix_(i, j) / this->delta_;
				// Why did we need to assure d > 0 ?
				if ((d > 0) && (d <= 1)) {
					objective += " + " + numberToString(d) + " " + IndexToVarName("x",i,j);
				}
			} 
		}
	}
	return objective;
}


void GlpkILP::ProblemConstraintsColumnSum() {
	for (size_t i = 0; i < this->distance_matrix_.NumRows(); i++) {
		std::string constraint = "C" + numberToString(i) + ": " + IndexToVarName("x",i,i);
		for (size_t j = 0; j < this->distance_matrix_.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->distance_matrix_(i, j);
				if (d <= this->delta_) {
					constraint += " + " + IndexToVarName("x",i,j);
				}
			}
		}
		constraint += " = 1 ";
		this->problem_.push_back(constraint);
	}
}


void GlpkILP::ProblemConstraintsCenter() {
	for (size_t i = 0; i < this->distance_matrix_.NumRows(); i++) {
		for (size_t j = 0; j < this->distance_matrix_.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->distance_matrix_(i, j);
				if (d <= this->delta_) {
					this->problem_.push_back(IndexToVarName("x",i,j) + " - " + IndexToVarName("x",j,j) + " <= 0");
				}
			}
		}
	}
}


void GlpkILP::ListBinaryVariables() {
	for (size_t i = 0; i < this->distance_matrix_.NumRows(); i++) {
		for (size_t j = 0; j < this->distance_matrix_.NumRows(); j++) {
			BaseFloat d = this->distance_matrix_(i,j);
			if (d <= this->delta_) {
				this->problem_.push_back(IndexToVarName("x",i,j));
			}
		}
	}
}

std::string GlpkILP::IndexToVarName(std::string prefix, int32 i, int32 j) { 
    return prefix + "_" + numberToString(i) + "_" + numberToString(j);
}


std::vector<int32> GlpkILP::VarNameToIndex(std::string& variableName){
    std::vector<std::string> fields = split(variableName, '_');
    std::vector<int32> indexes;
    indexes.push_back(std::atoi(fields[1].c_str()));
    indexes.push_back(std::atoi(fields[2].c_str()));
    return indexes;
}


void GlpkILP::Write(std::string ilp_dirname){
	std::string ilp_wxfilename = ilp_dirname + "/" + this->uttid_ + ".ilp";
	std::string ilp_scpfilename = ilp_dirname + "/" + "ilp.scp";
	std::ofstream fout;
	std::ofstream fscp;
	fout.open(ilp_wxfilename.c_str());
	fscp.open(ilp_scpfilename.c_str(), std::ios::app);
	for (size_t i =0; i < this->problem_.size(); i++){
		fout << this->problem_[i] << "\n";
	}
	fscp << ilp_wxfilename << "\n";
	fscp.close();
	fout.close();
	return;
}


std::vector<std::string> GlpkILP::ReadGlpkSolution(std::string glpkSolutionFile) {
	std::ifstream fin;
    fin.open(glpkSolutionFile.c_str());
    std::string line;
    std::vector<std::string> IlpClusterLabel;
    while (std::getline(fin, line)){
        if (line.find("*") != std::string::npos){
            std::vector<std::string> fields = split(line, ' ');
            std::vector<string> nonEmptyFields = returnNonEmptyFields(fields);
            std::vector<int32> varIndex = VarNameToIndex(nonEmptyFields[1]);
            int32 k = varIndex[0];
            int32 j = varIndex[1];
            if (k==j) {
            	if (nonEmptyFields[3] == "1") {
                	IlpClusterLabel.push_back(numberToString(k));
                } else {
                	IlpClusterLabel.push_back(numberToString(-1));
                }
            }

            if (k!=j && nonEmptyFields[3] == "1") {
                 IlpClusterLabel[k] = numberToString(j);
            }
        }
    }

    return IlpClusterLabel;
}


}