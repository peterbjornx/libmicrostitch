#pragma once

#include "solver.h"
#include "scanset.h"

class __declspec(dllexport) RelaxationSolver : public Solver
{
public:

	void setup(ScanSet& set, int maxSanityDiff);
	void run(int iters);

private:
	cv::Mat                posGrid;
	ScanSet* set = nullptr;
	int                    sanityNorm = -1;
	int maxSanityDiff = -1;
	int iterations = 0;
	void accumulateFromNeighbor(cv::Point2i pos, int dir, cv::Point2d& acc, int& n);

};

