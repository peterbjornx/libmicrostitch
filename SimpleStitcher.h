#pragma once

#include "solver.h"
#include "scanset.h"

class __declspec(dllexport) SimpleStitcher : public Solver
{
public:
	void run(ScanSet& set, std::string path, cv::Size cropSize, int decimation);
};

