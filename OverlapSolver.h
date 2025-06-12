#pragma once

#include "solver.h"
#include "scanset.h"

#define GUESS_STAGE  (0)
#define GUESS_RESULT (1)
#define GUESS_FIXED (2)

#define STEP_OVERLAPSY (1)
#define STEP_OVERLAPSX (2)
#define STEP_GRIDVEC   (3)

class __declspec(dllexport)  OverlapSolver : public Solver
{
public:
	void computeOverlapsX  ( ScanSet& set );
	void computeOverlapsY  ( ScanSet& set );
	float computeGridVector(ScanSet& set, int x, int y, int dir);
	void setFixedGuess( cv::Point2i guessH, cv::Point2i guessV);
	void setParameters(int guessMode, int maxDist, int logSteps, cv::Size cropSize, cv::Point2i rangeH, cv::Point2i rangeV);
	void applyInitialGrid(ScanSet& set);
private:

	float findOverlapPair(ScanImage& imageA, ScanImage& imageB, cv::Point2i guess, cv::Point2i range, cv::Point2i& dr);
	float findOverlapPair(ScanSet& set, int x, int y, int dir, cv::Point2i guess, cv::Point2i& dr);
	float findOverlapPair(ScanSet& set, int x, int y, int dir, cv::Point2i& dr);

	cv::Point2i getRange(int dir) const {
		return (dir == DISP_DOWN || dir == DISP_UP) ? rangeV : rangeH;
	}
	int         maxDistance = -1;
	int         guessMode = -1;
	int         logSteps = -1;
	cv::Size    cropSize;
	cv::Point2i rangeV;
	cv::Point2i rangeH;
	cv::Point2i guessV;
	cv::Point2i guessH;
};

