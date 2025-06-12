#include "pch.h"
#include "OverlapSolver.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "stitch.h"
#include <assert.h>
#include <omp.h>

using namespace cv;

#define BAD_SCORE (1e29)
#define NAN_SCORE NAN
static void cropImage(Size cropSize, Mat& in, Mat& out) {
	Size sourceSz = Size(in.cols, in.rows);
	Rect cropRect((Point2i(sourceSz) - Point2i(cropSize)) / 2, cropSize);
	out = in(cropRect);
}

float OverlapSolver::findOverlapPair( ScanImage& imageA, ScanImage& imageB, cv::Point2i guess, cv::Point2i range, cv::Point2i& dr)
{
	Mat im_a, im_b, im_ca, im_cb;

	/* Load images */
	if (!imageA.getImage(im_a)) {
		fatal("Could not load image for overlap: \"" + imageA.path + "\"");
		return NAN_SCORE;
	}
	if (!imageB.getImage(im_b)) {
		fatal("Could not load image for overlap: \"" + imageA.path + "\"");
		return NAN_SCORE;
	}

	/* Crop images */
	cropImage(cropSize, im_a, im_ca);
	cropImage(cropSize, im_b, im_cb);

	/* Compute score */
	return iterBestOverlapNC(im_ca, im_cb, guess, range, logSteps, dr);
}

float OverlapSolver::findOverlapPair(ScanSet &set, int x, int y, int dir, cv::Point2i guess, cv::Point2i& dr)
{
	ScanImage& imA = set.imageAt( x, y );
	ScanImage& imB = set.imageAt( x, y, dir );
	
	return findOverlapPair(imA, imB, guess, getRange(dir), dr);

}

float OverlapSolver::findOverlapPair(ScanSet& set, int x, int y, int dir, cv::Point2i& dr)
{
	float score;
	Point2i guess;
	Point2f stagePos;
	ScanImage& imA = set.imageAt(x, y);
	ScanImage& imB = set.imageAt(x, y, dir);

	if ( guessMode == GUESS_STAGE ) {
		stagePos = imB.gridPosition - imA.gridPosition;
		guess = stagePos.x * set.stageToImgX + stagePos.y * set.stageToImgY;
	}
	else if (guessMode == GUESS_RESULT) {
		guess = imB.stitchPosition - imA.stitchPosition;
	}
	else if (guessMode == GUESS_FIXED) {
		guess = (dir == DISP_DOWN || dir == DISP_UP) ? guessV : guessH;
		if (dir == DISP_UP || dir == DISP_LEFT)
			guess = -guess;
	}
	else
		assert(!"invalid guess mode");

	score = findOverlapPair(imA, imB, guess, getRange(dir), dr);

	/* Warn if overly large */
	if (norm(dr - guess) > maxDistance) {
		logf(SLOG_WARN,
			"overly large difference %f from guess encountered at (%3i,%3i)",
			x, y, norm(dr - guess));
	}

	return score;
}


void OverlapSolver::computeOverlapsY(ScanSet& set)
{
	log(SLOG_INFO, "Computing vertical overlaps...");
	progress(STEP_OVERLAPSY, 0, set.gridHeight - 1, "Computing overlaps");
	for (int y = 0; y < set.gridHeight - 1; y++) {
		
#pragma omp parallel for 
		for (int x = 0; x < set.gridWidth; x++) {
			Point2i dr(0,0);
			findOverlapPair(set, x, y, DISP_DOWN, dr);
			set.imageAt(x, y           ).displacements[DISP_DOWN] =  dr;
			set.imageAt(x, y, DISP_DOWN).displacements[DISP_UP]   = -dr;
		}
		progress(STEP_OVERLAPSY, y, set.gridHeight - 1, "Computing overlaps");
	}
}

void OverlapSolver::computeOverlapsX(ScanSet& set)
{
	log(SLOG_INFO, "Computing horizontal overlaps...");
	progress(STEP_OVERLAPSX, 0, set.gridWidth - 1, "Computing overlaps");
	for (int x = 0; x < set.gridWidth - 1; x++) {
#pragma omp parallel for 
		for (int y = 0; y < set.gridHeight; y++) {
			Point2i dr(0, 0);
			findOverlapPair(set, x, y, DISP_RIGHT, dr);
			set.imageAt(x, y).displacements[DISP_RIGHT]            =  dr;
			set.imageAt(x, y, DISP_RIGHT).displacements[DISP_LEFT] = -dr;
		}
		progress(STEP_OVERLAPSX, x, set.gridWidth - 1, "Computing overlaps");
	}
}

/**
 * Utility function used to fine tune the image coordinate system
 *
 * It measures the overlap between image (x,y) and it's dir neighbour, and
 * uses that to determine a column of the stage to image space matrix.
 *
 * This needs to be run before starting any further overlap measurements as
 * it sets up the initial positions of each tile before further fitting.
 *
 * These initial positions are used as the guess for the precision overlap
 * measurements.
 */
float OverlapSolver::computeGridVector(ScanSet& set, int x, int y, int dir)
{
	float score;
	Point2i dr;
	Point2f ds;

	progress(STEP_GRIDVEC, 0, 1, "Computing grid vector");

	/* Compute stage displacement */
	ds = set.imageAt(x, y, dir).gridPosition - set.imageAt(x, y).gridPosition;

	/* Solve for overlap */
	score = findOverlapPair(set, x, y, dir, dr);

	//TODO: Deal with stage motion rotated wrt grid!!!
	/* Store result */
	if (dir == DISP_DOWN || dir == DISP_UP)
		set.stageToImgY = Point2f(dr) / ds.y;
	else
		set.stageToImgX = Point2f(dr) / ds.x;

	progress(STEP_GRIDVEC, 1, 1, "Computed grid vector");
	return score;
}

void OverlapSolver::setFixedGuess(cv::Point2i guessH, cv::Point2i guessV)
{
	this->guessMode = GUESS_FIXED;
	this->guessH = guessH;
	this->guessV = guessV;
}

void OverlapSolver::setParameters(int guessMode, int maxDist, int logSteps, cv::Size cropSize, cv::Point2i rangeH, cv::Point2i rangeV)
{
	this->guessMode = guessMode;
	this->maxDistance = maxDist;
	this->logSteps = logSteps;
	this->cropSize = cropSize;
	this->rangeH = rangeH;
	this->rangeV = rangeV;
}

void OverlapSolver::applyInitialGrid(ScanSet& set) {
	for (int y = 0; y < set.gridHeight; y++) {
		for (int x = 0; x < set.gridWidth; x++) {
			Point2i pos = Point(x, y);
			Point2d stagePos = set.imageAt(pos).stagePosition - set.stageOrigin;
			set.imageAt(pos).stitchPosition = stagePos.x * set.stageToImgX + stagePos.y * set.stageToImgY;
		}
	}
}
