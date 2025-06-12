#include "pch.h"
#include "AffineOverlapSolver.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "stitch.h"
#include <assert.h>
#include <omp.h>
#include <iostream>

using namespace cv;

#define BAD_SCORE (1e29)
#define NAN_SCORE NAN
static void cropImage(Size cropSize, Mat& in, Mat& out) {
	Size sourceSz = Size(in.cols, in.rows);
	Rect cropRect((Point2i(sourceSz) - Point2i(cropSize)) / 2, cropSize);
	out = in(cropRect);
}

float AffineOverlapSolver::findOverlapPair(ScanImage& imageA, ScanImage& imageB, cv::Point2i guess, cv::Point2i range, cv::Point2i& dr)
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

float AffineOverlapSolver::findOverlapPair(ScanSet& set, int x, int y, int dir, cv::Point2i guess, cv::Point2i& dr)
{
	ScanImage& imA = set.imageAt(x, y);
	ScanImage& imB = set.imageAt(x, y, dir);

	return findOverlapPair(imA, imB, guess, getRange(dir), dr);

}

float AffineOverlapSolver::findOverlapPair(ScanSet& set, int x, int y, int dir, cv::Point2i& dr)
{
	float score;
	Point2i guess;
	Point2f stagePos;
	ScanImage& imA = set.imageAt(x, y);
	ScanImage& imB = set.imageAt(x, y, dir);

	if (guessMode == GUESS_STAGE) {
		stagePos = imB.stagePosition - imA.stagePosition;
		Vec2f gv = set.affineStageToImage * Vec3f(stagePos.x, stagePos.y, 1.);
		guess = Point2i(gv[0],gv[1]);
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


void AffineOverlapSolver::computeOverlapsY(ScanSet& set)
{
	log(SLOG_INFO, "Computing vertical overlaps...");
	progress(STEP_OVERLAPSY, 0, set.gridHeight - 1, "Computing overlaps");
	for (int y = 0; y < set.gridHeight - 1; y++) {

#pragma omp parallel for 
		for (int x = 0; x < set.gridWidth; x++) {
			Point2i dr(0, 0);
			findOverlapPair(set, x, y, DISP_DOWN, dr);
			set.imageAt(x, y).displacements[DISP_DOWN] = dr;
			set.imageAt(x, y, DISP_DOWN).displacements[DISP_UP] = -dr;
		}
		progress(STEP_OVERLAPSY, y, set.gridHeight - 1, "Computing overlaps");
	}
}

void AffineOverlapSolver::computeOverlapsX(ScanSet& set)
{
	log(SLOG_INFO, "Computing horizontal overlaps...");
	progress(STEP_OVERLAPSX, 0, set.gridWidth - 1, "Computing overlaps");
	for (int x = 0; x < set.gridWidth - 1; x++) {
#pragma omp parallel for 
		for (int y = 0; y < set.gridHeight; y++) {
			Point2i dr(0, 0);
			findOverlapPair(set, x, y, DISP_RIGHT, dr);
			set.imageAt(x, y).displacements[DISP_RIGHT] = dr;
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
float AffineOverlapSolver::computeMatrix(ScanSet& set, int x, int y)
{
	float score_b, score_c;
	int d1 = DISP_DOWN, d2 = DISP_RIGHT;

	/* Compute stage displacement */
	Point2f s[3];
	s[0] = Point2f(0, 0);
	s[1] = Point2f(set.imageAt(x, y, d1).stagePosition - set.imageAt(x, y).stagePosition);
	s[2] = Point2f(set.imageAt(x, y, d2).stagePosition - set.imageAt(x, y).stagePosition);

	progress(STEP_GRIDVEC, 0, 3, "Computing pair 1 overlap");

	Point2i p[3];
	p[0] = Point2i(0, 0);

	/* Solve for overlap */
	score_b = findOverlapPair(set, x, y, d1, p[1]);
	progress(STEP_GRIDVEC, 1, 3, "Computing pair 2 overlap");
	score_c = findOverlapPair(set, x, y, d2, p[2]);
	progress(STEP_GRIDVEC, 2, 3, "Computing stage->pixel affine matrix");
	Point2f pd[3] = { Point2d(p[0]), Point2d(p[1]), Point2d(p[2]) };

	set.affineStageToImage = getAffineTransform(s, pd);

	progress(STEP_GRIDVEC, 3, 3, "Computed affine matrix");
	std::cout << "Affine map:" << std::endl;
	std::cout << set.affineStageToImage << std::endl;
	return score_b + score_c;
}

void AffineOverlapSolver::setFixedGuess(cv::Point2i guessH, cv::Point2i guessV)
{
	this->guessMode = GUESS_FIXED;
	this->guessH = guessH;
	this->guessV = guessV;
}

void AffineOverlapSolver::setParameters(int guessMode, int maxDist, int logSteps, cv::Size cropSize, cv::Point2i rangeH, cv::Point2i rangeV)
{
	this->guessMode = guessMode;
	this->maxDistance = maxDist;
	this->logSteps = logSteps;
	this->cropSize = cropSize;
	this->rangeH = rangeH;
	this->rangeV = rangeV;
}

void AffineOverlapSolver::computeResidual(ScanSet& set, cv::Mat& mat) {
	Point2f stagePos;
	mat.create(set.gridHeight, set.gridWidth, CV_32F);
	Point2i stitchOrigin = set.imageAt(0, 0).stitchPosition;
	cv::Matx23f imageToStageAffine;
	invertAffineTransform(set.affineStageToImage, imageToStageAffine);
	for (int x = 0; x < set.gridWidth; x++) {
		for (int y = 0; y < set.gridHeight; y++) {
			stagePos = set.imageAt(x, y).stitchPosition - stitchOrigin;
			Vec2f gv = imageToStageAffine * Vec3f(stagePos.x, stagePos.y, 1.) - Vec2f(set.imageAt(x, y).stagePosition - set.stageOrigin);
			mat.at<float>(Point(x, y)) = sqrt(gv.dot(gv));
		}
	}
	double mi, ma;
	minMaxLoc(mat, &mi, &ma);
	std::cout << mi << "," << ma << std::endl;
}

void AffineOverlapSolver::applyInitialGrid(ScanSet& set) {
	for (int y = 0; y < set.gridHeight; y++) {
		for (int x = 0; x < set.gridWidth; x++) {
			Point2i pos = Point(x, y);
			Point2d stagePos = set.imageAt(pos).stagePosition - set.stageOrigin;
			Vec2f gv = set.affineStageToImage * Vec3f(stagePos.x, stagePos.y, 1.);
			set.imageAt(pos).stitchPosition = Point2i(gv[0], gv[1]);
		}
	}
}

void AffineOverlapSolver::computeMatrixFromStitch(ScanSet& set, Point2i ta, Point2i tb, Point2i tc)
{
	int d1 = DISP_DOWN, d2 = DISP_RIGHT;

	/* Compute stage displacement */
	Point2f s[3];
	s[0] = Point2f(0,0);
	s[1] = Point2f(set.imageAt(tb).stagePosition - set.imageAt(ta).stagePosition);
	s[2] = Point2f(set.imageAt(tc).stagePosition - set.imageAt(ta).stagePosition);

	Point2i spOrigin = set.imageAt(ta).stitchPosition;
	Point2i p[3];
	Point2f pd[3] = { Point2f(0,0), set.imageAt(tb).stitchPosition - spOrigin, set.imageAt(tc).stitchPosition - spOrigin };

	set.affineStageToImage = getAffineTransform(s, pd);
	std::cout << "Affine map:" << std::endl;
	std::cout << set.affineStageToImage << std::endl;
	return;
}