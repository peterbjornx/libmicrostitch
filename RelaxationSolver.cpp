#include "pch.h"
#include "RelaxationSolver.h"

using namespace cv;

void RelaxationSolver::setup(ScanSet& set, int maxSanityDiff)
{
	this->set = &set;
	this->iterations = 0;
	this->maxSanityDiff = maxSanityDiff;
	sanityNorm = 0;
	log(SLOG_INFO, "Relaxation: Initializing solver...");
	posGrid.create(set.gridHeight, set.gridWidth, CV_64FC2);
	for (int x = 0; x < set.gridWidth; x++)
		for (int y = 0; y < set.gridHeight; y++) {
			Point2i pos(x,y);
			this->posGrid.at<Point2d>(pos) = set.imageAt(pos).stitchPosition;
		}
	for (int x = 1; x < set.gridWidth - 1; x++) {
		for (int y = 1; y < set.gridHeight - 1; y++) {
			ScanImage& i = set.imageAt(x, y);
			for (int d = 0; d < 4; d++)
				sanityNorm += norm(i.displacements[d]) / 4;
		}
	}
	sanityNorm /= (set.gridWidth - 2) * (set.gridHeight - 2);
}

void RelaxationSolver::run(int iters)
{
	assert(set != nullptr);
	int n;
	double mt;
	Mat nextPos;

	log(SLOG_INFO, "Relaxation: Starting run of "+std::to_string(iters)+" iterations...");
	this->posGrid.copyTo(nextPos);
	for (int it = 0; it < iters; it++, iterations++) {
		mt = 0;
		for (int x = 0; x < set->gridWidth; x++) {
			for (int y = 0; y < set->gridHeight; y++) {
				Point2i gridPos(x, y);
				Point2d acc(0, 0);
				n = 0;

				/* Compute average of positions set by neigbors*/
				for (int d = 0; d < 4; d++)
					accumulateFromNeighbor(gridPos, d, acc, n);


				if (n == 0) {
					logf(SLOG_WARN, "No valid neighbors at %i, %i", x, y);
					continue;
				}

				acc /= n;

				/* Keep track of distance moved */
				mt += norm(this->posGrid.at<Point2d>(gridPos) - acc);

				nextPos.at<Point2d>(gridPos) = acc;
			}
		}
		nextPos.copyTo(this->posGrid);
		progress(0, it, iters, "Solving grid (current score="+std::to_string(mt)+")");
	}
	log(SLOG_INFO, "Relaxation: Committing results...");
	/* Commit solution to scan set */
	for (int x = 0; x < set->gridWidth; x++)
		for (int y = 0; y < set->gridHeight; y++) {
			set->imageAt(x, y).stitchPosition = this->posGrid.at<Point2d>(y, x);
		}

	/* Find scan set size*/
	Point2i min_xy, max_xy;
	double mm, mM;
	Mat grid[2];
	split(posGrid, grid);
	minMaxLoc(grid[0], &mm, &mM); min_xy.x = mm; max_xy.x = mM;
	minMaxLoc(grid[1], &mm, &mM); min_xy.y = mm; max_xy.y = mM;
	set->stitchRect = Rect(min_xy, max_xy);


	log(SLOG_INFO, "Relaxation done.");
}

void RelaxationSolver::accumulateFromNeighbor(cv::Point2i pos, int dir, cv::Point2d& acc, int& n)
{
	Point2i ds;
	ScanImage& img = set->imageAt(pos);

	if (!set->hasImageAt(pos, dir))
		return;

	ds = img.displacements[dir];
	if (cv_abs(norm(ds) - sanityNorm) > maxSanityDiff) {
		return;
	}

	acc += posGrid.at<Point2d>(pos + DISP_DIRECTIONS[dir]) - Point2d(ds);

	n++;
}
