#include "pch.h"
#include "scanset.h"
#include <set>
#include "stitch.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <omp.h>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

Point2i DISP_DIRECTIONS[4] = { Point2i(0,-1), Point2i(0,1), Point2i(-1,0), Point2i(1,0) };

/**
 * Adds an image to the scan set.
 * @param path      The filesystem path to the image data.
 * @param gridPos   The position within the logical grid. These need to be exact (see generateGrid)
 * @param stagePos  The stage position this image was taken at, if no stage feedback is available, this can be set to gridPos.
 */
void ScanSet::addImage(std::string path, cv::Point2i gridPos, cv::Point2f stagePos)
{
	ScanImage image;

	assert( gridGenerated == false );

	image.path          = path;
	image.stagePosition = stagePos;
	image.gridPosition  = gridPos;

	/* Add the image to our imagelist */
	m_Images.push_back(image);
}

static int findIndexInSet( set<int> &s, int value) {
	int j = 0;
	for (int i : s) {
		if (i == value)
			return j;
		else
			j++;
	}
	return -1;
}

/**
 * Generates a logical 2D grid from the list of images provided by addImage.
 * It does this in increasing order of the gridPosition coordinate, and it
 * is important that these are not noisy - every tile in the same row/column
 * should have the EXACT same value for the corresponding gridPosition axis.
 * 
 * When this function has been run, it is no longer possible to add new images.
 */
void ScanSet::generateGrid()
{
	Point2i g_min, g_max, g_size, g_step;
	set<int> x_pos, y_pos;
	int xi = -1, yi = -1;

	/* Build sorted list of unique grid coordinates*/
	for (ScanImage& img : m_Images) {
		x_pos.insert(img.gridPosition.x);
		y_pos.insert(img.gridPosition.y);
	}

	/* Find the extents and of that grid */
	g_min  = Point2i(*x_pos.cbegin(), *y_pos.cbegin());
	g_max  = Point2i(*x_pos.crbegin(), *y_pos.crbegin());
	g_size = g_max - g_min;
	g_step = Point2i(*++x_pos.cbegin(), *++y_pos.cbegin()) - g_min;
	gridWidth  = g_size.x / g_step.x + 1;
	gridHeight = g_size.y / g_step.y + 1;

	assert(g_size.x / g_step.x == x_pos.size() - 1);
	assert(g_size.y / g_step.y == y_pos.size() - 1);
	
	/* Create the image position map */
	idxGrid.create(gridHeight, gridWidth, CV_32S);
	for (int ii = 0; ii < m_Images.size(); ii++ ) {
		ScanImage& img = m_Images[ii];

		/* Find the index of that position in our list */
		xi = findIndexInSet(x_pos, (int) img.gridPosition.x);
		yi = findIndexInSet(y_pos, (int) img.gridPosition.y);
		assert( xi != -1 && yi != -1 );

		idxGrid.at<int>(yi, xi) = ii;
	}

	/* Mark that we are done */
	gridGenerated = true;

	stageOrigin = imageAt(0, 0).stagePosition;
}

void ScanSet::saveProject(std::string path, int flags)
{
	cv::FileStorage fs(path, cv::FileStorage::WRITE);
	if (flags & SAVE_FLAG_MATRIX) {
		fs << "stageToImgX" << stageToImgX;
		fs << "stageToImgY" << stageToImgY;
	}
	if (flags & SAVE_FLAG_GRID_SIZE) {
		fs << "gridWidth"   << gridWidth;
		fs << "gridHeight"  << gridHeight;
		fs << "stageOrigin" << stageOrigin;
	}

	if (flags & SAVE_FLAG_SOLVER_OPT )
		fs << "stitchRect" << stitchRect;
	
	fs << "images" << "[";
	for (ScanImage& si : m_Images) {
		fs << "{:";
		fs << "path" << si.path;
		fs << "grid"   << si.gridPosition;
		fs << "stage"  << si.stagePosition;
		if ( flags & SAVE_FLAG_SOLVER_OPT )
			fs << "stitch" << si.stitchPosition;
		if (flags & SAVE_FLAG_DISPLACEMENTS) {
			std::vector<Point2i> disps;
			for (int i = 0; i < 4; i++)
				disps.push_back(si.displacements[i]);
			fs << "displacements" << disps;
		}
		fs << "}";
	}
	fs << "]";
	fs.release();
}

void ScanSet::loadInput(std::string path)
{
	cv::FileStorage fs(path, cv::FileStorage::READ);
	cv::FileNode images = fs["images"];
	cv::FileNodeIterator it = images.begin(), it_end = images.end();
	// iterate through a sequence using FileNodeIterator
	for (; it != it_end; ++it )
	{
		Point2i gridPos;
		Point2f stagePos;
		std::string ipath = (*it)["path"];
		cv::FileNode gnode = (*it)["grid"];
		gridPos = Point2i((int)gnode[0], (int)gnode[1]);
		cv::FileNode snode = (*it)["stage"];
		stagePos = Point2f((float)snode[0], (float)snode[1]);
		addImage(ipath, gridPos, stagePos);
	}
	fs.release();

}

void ScanSet::saveOverlaps(std::string path)
{
	int sizes[3] = { gridWidth, gridHeight, 4 };
	Mat dispMap(3, sizes, CV_32SC2);
	for (int x = 0; x < gridWidth; x++) {
		for (int y = 0; y < gridHeight; y++) {
			ScanImage& si = imageAt(x, y);
			for (int d = 0; d < 4; d++)
				dispMap.at<Point2i>(x, y, d) = si.displacements[d];
		}
	}
	cv::FileStorage fs(path, cv::FileStorage::WRITE);
	fs << "displacements" << dispMap;
}

void ScanSet::loadOverlaps(std::string path)
{
	int sizes[3] = { gridWidth, gridHeight, 4 };
	cv::FileStorage fs(path, cv::FileStorage::READ);
	Mat dispMap;
	fs["displacements"] >> dispMap;
	for (int x = 0; x < gridWidth; x++) {
		for (int y = 0; y < gridHeight; y++) {
			ScanImage& si = imageAt(x, y);
			for (int d = 0; d < 4; d++)
				si.displacements[d] = dispMap.at<Point2i>(x, y, d);
		}
	}
}

void ScanSet::evictAllF32()
{
	for (int x = 0; x < gridWidth; x++)
		for (int y = 0; y < gridHeight; y++)
			imageAt(x,y).evictImageF32();
}

bool ScanImage::getImage(cv::Mat& image)
{
	if (!cached) {
		cachedImage = imread(String(path.c_str()), IMREAD_ANYDEPTH);
		if (cachedImage.data == nullptr)
			return false;
		cached = true;
	}
	image = cachedImage;
	return true;
}

bool ScanImage::getImageF32(cv::Mat& image)
{
	Mat unc;

	if (!cachedF32) {
		if (!getImage(unc))
			return false;
		unc.convertTo(cachedF32Img, CV_32F);
		cachedF32 = true;
	}
	image = cachedF32Img;
	return true;
}

void ScanImage::evictImage()
{
	evictImageF32();
	cachedImage.create(0, 0, CV_16F);
	cached = false;

}

void ScanImage::evictImageF32()
{
	cachedF32Img.create(0, 0, CV_16F);
	cachedF32 = false;
}

ScanImage& ScanSet::imageAt(cv::Point2i g) {
	return imageAt(g.x, g.y);
}

ScanImage& ScanSet::imageAt(cv::Point2i g, int dir) {
	cv::Point2i h = g + DISP_DIRECTIONS[dir];
	return imageAt(h.x, h.y);
}

ScanImage& ScanSet::imageAt(int x, int y, int dir) {
	return imageAt(Point2i(x, y), dir);
}

bool ScanSet::hasImageAt(cv::Point2i g, int dir)
{
	cv::Point2i h = g + DISP_DIRECTIONS[dir];
	return h.x >= 0 && h.y >= 0 && h.x < gridWidth && h.y < gridHeight;
}

ScanImage& ScanSet::imageAt(int x, int y) {
	assert(gridGenerated);
	assert(x >= 0 && x < gridWidth);
	assert(y >= 0 && y < gridHeight);
	int ii = idxGrid.at<int>(y, x);
	return m_Images[ii];
}

/*
void ScanSet::viewOverlap(int x, int y, int dir, cv::Point2i range)
{
	float score;
	Point2i dr, di, guess;
	Point2f stagePos;
	Mat i_a, i_b, i_render;
	di = DISP_DIRECTIONS[dir];
	ScanImage& im_a = imageAt(x, y);
	ScanImage& im_b = imageAt(x + di.x, y + di.y);
	stagePos = im_b.stagePosition - im_a.stagePosition;
	guess = stagePos.x * stageToImgX + stagePos.y * stageToImgY;
	score = findOverlapPair(im_a, im_b, guess, range, gridvecLogD, dr);
	if (!getOverlapRoi(im_a.cachedImage, im_b.cachedImage, dr, i_a, i_b))
		return;
	i_a.copyTo(i_render);
	i_render += i_b;
	i_render /= 2;
	imshow("Overlap", i_render);
	int k = waitKey();
	switch (k) {
	case 'q':
		return;
	case 'a':
		viewOverlap(x + 1, y, dir, range);
		return;
	case 's':
		viewOverlap(x, y + 1, dir, range);
		return;
	case 'd':
		viewOverlap(x - 1, y, dir, range);
		return;
	case 'w':
		viewOverlap(x, y - 1, dir, range);
		return;
	}

}
*/