#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <unordered_map>

class __declspec(dllexport) ScanImage;

#define DISP_UP    (0)
#define DISP_DOWN  (1)
#define DISP_LEFT  (2)
#define DISP_RIGHT (3)

#define SAVE_FLAG_DISPLACEMENTS (1)
#define SAVE_FLAG_SOLVER_OPT    (2)
#define SAVE_FLAG_MATRIX        (4)
#define SAVE_FLAG_GRID_SIZE     (8)
#define SAVE_FLAGS_ALL (0xF)
#define SAVE_FLAGS_INPUT (0)
#define SAVE_FLAGS_GRID  (SAVE_FLAG_GRID_SIZE)

extern cv::Point2i DISP_DIRECTIONS[4];
template class __declspec(dllexport) cv::Point_<double>;
template class __declspec(dllexport) cv::Point_<float>;
template class __declspec(dllexport) cv::Point_<int>;
template class __declspec(dllexport) cv::Rect_<int>;
template class __declspec(dllexport) cv::Size_<int>;
class __declspec(dllexport) cv::Mat;
template class __declspec(dllexport) std::basic_string<char, std::char_traits<char>, std::allocator<char>>;

class ScanImage
{
public:
	cv::Point2i     gridPosition;
	cv::Point2f     stagePosition;
	cv::Point2i     stitchPosition;
	std::string     path;
	cv::Point2i     displacements[4];

	bool            getImage(cv::Mat& out);
	bool            getImageF32(cv::Mat& out);
	void            evictImage();
	void            evictImageF32();
private:
	cv::Mat         cachedImage;
	cv::Mat         cachedF32Img;
	bool            cachedF32 = false;
	bool            cached = false;
};

template class __declspec(dllexport) std::_Vector_val<std::_Simple_types<ScanImage>>;
template class __declspec(dllexport) std::_Compressed_pair<std::allocator<ScanImage>, std::_Vector_val<std::_Simple_types<ScanImage>>, true>;
template class __declspec(dllexport) std::vector<ScanImage>;

class  __declspec(dllexport) ScanSet
{

private:
	bool                   gridGenerated = false;
	bool                   vecsGenerated = false;
	cv::Mat                idxGrid;
public:
	cv::Rect               stitchRect;
	cv::Point2f            stageOrigin;
	cv::Point2f            stageToImgX;
	cv::Point2f            stageToImgY;
	cv::Matx23f            affineStageToImage;
	int                    gridWidth = -1;
	int                    gridHeight = -1;
	std::vector<ScanImage> m_Images;

	void addImage(std::string path, cv::Point2i gridPosition, cv::Point2f stagePosition);

	void generateGrid();

	ScanImage& imageAt(cv::Point2i g);
	ScanImage &imageAt(int x, int y);
	ScanImage& imageAt(cv::Point2i g, int dir);
	ScanImage& imageAt(int x, int y, int dir);
	bool hasImageAt(cv::Point2i g, int dir);

	void saveOverlaps(std::string path);

	void saveProject(std::string path, int flags );

	void loadOverlaps(std::string path);

	void loadInput(std::string path);

	void evictAllF32();
};

