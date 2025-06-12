#pragma once

#include <opencv2/core.hpp>

bool getOverlapRoi(cv::Mat& imageA, cv::Mat& imageB, cv::Point2i dr, cv::Mat& roiA, cv::Mat& roiB);
float scoreOverlap(cv::Mat& imageA, cv::Mat& imageB, cv::Point2i dr);

/**
 * Finds the displacement best fitting two overlapping images together.
 *
 * @note  Images must be float32 type for this function to work properly.
 *
 * @param guess      Point to search around
 * @param range      Amount of pixels to deviate from the starting point
 * @param decimate   Factor by which to decimate the image before searching
 * @param dr         Displacement giving the best overlap
 */
float findBestOverlap(cv::Mat& imageA, cv::Mat& imageB, cv::Point2i guess, cv::Point2i range, int decimate, cv::Point2i& dr);

/**
 * Efficiently finds the displacement best fitting two overlapping images together.
 *
 * It does this by recursively performing finer grained searches on higher resolution
 * versions of the image.
 *
 * @param guess      Point to search around
 * @param range      Amount of pixels to deviate from the starting point
 * @param logd       log2 of the maximum decimation factor
 * @param dr         Displacement giving the best overlap
 */
float iterBestOverlap(cv::Mat& imageA, cv::Mat& imageB, cv::Point2i guess, cv::Point2i range, int logd, cv::Point2i& dr);
float iterBestOverlapNC(cv::Mat& imageA, cv::Mat& imageB, cv::Point2i guess, cv::Point2i range, int logd, cv::Point2i& dr);