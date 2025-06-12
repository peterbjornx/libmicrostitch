#include "pch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>

using namespace cv;

static Point2i pointCoordMin(Point2i a, Point2i b)
{
    return Point2i(MIN(a.x,b.x), MIN(a.y, b.y));
}

static Point2i pointCoordMax(Point2i a, Point2i b)
{
    return Point2i(MAX(a.x, b.x), MAX(a.y, b.y));
}

static Point2i clampRect(Rect2i& r, Point2i p)
{
    return pointCoordMax(r.tl(), pointCoordMin(r.br(), p));
}

bool getOverlapRoi(Mat& imageA, Mat& imageB, Point2i dr, Mat& roiA, Mat&roiB) {
    Point2i start_a, start_b, end_a, end_b;
    Point2i zero(0, 0);
    Rect2i  bounds_a(0, 0, imageA.cols, imageA.rows);
    Rect2i  bounds_b(0, 0, imageB.cols, imageB.rows);
    Rect2i  roi_a, roi_b;

    start_a = pointCoordMax(dr, zero);
    start_b = pointCoordMax(-dr, zero);
    end_a = clampRect(bounds_a, start_a + Point2i(bounds_b.size()) - start_b);
    end_b = clampRect(bounds_b, start_b + Point2i(bounds_a.size()) - start_a);
    roi_a = Rect2i(start_a, end_a);
    roi_b = Rect2i(start_b, end_b);

    /* Sanity check */
    if (roi_a.size() != roi_b.size() || roi_a.width == 0 || roi_a.height == 0)
        return false;

    roiA = imageA(roi_a);
    roiB = imageB(roi_b);

    return true;

}

float scoreOverlap(Mat& imageA, Mat& imageB, Point2i dr) {
    Mat     diff_roi, roiA, roiB;
    Scalar  norm;
    double  norm_f;

    /* Sanity check */
    if ( !getOverlapRoi( imageA, imageB, dr, roiA, roiB ) )
        return 1e29;

    /* Compute square difference */
   // absdiff(imageA(roi_a), imageB(roi_b), diff_roi);
   // diff_roi.mul(diff_roi);
    //diff_roi.mul(diff_roi); //TODO: Is this needed? python version was squaring twice*/
    norm_f = cv::norm(roiA, roiB, NORM_L2);//sum(diff_roi);
    norm_f = pow(norm_f, 3.3);
    //norm_f = norm.val[0] + norm.val[1] + norm.val[2];

    if (norm_f < 0.0)
        return 1e29;

    return (roiA.rows * roiA.cols) / norm_f;

}

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
float findBestOverlap(Mat& imageA, Mat& imageB, Point2i guess, Point2i range, int decimate, Point2i &dr) {
    Mat sc_a, sc_b;
    float best_score = 0, score;
    Point2i pos;

    /* Resample image to reduce workload */
    cv::resize( imageA, sc_a, Size(), 1. / decimate, 1. / decimate, INTER_LINEAR );
    cv::resize( imageB, sc_b, Size(), 1. / decimate, 1. / decimate, INTER_LINEAR );

    /* Search through range */
    for (int dx = guess.x - range.x; dx <= guess.x + range.x; dx+=decimate)
        for (int dy = guess.y - range.y; dy <= guess.y + range.y; dy+=decimate) {
            pos = Point2i(dx, dy);
            score = scoreOverlap(sc_a, sc_b, pos / decimate);
            if (score > best_score) {
                best_score = score;
                dr = pos;
            }
        }

    return best_score;
}

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
float iterBestOverlap(Mat& imageA, Mat& imageB, Point2i guess, Point2i range, int logd, Point2i& dr) {
    Mat fim_a, fim_b;
    float score;
    Point2i round_guess, round_range;

    /* Convert the images to float32 */
    //TODO: skip this and use the input if images are already float32
    imageA.convertTo(fim_a, CV_32F);
    imageB.convertTo(fim_b, CV_32F);

    round_guess = guess;
    for (int sf = logd; sf >= 0; sf--) {

        /* Determine the best overlap vector */
        score = findBestOverlap(fim_a, fim_b, round_guess, round_range, 1 << sf, dr);

        /* Search an area half as large around the result */
        round_guess = dr;
        round_range = (round_range / 3) + Point2i(1, 1);
    }

    return score;

}

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
float iterBestOverlapNC(Mat& imageA, Mat& imageB, Point2i guess, Point2i range, int logd, Point2i& dr) {
    float score;
    Point2i round_guess, round_range;

    round_guess = guess;
    round_range = range;
    for (int sf = logd; sf >= 0; sf--) {

        /* Determine the best overlap vector */
        score = findBestOverlap(imageA, imageB, round_guess, round_range, 1 << sf, dr);

        /* Search an area half as large around the result */
        round_guess = dr;
        round_range = (round_range / 4) + Point2i(1, 1);
    }

    return score;

}