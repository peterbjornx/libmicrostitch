#include "pch.h"
#include "SimpleStitcher.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;

void SimpleStitcher::run(ScanSet& set, std::string path, cv::Size cropSize, int decimate)
{
	Point2i src_sz(640, 512);//920, 1080);
	Rect    crop_rect((src_sz - Point2i(cropSize)) / 2, cropSize);
	Point2i out_sz = (set.stitchRect.br() + Point2i(cropSize)+Point2i(1,1)+ - set.stitchRect.tl()) / decimate;
	log(SLOG_INFO, "Stitcher: Assembling "+ std::to_string(out_sz.x)+
		" x " + std::to_string(out_sz.y) + " stitched image ("+std::to_string(decimate)+" times reduced resolution)");
	log(SLOG_INFO, "Stitcher: Allocating output image...");
	Mat out_img(out_sz.y , out_sz.x , CV_32S);
	Mat out_n(out_sz.y , out_sz.x , CV_8S);
	int total = set.gridWidth * set.gridHeight;
	log(SLOG_INFO, "Stitcher: Stitching "+std::to_string(total)+" tiles...");
	out_img = Scalar(0, 0, 0);
	out_n = Scalar(0);

	for (int x = 0; x < set.gridWidth; x++)
		for (int y = 0; y < set.gridHeight; y++) {
			progress(1, x * set.gridHeight + y, total, "Stitching tile "+std::to_string(x)+ ","+std::to_string(y));
			ScanImage& i = set.imageAt(x, y);
			Point2i im_p = i.stitchPosition - set.stitchRect.tl();
			Point2i im_pd = im_p / decimate;
			Range y_rd(MAX(0, im_pd.y), MAX(0, im_pd.y) + cropSize.height / decimate);
			Range x_rd(MAX(0, im_pd.x), MAX(0, im_pd.x) + cropSize.width / decimate);
			Mat srci, srcid;
			i.getImage(srci);
			cv::resize(srci(crop_rect), srcid, Size(), 1. / decimate, 1. / decimate);
			Mat f32;
			out_img(y_rd, x_rd) += srcid;
			out_n(y_rd, x_rd) += 1;
			i.evictImage();
		}
	log(SLOG_INFO, "Stitcher: completed combining tiles.");
	log(SLOG_INFO, "Stitcher: Masking zeros to prevent divide error...");
	Mat zeromask = out_n < 0.0000001;
	out_n.setTo(0.000001, zeromask);
	/*Mat out_nn[3], out_n3;
	out_nn[0] = out_n;
	out_nn[1] = out_n;
	out_nn[2] = out_n;
	merge(out_nn, 3, out_n3);*/
	log(SLOG_INFO, "Stitcher: Computing average of overlapped areas...");
	out_img /= out_n;//3;
	log(SLOG_INFO, "Stitcher: Filling background pixels...");
	out_img.setTo(0, zeromask);
	zeromask.release();
	out_n.release();
	//patchNaNs(out_img, 0.0);
	Mat out_cvt;
	progress(1, 3, 5, "Converting image");
	log(SLOG_INFO, "Stitcher: Converting image to 24bpp RGB...");
	out_img.convertTo(out_cvt, CV_16U/*8UC3*/, 1);
	out_img.release();
	progress(1, 4, 5, "Encoding output file");
	log(SLOG_INFO, "Stitcher: Encoding result into \""+path+"\"...");
	imwrite(path, out_cvt);
	log(SLOG_INFO, "Stitcher: Encoding completed!\a");
	progress(1, 5, 5, "Encoding output file");
}
