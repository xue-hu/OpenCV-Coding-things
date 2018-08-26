#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "SimpleOperator.h"

using namespace std;

int SimpleIO() {

	//read a image
	cv::Mat image = cv::imread("C:/Users/Iris/Desktop/1.jpg");

	//create image window named "My Image"

	cv::namedWindow("My Image");

	//show the image on window

	cv::imshow("My Image", image);

	// same process, flip the image
	cv::Mat result;
	cv::flip(image, result, 1);
	cv::namedWindow("out Img");
	cv::imshow("out Img", result);

	//wait key for 5000 ms

	cv::waitKey(5000);

	return 1;


}

void Salt(cv::Mat &Img, int n)
{
	for (int k = 0; k < n; k++)
	{
		int i = rand() % Img.rows;
		int j = rand() % Img.cols;
		if (Img.channels() == 1)
			Img.at<uchar>(i, j) = 255;
		else
		{
			Img.at<cv::Vec3b>(i, j)[0] = 255;
			Img.at<cv::Vec3b>(i, j)[1] = 255;
			Img.at<cv::Vec3b>(i, j)[2] = 255;
		}
	}

}

void Sharpen(cv::Mat &Img, cv::Mat &Result)
{

	Result.create(Img.rows, Img.cols, Img.type());
	int chl = Img.channels();

	for (int i = 1; i < Img.rows - 1; i++)
	{
		const uchar* current = Img.ptr<const uchar>(i);
		const uchar* pre = Img.ptr<const uchar>(i-1);
		const uchar* next = Img.ptr<const uchar>(i+1);
		uchar* output = Result.ptr<uchar>(i);

		for (int j = 1; j < chl*Img.cols - 1; j++)
		{
			output[j] = cv::saturate_cast<uchar>(5 * current[j] - pre[j] - next[j] - current[j- chl] - current[j+ chl]) ;
		}
	}

	Result.row(0).setTo(cv::Scalar(0));
	Result.row(Result.rows-1).setTo(cv::Scalar(0));
	Result.col(0).setTo(cv::Scalar(0));
	Result.col(Result.cols-1).setTo(cv::Scalar(0));

}

cv::Mat equalize( const cv::Mat &image)
{
	cv::Mat result(image.size(), CV_8UC1);
	cv::Mat img;
	cv::cvtColor(image, img, CV_RGB2GRAY);
	cv::equalizeHist(img, result);
	return result;
}

ColorDetector::ColorDetector():minDist(100) {
		target[0] = target[1] = target[2] = 0;

	}
void ColorDetector::setMinDist(int t) {
		if (t < 0)
		{
			t = 0;
		}
		minDist = t;

	}
int ColorDetector::getMinDist() const {

		return minDist;

	}
void ColorDetector::setTarget(cv::Vec3b color) {

		target = color;

	}
	void ColorDetector::setTarget(unsigned char r,
		unsigned char g,
		unsigned char b) {

		target[2] = r;
		target[1] = g;
		target[0] = b;

	}
cv::Vec3b ColorDetector::getTarget() const {

		return target;

	}
int ColorDetector::getDistance(cv::Vec3b a, cv::Vec3b b) {

		return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]);

	}
cv::Mat ColorDetector::process(const cv::Mat image) {
		result.create(image.size(), CV_8UC1);
		converted.create(image.size(), CV_8UC3);
		cv::Mat ctarget;
		ctarget.create(1,1,CV_8UC3);
		ctarget.at<cv::Vec3b>(0,0)[0] = target[0];
		ctarget.at<cv::Vec3b>(0,0)[1] = target[1];
		ctarget.at<cv::Vec3b>(0,0)[2] = target[2];
		
		cv::cvtColor(ctarget, ctarget,CV_BGR2Lab);
		cv::cvtColor(image, converted, CV_BGR2Lab);

		
		cv::Mat_<cv::Vec3b>::const_iterator it = converted.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::const_iterator itend = converted.end<cv::Vec3b>();
		cv::Mat_<uchar>::iterator itout = result.begin<uchar>();

		while (it != itend)
		{
			if (getDistance(*it, target) > minDist)
			{
				*itout = 255;
			}
			else
				*itout = 0;

			it++;
			itout++;
		}
		 
		return result;


	}
Histogram3D::Histogram3D() {
	bins[0] = 8 ;
	bins[1] = 8;
	bins[2] = 8;
	hranges[0] =  0.0 ;
	hranges[1] = 255.0 ;
	ranges[0] = hranges ;
	ranges[1] = hranges;
	ranges[2] = hranges;
	channels[0] = 0 ;
	channels[1] = 1;
	channels[2] = 2;
}
cv::MatND Histogram3D::getHistogram3D( const cv::Mat &image)
{
	cv::MatND hist;
	cv::calcHist(&image,
						1 ,
						channels,
						cv::Mat () ,
						hist,
						3,
						bins,
						ranges);
	
	return hist;

}
cv::Mat* Histogram3D::getHistgramImg(const cv::Mat &image)
{
	
	cv::MatND hist = getHistogram3D(image);
	double maxval = 0.0;
	double minval = 0.0;
	cv::minMaxLoc( hist, &minval, &maxval, 0, 0);
	cv::Mat  histImg[3];
	for (int i = 0; i < 3; i++)
		histImg[i] = cv::Mat(bins[0], bins[0], CV_8UC1, cv::Scalar(255));
	int hpt = static_cast<int>(0.9 * bins[0]);

	for (int h = 0; h < bins[0]; h++)
	{
		for (int cn = 0; cn < 3; cn++)
		{
			float val = hist.at<float>(cn, h);
			int intensity = static_cast<int>(val / maxval * hpt);
			for (int i = 0; i < 3; i++)
				cv::line(histImg[i],
				cv::Point(h, bins[0]),
				cv::Point(h, bins[0] - intensity),
				cv::Scalar::all(0));
		}
		
	}
	return histImg;
}
ContentFinder::ContentFinder() {
	threshold = 0.01f;
	ranges[0] = hranges;
	ranges[1] = hranges;
	ranges[2] = hranges;
	channels[0] = 0;
	channels[1] = 1;
	channels[2] = 2;
}
void ContentFinder::setThreshold(float t)
{
	if (t > 0.0)
		threshold = t;
	else
		threshold = 0;

}
float ContentFinder::getThreshold()
{
	return threshold;
}
void ContentFinder::setHistogram(const cv::MatND& histogram)
{
	hist = histogram;
	cv::normalize(hist, hist, 1.0);
}
cv::Mat ContentFinder::find(const cv::Mat& image,
	float minval,
	float maxval,
	int dim)
{
	
	cv::Mat result;
	hranges[0] = minval;
	hranges[1] = maxval;
	cv::calcBackProject(&image,1,
									channels,
									hist,
									result,
									ranges,
									255);
	if (threshold > 0.0)
	{
		cv::threshold(result, result,255*threshold, 255, cv::THRESH_BINARY );
	}
	return result;
}
MorphoFeatures::MorphoFeatures():threshold(40), 
							cros(5,5,CV_8U,cv::Scalar(0)),
							diamond(5, 5, CV_8U, cv::Scalar(1)), 
							square(5, 5, CV_8U, cv::Scalar(1)),
							x(5, 5, CV_8U, cv::Scalar(0)) {
	
	for (int i = 0; i < 5; i++)
	{
		cros.at<uchar>(2, i) = 1;
		cros.at<uchar>(2, i) = 1;
	}
	for (int i = 0; i < 5; i++)
	{
		x.at<uchar>(i, i) = 1;
		x.at<uchar>(4-i, i) = 1;
	}
	diamond.at<uchar>(0,0) = 0;
	diamond.at<uchar>(0,1 ) = 0;
	diamond.at<uchar>(1,0 ) = 0;
	diamond.at<uchar>(4,4 ) = 0;
	diamond.at<uchar>(3, 4) = 0;
	diamond.at<uchar>(4, 3) = 0;
	diamond.at<uchar>(0, 3) = 0;
	diamond.at<uchar>(0,4 ) = 0;
	diamond.at<uchar>(1, 4) = 0;
	diamond.at<uchar>(4, 0) = 0;
	diamond.at<uchar>(4, 1) = 0;
	diamond.at<uchar>(3, 0) = 0;
	
}
cv::Mat MorphoFeatures::getEdges(const cv::Mat &image) {
	cv::Mat result;
	cv::cvtColor(image,result,CV_BGR2GRAY);
	cv::morphologyEx(result, result, cv::MORPH_GRADIENT, cv::Mat());
	ApplyThreshold(result);
	return result;
}
void MorphoFeatures::ApplyThreshold(const cv::Mat &result) {
	if (threshold > 0)
		cv::threshold(result, result, threshold, 255, cv::THRESH_BINARY );
}
void MorphoFeatures::setThreshold(int t) {
	if (t > 0)
		threshold = t;
}
int MorphoFeatures::getThreshold() {
	return threshold;
}
cv::Mat MorphoFeatures::getCorners(const cv::Mat &image) {
	cv::Mat result;
	cv::cvtColor(image, result, CV_BGR2GRAY);
	cv::dilate(result,result,cros);
	cv::erode(result, result, diamond);
	cv::Mat result2;
	cv::cvtColor(image, result2, CV_BGR2GRAY);
	cv::dilate(result2, result2, x);
	cv::erode(result2, result2, square);
	cv::absdiff(result2, result, result);
	if (threshold > 0)
		ApplyThreshold(result);
	return result;
}
void WatershedSegmenter::setMarkers(const cv::Mat &image) {
	cv::Mat bimg;
	cv::cvtColor(image, bimg, CV_BGR2GRAY);
	cv::threshold(bimg, bimg, 40, 255, CV_THRESH_BINARY); 
	cv::Mat fg;
	cv::erode(bimg,fg,cv::Mat(),cv::Point(-1,-1),6);
	cv::Mat bg;
	cv::dilate(bimg, bg, cv::Mat(), cv::Point(-1, -1), 6);
	cv::threshold(bg,bg,1,128,CV_THRESH_BINARY_INV);
	markers = fg + bg;
	markers.convertTo(markers, CV_32S);

}
cv::Mat WatershedSegmenter::process(const cv::Mat &image) {
	cv::watershed(image, markers);
	markers.convertTo(markers,CV_8U);
	cv::threshold(markers,markers,130,255,CV_THRESH_BINARY);
	return markers;
}
LaplacianZC::LaplacianZC():aperture(3) {}
void LaplacianZC::setAperture(int t) {
	if (t > 0)
		aperture = t;
}
void LaplacianZC::computeLaplacian(const cv::Mat &image) {
	cv::Mat grayimg;
	cv::cvtColor(image, grayimg, CV_BGR2GRAY);
	cv::Laplacian(grayimg, laplace, CV_32FC1, aperture);

	this->image = image.clone();
}
cv::Mat LaplacianZC::getLaplacianImg(const cv::Mat &image) {
	computeLaplacian(image);
	double lapmin, lapmax;
	cv::minMaxLoc(laplace, &lapmin, &lapmax);
	double scale = 127 / max(-lapmin, lapmax);
	cv::Mat lapImg;
	laplace.convertTo(lapImg,CV_8U,scale,128);
	return lapImg;

}
cv::Mat LaplacianZC::getZeroCrossing(const cv::Mat &image,float threshold) {
	computeLaplacian(image);
	cv::Mat result(laplace.size(),CV_8U, cv::Scalar(255));
	cout << laplace.elemSize1()<< laplace.channels() << endl;
	cv::Mat_<float>::const_iterator it = laplace.begin<float>() + laplace.step1();
	cv::Mat_<float>::const_iterator itend = laplace.end<float>();
	cv::Mat_<float>::const_iterator itup = laplace.begin<float>() ;
	cv::Mat_<uchar>::iterator itout = result.begin<uchar>() + result.step1();

	while ( it != itend)
	{
		if (*it * *(it - 1) < threshold)
			*itout = 0;
		else if (*it * *itup < threshold)
			*itout = 0;
		it ++;
		itup ++;
		itout++;
	}
	
	return result;

}
LineFinder::LineFinder() :deltaR(1.0) , deltaT(PI/180),
										 minVote(300), minLength(1.), 
										 maxGap(1.){}
void LineFinder::setAccResolution(double r, double theta) {
	if (r > 0 && theta > 0)
	{
		deltaR = r;
		deltaT = theta;
	}
}
void LineFinder::setMinVote(int minval) {
	if (minval > 0)
		minVote = minval;

}
void LineFinder::setLengthGap(double len, double gap) {
	if (len > 0 && gap > 0)
	{
		minLength = len;
		maxGap = gap;
	}
}
vector<cv::Vec4i> LineFinder::findLines(const cv::Mat &img) {
	lines.clear();
	cv::Mat gryImg;
	cv::Mat contours;
	cv::Mat binary;
	cv::cvtColor(img,gryImg, CV_BGR2GRAY);
	cv::Canny(gryImg, contours, 125, 350);
	cv::threshold(contours, binary, 128, 255, CV_THRESH_BINARY_INV);
	cv::HoughLinesP(binary, lines, deltaR, deltaT, minVote, minLength, maxGap);
	return lines;
}
cv::Mat LineFinder::drawDetecLines(const cv::Mat &img) {
	lines = findLines(img);
	cv::Mat result = img.clone() ;
	cv::Scalar color = cv::Scalar(0, 0, 255);
	vector<cv::Vec4i>::const_iterator it = lines.begin();
	while (it != lines.end())
	{
		cv::Point pt1( (*it)[0],  (*it)[1] );
		cv::Point pt2( (*it)[2], (*it)[3]  );
		cv::line(result, pt1, pt2,color,3);
		it++;
	}
	return result;

}
HarrisDetector::HarrisDetector(): neighbor(3), aperture(3), k(0.01), maxStrength(0.), threshold(0.01), nonMaxSize(3){
	setLocalMaxWindowSize(nonMaxSize);
}
void HarrisDetector::setLocalMaxWindowSize(int nonMaxSize){
	


}
void HarrisDetector::detect(const cv::Mat &image) {
	cv::Mat gryImg;
	cv::cvtColor(image, gryImg, CV_BGR2GRAY);
	cv::cornerHarris(gryImg, cornerStrength, neighbor, aperture, k);
	double minStrength;
	cv::minMaxLoc(cornerStrength, &minStrength, &maxStrength);
	cv::Mat dilated;
	cv::dilate(cornerStrength, dilated, cv::Mat());
	cv::compare(cornerStrength, dilated, localMax, cv::CMP_EQ);
}
cv::Mat HarrisDetector::getCornerMap(double qualityLevel) {
	cv::Mat CornerMap;
	threshold = maxStrength * qualityLevel; 
	cv::threshold(cornerStrength, cornerThrd, threshold, 255, CV_THRESH_BINARY );
	cornerThrd.convertTo( CornerMap, CV_8U);
	cv:: bitwise_and( CornerMap, localMax, CornerMap);
	return CornerMap;
}
void HarrisDetector::getCorner(std::vector<cv::Point> &points, double qualityLevel)
{
	cv::Mat cornerMap = getCornerMap(qualityLevel);
	for (int i = 0; i < cornerMap.rows; ++i)
	{
		const uchar* rowptr = cornerMap.ptr<uchar>(i);
		for (int j = 0; j < cornerMap.cols; ++j)
		{
			if (rowptr[j])
				points.push_back( cv::Point(j,i) );
		}
	}
}
void HarrisDetector::DrawOnImg(cv::Mat &image, vector < cv::Point > points,
														cv::Scalar color, 
														int radius , int thickness) {

	vector<cv::Point>::const_iterator it = points.begin();
	vector<cv::Point>::const_iterator itend = points.end();
	while (it != itend)
	{
		cv::circle(image,(*it),radius, color, thickness);
		++it;
	}

}