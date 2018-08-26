#pragma once
#ifndef SIMPLEOPERATOR_H
#define SIMPLEOPERATOR_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;

#define PI 3.14159

int SimpleIO();
void Salt(cv::Mat &Img, int n);
void Sharpen(cv::Mat &Img, cv::Mat &Result);
cv::Mat equalize(const cv::Mat &image);

class ColorDetector {

private:
	int minDist;
	cv::Vec3b target;
	cv::Mat result;
	cv::Mat converted;
	int getDistance(cv::Vec3b a, cv::Vec3b b);

public:
	ColorDetector();
	void setMinDist(int t);
	int getMinDist() const;
	void setTarget(cv::Vec3b color);
	void setTarget(unsigned char r, unsigned char g, unsigned char b);
	cv::Vec3b getTarget() const;
	cv::Mat process(const cv::Mat image);
};

class Histogram3D {

private:
	int bins[3];
	float hranges[2];
	const float *ranges[3];
	int channels[3];

public:

	Histogram3D();
	cv::MatND getHistogram3D( const cv::Mat &image);
	cv::Mat* getHistgramImg(const cv::Mat &image );
};

class ContentFinder {
private:
	float hranges[2];
	const float* ranges[3];
	int channels[3];
	float threshold;
	cv::MatND hist;

public:
	ContentFinder();
	void setThreshold(float t);
	float getThreshold();
	void setHistogram(const cv::MatND& histogram);
	cv::Mat find(const cv::Mat &image ,
							float minval,
							float maxval,
							int dim);
};

class MorphoFeatures {
private:
	int threshold;
	cv::Mat cros;
	cv::Mat diamond;
	cv::Mat square;
	cv::Mat x;
public:
	MorphoFeatures();
	void setThreshold(int t);
	int getThreshold();
	void ApplyThreshold(const cv::Mat &result);
	cv::Mat getEdges(const cv::Mat &image);
	cv::Mat getCorners(const cv::Mat &image);
};

class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	void setMarkers( const cv::Mat &markerImg);
	cv::Mat process( const cv::Mat &image);
};

class LaplacianZC {
private:
	cv::Mat image;
	cv::Mat laplace;
	int aperture;

public:
	LaplacianZC();
	void setAperture(int t);
	void computeLaplacian(const cv::Mat &iamge);
	cv::Mat getLaplacianImg(const cv::Mat &iamge);
	cv::Mat getZeroCrossing(const cv::Mat &image,float threshold);
};

class LineFinder {
private:
	cv::Mat img;
	vector<cv::Vec4i> lines;
	double deltaR;
	double deltaT;
	int minVote;
	double minLength;
	double maxGap;
public:
	LineFinder();
	void setAccResolution(double r, double theta);
	void setMinVote(int minval);
	void setLengthGap(double len, double gap);
	vector<cv::Vec4i> findLines(const cv::Mat &img);
	cv::Mat drawDetecLines(const cv::Mat &img);
};
class HarrisDetector {
private:
	cv::Mat cornerStrength;
	cv::Mat cornerThrd;
	cv::Mat localMax;
	int neighbor;
	int aperture;
	double k;
	double maxStrength;
	double threshold;
	int nonMaxSize;
	cv::Mat kernel;
public:
	HarrisDetector();
	void setLocalMaxWindowSize( int nonMaxSize);
	void detect( const cv::Mat &image);
	cv::Mat getCornerMap(double qualityLevel);
	void getCorner(std::vector<cv::Point> &points, double qualityLevel);
	void DrawOnImg(cv::Mat &image, std::vector < cv::Point > points, 
									cv::Scalar color=cv::Scalar(255,255,255), 
									int radius= 3, int thickness=2);
};




#endif 