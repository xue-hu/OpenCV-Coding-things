#pragma once
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>


int SimpleIO();
void Salt(cv::Mat &Img, int n);
void Sharpen(cv::Mat &Img, cv::Mat &Result);

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
