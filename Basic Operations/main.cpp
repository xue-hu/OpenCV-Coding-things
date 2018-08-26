#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "SimpleOperator.h"
#include <iostream>
#include<windows.h>
using namespace std;

int main() {

	cv::Mat image = cv::imread("C:/Users/Iris/Desktop/2.jpg");
	HarrisDetector detector;
	detector.detect(image);
	double qualityLevel = 0.01;
	vector < cv::Point > points;
	detector.getCorner(points, qualityLevel);
	detector.DrawOnImg(image, points);
	//cv::namedWindow("my image");
	//cv::imshow("my image", ROI); 
	cv::namedWindow("my result");
	cv::imshow("my result", image);
	cv::waitKey(0);
	return 0;


}

