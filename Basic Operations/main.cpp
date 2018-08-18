#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "SimpleOperator.h"
#include <iostream>
using namespace std;

int main() {

	cv::Mat image = cv::imread("C:/Users/Iris/Desktop/1.jpg");
	ColorDetector cdetect;
	cdetect.setTarget(130,190,230);
	cv::namedWindow("my image");
	cv::imshow("my image",image);
	cv::namedWindow("my result");
	cv::imshow("my result", cdetect.process(image));
	cv::waitKey(0);
	return 1;


}

