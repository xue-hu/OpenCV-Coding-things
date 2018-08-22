#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "SimpleOperator.h"
#include <iostream>
#include<windows.h>
using namespace std;

int main() {

	cv::Mat image = cv::imread("C:/Users/Iris/Desktop/1.jpg");
	cv::Mat result;
	LaplacianZC laplace;
	laplace.setAperture(7);
	result = laplace.getZeroCrossing(image, 0);
	cv::namedWindow("my result");
	cv::imshow("my result", result);
	cv::waitKey(0);
	return 0;


}

