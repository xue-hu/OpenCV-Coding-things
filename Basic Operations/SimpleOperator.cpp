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


ColorDetector::ColorDetector() {
		minDist = 100;
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




