#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "videoCap.h"
#include <iostream>
#include<windows.h>

using namespace std;

void canny(cv::Mat& img, cv::Mat& out) {
	if (img.channels() == 3)
		cv::cvtColor(img, out, CV_BGR2GRAY);
	cv::Canny(out, out, 100, 100);
	cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
}



videoProcessor::videoProcessor():callIt(true), delay(0),
														fnumber(0), stop(false), frameToStop(-1){
	cv::destroyAllWindows();

}


bool videoProcessor::setInput(string file_path) {
	fnumber = 0;
	capture.release();
	return capture.open(file_path);
	
}

void videoProcessor::displayInput(string In_Win) {
	cv::destroyWindow(In_Win);
	InWinName = In_Win;
	cv::namedWindow(In_Win);
}

void videoProcessor::displayOutput(string Out_Win) {
	cv::destroyWindow(Out_Win);
	OutWinName = Out_Win;
	cv::namedWindow(Out_Win);
}

void videoProcessor::stopProcess() {
	stop = true;
}

void videoProcessor::StopCallIt() {
	callIt = false;
}

void videoProcessor::CallIt() {
	callIt = true;
}

long videoProcessor::getFrameRate() {
	long t = static_cast<long>(
					capture.get(CV_CAP_PROP_FRAME_COUNT));
	return t;
}

void videoProcessor::run() {
	cv::Mat frame;
	cv::Mat out;
	if (!capture.isOpened())
		return;
	stop = false;
	while (capture.isOpened())
	{
		if (!capture.read(frame))
			break;
		if (InWinName.length() != 0)
			cv::imshow(InWinName, frame);
		if (callIt)
		{
			process(frame, out);
			fnumber++;
		}
		else 
		{
			out = frame;
		}
		if (OutWinName.length() != 0)
			cv::imshow(OutWinName, out);
		if (delay >= 0 && cv::waitKey(delay) >= 0)
			stop = false;
		if (frameToStop >= 0 && getFrameNumber() == frameToStop)
			stop = true;
	}

}

void videoProcessor::setDelay(int t) {
	delay = t;
}

long videoProcessor::getFrameNumber() {
	long fnumber = static_cast<long>(
								capture.get(CV_CAP_PROP_POS_FRAMES));
	return fnumber;
}

void videoProcessor::stopAtFrameNo(long frame) {
	frameToStop = frame;
}

void videoProcessor::setFrameProcessor(
	void(*frameProcessingCallback)(cv::Mat&, cv::Mat&) ){
		process = frameProcessingCallback;
	}
