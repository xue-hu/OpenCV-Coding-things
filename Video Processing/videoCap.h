#pragma once


#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;

void canny(cv::Mat& img, cv::Mat& out);
void processFrame(cv::Mat& frame, cv::Mat& out);

class videoProcessor {
private :
	cv::VideoCapture capture;
	void(*process)(cv::Mat&, cv::Mat&);
	bool callIt;
	string InWinName;
	string OutWinName;
	int delay;
	long fnumber;
	long frameToStop;
	bool stop;

public:
	videoProcessor();
	bool setInput(string file_path);
	void displayInput(string In_Win);
	void displayOutput(string Out_Win);
	long getFrameRate();
	void setDelay(int t);
	void stopProcess();
	void StopCallIt();
	void CallIt();
	void stopAtFrameNo(long frame);
	void setFrameProcessor(void(*frameProcessingCallback)(cv::Mat&, cv::Mat&));
	long getFrameNumber();
	void run();

};