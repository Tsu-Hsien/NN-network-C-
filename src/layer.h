#ifndef LAYER_H
#define LAYER_H

#include <opencv2/opencv.hpp>

class layer {
protected:
	cv::Mat kernel;
	cv::Mat grad;
	cv::Mat input;
	int neural;

public:
	layer(cv::Mat input, int neural);
	layer(int neural);
	virtual cv::Mat forward(cv::Mat) = 0;
	virtual cv::Mat backward(cv::Mat) = 0;
};

class mutiLayer:public layer
{
public:
	cv::Mat forward(cv::Mat);
	cv::Mat backward(cv::Mat loss);
	mutiLayer(int neural);
};

class addLayer :public layer
{
public:
	cv::Mat forward(cv::Mat);
	cv::Mat backward(cv::Mat loss);
	addLayer(int neural);
	//~addLayer();
};

class nnLayer: public layer
{
private:
	mutiLayer* multi;
	addLayer* add;
public:
	cv::Mat forward(cv::Mat);
	cv::Mat backward(cv::Mat loss);
	nnLayer(int neural);
};

#endif