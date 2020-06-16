#include "layer.h"
#include <iostream>

using namespace std;
using namespace cv;

layer::layer(Mat input, int neural) {
	this->input = input;
	this->neural = neural;
}

layer::layer(int neural) {
	this->neural = neural;
}

layer::layer(){}

mutiLayer::mutiLayer(int neural) :layer( neural) {}

Mat mutiLayer::backward(Mat loss) {
	this->grad = this->input.t() * loss;
	Mat tmp = loss * this->kernel.t();
	return tmp;
}

Mat mutiLayer::forward(Mat input) {
	this->input = input.clone();

	if (this->kernel.empty()){
		this->kernel = Mat::zeros(input.cols, neural, CV_32F);
		randn(this->kernel, 0, 1);
	}

	return this->input * this->kernel;
}

addLayer::addLayer(int neural) :layer(neural) {}

Mat addLayer::backward(Mat loss) {
	this->grad =  loss;
	return loss;
}

Mat addLayer::forward(Mat input) {
	this->input = input.clone();

	if (this->kernel.empty())
	{
		this->kernel = Mat::zeros(1, neural, CV_32F);
		randn(this->kernel, 0, 1);
	}

	return this->input + this->kernel;
}


nnLayer::nnLayer(int neural):layer(neural){
	this->add = new addLayer(neural);
	this->multi = new mutiLayer(neural);
}

Mat nnLayer::backward(Mat loss) {
	loss = this->add->backward(loss);
	loss = this->multi->backward(loss);
	return loss;
}

Mat nnLayer::forward(Mat input){
	input = this->multi->forward(input);
	input = this->add->forward(input);
	return input;
}

