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

mutiLayer::mutiLayer(int neural) :layer( neural) {}

Mat mutiLayer::backward(Mat loss) {
	this->grad = this->input.t() * loss;
	return loss * this->kernel.t();
}

Mat mutiLayer::forward(Mat input) {
	this->input = input.clone();
	this->kernel = Mat::zeros(input.cols, neural, CV_32F);
	randn(this->kernel, 0, 1);
	cout << this->kernel.type() << "　" << this->input.type() << endl;
	return this->input * this->kernel;
}

addLayer::addLayer(int neural) :layer(neural) {}

Mat addLayer::backward(Mat loss) {
	this->grad =  loss;
	return loss ;
}

Mat addLayer::forward(Mat input) {
	this->input = input;
	this->kernel = Mat::zeros(1, this->neural, CV_32F);
	return this->input + this->kernel;
}


nnLayer::nnLayer(int neural):layer(neural){
	this->add = new addLayer(neural);
	this->multi = new mutiLayer(neural);
}

Mat nnLayer::backward(Mat loss) {
	return loss ;
}

Mat nnLayer::forward(Mat input){
	input = this->multi->forward(input);
	input = this->add->forward(input);
	return input;
}