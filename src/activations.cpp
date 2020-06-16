#include "layer.h"
#include <iostream>

using namespace std;
using namespace cv;

Relu::Relu(){}

Mat Relu::forward(Mat input)
{
    this->kernel = Mat::zeros(input.rows, input.cols, CV_32F);
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++)
            if (input.at<double>(i, j) <= 0){
                this->kernel = 1;
                this->input = 0;
            }

    return input;
}

Mat Relu::backward(Mat loss)
{
    for (int i = 0; i < this->kernel.rows; i++)
        for (int j = 0; j < this->kernel.cols; j++)
            loss.at<double>(i, j) = 0 ? this->kernel.at<double>(i, j) == 1 : loss.at<double>(i, j);

    return loss;
}