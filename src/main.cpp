#include "main.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    Mat input = imread("img/img_1.jpg", 0);

    input = input.reshape(0, 1);
    input.convertTo(input, CV_32F);

    Mat temp = input.clone();

    vector<layer*> layerList;
    layerList.push_back(new nnLayer(128));
    layerList.push_back(new Relu());
    layerList.push_back(new nnLayer(64));
    layerList.push_back(new Relu());
    layerList.push_back(new nnLayer(1));

    for(auto layer:layerList){
        temp = layer->forward(temp);
    }
// loss function
    for (auto layer = layerList.rbegin(); layer != layerList.rend(); layer++)
    {
        temp = (*layer)->backward(temp);
    }
// optimal

    system("pause");

    return 0;
}