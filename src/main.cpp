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
    layerList.push_back(new nnLayer(64));
    layerList.push_back(new nnLayer(1));

    for(auto layer:layerList){
        temp = layer->forward(temp);
    }
    // layer *NN_layer1 = new nnLayer(128);
    // layer *NN_layer2 = new nnLayer(64);
    // layer *NN_layer3 = new nnLayer(1);

    // temp = NN_layer1->forward(input);
    // temp = NN_layer2->forward(temp);
    // temp = NN_layer3->forward(temp);

    system("pause");

    return 0;
}