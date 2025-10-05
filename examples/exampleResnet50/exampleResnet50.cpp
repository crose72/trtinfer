#include "ResNet50.h"

int main(int argc, char **argv)
{
    ResNet50 resnet50Classifier("/workspace/examples/exampleResnet50/resnet50.engine");
    resnet50Classifier.init();
    std::vector<std::string> testImgsList;

    testImgsList.push_back("/workspace/sampleData/elephant.jpg");
    testImgsList.push_back("/workspace/sampleData/border-collie.jpg");
    testImgsList.push_back("/workspace/sampleData/squirrel.jpg");

    for (const auto testImg : testImgsList)
    {
        cv::Mat img = cv::imread(testImg);
        std::cout << "Image: " << testImg << std::endl;
        resnet50Classifier.classify(img);
    }

    return 0;
}
