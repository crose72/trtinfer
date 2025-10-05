#include "ResNet50.h"

int main(int argc, char **argv)
{
    ResNet50 resnet50Classifier("/workspace/examples/models/resnet50.engine");
    resnet50Classifier.init();
    std::vector<std::string> testImgsList;

    testImgsList.push_back("/workspace/examples/sampleData/elephant.jpg");
    testImgsList.push_back("/workspace/examples/sampleData/border-collie.jpg");
    testImgsList.push_back("/workspace/examples/sampleData/squirrel.jpg");

    for (const auto testImg : testImgsList)
    {
        cv::Mat img = cv::imread(testImg);
        std::cout << "Image: " << testImg << std::endl;
        resnet50Classifier.classify(img);
    }

    return 0;
}
