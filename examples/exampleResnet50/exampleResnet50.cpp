#include "ResNet50.h"

int main(int argc, char **argv)
{
    ResNet50 resnet50Classifier("/workspace/examples/exampleResnet50/resnet50.engine");
    resnet50Classifier.init();
    std::vector<std::string> testImages;

    testImages.push_back("/workspace/examples/exampleResnet50/elephant.jpg");
    testImages.push_back("/workspace/examples/exampleResnet50/border-collie.jpg");
    testImages.push_back("/workspace/examples/exampleResnet50/squirrel.jpg");

    std::vector<cv::Mat> testImagesCV;

    for (const auto image : testImages)
    {
        std::cout << "Image: " << image << std::endl;
        resnet50Classifier.classify(cv::imread(image));
    }

    return 0;
}
