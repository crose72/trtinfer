#include "ResNet50.h"

int main(int argc, char **argv)
{
    ResNet50 resnet50Classifier("/workspace/examples/exampleResnet50_v2/resnet_engine_intro.engine");
    resnet50Classifier.init();
    std::vector<std::string> testImages;

    testImages.push_back("/workspace/examples/exampleResnet50_v2/elephant-1.jpg");
    testImages.push_back("/workspace/examples/exampleResnet50_v2/border-collie-1.jpg");
    testImages.push_back("/workspace/examples/exampleResnet50_v2/squirrel-1.jpg");

    std::vector<cv::Mat> testImagesCV;

    for (const auto image : testImages)
    {
        std::cout << "Image: " << image << std::endl;
        resnet50Classifier.classify(cv::imread(image));
    }

    return 0;
}
