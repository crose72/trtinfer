#include "YOLOv8.h"

int main(void)
{
    // Create engine directly from engine file path
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/exampleYOLOv8/yolov8s.engine", config);
    // yolo.printEngineInfo();
    std::string image = "/workspace/examples/exampleYOLOv8/elephant.jpg";
    cv::Mat testImage = cv::imread(image);
    std::vector<Object> detections = yolo.detectObjects(testImage);
    yolo.drawObjectLabels(testImage, detections);
    cv::imwrite("result.jpg", testImage);
    yolo.printEngineInfo();
    return 0;
}