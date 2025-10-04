#include "YOLOv8.h"

int main(void)
{
    // Need to preprocess images with yolo first
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/models/yolov8s_seg_batch_fp32.engine", config);
    yolo.printEngineInfo();

    std::vector<cv::Mat> imgs_cpu;

    std::vector<std::string> imgPaths = {
        "/workspace/examples/media/elephant.jpg",
        "/workspace/examples/media/squirrel.jpg",
        "/workspace/examples/media/border-collie.jpg",
        "/workspace/examples/media/people.jpg"};

    for (const auto &path : imgPaths)
    {
        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            std::cerr << "Failed to load " << path << std::endl;
            continue;
        }
        imgs_cpu.push_back(img);
    }

    std::vector<std::vector<Object>> detections = yolo.detectObjects(imgs_cpu);

    for (int i = 0; i < imgs_cpu.size(); ++i)
    {
        // Draw results
        yolo.drawObjectLabels(imgs_cpu[i], detections[i]);

        // Show the frame
        cv::imshow("YOLOv8 Detection", imgs_cpu[i]);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();

    return 0;
}