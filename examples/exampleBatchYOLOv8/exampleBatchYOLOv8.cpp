#include "YOLOv8.h"

int main(void)
{
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/exampleBatchYOLOv8/yolov8s_batch.engine", config);

    // ---- [1] Define your batch of images ----
    std::vector<std::string> imgPaths = {
        "/workspace/examples/exampleBatchYOLOv8/elephant.jpg",
        "/workspace/examples/exampleBatchYOLOv8/squirrel.jpg",
        "/workspace/examples/exampleBatchYOLOv8/border-collie.jpg"};

    // ---- [2] Load images into cv::Mat ----
    std::vector<cv::Mat> imgs_cpu;
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

    if (imgs_cpu.empty())
    {
        std::cerr << "No valid images loaded!" << std::endl;
        return 1;
    }

    // ---- [3] Upload images to GPU (cv::cuda::GpuMat) ----
    std::vector<cv::cuda::GpuMat> imgs_gpu;
    for (const auto img : imgs_cpu)
    {
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(img);
        imgs_gpu.push_back(gpuImg);
    }

    // ---- [4] Run batch inference ----
    auto batchResults = yolo.detectObjects(imgs_gpu);
    return 0;
}