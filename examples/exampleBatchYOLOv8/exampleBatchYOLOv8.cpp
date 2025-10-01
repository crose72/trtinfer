#include "YOLOv8.h"

void getBatchInputs(std::vector<std::string> imgPaths,
                    std::vector<cv::cuda::GpuMat> *imgs_gpu,
                    std::vector<cv::Mat> *imgs_cpu)
{
    // ---- [2] Load images into cv::Mat ----
    for (const auto &path : imgPaths)
    {
        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            std::cerr << "Failed to load " << path << std::endl;
            continue;
        }
        imgs_cpu->push_back(img); // Now this matches vector<cv::Mat>
    }

    if (imgs_cpu->empty())
    {
        std::cerr << "No valid images loaded!" << std::endl;
    }

    // ---- [3] Upload images to GPU (cv::cuda::GpuMat) ----
    for (const auto &img : *imgs_cpu)
    {
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(img);
        imgs_gpu->push_back(gpuImg);
    }

    std::cout << "Num Images = " << imgs_gpu->size() << std::endl;
}

int main(void)
{
    // Get images to perform inference on
    std::vector<cv::cuda::GpuMat> imgs_gpu;
    std::vector<cv::Mat> imgs_cpu;

    std::vector<std::string> imgPaths = {
        "/workspace/trtinfer/examples/scratch/elephant.jpg",
        "/workspace/trtinfer/examples/scratch/squirrel.jpg",
        "/workspace/trtinfer/examples/scratch/border-collie.jpg",
        "/workspace/trtinfer/examples/scratch/people.jpg"};

    getBatchInputs(imgPaths, &imgs_gpu, &imgs_cpu);

    // Setup yolo
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/trtinfer/examples/scratch/yolov8s_batch.engine", config);
    yolo.printEngineInfo();

    // Perform preprocessing, inference, and postprocessing
    // Can provide cpu or gpu images
    std::vector<std::vector<Object>> detections = yolo.detectObjects(imgs_cpu);

    for (int i = 0; i < imgs_gpu.size(); ++i)
    {
        cv::Mat cpuImg;

        imgs_gpu[i].download(cpuImg);
        // Draw results
        yolo.drawObjectLabels(cpuImg, detections[i]);

        // Show the frame
        cv::imshow("YOLOv8 Detection", cpuImg);
        cv::waitKey(0);
    }

    return 0;
}
