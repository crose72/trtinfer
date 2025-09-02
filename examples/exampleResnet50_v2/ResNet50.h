#pragma once

#include "TRTEngine.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <array>
#include <vector>

class ResNet50
{
public:
    ResNet50(const std::string &engine_path);

    // Run inference: input is CPU or GPU Mat (single image)
    std::vector<float> classify(const cv::Mat &img_cpu);
    std::vector<float> classify(const cv::cuda::GpuMat &img_gpu);

    void init(void) { mEngine.loadNetwork(
        mEnginePath,
        mMean,
        mStd,
        mNormalize); };

private:
    TRTEngine<float> mEngine;
    std::array<float, 3> mMean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> mStd = {0.229f, 0.224f, 0.225f};
    bool mNormalize = true; // normalize to [0,1] before mean/std
    std::string mEnginePath;

    static cv::Mat preprocess(const cv::Mat &img, int target_w = 224, int target_h = 224);
    static void postprocess(const std::vector<float> &scores);
};
