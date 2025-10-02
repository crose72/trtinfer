#pragma once

#include "TRTEngine.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <array>
#include <vector>

struct Object
{
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};

class YOLOv8
{
public:
    // Config the behavior of the YOLOv8 detector.
    // Can pass these arguments as command line parameters.
    struct Config
    {
        // Probability threshold used to filter detected objects
        float probabilityThreshold = (float)0.25;
        // Non-maximum suppression threshold
        float nmsThreshold = (float)0.65;
        // Max number of detected objects to return
        int topK = 100;
        // Segmentation config options
        int segChannels = 32;
        int segH = 160;
        int segW = 160;
        float segmentationThreshold = (float)0.5;
        // Pose estimation options
        int numKPS = 17;
        float kpsThreshold = (float)0.5;
        // Class thresholds (default are COCO classes)
        std::vector<std::string> classNames = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"};
    };

    enum PostProcessType
    {
        YOLO_DET,
        YOLO_SEG,
        YOLO_POSE
    };

    YOLOv8(const std::string &trtModelPath, const Config &config);

    std::vector<Object> detectObjects(const cv::Mat &inputImageBGR);
    std::vector<Object> detectObjects(const cv::cuda::GpuMat &inputImageBGR);
    std::vector<std::vector<Object>> detectObjects(const std::vector<cv::Mat> &batchImgs);
    std::vector<std::vector<Object>> detectObjects(const std::vector<cv::cuda::GpuMat> &batchImgs);
    void decodeYOLOAnchors(
        const std::vector<float> &output,
        int numAnchors,
        int numClasses,
        float detectionThreshold,
        float aspectScaleFactor,
        float imgWidth,
        float imgHeight,
        std::vector<cv::Rect> &bboxes,
        std::vector<float> &scores,
        std::vector<int> &labels,
        PostProcessType type,
        // Optional: mask coeffs and keypoints containers (pass nullptr if unused)
        std::vector<cv::Mat> *maskConfs = nullptr,
        std::vector<std::vector<float>> *kpss = nullptr,
        int numMaskChannels = 0,
        int numKeypoints = 0);
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale = 2);
    void printEngineInfo(void) { mEngine->printEngineInfo(); };
    void init(void) { mEngine->loadNetwork(
        mEnginePath,
        mMean,
        mStd,
        mNormalize); };

private:
    // Overloaded function for single input preprocessing
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg);
    // Overloaded function for batch input preprocessing
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const std::vector<cv::cuda::GpuMat> &gpuImgs);

    // Overloaded function for single input object detection post processing
    std::vector<Object> postprocessDetect(std::vector<float> &featureVector);
    // Overloaded function for batch input object detection post processing
    std::vector<Object> postprocessDetect(std::vector<float> &featureVector, int imageInBatch);
    // Postprocess the output for pose model
    std::vector<Object> postprocessPose(std::vector<float> &featureVector);
    // Postprocess the output for segmentation model
    std::vector<Object> postProcessSegmentation(std::vector<std::vector<float>> &featureVectors);

    // Engine input/output parameters
    int64_t mEngineInputHeight;
    int64_t mEngineInputWidth;
    int64_t mEngineBatchSize;
    int64_t mNumAnchorFeatures;
    int64_t mNumAnchors;
    size_t mNumOutputTensors;

    // Engine instance
    std::string mEnginePath;
    std::unique_ptr<TRTEngine<float>> mEngine;

    // Mean and Std to subtract and divie each pixel by before
    // passing input image to the engine
    std::array<float, 3> mMean = {(float)0.0, (float)0.0, (float)0.0};
    std::array<float, 3> mStd = {(float)1.0, (float)1.0, (float)1.0};
    bool mNormalize = true; // normalize to [0,1] before mean/std

    // single input image parameters
    float mInputImgHeight = (float)0.0;
    float mInputImgWidth = (float)0.0;
    float mAspectScaleFactor = (float)1.0;

    // batch input image parameters
    int mActualBatchSize;
    std::vector<float> mInputImgHeights;
    std::vector<float> mInputImgWidths;
    std::vector<float> mAspectScaleFactors;

    const float mNMSThreshold = (float)0.65; // Non-maximum suppression threshold
    const int mTopK = 100;                   // Max number of detected objects to return
    int mNumClasses = 80;                    // default for COCO dataset
    int mNumPoseAnchorFeatures = 56;         // default for pose estimation
    int mNumDetectAnchorFeatures = 84;       // default for object detection

    // Object classes as strings
    const std::vector<std::string> mClassNames;

    // detection params
    const float mDetectionThreshold = (float)0.25;

    // Segmentation constants
    const int mSegChannels;
    const int mSegHeight;
    const int mSegWidth;
    const float mSegThreshold;

    // Pose estimation constant
    const int mNumKPS;
    const float mKPSThresh;

    // Color list for drawing objects
    const std::vector<std::vector<float>> mColorList = {{1, 1, 1},
                                                        {0.098, 0.325, 0.850},
                                                        {0.125, 0.694, 0.929},
                                                        {0.556, 0.184, 0.494},
                                                        {0.188, 0.674, 0.466},
                                                        {0.933, 0.745, 0.301},
                                                        {0.184, 0.078, 0.635},
                                                        {0.300, 0.300, 0.300},
                                                        {0.600, 0.600, 0.600},
                                                        {0.000, 0.000, 1.000},
                                                        {0.000, 0.500, 1.000},
                                                        {0.000, 0.749, 0.749},
                                                        {0.000, 1.000, 0.000},
                                                        {1.000, 0.000, 0.000},
                                                        {1.000, 0.000, 0.667},
                                                        {0.000, 0.333, 0.333},
                                                        {0.000, 0.667, 0.333},
                                                        {0.000, 1.000, 0.333},
                                                        {0.000, 0.333, 0.667},
                                                        {0.000, 0.667, 0.667},
                                                        {0.000, 1.000, 0.667},
                                                        {0.000, 0.333, 1.000},
                                                        {0.000, 0.667, 1.000},
                                                        {0.000, 1.000, 1.000},
                                                        {0.500, 0.333, 0.000},
                                                        {0.500, 0.667, 0.000},
                                                        {0.500, 1.000, 0.000},
                                                        {0.500, 0.000, 0.333},
                                                        {0.500, 0.333, 0.333},
                                                        {0.500, 0.667, 0.333},
                                                        {0.500, 1.000, 0.333},
                                                        {0.500, 0.000, 0.667},
                                                        {0.500, 0.333, 0.667},
                                                        {0.500, 0.667, 0.667},
                                                        {0.500, 1.000, 0.667},
                                                        {0.500, 0.000, 1.000},
                                                        {0.500, 0.333, 1.000},
                                                        {0.500, 0.667, 1.000},
                                                        {0.500, 1.000, 1.000},
                                                        {1.000, 0.333, 0.000},
                                                        {1.000, 0.667, 0.000},
                                                        {1.000, 1.000, 0.000},
                                                        {1.000, 0.000, 0.333},
                                                        {1.000, 0.333, 0.333},
                                                        {1.000, 0.667, 0.333},
                                                        {1.000, 1.000, 0.333},
                                                        {1.000, 0.000, 0.667},
                                                        {1.000, 0.333, 0.667},
                                                        {1.000, 0.667, 0.667},
                                                        {1.000, 1.000, 0.667},
                                                        {1.000, 0.000, 1.000},
                                                        {1.000, 0.333, 1.000},
                                                        {1.000, 0.667, 1.000},
                                                        {0.000, 0.000, 0.333},
                                                        {0.000, 0.000, 0.500},
                                                        {0.000, 0.000, 0.667},
                                                        {0.000, 0.000, 0.833},
                                                        {0.000, 0.000, 1.000},
                                                        {0.000, 0.167, 0.000},
                                                        {0.000, 0.333, 0.000},
                                                        {0.000, 0.500, 0.000},
                                                        {0.000, 0.667, 0.000},
                                                        {0.000, 0.833, 0.000},
                                                        {0.000, 1.000, 0.000},
                                                        {0.167, 0.000, 0.000},
                                                        {0.333, 0.000, 0.000},
                                                        {0.500, 0.000, 0.000},
                                                        {0.667, 0.000, 0.000},
                                                        {0.833, 0.000, 0.000},
                                                        {1.000, 0.000, 0.000},
                                                        {0.000, 0.000, 0.000},
                                                        {0.143, 0.143, 0.143},
                                                        {0.286, 0.286, 0.286},
                                                        {0.429, 0.429, 0.429},
                                                        {0.571, 0.571, 0.571},
                                                        {0.714, 0.714, 0.714},
                                                        {0.857, 0.857, 0.857},
                                                        {0.741, 0.447, 0.000},
                                                        {0.741, 0.717, 0.314},
                                                        {0.000, 0.500, 0.500}};
    const std::vector<std::vector<unsigned int>> mKPSColors = {
        {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

    const std::vector<std::vector<unsigned int>> mSkeleton = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

    const std::vector<std::vector<unsigned int>> mLimbColors = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}};
};