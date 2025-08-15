#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "util.h"

// For postprocessing
#include <algorithm> // partial_sort, min
#include <numeric>   // iota
#include <vector>

// For cropping and saving image as ppm
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

#include <filesystem>
#include <cuda_fp16.h>

//!
//! \class TensorInfer
//!
//! \brief Implements semantic segmentation using FCN-ResNet101 ONNX model.
//!
class TensorInfer
{
public:
    struct Config
    {
        std::string engineFilename;
        nvinfer1::TensorFormat tensorFormat = nvinfer1::TensorFormat::kLINEAR;
        int32_t batchSize = 1;
        int32_t numChannels = 3;
        int32_t height = 224;
        int32_t width = 224;
        nvinfer1::DataType precision = nvinfer1::DataType::kFLOAT;
        std::string inputTensorName;                // Optional override
        std::string outputTensorName;               // Optional override
        std::vector<std::string> outputTensorNames; // If model has multiple outputs (e.g. YOLO: boxes, scores, classes)
    };

    TensorInfer(const std::string &engineFilename);
    TensorInfer(const Config &config);
    ~TensorInfer();
    std::unique_ptr<float> infer(const std::unique_ptr<float> &input);
    std::unique_ptr<__half> infer(const std::unique_ptr<__half> &input);
    bool init(void);

private:
    Config mConfig;

    std::string mEngineFilename;                           //!< Filename of the serialized engine.
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;          //!< The TensorRT runtime used to run the network
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;        //!< The TensorRT engine used to run the network
    std::unique_ptr<nvinfer1::IExecutionContext> mContext; //!< The TensorRT execution mContext
    nvinfer1::Dims mInputDims;                             //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;                            //!< The dimensions of the output to the network.
    int32_t mBatchSize = 1;
    int32_t mNumChannels = 3;
    int32_t mHeight = 224;
    int32_t mWidtch = 224;
    char const *mInputName;
    char const *mOutputName;
    size_t mInputSize;
    size_t mOutputSize;

    // TODO make static? or just instantiate class at global scope?
    // Should persist for duration of program
    // Clean up in destructor?
    cudaStream_t mStream = nullptr;
    void *mInputMem = nullptr;
    void *mOutputMem = nullptr;

    char const *getNodeName(nvinfer1::TensorIOMode nodeType);
    size_t dataMemSize(nvinfer1::DataType dataType);
};