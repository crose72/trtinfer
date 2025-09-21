#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <NvInferRuntimePlugin.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>

#include "logging.h"
#include "IEngine.h"

template <typename T>
class TRTEngine : public IEngine<T>
{
public:
    // Precision to build TRT engine with for GPU inference
    enum class Precision
    {
        // Full precision floating point value
        FP32,
        // Half prevision floating point value
        FP16,
        // Int8 quantization.
        // Has reduced dynamic range, may result in slight loss in accuracy.
        // If INT8 is selected, must provide path to calibration dataset directory.
        INT8,
    };

    // BuildOptions for building a TRT engine
    struct BuildOptions
    {
        // Precision to use for GPU inference.
        Precision precision = Precision::FP16;
        // If INT8 precision is selected, must provide path to calibration dataset
        // directory.
        std::string calibrationDataDirectoryPath;
        // The batch size to be used when computing calibration data for INT8
        // inference. Should be set to as large a batch number as your GPU will
        // support.
        int32_t calibrationBatchSize = 128;
        // The batch size which should be optimized for.
        int32_t optBatchSize = 1;
        // Maximum allowable batch size
        int32_t maxBatchSize = 16;
        // GPU device index
        int deviceIndex = 0;
        // Directory where the engine file should be saved
        std::string engineFileDir = ".";
        // Maximum allowed input width
        int32_t maxInputWidth = -1; // Default to -1 --> expecting fixed input size
        // Minimum allowed input width
        int32_t minInputWidth = -1; // Default to -1 --> expecting fixed input size
        // Optimal input width
        int32_t optInputWidth = -1; // Default to -1 --> expecting fixed input size
    };

    TRTEngine(const std::string &engineFilename);
    TRTEngine();
    ~TRTEngine();

    void printEngineInfo(void) const;
    const std::vector<std::string> &getInputNames() const override { return mInputNames; };
    const std::vector<std::string> &getOutputNames() const override { return mOutputNames; };
    const std::vector<nvinfer1::Dims> &getInputDims() const override { return mInputDims; };
    const std::vector<nvinfer1::Dims> &getOutputDims() const override { return mOutputDims; };
    const std::vector<nvinfer1::TensorFormat> &getInputTensorFormats() const override { return mInputTensorFormats; };
    const std::vector<nvinfer1::TensorFormat> &getOutputTensorFormats() const override { return mOutputTensorFormats; };
    const std::vector<nvinfer1::DataType> &getInputDataType() const override { return mInputDataTypes; };
    const std::vector<nvinfer1::DataType> &getOutputDataType() const override { return mOutputDataTypes; };

    // Build the onnx model into a TensorRT engine file, cache the model to disk
    // (to avoid rebuilding in future), and then load the model into memory The
    // default implementation will normalize values between [0.f, 1.f] Setting the
    // normalize flag to false will leave values between [0.f, 255.f] (some
    // converted models may require this). If the model requires values to
    // be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool buildLoadNetwork(const std::string &onnxModelPath,
                          const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                          const std::array<float, 3> &divVals = {1.f, 1.f, 1.f},
                          bool normalize = true);

    // Load a TensorRT engine file from disk into memory
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this). If the model requires values to
    // be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool loadNetwork(const std::string &trtModelPath,
                     const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                     const std::array<float, 3> &divVals = {1.f, 1.f, 1.f},
                     bool normalize = true);
    bool loadNetwork(void);

    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                      std::vector<std::vector<std::vector<T>>> &featureVectors);

private:
    void getEngineInfo(void);
    bool build(
        std::string onnxModelPath,
        const std::array<float, 3> &subVals,
        const std::array<float, 3> &divVals,
        bool normalize);
    // Convert NHWC to NCHW and apply scaling and mean subtraction
    static cv::cuda::GpuMat blobFromGpuMats(
        const std::vector<cv::cuda::GpuMat> &batchInput,
        const std::array<float, 3> &subVals,
        const std::array<float, 3> &divVals,
        bool normalize,
        bool swapRB = false);
    static cv::cuda::GpuMat packBatchToNCHW(const std::vector<cv::cuda::GpuMat> &batch, int H, int W);
    void getDeviceNames(std::vector<std::string> &deviceNames);
    void clearGpuBuffers();

    // Members
    BuildOptions mOptions;
    std::array<float, 3> mSubVals{};
    std::array<float, 3> mDivVals{};
    bool mNormalize = true;
    int mDeviceIndex = 0;
    int32_t mMaxBatchSize = 1;
    int32_t mOptProfileIndex = 0;
    int32_t mInputBatchSize = 1;
    std::string mEngineFilename;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    std::vector<nvinfer1::Dims> mInputDims;
    std::vector<nvinfer1::Dims> mOutputDims;
    std::vector<nvinfer1::TensorFormat> mInputTensorFormats;
    std::vector<nvinfer1::TensorFormat> mOutputTensorFormats;
    std::vector<nvinfer1::DataType> mInputDataTypes;
    std::vector<nvinfer1::DataType> mOutputDataTypes;
    std::vector<uint32_t> mOutputLengths{};
    std::vector<void *> mBuffers;
    std::vector<std::string> mIOTensorNames;
    std::vector<nvinfer1::TensorIOMode> mTensorTypes;
    Logger mLogger;

    // Must keep IRuntime around for inference, see:
    // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime>
        mRuntime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext = nullptr;
    // std::unique_ptr<Int8EntropyCalibrator2> mCalibrator = nullptr;
};

#include "TRTEngine.inl"