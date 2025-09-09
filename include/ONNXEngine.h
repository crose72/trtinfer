#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "IEngine.h"

// You may want to move this into a .cpp for non-header-only use!
template <typename T>
class ONNXEngine : public IEngine<T>
{
public:
    struct Options
    {
        int deviceIndex = 0;                            // CUDA device, if using GPU
        bool useCUDA = true;                            // Use CUDA (otherwise CPU)
        std::string provider = "CUDAExecutionProvider"; // OR "CPUExecutionProvider"
        std::string modelPath;                          // Path to .onnx file
        int numThreads = 1;                             // For CPU only
    };

    // Construct and initialize session
    ONNXEngine(const std::string &modelPath, const Options &opts = Options())
        : mOptions(opts)
    {
        mOptions.modelPath = modelPath;
        init();
    }

    ONNXEngine(const Options &opts = Options())
        : mOptions(opts)
    {
        if (!mOptions.modelPath.empty())
            init();
    }

    ~ONNXEngine() {}

    void printEngineInfo(void) const override
    {
        spdlog::info("=== ONNX Engine Information ===");
        spdlog::info("Model: {}", mOptions.modelPath);
        spdlog::info("Provider: {}", mOptions.provider);
        for (size_t i = 0; i < mInputNames.size(); ++i)
            spdlog::info("Input[{}]: {} shape={}", i, mInputNames[i], vecToStr(mInputShapes[i]));
        for (size_t i = 0; i < mOutputNames.size(); ++i)
            spdlog::info("Output[{}]: {} shape={}", i, mOutputNames[i], vecToStr(mOutputShapes[i]));
        spdlog::info("===============================");
    }

    const std::vector<std::string> &getInputNames() const override { return mInputNames; }
    const std::vector<std::string> &getOutputNames() const override { return mOutputNames; }
    const std::vector<std::vector<int64_t>> &getInputDims() const { return mInputShapes; }
    const std::vector<std::vector<int64_t>> &getOutputDims() const { return mOutputShapes; }

    // Example: for detection model, batch size 1, 1 image input (NHWC as float)
    // input: [input][batch][cv::cuda::GpuMat], output: [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                      std::vector<std::vector<std::vector<T>>> &featureVectors)
    {
        // For batch size 1, single input, this will be one element.
        if (inputs.empty() || inputs[0].empty())
            return false;

        // Convert cv::cuda::GpuMat -> cv::Mat -> std::vector<float>
        // (for ONNX Runtime, usually expects NCHW float input, CPU memory)
        std::vector<std::vector<float>> preparedInputs;
        for (const auto &batch : inputs)
        {
            for (const auto &gpuMat : batch)
            {
                cv::Mat cpuMat;
                gpuMat.download(cpuMat);
                cpuMat.convertTo(cpuMat, CV_32F, 1.0f / 255.f);
                // Numpy-style flatten (row-major)
                preparedInputs.push_back(std::vector<float>((float *)cpuMat.datastart, (float *)cpuMat.dataend));
            }
        }

        size_t batchSize = preparedInputs.size();
        size_t inputSize = preparedInputs[0].size();

        // Create Ort tensors
        std::vector<Ort::Value> ortInputs;
        std::vector<const char *> inputNamesChar;
        for (size_t i = 0; i < mInputNames.size(); ++i)
            inputNamesChar.push_back(mInputNames[i].c_str());

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Note: this example assumes only 1 input (update for multiple inputs!)
        std::vector<int64_t> inputShape = mInputShapes[0];
        inputShape[0] = batchSize; // Dynamic batch

        ortInputs.emplace_back(Ort::Value::CreateTensor<float>(memInfo,
                                                               preparedInputs[0].data(), inputSize * batchSize, inputShape.data(), inputShape.size()));

        // Output allocation (let ONNX Runtime allocate)
        auto outputNamesChar = getOutputNamesChar();

        // Run inference
        auto ortOutputs = mSession->Run(Ort::RunOptions{nullptr},
                                        inputNamesChar.data(), ortInputs.data(), ortInputs.size(),
                                        outputNamesChar.data(), mOutputNames.size());

        // Convert outputs to [batch][output][feature_vector] (mirror TRTEngine)
        featureVectors.clear();
        for (size_t i = 0; i < ortOutputs.size(); ++i)
        {
            float *outputData = ortOutputs[i].GetTensorMutableData<float>();
            auto &shape = mOutputShapes[i];
            size_t outputSize = 1;
            for (auto d : shape)
                outputSize *= d;
            // TODO: For batch output, split properly (for now, batch size 1)
            std::vector<std::vector<T>> outputBatch;
            outputBatch.push_back(std::vector<T>(outputData, outputData + outputSize));
            featureVectors.push_back(outputBatch);
        }
        return true;
    }

    // Optionally, add overloaded runInference to support cv::Mat or vector<float> input

private:
    void init()
    {
        Ort::Env &env = getOrtEnv();
        mSessionOptions = std::make_unique<Ort::SessionOptions>();
        if (mOptions.useCUDA)
        {
            OrtCUDAProviderOptions cuda_options{};
            mSessionOptions->AppendExecutionProvider_CUDA(mOptions.deviceIndex);
            mOptions.provider = "CUDAExecutionProvider";
        }
        else
        {
            mSessionOptions->SetIntraOpNumThreads(mOptions.numThreads);
            mOptions.provider = "CPUExecutionProvider";
        }
        mSession = std::make_unique<Ort::Session>(env, mOptions.modelPath.c_str(), *mSessionOptions);
        // Extract I/O info
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputs = mSession->GetInputCount();
        mInputNames.clear();
        mInputShapes.clear();
        for (size_t i = 0; i < numInputs; ++i)
        {
            char *name = mSession->GetInputName(i, allocator);
            mInputNames.push_back(name);
            Ort::TypeInfo type_info = mSession->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            mInputShapes.push_back(tensor_info.GetShape());
            allocator.Free(name);
        }
        size_t numOutputs = mSession->GetOutputCount();
        mOutputNames.clear();
        mOutputShapes.clear();
        for (size_t i = 0; i < numOutputs; ++i)
        {
            char *name = mSession->GetOutputName(i, allocator);
            mOutputNames.push_back(name);
            Ort::TypeInfo type_info = mSession->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            mOutputShapes.push_back(tensor_info.GetShape());
            allocator.Free(name);
        }
    }

    std::vector<const char *> getOutputNamesChar() const
    {
        std::vector<const char *> out;
        for (const auto &n : mOutputNames)
            out.push_back(n.c_str());
        return out;
    }

    std::string vecToStr(const std::vector<int64_t> &v) const
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (i > 0)
                oss << ", ";
            oss << v[i];
        }
        oss << "]";
        return oss.str();
    }

    static Ort::Env &getOrtEnv()
    {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXEngine");
        return env;
    }

    // Members
    Options mOptions;
    std::unique_ptr<Ort::SessionOptions> mSessionOptions;
    std::unique_ptr<Ort::Session> mSession;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    std::vector<std::vector<int64_t>> mInputShapes;
    std::vector<std::vector<int64_t>> mOutputShapes;
};
