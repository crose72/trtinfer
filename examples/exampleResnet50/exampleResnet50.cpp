/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

std::filesystem::path path_relative_to_exe(const std::filesystem::path &rel)
{
    auto exe = std::filesystem::canonical("/proc/self/exe");
    auto exe_dir = exe.parent_path(); // .../toplevel/build/exampleResnet50
    return std::filesystem::weakly_canonical(exe_dir / rel);
}

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;

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
        int32_t batchSize = 1;
        int32_t numChannels = 3;
        int32_t height = 224;
        int32_t width = 224;
        std::string precision = "fp32";
        std::string inputTensorName;                // Optional override
        std::string outputTensorName;               // Optional override
        std::vector<std::string> outputTensorNames; // If model has multiple outputs (e.g. YOLO: boxes, scores, classes)
    };

    TensorInfer(const std::string &engineFilename);
    ~TensorInfer();
    std::unique_ptr<float> infer(const std::unique_ptr<float> &input);
    bool init(void);

private:
    Config mConfig;

    std::string mEngineFilename; //!< Filename of the serialized engine.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to run the network
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    std::unique_ptr<nvinfer1::IExecutionContext> mContext; //!< The TensorRT execution mContext

    cudaStream_t mStream = nullptr;

    char const *mInputName;
    char const *mOutputName;

    size_t mInputSize;
    size_t mOutputSize;

    int32_t mBatchSize = 1;
    int32_t mNumChannels = 3;
    int32_t mHeight = 224;
    int32_t mWidtch = 224;

    // TODO make static? or just instantiate class at global scope?
    // Should persist for duration of program
    // Clean up in destructor?
    void *mInputMem = nullptr;
    void *mOutputMem = nullptr;

    char const *getNodeName(nvinfer1::TensorIOMode nodeType);
    size_t dataMemSize(nvinfer1::DataType dataType);
};

size_t TensorInfer::dataMemSize(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4UL; // 32-bit float
    case nvinfer1::DataType::kHALF:
        return 2UL; // 16-bit float
    case nvinfer1::DataType::kINT8:
        return 1UL; // 8-bit int
    case nvinfer1::DataType::kINT32:
        return 4UL; // 32-bit int
    case nvinfer1::DataType::kBOOL:
        return 1UL; // 8-bit bool
    case nvinfer1::DataType::kUINT8:
        return 1UL; // 8-bit unsigned int
    case nvinfer1::DataType::kFP8:
        return 1UL; // 8-bit float
    case nvinfer1::DataType::kBF16:
        return 2UL; // 16-bit brain float
    case nvinfer1::DataType::kINT64:
        return 8UL; // 64-bit int
    case nvinfer1::DataType::kINT4:
        return 0UL; // 4-bit int (packed)
    default:
        return 0UL;
    }
}

//!
//! \class TensorInfer
//!
//! \brief Finds name of a node in a TensorRT engine.
//!
char const *TensorInfer::getNodeName(nvinfer1::TensorIOMode nodeType)
{
    for (int i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        const char *name = mEngine->getIOTensorName(i);

        if (mEngine->getTensorIOMode(name) == nodeType)
        {
            return name;
        }
    }
}

TensorInfer::TensorInfer(const std::string &engineFilename)
    : mEngineFilename(engineFilename), mEngine(nullptr)
{
    std::filesystem::path enginePath = path_relative_to_exe(engineFilename);

    gLogInfo << "[Engine] Resolved path = " << enginePath << std::endl;
    gLogInfo << "[Engine] Exists? "
             << (std::filesystem::exists(enginePath) ? "YES" : "NO") << std::endl;

    // De-serialize engine from file
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (engineFile.fail())
    {
        gLogError << "[Engine] Cannot open engine file: " << enginePath << "\n";
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    mRuntime.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), fsize));
    assert(mEngine.get() != nullptr);

    // TODO put in init?
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return;
    }

    mInputName = getNodeName(nvinfer1::TensorIOMode::kINPUT);
    size_t inputDataSize = dataMemSize(mEngine->getTensorDataType(mInputName));
    assert(mEngine->getTensorDataType(mInputName) == nvinfer1::DataType::kFLOAT);
    mInputDims = nvinfer1::Dims4{mBatchSize, mNumChannels, 224, 224};
    mContext->setInputShape(mInputName, mInputDims);
    mInputSize = (mInputDims.d[0] * mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3] * inputDataSize);

    mOutputName = getNodeName(nvinfer1::TensorIOMode::kOUTPUT);
    size_t outputDataSize = dataMemSize(mEngine->getTensorDataType(mOutputName));
    assert(mEngine->getTensorDataType(mOutputName) == nvinfer1::DataType::kFLOAT);
    mOutputDims = mContext->getTensorShape(mOutputName);
    mOutputSize = util::getMemorySize(mOutputDims, outputDataSize);

    // Allocate CUDA memory for input and output bindings
    if (cudaMalloc(&mInputMem, mInputSize) != cudaSuccess)
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << mInputSize << " bytes" << std::endl;
        return;
    }

    if (cudaMalloc(&mOutputMem, mOutputSize) != cudaSuccess)
    {
        gLogError << "ERROR: output cuda memory allocation failed, size = " << mOutputSize << " bytes" << std::endl;
        return;
    }

    if (cudaStreamCreate(&mStream) != cudaSuccess)
    {
        gLogError << "ERROR: cuda mStream creation failed." << std::endl;
        return;
    }
}

TensorInfer::~TensorInfer()
{
}

//!
//! \brief Initialize the TensorRT inference.
//!
//! \details Allocate input and output memory.
//!
bool TensorInfer::init(void)
{

    return false;
}

//!
//! \brief Runs the TensorRT inference.
//!
//! \details Allocate input and output memory, and executes the engine.
//!
std::unique_ptr<float> TensorInfer::infer(const std::unique_ptr<float> &input)
{
    // Copy image data to input binding memory
    if (cudaMemcpyAsync(mInputMem, input.get(), mInputSize, cudaMemcpyHostToDevice, mStream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " << mInputSize << " bytes" << std::endl;
        // return;
    }
    mContext->setTensorAddress(mInputName, mInputMem);
    mContext->setTensorAddress(mOutputName, mOutputMem);

    // Run TensorRT inference
    bool status = mContext->enqueueV3(mStream);
    if (!status)
    {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        // return;
    }

    // Copy predictions from output binding memory
    std::unique_ptr<float> output = std::unique_ptr<float>{new float[mOutputSize]};
    if (cudaMemcpyAsync(output.get(), mOutputMem, mOutputSize, cudaMemcpyDeviceToHost, mStream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << mOutputSize << " bytes" << std::endl;
        // return;
    }
    cudaStreamSynchronize(mStream);

    // Free CUDA resources
    // cudaFree(mInputMem);
    // cudaFree(mOutputMem);
    return output;
}

// Resize shortest side to 256 (keep aspect), center-crop to HxW, convert to RGB,
// and save as binary PPM (P6).
inline void preprocess_to_ppm(const std::string &in_path,
                              const std::string &out_ppm_path,
                              int target_w, int target_h)
{
    cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR); // BGR uint8
    if (img.empty())
        throw std::runtime_error("Could not read image: " + in_path);

    // BGR -> RGB (PPM expects RGB; your reader assumes RGB)
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 1) Resize shortest side to 256 (keep aspect ratio)
    int new_w, new_h;
    if (img.cols < img.rows)
    {
        new_w = 256;
        new_h = static_cast<int>(std::round(img.rows * (256.0 / img.cols)));
    }
    else
    {
        new_h = 256;
        new_w = static_cast<int>(std::round(img.cols * (256.0 / img.rows)));
    }
    cv::resize(img, img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 2) Center-crop to target_w x target_h
    if (img.cols < target_w || img.rows < target_h)
    {
        throw std::runtime_error("Image too small after resize for requested crop.");
    }
    int x = (img.cols - target_w) / 2;
    int y = (img.rows - target_h) / 2;
    cv::Rect roi(x, y, target_w, target_h);
    img = img(roi).clone(); // now exactly HxW in RGB, uint8

    // 3) Save as PPM (P6). OpenCV picks P6 for 8-bit 3-channel PPM by default.
    if (!cv::imwrite(out_ppm_path, img))
    {
        throw std::runtime_error("Failed to write PPM: " + out_ppm_path);
    }
}

inline std::unique_ptr<float> process_input(const std::string &input_filename, nvinfer1::Dims dims)
{
    // Read image data from file and mean-normalize it
    const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    auto input_image{util::RGBImageReader(input_filename, dims, mean, stddev)};
    input_image.read();
    return input_image.process();
}

void process_output(std::unique_ptr<float> output)
{
    // ---- Top-K on FP32 scores from `output_buffer` ----
    float *scores = output.get(); // raw pointer for lambda
    int32_t mOutputSize = 1 * 3 * 224 * 224;

    std::vector<size_t> idx(mOutputSize);
    std::iota(idx.begin(), idx.end(), size_t{0});
    size_t topk = std::min<size_t>(5, mOutputSize);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [scores](size_t a, size_t b)
                      { return scores[a] > scores[b]; });

    for (size_t i = 0; i < topk; ++i)
    {
        size_t k = idx[i];
        std::cout << "#" << (i + 1) << ": class " << k
                  << " score " << scores[k] << "\n";
    }
}

int main(int argc, char **argv)
{
    int32_t width{224};
    int32_t height{224};
    nvinfer1::Dims input_dims = nvinfer1::Dims4{1, 3, 224, 224};

    TensorInfer sample("../../exampleResnet50/resnet_engine_intro.engine");

    std::string prepped_img = "../exampleResnet50/squirrel-out.ppm";

    preprocess_to_ppm(
        "../exampleResnet50/elephant.jpg",
        prepped_img,
        width,
        height);

    std::unique_ptr<float> input = process_input(prepped_img, input_dims);

    gLogInfo
        << "Running TensorRT inference for ResNet50" << std::endl;

    std::unique_ptr<float> output = sample.infer(input);
    process_output(std::move(output));

    return 0;
}
