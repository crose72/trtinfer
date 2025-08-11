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
//! \class Resnet50
//!
//! \brief Implements semantic segmentation using FCN-ResNet101 ONNX model.
//!
class Resnet50
{

public:
    Resnet50(const std::string &engineFilename);
    bool infer(const std::string &input_filename, int32_t width, int32_t height, const std::string &output_filename);
    bool init(void);

private:
    std::string mEngineFilename; //!< Filename of the serialized engine.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to run the network
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    std::unique_ptr<nvinfer1::IExecutionContext> mContext; //!< The TensorRT execution mContext

    char const *getNodeName(nvinfer1::TensorIOMode nodeType);
    size_t dataMemSize(nvinfer1::DataType dataType);
};

size_t Resnet50::dataMemSize(nvinfer1::DataType dataType)
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

char const *Resnet50::getNodeName(nvinfer1::TensorIOMode nodeType)
{
    char const *input_name = nullptr;

    for (int i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        const char *name = mEngine->getIOTensorName(i);

        if (mEngine->getTensorIOMode(name) == nodeType)
        {
            input_name = name;
            break;
        }
    }
}

Resnet50::Resnet50(const std::string &engineFilename)
    : mEngineFilename(engineFilename), mEngine(nullptr)
{
    auto enginePath = path_relative_to_exe(engineFilename);

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
}

//!
//! \brief Initialize the TensorRT inference.
//!
//! \details Allocate input and output memory.
//!
bool Resnet50::init(void)
{
}

//!
//! \brief Runs the TensorRT inference.
//!
//! \details Allocate input and output memory, and executes the engine.
//!
bool Resnet50::infer(const std::string &input_filename, int32_t width, int32_t height, const std::string &output_filename)
{
    auto mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }

    char const *input_name = getNodeName(nvinfer1::TensorIOMode::kINPUT);
    size_t inputDataSize = dataMemSize(mEngine->getTensorDataType(input_name));
    assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    nvinfer1::Dims input_dims = nvinfer1::Dims4{/* batch size */ 1, /* channels */ 3, height, width};
    mContext->setInputShape(input_name, input_dims);
    size_t input_size = (input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * inputDataSize);

    char const *output_name = getNodeName(nvinfer1::TensorIOMode::kOUTPUT);
    size_t outputDataSize = dataMemSize(mEngine->getTensorDataType(output_name));
    assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);
    nvinfer1::Dims output_dims = mContext->getTensorShape(output_name);
    size_t output_size = util::getMemorySize(output_dims, outputDataSize);

    // Allocate CUDA memory for input and output bindings
    void *input_mem{nullptr};
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    void *output_mem{nullptr};
    if (cudaMalloc(&output_mem, output_size) != cudaSuccess)
    {
        gLogError << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }

    // Read image data from file and mean-normalize it
    const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    auto input_image{util::RGBImageReader(input_filename, input_dims, mean, stddev)};
    input_image.read();
    auto input_buffer = input_image.process();
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    // Copy image data to input binding memory
    if (cudaMemcpyAsync(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    mContext->setTensorAddress(input_name, input_mem);
    mContext->setTensorAddress(output_name, output_mem);

    // Run TensorRT inference
    bool status = mContext->enqueueV3(stream);
    if (!status)
    {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    // Copy predictions from output binding memory
    auto output_buffer = std::unique_ptr<float>{new float[output_size]};
    if (cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // ---- Top-K on FP32 scores from `output_buffer` ----
    float *scores = output_buffer.get(); // raw pointer for lambda

    std::vector<size_t> idx(output_size);
    std::iota(idx.begin(), idx.end(), size_t{0});

    size_t topk = std::min<size_t>(5, output_size);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [scores](size_t a, size_t b)
                      { return scores[a] > scores[b]; });

    for (size_t i = 0; i < topk; ++i)
    {
        size_t k = idx[i];
        std::cout << "#" << (i + 1) << ": class " << k
                  << " score " << scores[k] << "\n";
    }

    // Free CUDA resources
    cudaFree(input_mem);
    cudaFree(output_mem);
    return true;
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

int main(int argc, char **argv)
{
    int32_t width{224};
    int32_t height{224};

    Resnet50 sample("../../exampleResnet50/resnet_engine_intro.engine");

    std::string prepped_img = "../exampleResnet50/squirrel-out.ppm";

    preprocess_to_ppm(
        "../exampleResnet50/elephant.jpg",
        prepped_img,
        width,
        height);

    gLogInfo
        << "Running TensorRT inference for ResNet50" << std::endl;

    if (!sample.infer(prepped_img, width, height, "../exampleResnet50/output.ppm"))
    {
        return -1;
    }

    return 0;
}
