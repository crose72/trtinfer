#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <sstream> // Optional, for std::ostringstream if you want to build strings
#include <spdlog/spdlog.h>
#include <NvInfer.h> // For nvinfer1::DataType, nvinfer1::TensorFormat
#include <algorithm>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

    void launch_pack_nchw_kernel(
        const float *const *imgs, float *batch_blob,
        int N, int C, int H, int W,
        int threadsPerBlock);

    void launch_preprocess_and_pack_nchw_kernel(
        const float *const *imgs, float *batch_blob,
        int N, int C, int H, int W,
        int img_stride, // <-- Add stride as param!
        float mean0, float mean1, float mean2,
        float std0, float std1, float std2,
        float norm_factor,
        int threadsPerBlock);

#ifdef __cplusplus
}
#endif

inline void checkCudaErrorCode(cudaError_t code);
inline std::string tensorDataTypeStr(nvinfer1::DataType dataType);
inline std::string tensorFormatStr(nvinfer1::TensorFormat format);
inline bool doesFileExist(const std::string &name);
inline std::filesystem::path relativePath(const std::filesystem::path &rel);
inline std::vector<std::string> getFilesInDirectory(const std::string &dirPath);
inline cv::cuda::GpuMat letterbox(
    const cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));
inline void transformOutput(const std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<float>> &output);
inline void transformOutput(const std::vector<std::vector<std::vector<float>>> &input, std::vector<float> &output);
inline void transformOutput(const std::vector<std::vector<std::vector<__half>>> &input, std::vector<std::vector<float>> &output);
inline void transformOutput(const std::vector<std::vector<std::vector<__half>>> &input, std::vector<float> &output);

#define CHECK(condition)                                                                                                           \
    do                                                                                                                             \
    {                                                                                                                              \
        if (!(condition))                                                                                                          \
        {                                                                                                                          \
            spdlog::error("Assertion failed: ({}), function {}, file {}, line {}.", #condition, __FUNCTION__, __FILE__, __LINE__); \
            abort();                                                                                                               \
        }                                                                                                                          \
    } while (false);

/**
 * @brief Checks the CUDA error code and throws on failure.
 * @param code CUDA error code to check.
 * @throws std::runtime_error if code is not cudaSuccess.
 */
inline void checkCudaErrorCode(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        spdlog::error(errMsg);
        throw std::runtime_error(errMsg);
    }
}

/**
 * @brief Check if a file exists at the specified path.
 * @param name File path to check.
 * @return true if file exists, false otherwise.
 */
inline bool doesFileExist(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

/**
 * @brief Returns a list of all regular files in a directory.
 * @param dirPath Directory to search.
 * @return Vector of file paths (as strings).
 */
inline std::vector<std::string> getFilesInDirectory(const std::string &dirPath)
{
    std::vector<std::string> fileNames;
    for (const auto &entry : std::filesystem::directory_iterator(dirPath))
    {
        if (entry.is_regular_file())
            fileNames.push_back(entry.path().string());
    }
    return fileNames;
}

/**
 * @brief Computes a path relative to the running executable.
 * @param rel Relative path from executable.
 * @return Absolute path.
 */
inline std::filesystem::path relativePath(const std::filesystem::path &rel)
{
    auto exe = std::filesystem::canonical("/proc/self/exe");
    auto exe_dir = exe.parent_path(); // .../toplevel/build/exampleResnet50
    return std::filesystem::weakly_canonical(exe_dir / rel);
}

/**
 * @brief Converts TensorRT DataType to human-readable string.
 * @param dataType TensorRT data type.
 * @return String representation ("FP32", "FP16", etc.).
 */
inline std::string tensorDataTypeStr(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FP32";
    case nvinfer1::DataType::kHALF:
        return "FP16";
    case nvinfer1::DataType::kINT8:
        return "INT8";
    case nvinfer1::DataType::kINT32:
        return "INT32";
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Converts TensorRT TensorFormat to human-readable string.
 * @param format Tensor format enum.
 * @return String representation ("LINEAR", "CHW16", etc.).
 */
inline std::string tensorFormatStr(nvinfer1::TensorFormat format)
{
    switch (format)
    {
    case nvinfer1::TensorFormat::kLINEAR:
        return "LINEAR";
    case nvinfer1::TensorFormat::kCHW2:
        return "CHW2";
    case nvinfer1::TensorFormat::kHWC8:
        return "HWC8";
    case nvinfer1::TensorFormat::kCHW4:
        return "CHW4";
    case nvinfer1::TensorFormat::kCHW16:
        return "CHW16";
    case nvinfer1::TensorFormat::kCHW32:
        return "CHW32";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Resize a CUDA image to the target size while maintaining aspect ratio, padding right/bottom as needed.
 * @param input Input CUDA GpuMat image.
 * @param height Target height.
 * @param width Target width.
 * @param bgcolor Color used for padding (default: black).
 * @return Resized and padded CUDA image (GpuMat).
 */
inline cv::cuda::GpuMat letterbox(
    const cv::cuda::GpuMat &input,
    size_t height, size_t width,
    const cv::Scalar &bgcolor)
{
    // Compute resize ratio
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;

    // Resize the input image with aspect ratio preserved
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());

    // Create an output image filled with bg color
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);

    // Copy resized image to top-left corner of output
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

/**
 * @brief Flattens a nested 3D feature vector (size 1x1xN) into a 1D output vector.
 * @param input Input feature vector (nested).
 * @param output Output vector (flattened).
 */
inline void transformOutput(const std::vector<std::vector<std::vector<float>>> &input, std::vector<float> &output)
{
    if (input.size() != 1 || input[0].size() != 1)
    {
        auto msg = "The feature vector has incorrect dimensions!";
        spdlog::error(msg);
    }
    output = std::move(input[0][0]);
}

/**
 * @brief Flattens a nested 3D feature vector (size BxMxN) into a 1D output vector.
 * @param input Input feature vector (nested).
 * @param output Output vector (flattened).
 */
inline void transformOutput(const std::vector<std::vector<std::vector<float>>> &input,
                            std::vector<std::vector<float>> &output)
{
    output.clear();

    for (const auto &batch_elem : input) // batch_elem is [C, N]
    {
        // Flatten [C, N] into one vector of C*N elements, channel-major order
        std::vector<float> flat;

        for (size_t c = 0; c < batch_elem.size(); ++c)
        {
            flat.insert(flat.end(), batch_elem[c].begin(), batch_elem[c].end());
        }

        output.push_back(std::move(flat));
    }
}

/**
 * @brief Flattens a nested 3D feature vector (size 1x1xN) into a 1D output vector.
 * @param input Input feature vector (nested).
 * @param output Output vector (flattened).
 * @throws std::logic_error if input is not 1x1.
 */
inline void transformOutput(const std::vector<std::vector<std::vector<__half>>> &input, std::vector<float> &output)
{
    if (input.size() != 1 || input[0].size() != 1)
    {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }
    auto &src = input[0][0];
    output.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        output[i] = __half2float(src[i]);
}

/**
 * @brief Flattens a nested 3D feature vector (size 1xNxM) into a 2D output vector.
 * @param input Input feature vector (nested).
 * @param output Output vector (2D, flattened batch).
 * @throws std::logic_error if input batch size is not 1.
 */
inline void transformOutput(const std::vector<std::vector<std::vector<__half>>> &input, std::vector<std::vector<float>> &output)
{
    if (input.size() != 1)
        throw std::logic_error("The feature vector has incorrect dimensions!");
    output.resize(input[0].size());
    for (size_t i = 0; i < input[0].size(); ++i)
    {
        auto &src = input[0][i];
        output[i].resize(src.size());
        for (size_t j = 0; j < src.size(); ++j)
            output[i][j] = __half2float(src[j]);
    }
}

template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch
{
    typename Clock::time_point start_point;

public:
    Stopwatch() : start_point(Clock::now()) {}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsedTime() const
    {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;