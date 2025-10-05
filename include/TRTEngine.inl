#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include "utils.h"

/**
 * @brief Constructor for TRTEngine.
 * @param engineFilename Path to the TensorRT engine file.
 */
template <typename T>
TRTEngine<T>::TRTEngine(const std::string &engineFilename) : mEngineFilename(engineFilename) {}

/**
 * @brief Default constructor for TRTEngine.
 *        The engine filename must be set before loading.
 */
template <typename T>
TRTEngine<T>::TRTEngine(void) {}

/**
 * @brief Destructor for TRTEngine. Frees all GPU resources.
 */
template <typename T>
TRTEngine<T>::~TRTEngine(void) {}

/**
 * @brief Load the TensorRT engine and allocate GPU buffers using stored parameters.
 * @return True on success, false on failure.
 */
template <typename T>
bool TRTEngine<T>::loadNetwork()
{
    return loadNetwork(mEngineFilename, mSubVals, mDivVals, mNormalize);
}

/**
 * @brief Load the TensorRT engine from file and allocate GPU buffers.
 * @param trtModelPath Path to the serialized TensorRT engine.
 * @param subVals      Channel-wise mean values for input normalization.
 * @param divVals      Channel-wise stddev values for input normalization.
 * @param normalize    Whether to normalize input images to [0,1].
 * @return True on success, false on failure.
 */
template <typename T>
bool TRTEngine<T>::loadNetwork(const std::string &trtModelPath,
                               const std::array<float, 3> &subVals,
                               const std::array<float, 3> &divVals,
                               bool normalize)
{
    mSubVals = subVals;
    mDivVals = divVals;
    mNormalize = normalize;

    // Read the serialized model from disk
    if (!doesFileExist(trtModelPath))
    {
        auto msg = "Error, unable to read TensorRT model at path: " + trtModelPath;
        spdlog::error(msg);
        return false;
    }
    else
    {
        auto msg = "Loading TensorRT engine file at path: " + trtModelPath;
        spdlog::info(msg);
    }

    std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        auto msg = "Error, unable to read engine file";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Create a runtime to deserialize the engine file.
    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    if (!mRuntime)
    {
        return false;
    }

    // Set the GPU device index
    auto ret = cudaSetDevice(mDeviceIndex);
    if (ret != 0)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(mDeviceIndex) + ". Note, your device has " +
                      std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        spdlog::error(errMsg);
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model - loads the engine into memory
    mEngine.reset(mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!mEngine)
    {
        return false;
    }

    // The execution context contains all of the state associated with a
    // particular invocation
    mContext.reset(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }

    getEngineInfo();

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    clearGpuBuffers();
    mBuffers.resize(mTensorTypes.size());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));
    int outputIndex = 0;

    for (int i = 0; i < mTensorTypes.size(); ++i)
    {
        if (mTensorTypes[i] == nvinfer1::TensorIOMode::kOUTPUT)
        {
            // Now size the output buffer appropriately, taking into account the max
            // possible batch size (although we could actually end up using less
            // memory)
            checkCudaErrorCode(cudaMallocAsync(&mBuffers[i], mOutputLengths[outputIndex] * mMaxBatchSize * sizeof(T), stream));
            ++outputIndex;
        }
        else // INPUT
        {
            mBuffers[i] = nullptr; // EXPLICITLY SET INPUT BUFFERS TO NULLPTR!
        }
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

/**
 * @brief Run inference on a batch of input images.
 * @param inputs         Batched input images for each input tensor (NHWC, on GPU).
 * @param featureVectors Output feature vectors (will be filled on CPU).
 * @return True on success, false on failure.
 */
template <typename T>
bool TRTEngine<T>::runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                                std::vector<std::vector<std::vector<T>>> &featureVectors)
{
    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    std::vector<cv::cuda::GpuMat> preprocessedInputs;
    const int32_t numInputs = mInputDims.size();

    // Preprocess all the input tensors
    for (size_t i = 0; i < mInputDims.size(); ++i)
    {
        const auto &batchInput = inputs[i];
        // OpenCV reads images into memory in NHWC format, while TensorRT expects
        // images in NCHW format. The following method converts NHWC to NCHW. Even
        // though TensorRT expects NCHW at IO, during optimization, it can
        // internally use NHWC to optimize cuda kernels See:
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        auto mfloat = blobFromGpuMats(batchInput, mSubVals, mDivVals, mNormalize);
        preprocessedInputs.push_back(mfloat);
        mBuffers[i] = mfloat.ptr<void>();
    }

    for (size_t i = 0; i < mInputNames.size(); ++i)
    {
        nvinfer1::Dims dims = mInputDims[i]; // e.g. dims.nbDims == 4 for [N,3,640,640]
        dims.d[0] = inputs[i].size();        // set N = batch size (for input i)

        bool success = mContext->setInputShape(mInputNames[i].c_str(), dims);
        if (!success)
        {
            throw std::runtime_error("Failed to set input shape for " + std::string(mInputNames[i]));
        }
    }

    // Ensure all dynamic bindings have been defined.
    if (!mContext->allInputDimensionsSpecified())
    {
        auto msg = "Error, not all required dimensions specified.";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < mBuffers.size(); ++i)
    {
        bool status = mContext->setTensorAddress(mIOTensorNames[i].c_str(), mBuffers[i]);
        if (!status)
        {
            return false;
        }
    }

    // Run inference.
    bool status = mContext->enqueueV3(inferenceCudaStream);
    if (!status)
    {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    for (int batch = 0; batch < batchSize; ++batch)
    {
        // Batch
        std::vector<std::vector<T>> batchOutputs{};
        for (int32_t outputBinding = numInputs; outputBinding < mEngine->getNbIOTensors(); ++outputBinding)
        {
            // We start at index mInputDims.size() to account for the inputs in our
            // mBuffers
            std::vector<T> output;
            auto outputLength = mOutputLengths[outputBinding - numInputs];
            output.resize(outputLength);
            // Copy the output
            checkCudaErrorCode(cudaMemcpyAsync(output.data(),
                                               static_cast<char *>(mBuffers[outputBinding]) + (batch * sizeof(T) * outputLength),
                                               outputLength * sizeof(T), cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}

/**
 * @brief Free all GPU buffers associated with outputs.
 */
template <typename T>
void TRTEngine<T>::clearGpuBuffers()
{
    if (!mBuffers.empty())
    {
        // Free GPU memory of outputs
        const auto numInputs = mInputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < mEngine->getNbIOTensors(); ++outputBinding)
        {
            checkCudaErrorCode(cudaFree(mBuffers[outputBinding]));
        }
        mBuffers.clear();
    }
}

/**
 * @brief Convert and preprocess input GPU images to a single input tensor (NCHW, float, mean/std).
 * @param batchInput Batch of input images (as cv::cuda::GpuMat).
 * @param subVals    Channel-wise mean values.
 * @param divVals    Channel-wise stddev values.
 * @param normalize  Whether to normalize to [0,1].
 * @param swapRB     Whether to swap R/B channels (BGR <-> RGB).
 * @return Preprocessed input as cv::cuda::GpuMat.
 */
template <typename T>
cv::cuda::GpuMat TRTEngine<T>::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput,
                                               const std::array<float, 3> &subVals,
                                               const std::array<float, 3> &divVals,
                                               bool normalize,
                                               bool swapRB)
{
    /*
     * ----------------------------------------
     * Batch Image Packing for TensorRT Inference
     * ----------------------------------------
     *
     * This function packs a batch of images (each as a cv::cuda::GpuMat, shape HxW, 3 channels)
     * into a single, flat, contiguous CUDA buffer (GpuMat) suitable for TensorRT NCHW input.
     *
     * Memory Layout:
     *   - The output buffer (gpu_dst) is a single row, with width = H * W * batch_size, 3 channels.
     *   - For each image in the batch:
     *       - The R, G, and B channel data are placed contiguously at specific offsets
     *         (i.e., all R for image 0, all G for image 0, all B for image 0, then repeat for image 1, etc).
     *   - Each channel of each image is "viewed" as a GpuMat at the correct offset inside the buffer,
     *     so no extra memory allocation or copying is needed.
     *
     * Usage of cv::cuda::split:
     *   - For each image, we create three GpuMat views (one per channel) at the appropriate offset in the flat buffer.
     *   - cv::cuda::split splits an input image into R, G, and B channel planes, storing them directly into the preallocated regions.
     *   - This results in the correct NCHW layout for inference: [R_plane][G_plane][B_plane] for each image, contiguous for the batch.
     *
     * Why not just interleaved?
     *   - OpenCV stores images as interleaved (RGBRGB...), but most deep learning inference engines (like TensorRT)
     *     expect the data in planar format: [all R][all G][all B], row-major within each channel.
     *   - This packing function prepares exactly that format, compatible with TensorRT NCHW input.
     *
     * Note:
     *   - The critical part is using GpuMat "views" (with custom data pointers) for each channel at the right offset,
     *     so the split writes directly into the batch buffer.
     */
    CHECK(!batchInput.empty())
    CHECK(batchInput[0].channels() == 3)

    cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

    size_t width = batchInput[0].cols * batchInput[0].rows;

    if (swapRB)
    {
        for (size_t img = 0; img < batchInput.size(); ++img)
        {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        }
    }
    else
    {
        for (size_t img = 0; img < batchInput.size(); ++img)
        {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        }
    }
    cv::cuda::GpuMat mfloat;
    if (normalize)
    {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    }
    else
    {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

/**
 * @brief Examine engine structure and extract tensor names, shapes, formats, and types.
 *        Populates internal vectors for all I/O tensors.
 */
template <typename T>
void TRTEngine<T>::getEngineInfo(void)
{
    // Total number of input/output tensors
    int32_t numTensors = mEngine->getNbIOTensors();

    // Search all input and output tensors for names, shapes, types, etc
    for (int tensor = 0; tensor < numTensors; ++tensor)
    {
        // Get tensor name (could be an input or output)
        char const *tensorName = mEngine->getIOTensorName(tensor);
        mIOTensorNames.emplace_back(tensorName);

        // Get tensor type (input or output) and other info
        const nvinfer1::TensorIOMode tensorType = mEngine->getTensorIOMode(tensorName);
        const nvinfer1::Dims tensorShape = mEngine->getTensorShape(tensorName);
        const nvinfer1::DataType tensorDataType = mEngine->getTensorDataType(tensorName);
        mTensorTypes.push_back(tensorType);
        int profileIndex = 0; // index for looping through optimization profiles - if engine has > 1

        // Tensor is an input
        if (tensorType == nvinfer1::TensorIOMode::kINPUT)
        {
            // The implementation currently only supports inputs of type float
            if (mEngine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT)
            {
                std::string errMsg = "Error, the implementation currently only supports FP32 inputs";
                spdlog::error(errMsg);
                throw std::runtime_error(errMsg);
            }

            // Don't need to allocate memory for inputs as we will be using the OpenCV
            // GpuMat buffer directly - could be something done in the future

            // Populate engine info
            mInputDims.emplace_back(tensorShape);
            mInputNames.emplace_back(tensorName);
            mInputTensorFormats.emplace_back(mEngine->getTensorFormat(tensorName));
            mInputDataTypes.emplace_back(mEngine->getTensorDataType(tensorName));
            // first dim is typically batch size.  Is -1 for dynamic batches
            // Getting the max batch size from the optimization profile
            // TODO: support multiple optimization profiles
            nvinfer1::Dims maxDims;
            maxDims = mEngine->getProfileShape(tensorName, profileIndex, nvinfer1::OptProfileSelector::kMAX);
            mInputBatchSize = tensorShape.d[0];
            mMaxBatchSize = std::max((int32_t)maxDims.d[0], mMaxBatchSize);
        }
        else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT)
        {
            // Tensor is an output

            // Ensure the model output data type matches the template argument
            // specified by the user
            if (tensorDataType == nvinfer1::DataType::kFLOAT && !std::is_same<float, T>::value)
            {
                auto msg = "Error, the model has expected output of type float. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }
            else if (tensorDataType == nvinfer1::DataType::kHALF && !std::is_same<__half, T>::value)
            {
                auto msg = "Error, the model has expected output of type __half. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }
            else if (tensorDataType == nvinfer1::DataType::kINT8 && !std::is_same<int8_t, T>::value)
            {
                auto msg = "Error, the model has expected output of type int8_t. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }
            else if (tensorDataType == nvinfer1::DataType::kINT32 && !std::is_same<int32_t, T>::value)
            {
                auto msg = "Error, the model has expected output of type int32_t. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }
            else if (tensorDataType == nvinfer1::DataType::kBOOL && !std::is_same<bool, T>::value)
            {
                auto msg = "Error, the model has expected output of type bool. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }
            else if (tensorDataType == nvinfer1::DataType::kUINT8 && !std::is_same<uint8_t, T>::value)
            {
                auto msg = "Error, the model has expected output of type uint8_t. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }
            else if (tensorDataType == nvinfer1::DataType::kFP8)
            {
                auto msg = "Error, the model has expected output of type kFP8. This is not supported by the Engine class.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }

            // Calculate the length of the output to allocate for the output
            // Output memory reserved for output buffer will be the outputLength * sizeof(dataType)
            uint32_t outputLength = 1;

            for (int j = 1; j < tensorShape.nbDims; ++j)
            {
                // Ignore j = 0 because that is the batch size, and we will take that
                // into account when sizing the buffer
                outputLength *= tensorShape.d[j];
            }

            // Populate engine info
            mOutputDims.push_back(tensorShape);
            mOutputLengths.push_back(outputLength);
            mOutputNames.emplace_back(tensorName);
            mOutputTensorFormats.emplace_back(mEngine->getTensorFormat(tensorName));
            mOutputDataTypes.emplace_back(mEngine->getTensorDataType(tensorName));
        }
        else
        {
            auto msg = "Error, IO Tensor is neither an input or output!";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }
}

/**
 * @brief Print a summary of the engine's input/output tensors, dimensions, types, and properties.
 *        Outputs a single log entry using spdlog.
 */
template <typename T>
void TRTEngine<T>::printEngineInfo() const
{
    std::ostringstream oss;
    oss << "\n=== Engine Information ===\n";

    // Input tensor names
    const auto inputNames = getInputNames();
    oss << "\nInput tensors (" << inputNames.size() << "):\n";
    for (size_t i = 0; i < inputNames.size(); ++i)
    {

        oss << "  [" << i << "] " << inputNames[i] << "\n";
    }

    // Output tensor names
    const auto outputNames = getOutputNames();
    oss << "\nOutput tensors (" << outputNames.size() << "):\n";
    for (size_t i = 0; i < outputNames.size(); ++i)
    {
        oss << "  [" << i << "] " << outputNames[i] << "\n";
    }

    // Input dims
    const auto inputDims = getInputDims();
    oss << "\nInput dimensions:\n";
    for (size_t i = 0; i < inputDims.size(); ++i)
    {
        oss << "  [" << i << "] [";
        for (int j = 0; j < inputDims[i].nbDims; ++j)
        {
            if (j > 0)
                oss << ", ";
            oss << inputDims[i].d[j];
        }
        oss << "]\n";
    }

    // Output dims
    const auto outputDims = getOutputDims();
    oss << "\nOutput dimensions:\n";
    for (size_t i = 0; i < outputDims.size(); ++i)
    {
        oss << "  [" << i << "] [";
        for (int j = 0; j < outputDims[i].nbDims; ++j)
        {
            if (j > 0)
                oss << ", ";
            oss << outputDims[i].d[j];
        }
        oss << "]\n";
    }

    // Data types
    const auto inputDataTypes = getInputDataType();
    const auto outputDataTypes = getOutputDataType();
    oss << "\nData Types:\n";
    for (size_t i = 0; i < inputNames.size(); ++i)
    {
        oss << "  Input " << i << " (" << inputNames[i] << "): "
            << tensorDataTypeStr(inputDataTypes[i]) << "\n";
    }
    for (size_t i = 0; i < outputNames.size(); ++i)
    {
        oss << "  Output " << i << " (" << outputNames[i] << "): "
            << tensorDataTypeStr(outputDataTypes[i]) << "\n";
    }

    // Tensor formats
    const auto inputTensorFormats = getInputTensorFormats();
    const auto outputTensorFormats = getOutputTensorFormats();
    oss << "\nTensor Formats:\n";
    for (size_t i = 0; i < inputTensorFormats.size(); ++i)
    {
        oss << "  Input [" << i << "]: "
            << tensorFormatStr(inputTensorFormats[i]) << "\n";
    }
    for (size_t i = 0; i < outputTensorFormats.size(); ++i)
    {
        oss << "  Output [" << i << "]: "
            << tensorFormatStr(outputTensorFormats[i]) << "\n";
    }

    // Engine properties
    oss << "\nEngine Properties:\n";
    oss << "  Number of layers: " << mEngine->getNbLayers() << "\n";
    oss << "  Number of I/O tensors: " << mEngine->getNbIOTensors() << "\n";
    oss << "=========================\n";

    // Print the entire summary as a single log entry
    spdlog::info(oss.str());
}