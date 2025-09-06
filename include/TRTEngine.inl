#include <opencv2/cudaarithm.hpp>
#include <algorithm>

#define CHECK(condition)                                                                                                           \
    do                                                                                                                             \
    {                                                                                                                              \
        if (!(condition))                                                                                                          \
        {                                                                                                                          \
            spdlog::error("Assertion failed: ({}), function {}, file {}, line {}.", #condition, __FUNCTION__, __FILE__, __LINE__); \
            abort();                                                                                                               \
        }                                                                                                                          \
    } while (false);

inline bool doesFileExist(const std::string &name);
inline void checkCudaErrorCode(cudaError_t code);
inline std::vector<std::string> getFilesInDirectory(const std::string &dirPath);
inline std::string getDataTypeString(nvinfer1::DataType dataType);
inline std::string cnvrtTensorFormatToString(nvinfer1::TensorFormat format);
inline std::filesystem::path path_relative_to_exe(const std::filesystem::path &rel);

// Utility functions
inline bool doesFileExist(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

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

inline std::string getDataTypeString(nvinfer1::DataType dataType)
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

inline std::string cnvrtTensorFormatToString(nvinfer1::TensorFormat format)
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

inline std::filesystem::path path_relative_to_exe(const std::filesystem::path &rel)
{
    auto exe = std::filesystem::canonical("/proc/self/exe");
    auto exe_dir = exe.parent_path(); // .../toplevel/build/exampleResnet50
    return std::filesystem::weakly_canonical(exe_dir / rel);
}

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

// TensorRT engine functions

template <typename T>
TRTEngine<T>::TRTEngine(const std::string &engineFilename) : mEngineFilename(engineFilename) {}

template <typename T>
TRTEngine<T>::TRTEngine(void) {}

template <typename T>
TRTEngine<T>::~TRTEngine(void) {}

template <typename T>
void TRTEngine<T>::getEngineInfo(void)
{
    int32_t numTensors = mEngine->getNbIOTensors();

    // Search all input and output tensors for names, shapes, types, etc
    for (int tensor = 0; tensor < numTensors; ++tensor)
    {
        char const *tensorName = mEngine->getIOTensorName(tensor);
        mIOTensorNames.emplace_back(tensorName);
        const nvinfer1::TensorIOMode tensorType = mEngine->getTensorIOMode(tensorName);
        const nvinfer1::Dims tensorShape = mEngine->getTensorShape(tensorName);
        const nvinfer1::DataType tensorDataType = mEngine->getTensorDataType(tensorName);
        mTensorTypes.push_back(tensorType);

        // Tensor is an input
        if (tensorType == nvinfer1::TensorIOMode::kINPUT)
        {
            // The implementation currently only supports inputs of type float
            if (mEngine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT)
            {
                std::cerr << "Error, the implementation currently only supports float inputs" << std::endl;
                // spdlog::error(msg);
            }

            // Don't need to allocate memory for inputs as we will be using the OpenCV
            // GpuMat buffer directly.

            // Populate the member variables of the class with engine information
            nvinfer1::Dims3 inputDims(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            mInputDims.emplace_back(inputDims);
            mInputNames.emplace_back(tensorName);
            mInputTensorFormats.emplace_back(mEngine->getTensorFormat(tensorName));
            mInputDataTypes.emplace_back(mEngine->getTensorDataType(tensorName));
            mInputBatchSize = tensorShape.d[0];
            /* only works for tensors that are shape bindings - dynamic shapes
            int32_t const *inputProfileMaxSize = mEngine->getProfileTensorValues(
                tensorName,
                mOptProfileIndex, // profile index: 0 if you only have one
                nvinfer1::OptProfileSelector::kMAX);*/
            std::cout << "TensorShape[d[0]]: " << tensorShape.d[0] << std::endl;
            mMaxBatchSize = std::max((int32_t)tensorShape.d[0], mMaxBatchSize);
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

            // The binding is an output
            uint32_t outputLength = 1;
            mOutputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j)
            {
                // This will be the size of memory in bytes
                // We ignore j = 0 because that is the batch size, and we will take that
                // into account when sizing the buffer
                outputLength *= tensorShape.d[j];
            }

            if (tensorShape.d[0] == 1)
            {
                outputLength = outputLength / tensorShape.d[0];
            }

            // Populate the member variables of the class with engine information
            mOutputLengths.push_back(outputLength);
            mOutputNames.emplace_back(tensorName);
            mOutputTensorFormats.emplace_back(mEngine->getTensorFormat(tensorName));
            mOutputDataTypes.emplace_back(mEngine->getTensorDataType(tensorName));
            // Now size the output buffer appropriately, taking into account the max
            // possible batch size (although we could actually end up using less
            // memory)
            // TODO - could keep as a max memory size limit? probably delete though
            // was used for allocating the output memory before inference - move the logic?
            // checkCudaErrorCode(cudaMallocAsync(&mBuffers[tensor], outputLength * mMaxBatchSize * sizeof(T), stream));
        }
        else
        {
            auto msg = "Error, IO Tensor is neither an input or output!";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }
}

template <typename T>
void TRTEngine<T>::printEngineInfo() const
{
    std::cout << "=== Engine Information ===" << std::endl;

    // Print input/output tensor names
    const std::vector<std::string> inputNames = this->getInputNames();
    const std::vector<std::string> outputNames = this->getOutputNames();

    std::cout << "\nInput tensors (" << inputNames.size() << "):" << std::endl;
    for (size_t i = 0; i < inputNames.size(); ++i)
    {
        std::cout << "  [" << i << "] " << inputNames[i] << std::endl;
    }

    std::cout << "\nOutput tensors (" << outputNames.size() << "):" << std::endl;
    for (size_t i = 0; i < outputNames.size(); ++i)
    {
        std::cout << "  [" << i << "] " << outputNames[i] << std::endl;
    }

    // Print input/output tensor dimensions
    const std::vector<nvinfer1::Dims> inputDims = getInputDims();
    const std::vector<nvinfer1::Dims> outputDims = getOutputDims();

    std::cout << "\nInput dimensions:" << std::endl;

    for (size_t i = 0; i < inputDims.size(); ++i)
    {
        std::cout << "  [" << i << "] [";
        for (int j = 0; j < inputDims[i].nbDims; ++j)
        {
            if (j > 0)
                std::cout << ", ";
            std::cout << inputDims[i].d[j];
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nOutput dimensions:" << std::endl;

    for (size_t i = 0; i < outputDims.size(); ++i)
    {
        std::cout << "  [" << i << "] [";
        for (int j = 0; j < outputDims[i].nbDims; ++j)
        {
            if (j > 0)
                std::cout << ", ";
            std::cout << outputDims[i].d[j];
        }
        std::cout << "]" << std::endl;
    }

    // Print input/output tensor data types
    const std::vector<nvinfer1::DataType> inputDataTypes = getInputDataType();
    const std::vector<nvinfer1::DataType> outputDataTypes = getInputDataType();

    std::cout << "\nData Types:" << std::endl;

    for (size_t i = 0; i < inputNames.size(); ++i)
    {
        std::cout << "  Input " << i << " (" << inputNames[i] << "): "
                  << getDataTypeString(inputDataTypes[i]) << std::endl;
    }

    for (size_t i = 0; i < outputNames.size(); ++i)
    {
        std::cout << "  Output " << i << " (" << outputNames[i] << "): "
                  << getDataTypeString(outputDataTypes[i]) << std::endl;
    }

    // Print input/output tensor formats

    const std::vector<nvinfer1::TensorFormat> inputTensorFormats = getInputTensorFormat();
    const std::vector<nvinfer1::TensorFormat> outputTensorFormats = getOutputTensorFormat();

    std::cout << "\nTensor Formats:" << std::endl;

    for (size_t i = 0; i < inputTensorFormats.size(); ++i)
    {
        std::cout << "  Input " << "[" << i << "]: " << cnvrtTensorFormatToString(inputTensorFormats[i])
                  << std::endl;
    }

    for (size_t i = 0; i < outputTensorFormats.size(); ++i)
    {
        std::cout << "  Output " << "[" << i << "]: " << cnvrtTensorFormatToString(outputTensorFormats[i])
                  << std::endl;
    }

    // Print basic engine info
    std::cout << "\nEngine Properties:" << std::endl;
    std::cout << "  Number of layers: " << mEngine->getNbLayers() << std::endl;
    std::cout << "  Number of I/O tensors: " << mEngine->getNbIOTensors() << std::endl;

    std::cout << "=========================" << std::endl;
}

template <typename T>
bool TRTEngine<T>::loadNetwork()
{
    return loadNetwork(mEngineFilename, mSubVals, mDivVals, mNormalize);
}

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
            std::cout << "[DEBUG] Allocating OUTPUT buffer " << i
                      << " size: " << mOutputLengths[outputIndex] << " * " << mMaxBatchSize
                      << " * " << sizeof(T) << " = " << (mOutputLengths[outputIndex] * mMaxBatchSize * sizeof(T)) << " bytes\n";
            for (int j = 0; j < mOutputLengths.size(); ++j)
            {
                std::cout << "mOutputLengths " << mOutputLengths[outputIndex] << " " << std::endl;
            }
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

template <typename T>
cv::cuda::GpuMat TRTEngine<T>::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput,
                                               const std::array<float, 3> &subVals,
                                               const std::array<float, 3> &divVals,
                                               bool normalize,
                                               bool swapRB)
{

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