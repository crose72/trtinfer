#include "engine.h"

std::string getDataTypeString(nvinfer1::DataType dataType)
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

std::string cnvrtTensorFormatToString(nvinfer1::TensorFormat format)
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

// Define the global logger
Logger gLogger{nvinfer1::ILogger::Severity::kINFO};

std::filesystem::path path_relative_to_exe(const std::filesystem::path &rel)
{
    auto exe = std::filesystem::canonical("/proc/self/exe");
    auto exe_dir = exe.parent_path(); // .../toplevel/build/exampleResnet50
    return std::filesystem::weakly_canonical(exe_dir / rel);
}

template <typename T>
Engine<T>::Engine(const std::string &engineFilename)
    : mEngineFilename(engineFilename),
      mEngine(nullptr)
{
    std::filesystem::path enginePath = mEngineFilename; // path_relative_to_exe(mEngineFilename);

    std::cout << "[Engine] Resolved path = " << enginePath << std::endl;
    std::cout << "[Engine] Exists? "
              << (std::filesystem::exists(enginePath) ? "YES" : "NO") << std::endl;

    // De-serialize engine from file
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (engineFile.fail())
    {
        std::cerr << "[Engine] Cannot open engine file: " << enginePath << "\n";
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    mRuntime.reset(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), fsize));
    assert(mEngine.get() != nullptr);

    // TODO put in init?
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    if (!mContext)
    {
        return;
    }

    getEngineInfo();
}

template <typename T>
Engine<T>::Engine(void) {}

template <typename T>
Engine<T>::~Engine(void) {}

template <typename T>
void Engine<T>::getEngineInfo(void)
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
        }
        else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT)
        {
            // Tensor is an output

            //  Ensure the model output data type matches the template argument
            //  specified by the user
            if (tensorDataType == nvinfer1::DataType::kFLOAT && !std::is_same<float, T>::value)
            {
                std::cerr << "Error, the model has expected output of type float. "
                          << "Engine class template parameter must be adjusted." << std::endl;
            }
            else if (tensorDataType == nvinfer1::DataType::kHALF && !std::is_same<__half, T>::value)
            {
                std::cerr << "Error, the model has expected output of type __half. "
                          << "Engine class template parameter must be adjusted." << std::endl;
            }
            else if (tensorDataType == nvinfer1::DataType::kINT8 && !std::is_same<int8_t, T>::value)
            {
                std::cerr << "Error, the model has expected output of type int8_t. "
                          << "Engine class template parameter must be adjusted." << std::endl;
            }
            else if (tensorDataType == nvinfer1::DataType::kINT32 && !std::is_same<int32_t, T>::value)
            {
                std::cerr << "Error, the model has expected output of type int32_t. "
                          << "Engine class template parameter must be adjusted." << std::endl;
            }
            else if (tensorDataType == nvinfer1::DataType::kBOOL && !std::is_same<bool, T>::value)
            {
                std::cerr << "Error, the model has expected output of type bool. "
                          << "Engine class template parameter must be adjusted." << std::endl;
            }
            else if (tensorDataType == nvinfer1::DataType::kUINT8 && !std::is_same<uint8_t, T>::value)
            {
                std::cerr << "Error, the model has expected output of type uint8_t. "
                          << "Engine class template parameter must be adjusted." << std::endl;
            }
            else if (tensorDataType == nvinfer1::DataType::kFP8)
            {
                std::cerr << "Error, the model has expected output of type kFP8. "
                          << "This is not supported by the Engine class." << std::endl;
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

            // Populate the member variables of the class with engine information
            mOutputLengths.push_back(outputLength);
            mOutputNames.emplace_back(tensorName);
            mOutputTensorFormats.emplace_back(mEngine->getTensorFormat(tensorName));
            mOutputDataTypes.emplace_back(mEngine->getTensorDataType(tensorName));
            // Now size the output buffer appropriately, taking into account the max
            // possible batch size (although we could actually end up using less
            // memory)
            // TODO - could keep as a max memory size limit? probably delete though
            // Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[tensor], outputLength * m_options.maxBatchSize * sizeof(T), stream));
        }
        else
        {
            std::cerr << "Error, IO Tensor is neither an input or output!" << std::endl;
        }
    }
}

template <typename T>
void Engine<T>::printEngineInfo() const
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

cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                        const cv::Scalar &bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void transformOutput(std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<float>> &output) {
    if (input.size() != 1) {
        std::cerr << "The feature vector has incorrect dimensions!" << std::endl;
    }

    output = std::move(input[0]);
}

void transformOutput(std::vector<std::vector<std::vector<float>>> &input, std::vector<float> &output) {
    if (input.size() != 1 || input[0].size() != 1) {
        std::cerr << "The feature vector has incorrect dimensions!" << std::endl;
    }

    output = std::move(input[0][0]);
}


// Add explicit template instantiations for Engine<float>
template Engine<float>::Engine(const std::string &engineFilename);
template Engine<float>::~Engine();
template void Engine<float>::getEngineInfo();
template void Engine<float>::printEngineInfo() const;
