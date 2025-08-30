#include "tensorrt_infer.h"

extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogInfo;

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

TensorInfer::TensorInfer(const Config &config) : mConfig(config)
{
    mConfig;
    mEngineFilename = mConfig.engineFilename;
    mInputDims = nvinfer1::Dims4(mConfig.batchSize, mConfig.numChannels, mConfig.height, mConfig.width);
    nvinfer1::Dims mOutputDims;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    cudaStream_t mStream = nullptr;

    std::filesystem::path enginePath = path_relative_to_exe(mEngineFilename);

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

    mRuntime.reset(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
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

    mRuntime.reset(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
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
    // Free CUDA resources
    cudaFree(mInputMem);
    cudaFree(mOutputMem);
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

    return output;
}
