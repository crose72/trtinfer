#pragma once

#include <vector>
#include <string>
#include <NvInfer.h>

template <typename T>
class IEngine
{
public:
    virtual ~IEngine() = default;

    virtual const std::vector<std::string> &getInputNames() const = 0;
    virtual const std::vector<std::string> &getOutputNames() const = 0;
    virtual const std::vector<nvinfer1::Dims> &getInputDims() const = 0;
    virtual const std::vector<nvinfer1::Dims> &getOutputDims() const = 0;
    virtual const std::vector<nvinfer1::TensorFormat> &getInputTensorFormat() const = 0;
    virtual const std::vector<nvinfer1::TensorFormat> &getOutputTensorFormat() const = 0;
    virtual const std::vector<nvinfer1::DataType> &getInputDataType() const = 0;
    virtual const std::vector<nvinfer1::DataType> &getOutputDataType() const = 0;
};
