### TensorRT getting started guide example
https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#export-from-pytorch


### Downloaded Resnet50 onnx model
wget https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz
tar xzf resnet50.tar.gz

### Convert onnx to TRT engine
/usr/src/tensorrt/bin/trtexec \
  --onnx=resnet50/model.onnx \
  --saveEngine=resnet50.engine \
  --stronglyTyped \
  --allowWeightStreaming