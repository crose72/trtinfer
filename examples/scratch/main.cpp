#include "YOLOv8.h"

std::vector<cv::Mat> imgs_cpu;
std::vector<cv::cuda::GpuMat> imgs_gpu;

void getBatchInputs(void)
{
    // ---- [1] Define your batch of images ----
    std::vector<std::string> imgPaths = {
        "/workspace/examples/scratch/elephant.jpg",
        "/workspace/examples/scratch/squirrel.jpg",
        "/workspace/examples/scratch/border-collie.jpg",
        "/workspace/examples/scratch/people.jpg"};

    // ---- [2] Load images into cv::Mat ----
    for (const auto &path : imgPaths)
    {
        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            std::cerr << "Failed to load " << path << std::endl;
            continue;
        }
        imgs_cpu.push_back(img);
    }

    if (imgs_cpu.empty())
    {
        std::cerr << "No valid images loaded!" << std::endl;
    }

    // ---- [3] Upload images to GPU (cv::cuda::GpuMat) ----
    for (const auto img : imgs_cpu)
    {
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(img);
        imgs_gpu.push_back(gpuImg);
    }

    std::cout << "Num Images = " << imgs_gpu.size() << std::endl;
}

cv::cuda::GpuMat blobFromGpuMatsTest(const std::vector<cv::cuda::GpuMat> &batchInput,
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

void packBatchToBuffer(
    const std::vector<cv::cuda::GpuMat> &batchImgs, // [N images], each HxWx3 float32 (or 8U if you convert)
    float *batchBuffer,                             // Device ptr, cudaMalloc-ed, size = N*3*H*W floats
    int H, int W)
{
    int N = batchImgs.size();
    for (int n = 0; n < N; ++n)
    {
        // Assumption: image is CV_32FC3, size HxW
        std::vector<cv::cuda::GpuMat> channels(3);
        cv::cuda::split(batchImgs[n], channels); // Each: HxW, float32

        for (int c = 0; c < 3; ++c)
        {
            size_t offset = (n * 3 + c) * H * W; // [N,C,H,W]
            // Copy channel to the correct offset of the batch buffer
            cudaError_t err = cudaMemcpy(
                batchBuffer + offset,
                channels[c].ptr<float>(),
                H * W * sizeof(float),
                cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
}

void saveImg2Bin(cv::cuda::GpuMat blob, std::string fileName)
{

    if (doesFileExist(fileName))
    {
        std::string errMsg = "File exists";
        spdlog::error(errMsg);
        return;
    }

    cv::Mat blobCPU;
    blob.download(blobCPU);

    std::ofstream ofs(fileName, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(blobCPU.data), blobCPU.total() * blobCPU.elemSize());
    ofs.close();
}

cv::cuda::GpuMat loadImgFromBin(cv::cuda::GpuMat blob, std::string fileName)
{
    cv::Mat blobCPU;
    blob.download(blobCPU);

    std::vector<float> blobData(blobCPU.total() * blobCPU.channels()); // or same shape as original
    std::ifstream ifs(fileName, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(blobData.data()), blobData.size() * sizeof(float));
    ifs.close();
}

bool compareBlobWithBin(const cv::cuda::GpuMat &blob, const std::string &fileName)
{
    // 1. Download the current blob to CPU
    cv::Mat blobCPU;
    blob.download(blobCPU);

    // 2. Load the saved binary data
    size_t numFloats = blobCPU.total() * blobCPU.channels();
    std::vector<float> loadedData(numFloats);

    std::ifstream ifs(fileName, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "Failed to open " << fileName << std::endl;
        return false;
    }
    ifs.read(reinterpret_cast<char *>(loadedData.data()), numFloats * sizeof(float));
    ifs.close();

    // 3. Compare the memory (element-wise float comparison)
    const float *blobPtr = reinterpret_cast<const float *>(blobCPU.data);

    // Choose a tolerance for floating point (optional)
    float maxDiff = 0.0f;
    for (size_t i = 0; i < numFloats; ++i)
    {
        float diff = std::abs(blobPtr[i] - loadedData[i]);
        if (diff > maxDiff)
            maxDiff = diff;
        if (diff > 1e-5f)
        { // Change tolerance as needed
            std::cout << "Difference at index " << i << ": "
                      << blobPtr[i] << " vs " << loadedData[i] << std::endl;
            return false; // or break to inspect more differences
        }
    }
    std::cout << "Blobs match! Max difference: " << maxDiff << std::endl;
    return true;
}

bool compareBatchBlobWithBin(const float *d_batch_blob, const std::string &fileName, int N, int C, int H, int W, float tol = 1e-5f)
{
    size_t total = N * C * H * W;
    std::vector<float> host_blob(total);
    cudaMemcpy(host_blob.data(), d_batch_blob, total * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> loadedData(total);
    std::ifstream ifs(fileName, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "Failed to open " << fileName << std::endl;
        return false;
    }
    ifs.read(reinterpret_cast<char *>(loadedData.data()), total * sizeof(float));
    ifs.close();

    float maxDiff = 0.0f;
    int nDiffs = 0;
    for (size_t i = 0; i < total; ++i)
    {
        float diff = std::abs(host_blob[i] - loadedData[i]);
        if (diff > maxDiff)
            maxDiff = diff;
        if (diff > tol)
        {
            if (++nDiffs < 10)
                std::cout << "Diff at " << i << ": " << host_blob[i] << " vs " << loadedData[i] << std::endl;
        }
    }
    std::cout << "Total diffs: " << nDiffs << ", Max diff: " << maxDiff << std::endl;
    return nDiffs == 0;
}

void concatRawFiles(const std::vector<std::string> &inFiles, const std::string &outFile)
{
    std::ofstream out(outFile, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file!");
    }

    for (const auto &inFile : inFiles)
    {
        std::ifstream in(inFile, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("Failed to open input file: " + inFile);
        }
        out << in.rdbuf(); // Efficient file-to-file copy
        in.close();
    }
    out.close();
}

void testNCHWPackSingleKernel(void)
{
    /* NCHW pack only kernel
    std::vector<float *> img_ptrs(N);
    for (int i = 0; i < N; ++i)
    {
        img_ptrs[i] = reinterpret_cast<float *>(images_gpu[i].data); // pointer is already device ptr!
    }

    float **d_img_ptrs;
    cudaMalloc(&d_img_ptrs, N * sizeof(float *));
    cudaMemcpy(d_img_ptrs, img_ptrs.data(), N * sizeof(float *), cudaMemcpyHostToDevice);

    float *d_batch_blob;
    cudaMalloc(&d_batch_blob, N * C * H * W * sizeof(float));

    int threadsPerBlock = 256;
    launch_pack_nchw_kernel(d_img_ptrs, d_batch_blob, N, C, H, W, threadsPerBlock);
    */

    /* Pack and normalize kernel */
    // Suppose images_gpu is std::vector<cv::cuda::GpuMat> of size N, type CV_32FC3, already uploaded to GPU.
}

void testNCHWPackBatchKernel(void)
{
    // Need to preprocess images with yolo first
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/exampleYOLOv8/yolov8s.engine", config);

    getBatchInputs();

    // cv::imshow("Before preprocessing", imgs_cpu[0]);
    // cv::waitKey(0);

    // YOLOv8 preprocessing
    // Converting image from BGR to RGB
    // Resizing and padding while preserving aspect ratio
    std::vector<std::vector<cv::cuda::GpuMat>> preprocessedImgs = yolo.preprocess(imgs_gpu[0]);

    std::cout << "Rows: " << imgs_gpu[0].rows
              << ", Cols: " << imgs_gpu[0].cols
              << ", Channels: " << imgs_gpu[0].channels()
              << ", Type: " << imgs_gpu[0].type()
              << std::endl;

    // Preprocessing before passing to engine
    // Convert format from HWC (OpenCV default) -> CHW (TensorRT engine required)
    // Convert to 32FC3 image type
    std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> std = {1.0f, 1.0f, 1.0f};
    bool normalize = true; // normalize to [0,1] before mean/std
    cv::cuda::GpuMat blob = blobFromGpuMatsTest(preprocessedImgs[0], mean, std, normalize, false);

    std::cout << "Rows: " << blob.rows
              << ", Cols: " << blob.cols
              << ", Channels: " << blob.channels()
              << ", Type: " << blob.type()
              << std::endl;

    cv::Mat cpuMat;
    blob.download(cpuMat); // Copy from GPU to CPU

    float *ptr = reinterpret_cast<float *>(cpuMat.data);
    for (int i = 0; i < 10; ++i)
    {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;

    // Testing with CUDA kernal now

    // Single-image batch from preprocessedImgs[0][0]
    std::vector<cv::cuda::GpuMat> testGpuBatch{preprocessedImgs[0][0]};
    int N = testGpuBatch.size();
    int C = 3, H = testGpuBatch[0].rows, W = testGpuBatch[0].cols;

    // Build host array of device pointers (already device ptrs)
    std::vector<float *> img_ptrs(N);
    for (int i = 0; i < N; ++i)
        img_ptrs[i] = reinterpret_cast<float *>(testGpuBatch[i].data);

    float **d_img_ptrs;
    cudaMalloc(&d_img_ptrs, N * sizeof(float *));
    cudaMemcpy(d_img_ptrs, img_ptrs.data(), N * sizeof(float *), cudaMemcpyHostToDevice);

    // Allocate output buffer:
    float *d_batch_blob;
    cudaMalloc(&d_batch_blob, N * C * H * W * sizeof(float));

    // Kernel call
    float mean0 = 0.0f, mean1 = 0.0f, mean2 = 0.0f;
    float std0 = 1.0f, std1 = 1.0f, std2 = 1.0f;
    float norm_factor = 1.f / 255.f; // or 1.0 if already normalized

    int threadsPerBlock = 256;
    int img_stride = testGpuBatch[0].step / sizeof(float);
    launch_preprocess_and_pack_nchw_kernel(
        d_img_ptrs, d_batch_blob,
        N, C, H, W, img_stride,
        mean0, mean1, mean2,
        std0, std1, std2,
        norm_factor,
        threadsPerBlock);

    size_t total = N * C * H * W;
    std::vector<float> host_blob(total);
    cudaMemcpy(host_blob.data(), d_batch_blob, total * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "First 10 values from kernel output: ";
    for (int i = 0; i < 1228800; ++i)
    {
        std::cout << host_blob[i] << " ";
    }
    std::cout << std::endl;

    // Now d_batch_blob contains your fully-preprocessed, NCHW-packed batch, ready for TensorRT or file comparison!
    /*
    bool match = compareBatchBlobWithBin(d_batch_blob, "elephant.raw", N, C, H, W);
    if (match)
    {
        std::cout << "Batch blob matches reference file!" << std::endl;
    }
    else
    {
        std::cout << "Batch blob DOES NOT match reference!" << std::endl;
    }*/
}

void testYOLO(void)
{
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/exampleYOLOv8/yolov8s.engine", config);

    getBatchInputs();

    // cv::imshow("Before preprocessing", imgs_cpu[0]);
    // cv::waitKey(0);

    // YOLOv8 preprocessing
    std::vector<std::vector<cv::cuda::GpuMat>> preprocessedImgs = yolo.preprocess(imgs_gpu[2]);

    cv::Mat preProcImage;
    preprocessedImgs[0][0].download(preProcImage);

    // cv::imshow("After preprocessing", preProcImage);
    // cv::waitKey(0);

    // TRT engine preprocessing - this is what gets fed to the engine
    std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> std = {1.0f, 1.0f, 1.0f};
    bool normalize = true; // normalize to [0,1] before mean/std
    std::vector<void *> buffers{nullptr};
    std::vector<cv::cuda::GpuMat> preprocessedInputs;
    int n = preprocessedImgs.size();

    // Preprocess all the input tensors
    for (size_t i = 0; i < 1; ++i) /* number of output tensors */
    {
        // const auto &batchInput = inputs[i];
        //  OpenCV reads images into memory in NHWC format, while TensorRT expects
        //  images in NCHW format. The following method converts NHWC to NCHW. Even
        //  though TensorRT expects NCHW at IO, during optimization, it can
        //  internally use NHWC to optimize cuda kernels See:
        //  https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        //  Copy over the input data and perform the preprocessing
        // for (int j = 0; j < n; ++j)
        //{
        auto blob = blobFromGpuMatsTest(preprocessedImgs[0], mean, std, normalize, false);
        preprocessedInputs.push_back(blob);
        //}

        buffers[i] = blob.ptr<void>();
        // compareBlobWithBin(blob, "border-collie.raw");
        //  saveImg2Bin(blob, "border-collie.raw");
    }
}

void testYOLO_Batch(void)
{
    // Need to preprocess images with yolo first
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/scratch/yolov8s_batch.engine", config);

    getBatchInputs();

    std::vector<std::vector<Object>> detections = yolo.detectObjects(imgs_gpu);

    for (int i = 0; i < imgs_gpu.size(); ++i)
    {
        cv::Mat cpuImg;

        imgs_gpu[i].download(cpuImg);
        // Draw results
        yolo.drawObjectLabels(cpuImg, detections[i]);

        // Show the frame
        cv::imshow("YOLOv8 Detection", cpuImg);
    }

    return;
    // cv::imshow("Before preprocessing", imgs_cpu[0]);
    // cv::waitKey(0);

    // YOLOv8 preprocessing
    // Converting image from BGR to RGB
    // Resizing and padding while preserving aspect ratio
    std::vector<std::vector<cv::cuda::GpuMat>> preprocessedImgs = yolo.preprocess(imgs_gpu);

    std::cout << "batch " << preprocessedImgs[0].size() << std::endl;

    std::cout << "Rows: " << imgs_gpu[0].rows
              << ", Cols: " << imgs_gpu[0].cols
              << ", Channels: " << imgs_gpu[0].channels()
              << ", Type: " << imgs_gpu[0].type()
              << std::endl;

    // Preprocessing before passing to engine
    // Convert format from HWC (OpenCV default) -> CHW (TensorRT engine required)
    // Convert to 32FC3 image type
    std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> std = {1.0f, 1.0f, 1.0f};
    bool normalize = true; // normalize to [0,1] before mean/std
    cv::cuda::GpuMat blob = blobFromGpuMatsTest(preprocessedImgs[0], mean, std, normalize, false);

    std::cout << "Rows: " << blob.rows
              << ", Cols: " << blob.cols
              << ", Channels: " << blob.channels()
              << ", Type: " << blob.type()
              << std::endl;

    cv::Mat cpuMat;
    blob.download(cpuMat); // Copy from GPU to CPU

    float *ptr = reinterpret_cast<float *>(cpuMat.data);
    for (int i = 0; i < 10; ++i)
    {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;
}

int main(void)
{
    // concatRawFiles({"elephant.raw", "squirrel.raw", "border-collie.raw"}, "batch_blob.raw");
    testYOLO_Batch();

    return 0;
}