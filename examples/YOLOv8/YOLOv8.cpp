#include "YOLOv8.h"
#include <algorithm>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "utils.h"

/**
 * @brief YOLOv8 constructor. Loads a TensorRT engine and parameters
 *        from a model path and config.
 * @param trtModelPath Path to TensorRT engine file.
 * @param config YOLOv8 configuration parameters.
 */
YOLOv8::YOLOv8(const std::string &trtModelPath, const Config &config)
    : mDetectionThreshold(config.probabilityThreshold),
      mNMSThreshold(config.nmsThreshold),
      mTopK(config.topK),
      mSegChannels(config.segChannels),
      mSegHeight(config.segH),
      mSegWidth(config.segW),
      mSegThreshold(config.segmentationThreshold),
      mClassNames(config.classNames),
      mNumKPS(config.numKPS),
      mKPSThresh(config.kpsThreshold)
{
    if (!trtModelPath.empty())
    {
        mEngine = std::make_unique<TRTEngine<float>>(trtModelPath);

        bool engineLoaded = mEngine->loadNetwork(trtModelPath, mMean, mStd, mNormalize);

        if (!engineLoaded)
        {
            const std::string msg = "Error: Unable to load TensorRT engine from " + trtModelPath;
            spdlog::error(msg);
        }
    }
    else
    {
        const std::string msg = "No TensorRT engine path provided.";
        spdlog::error(msg);
    }

    // Get engine parameters
    const std::vector<nvinfer1::Dims> inputDims = mEngine->getInputDims();

    if (inputDims[0].nbDims == 4)
    {
        // [N, C, H, W]
        mEngineInputHeight = inputDims[0].d[2];
        mEngineInputWidth = inputDims[0].d[3];
    }
    else if (inputDims[0].nbDims == 3)
    {
        // [C, H, W] (single image, N omitted)
        mEngineInputHeight = inputDims[0].d[1];
        mEngineInputWidth = inputDims[0].d[2];
    }
    else
    {
        // Unexpected engine format
        const std::string msg = "Unsupported input tensor for YOLO, actual input tensors is: " + std::to_string(inputDims[0].nbDims);
        spdlog::error(msg);
    }

    const std::vector<nvinfer1::Dims> outputDims = mEngine->getOutputDims();

    mNumOutputTensors = outputDims.size();

    if (outputDims[0].nbDims == 3)
    {
        mEngineBatchSize = outputDims[0].d[0];
        mNumAnchorFeatures = outputDims[0].d[1]; // 84 for detection - proposal [x,y,h,w,objectness,class 0,...,class 79]
        mNumAnchors = outputDims[0].d[2];        // 8400 for detection - number of anchors (proposals)
    }
    else
    {
        // Unexpected engine format
        const std::string msg = "Unsupported output tensor for YOLO, actual output tensors is: " + std::to_string(outputDims[0].nbDims);
        spdlog::error(msg);
    }

    mNumClasses = mClassNames.size();
}

/**
 * @brief Preprocesses a CUDA BGR image for YOLOv8 inference.
 * @param gpuImg Input CUDA image (BGR).
 * @return Single
 */
std::vector<std::vector<cv::cuda::GpuMat>> YOLOv8::preprocess(const cv::cuda::GpuMat &gpuImg)
{
    // Convert the image from BGR to RGB
    // This assumes that the image is being read as the OpenCV standard BGR format
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    // Initialize the resized image
    cv::cuda::GpuMat resizedImg = rgbMat;

    // Save the size of the image
    // These params will be used in the post-processing stage
    mInputImgHeight = rgbMat.rows;
    mInputImgWidth = rgbMat.cols;

    // How much the image is scaled up/down to required YOLO size
    mAspectScaleFactor = (float)1.0 / std::min(mEngineInputWidth / static_cast<float>(mInputImgWidth),
                                               mEngineInputHeight / static_cast<float>(mInputImgHeight));

    // Resize to the model expected input size while maintaining the
    // aspect ratio with the use of padding
    if (mInputImgHeight != mEngineInputHeight || mInputImgWidth != mEngineInputWidth)
    {
        // Only resize if not already the right size to avoid unecessary copy
        resizedImg = letterbox(rgbMat, mEngineInputHeight, mEngineInputWidth);
    }

    // Convert to format expected by the tensorrt inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> preprocessedImg{std::move(resizedImg)};
    std::vector<std::vector<cv::cuda::GpuMat>> preprocessedBatch{std::move(preprocessedImg)};

    return preprocessedBatch;
}

/**
 * @brief Preprocesses a batch of CUDA BGR images for YOLOv8 inference.
 * @param gpuImg Input vector of CUDA images (BGR).
 * @return Nested vector of resized images with RGB format and padding
 */
std::vector<std::vector<cv::cuda::GpuMat>> YOLOv8::preprocess(const std::vector<cv::cuda::GpuMat> &gpuImgs)
{
    mActualBatchSize = gpuImgs.size();
    mInputImgHeights.clear();
    mInputImgWidths.clear();
    mAspectScaleFactors.clear();

    std::vector<cv::cuda::GpuMat> resizedImgs;

    for (const auto &gpuImg : gpuImgs)
    {
        float inputImgHeight = gpuImg.rows;
        float inputImgWidth = gpuImg.cols;

        // Convert the image from BGR to RGB
        // This assumes that the image is being read as the OpenCV standard BGR format
        cv::cuda::GpuMat rgbMat;
        cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

        // Initialize the resized image
        cv::cuda::GpuMat resizedImg = rgbMat;

        // Resize to the model expected input size while maintaining the
        // aspect ratio with the use of padding
        if (inputImgHeight != mEngineInputHeight || inputImgWidth != mEngineInputWidth)
        {
            resizedImg = letterbox(rgbMat, mEngineInputHeight, mEngineInputWidth);
        }

        resizedImgs.push_back(resizedImg);

        // Save the size of the image
        // These params will be used in the post-processing stage
        mInputImgHeights.push_back(inputImgHeight);
        mInputImgWidths.push_back(inputImgWidth);

        // How much the image is scaled up/down to required YOLO size
        mAspectScaleFactors.push_back(
            (float)1.0 / std::min(mEngineInputWidth / static_cast<float>(inputImgWidth),
                                  mEngineInputHeight / static_cast<float>(inputImgHeight)));
    }

    // For YOLOv8, each element of the outer vector is a model input tensor (only 1 for normal YOLO).
    // Each inner vector is a batch. So we want [ [img1,img2,...,imgN] ]
    std::vector<std::vector<cv::cuda::GpuMat>> preprocessedBatch;
    preprocessedBatch.push_back(std::move(resizedImgs));

    return preprocessedBatch;
}

/**
 * @brief Run YOLOv8 inference and return detected objects (CUDA input).
 * @param inputImgBGR Input CUDA image (BGR).
 * @return Vector of detected objects.
 */
std::vector<Object> YOLOv8::detectObjects(const cv::cuda::GpuMat &inputImgBGR)
{
    // Start timer to clock preprocessing time
#ifdef ENABLE_BENCHMARKS
    static int numIts = 1;
    preciseStopwatch s1;
#endif

    // Preprocess the input image
    const std::vector<std::vector<cv::cuda::GpuMat>> engineInputBatch = preprocess(inputImgBGR);

    // End timer to clock preprocessing time
#ifdef ENABLE_BENCHMARKS
    static long long t1 = 0;
    t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
    spdlog::info("Avg Preprocess time: {:.3f} ms", (t1 / numIts) / 1000.f);
#endif

    // Start timer to clock inference time
#ifdef ENABLE_BENCHMARKS
    preciseStopwatch s2;
#endif

    // Run inference using the TensorRT engine
    // Raw engine outputs
    std::vector<std::vector<std::vector<float>>> featureVectors;

    bool inferenceSuccessful = mEngine->runInference(engineInputBatch, featureVectors);

    if (!inferenceSuccessful)
    {
        const std::string msg = "Error: Unable to run inference.";
        spdlog::error(msg);
    }

    // End timer to clock inference time
#ifdef ENABLE_BENCHMARKS
    static long long t2 = 0;
    t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
    spdlog::info("Avg Inference time: {:.3f} ms", (t2 / numIts) / 1000.f);

    // Start timer to clock postprocessing time
    preciseStopwatch s3;
#endif

    // Check if our model does only object detection or also supports segmentation
    std::vector<Object> detections;

    if (mNumOutputTensors == 1)
    {
        // Object detection or pose estimation
        // Since we have a batch size of 1 and only 1 output,
        // we must convert the output from a 3D array to a 1D array.
        // TODO: change single image pre/post processing and inference
        // to not use nested vector
        std::vector<float> featureVector;
        transformOutput(featureVectors, featureVector);

        // TODO: Need to improve this to make it more generic (don't use magic number).
        // For now it works with Ultralytics pretrained models.
        if (mNumAnchorFeatures == mNumPoseAnchorFeatures)
        {
            // Pose estimation
            detections = postprocessPose(featureVector);
        }
        else
        {
            // Object detection
            detections = postprocessDetect(featureVector);
        }
    }
    else
    {
        // Segmentation
        // Since we have a batch size of 1 and 2 outputs, we must convert the output from a 3D array to a 2D array.
        std::vector<std::vector<float>> featureVector;
        transformOutput(featureVectors, featureVector);
        detections = postProcessSegmentation(featureVector);
    }

    // End timer to clock postprocessing time
#ifdef ENABLE_BENCHMARKS
    static long long t3 = 0;
    t3 += s3.elapsedTime<long long, std::chrono::microseconds>();
    spdlog::info("Avg Postprocess time: {:.3f} ms", (t3 / numIts++) / 1000.f);
#endif

    return detections;
}

std::vector<std::vector<Object>> YOLOv8::detectObjects(const std::vector<cv::cuda::GpuMat> &batchInputImgsBGR)
{
    // Preprocess the input image
#ifdef ENABLE_BENCHMARKS
    static int numIts = 1;
    preciseStopwatch s1;
#endif

    const std::vector<std::vector<cv::cuda::GpuMat>> engineInputBatch = preprocess(batchInputImgsBGR);

    // End timer to clock preprocessing time
#ifdef ENABLE_BENCHMARKS
    static long long t1 = 0;
    t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
    spdlog::info("Avg Preprocess time: {:.3f} ms", (t1 / numIts) / 1000.f);
#endif

    // Start timer to clock inference time
#ifdef ENABLE_BENCHMARKS
    preciseStopwatch s2;
#endif

    // Run inference using the TensorRT engine
    // Raw engine outputs
    std::vector<std::vector<std::vector<float>>> featureVectors;

    bool inferenceSuccessful = mEngine->runInference(engineInputBatch, featureVectors);

    if (!inferenceSuccessful)
    {
        const std::string msg = "Error: Unable to run inference.";
        spdlog::error(msg);
    }

    // End timer to clock inference time
#ifdef ENABLE_BENCHMARKS
    static long long t2 = 0;
    t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
    spdlog::info("Avg Inference time: {:.3f} ms", (t2 / numIts) / 1000.f);

    // Start timer to clock postprocessing time
    preciseStopwatch s3;
#endif

    // Check if our model does only object detection or also supports segmentation
    std::vector<std::vector<Object>> detections;

    if (mNumOutputTensors == 1)
    {
        // Object detection or pose estimation
        // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
        std::vector<std::vector<float>> featureVector;
        transformOutput(featureVectors, featureVector);

        if (mNumAnchorFeatures == mNumPoseAnchorFeatures)
        {
            // TODO: add support batch of images
            // Pose estimation
            // detections = postprocessPose(featureVector);
        }
        else
        {
            // Object detection
            for (int i = 0; i < mActualBatchSize; ++i)
            {
                detections.push_back(postprocessDetect(featureVector[i], i));
            }
        }
    }
    else
    {
        // TODO: add support batch of images
        // Segmentation
        // Since we have a batch size of 1 and 2 outputs, we must convert the output from a 3D array to a 2D array.
        // std::vector<std::vector<float>> featureVector;
        // transformOutput(featureVectors, featureVector);
        // detections = postProcessSegmentation(featureVector);
    }

    // End timer to clock postprocessing time
#ifdef ENABLE_BENCHMARKS
    static long long t3 = 0;
    t3 += s3.elapsedTime<long long, std::chrono::microseconds>();
    spdlog::info("Avg Postprocess time: {:.3f} ms", (t3 / numIts++) / 1000.f);
#endif

    return detections;
}

/**
 * @brief Run YOLOv8 inference and return detected objects (CPU input).
 * @param inputImgBGR Input OpenCV image (BGR, CPU memory).
 * @return Vector of detected objects for single image.
 */
std::vector<Object> YOLOv8::detectObjects(const cv::Mat &inputImgBGR)
{
    // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImgBGR);

    // Call detectObjects with the GPU image
    return detectObjects(gpuImg);
}

/**
 * @brief Run YOLOv8 inference and return detected objects (CPU input).
 * @param batchInputImgsBGR Input OpenCV images (BGR, CPU memory).
 * @return Vector of detected objects for a batch of images.
 */
std::vector<std::vector<Object>> YOLOv8::detectObjects(const std::vector<cv::Mat> &batchInputImgsBGR)
{
    // Upload the image to GPU memory
    std::vector<cv::cuda::GpuMat> gpuImgs;

    for (const auto &cpuImg : batchInputImgsBGR)
    {
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);

        gpuImgs.push_back(gpuImg);
    }

    // Call detectObjects with the batch of GPU images
    return detectObjects(gpuImgs);
}

/**
 * @brief Post-process segmentation output tensors into object vector (YOLOv8 Segmentation).
 * @param featureVectors Model output tensors (segmentation).
 * @return Vector of detected objects with segmentation masks.
 */
std::vector<Object> YOLOv8::postProcessSegmentation(std::vector<std::vector<float>> &featureVectors)
{
    const int numAnchorPoseFeatures = (int)mNumAnchorFeatures - mSegChannels - 4;

    // Ensure the output lengths are correct
    if (featureVectors[0].size() != static_cast<size_t>(mNumAnchorFeatures) * mNumAnchors)
    {
        const std::string msg = "Output at index 0 has incorrect length";
        spdlog::error(msg);
    }

    if (featureVectors[1].size() != static_cast<size_t>(mSegChannels) * mSegHeight * mSegWidth)
    {

        const std::string msg = "Output at index 1 has incorrect length";
        spdlog::error(msg);
    }

    cv::Mat output = cv::Mat(mNumAnchorFeatures, mNumAnchors, CV_32F, featureVectors[0].data());
    output = output.t();

    cv::Mat protos = cv::Mat(mSegChannels, mSegHeight * mSegWidth, CV_32F, featureVectors[1].data());

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    // Object the bounding boxes and class labels
    for (int i = 0; i < mNumAnchors; i++)
    {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maskConfsPtr = rowPtr + 4 + numAnchorPoseFeatures;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numAnchorPoseFeatures);
        float score = *maxSPtr;
        if (score > mDetectionThreshold)
        {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * mAspectScaleFactor, 0.f, mInputImgWidth);
            float y0 = std::clamp((y - 0.5f * h) * mAspectScaleFactor, 0.f, mInputImgHeight);
            float x1 = std::clamp((x + 0.5f * w) * mAspectScaleFactor, 0.f, mInputImgWidth);
            float y1 = std::clamp((y + 0.5f * h) * mAspectScaleFactor, 0.f, mInputImgHeight);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat maskConf = cv::Mat(1, mSegChannels, CV_32F, maskConfsPtr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            maskConfs.push_back(maskConf);
        }
    }

    // Require OpenCV 4.7 for this function
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, mDetectionThreshold, mNMSThreshold, indices);

    // Obtain the segmentation masks
    cv::Mat masks;
    std::vector<Object> objs;
    int cnt = 0;
    for (auto &i : indices)
    {
        if (cnt >= mTopK)
        {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.probability = scores[i];
        masks.push_back(maskConfs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    // Convert segmentation mask to original frame
    if (!masks.empty())
    {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), {mSegWidth, mSegHeight});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        const auto inputDims = mEngine->getInputDims();

        cv::Rect roi;
        if (mInputImgHeight > mInputImgWidth)
        {
            roi = cv::Rect(0, 0, mSegWidth * mInputImgWidth / mInputImgHeight, mSegHeight);
        }
        else
        {
            roi = cv::Rect(0, 0, mSegWidth, mSegHeight * mInputImgHeight / mInputImgWidth);
        }

        for (size_t i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size(static_cast<int>(mInputImgWidth), static_cast<int>(mInputImgHeight)), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > mSegThreshold;
        }
    }

    return objs;
}

/**
 * @brief Post-process pose estimation output tensors into object vector.
 * @param featureVector Model output tensor.
 * @return Vector of detected objects with keypoints.
 */
std::vector<Object> YOLOv8::postprocessPose(std::vector<float> &featureVector)
{
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(mNumAnchorFeatures, mNumAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < mNumAnchors; i++)
    {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto kps_ptr = rowPtr + 5;
        float score = *scoresPtr;
        if (score > mDetectionThreshold)
        {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * mAspectScaleFactor, 0.f, mInputImgWidth);
            float y0 = std::clamp((y - 0.5f * h) * mAspectScaleFactor, 0.f, mInputImgHeight);
            float x1 = std::clamp((x + 0.5f * w) * mAspectScaleFactor, 0.f, mInputImgWidth);
            float y1 = std::clamp((y + 0.5f * h) * mAspectScaleFactor, 0.f, mInputImgHeight);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            std::vector<float> kps;
            for (int k = 0; k < mNumKPS; k++)
            {
                float kpsX = *(kps_ptr + 3 * k) * mAspectScaleFactor;
                float kpsY = *(kps_ptr + 3 * k + 1) * mAspectScaleFactor;
                float kpsS = *(kps_ptr + 3 * k + 2);
                kpsX = std::clamp(kpsX, 0.f, mInputImgWidth);
                kpsY = std::clamp(kpsY, 0.f, mInputImgHeight);
                kps.push_back(kpsX);
                kps.push_back(kpsY);
                kps.push_back(kpsS);
            }

            bboxes.push_back(bbox);
            labels.push_back(0); // All detected objects are people
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, mDetectionThreshold, mNMSThreshold, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices)
    {
        if (cnt >= mTopK)
        {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        obj.kps = kpss[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;
}

std::vector<Object> YOLOv8::postprocessDetect(std::vector<float> &featureVector, int imageInBatch)
{
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    for (int anchor = 0; anchor < mNumAnchors; ++anchor)
    {
        // Fetch bbox
        float x = featureVector[anchor + 0 * mNumAnchors];
        float y = featureVector[anchor + 1 * mNumAnchors];
        float w = featureVector[anchor + 2 * mNumAnchors];
        float h = featureVector[anchor + 3 * mNumAnchors];

        // Fetch class scores
        auto class_start = 4 * mNumAnchors + anchor;
        float max_score = -1;
        int max_label = -1;
        for (int cls = 0; cls < mNumClasses; ++cls)
        {
            float score = featureVector[class_start + cls * mNumAnchors];
            if (score > max_score)
            {
                max_score = score;
                max_label = cls;
            }
        }

        if (max_score > mDetectionThreshold)
        {
            // Undo normalization/clipping as you do
            float x0 = std::clamp((x - 0.5f * w) * mAspectScaleFactors[imageInBatch], 0.f, mInputImgWidths[imageInBatch]);
            float y0 = std::clamp((y - 0.5f * h) * mAspectScaleFactors[imageInBatch], 0.f, mInputImgHeights[imageInBatch]);
            float x1 = std::clamp((x + 0.5f * w) * mAspectScaleFactors[imageInBatch], 0.f, mInputImgWidths[imageInBatch]);
            float y1 = std::clamp((y + 0.5f * h) * mAspectScaleFactors[imageInBatch], 0.f, mInputImgHeights[imageInBatch]);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(max_label);
            scores.push_back(max_score);
        }
    }

    // NMS and rest is fine...
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, mDetectionThreshold, mNMSThreshold, indices);
    std::vector<Object> objects;
    int cnt = 0;
    for (auto &chosenIdx : indices)
    {
        if (cnt >= mTopK)
            break;
        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);
        cnt += 1;
    }

    return objects;
}

/**
 * @brief Post-process detection output tensors into object vector.
 * @param featureVector Model output tensor.
 * @return Vector of detected objects with bounding boxes.
 */
std::vector<Object> YOLOv8::postprocessDetect(std::vector<float> &featureVector)
{
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output = cv::Mat(mNumAnchorFeatures, mNumAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < mNumAnchors; i++)
    {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + mNumClasses);
        float score = *maxSPtr;
        if (score > mDetectionThreshold)
        {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * mAspectScaleFactor, 0.f, mInputImgWidth);
            float y0 = std::clamp((y - 0.5f * h) * mAspectScaleFactor, 0.f, mInputImgHeight);
            float x1 = std::clamp((x + 0.5f * w) * mAspectScaleFactor, 0.f, mInputImgWidth);
            float y1 = std::clamp((y + 0.5f * h) * mAspectScaleFactor, 0.f, mInputImgHeight);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, mDetectionThreshold, mNMSThreshold, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices)
    {
        if (cnt >= mTopK)
        {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;
}

/**
 * @brief Draw object bounding boxes, labels, and masks (if present) on an image.
 * @param image OpenCV image (CPU memory) to annotate.
 * @param objects Vector of detected objects.
 * @param scale Drawing scale (default: 1).
 */
void YOLOv8::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale)
{
    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].boxMask.empty())
    {
        cv::Mat mask = image.clone();
        for (const auto &object : objects)
        {
            // Choose the color
            int colorIndex = object.label % mColorList.size(); // We have only defined 80 unique colors
            cv::Scalar color = cv::Scalar(mColorList[colorIndex][0], mColorList[colorIndex][1], mColorList[colorIndex][2]);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto &object : objects)
    {
        // Choose the color
        int colorIndex = object.label % mColorList.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(mColorList[colorIndex][0], mColorList[colorIndex][1], mColorList[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5)
        {
            txtColor = cv::Scalar(0, 0, 0);
        }
        else
        {
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto &rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", mClassNames[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        // Pose estimation
        if (!object.kps.empty())
        {
            auto &kps = object.kps;
            for (int k = 0; k < mNumKPS + 2; k++)
            {
                if (k < mNumKPS)
                {
                    int kpsX = std::round(kps[k * 3]);
                    int kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > mKPSThresh)
                    {
                        cv::Scalar kpsColor = cv::Scalar(mKPSColors[k][0], mKPSColors[k][1], mKPSColors[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                auto &ske = mSkeleton[k];
                int pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > mKPSThresh && pos2S > mKPSThresh)
                {
                    cv::Scalar limbColor = cv::Scalar(mLimbColors[k][0], mLimbColors[k][1], mLimbColors[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}
