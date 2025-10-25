// Build example (adjust include/library paths as needed):
// g++ -std=c++17 klt_yolo_video.cpp -o klt_yolo_video \
//     `pkg-config --cflags --libs opencv4` \
//     -lvpi -lvpi_opencv_interop -lvpi_host
//
// Run:
// ./klt_yolo_video <cpu|cuda|pva> </path/to/video> </path/to/yolov8.engine> [--conf=0.25] [--class=-1] [--show] [--reseed=0] [--pad=0.12] [--minwh=20] [--tmpl=64]

#include <opencv2/opencv.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/KLTFeatureTracker.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>

// ---- Your detector ----
#include "YOLOv8.h" // assumes Object { cv::Rect2f rect; float probability; int label; }

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0)

static inline cv::Rect clampRect(const cv::Rect2f &r, int W, int H)
{
    int x = std::max(0, (int)std::floor(r.x));
    int y = std::max(0, (int)std::floor(r.y));
    int w = std::max(0, (int)std::round(r.width));
    int h = std::max(0, (int)std::round(r.height));
    if (x + w > W)
        w = std::max(0, W - x);
    if (y + h > H)
        h = std::max(0, H - y);
    return cv::Rect(x, y, w, h);
}

static inline cv::Rect padAndClamp(const cv::Rect &r, int W, int H, float padFrac, int minwh)
{
    float cx = r.x + 0.5f * r.width;
    float cy = r.y + 0.5f * r.height;
    float nw = r.width * (1.f + 2.f * padFrac);
    float nh = r.height * (1.f + 2.f * padFrac);
    cv::Rect2f rf(cx - 0.5f * nw, cy - 0.5f * nh, nw, nh);
    cv::Rect rc = clampRect(rf, W, H);

    // enforce minimum size (helps KLT corner extraction)
    if (rc.width < minwh)
    {
        int d = (minwh - rc.width);
        rc.x = std::max(0, rc.x - d / 2);
        rc.width = std::min(W - rc.x, minwh);
    }
    if (rc.height < minwh)
    {
        int d = (minwh - rc.height);
        rc.y = std::max(0, rc.y - d / 2);
        rc.height = std::min(H - rc.y, minwh);
    }

    return clampRect(rc, W, H);
}

// Draw KLT-estimated current boxes over a BGR frame produced from the grayscale VPI image
static cv::Mat DrawKLT(VPIImage imgGray, VPIArray boxes, VPIArray preds)
{
    // Convert VPI gray image to BGR Mat for drawing
    cv::Mat out;
    {
        VPIImageData data;
        CHECK_STATUS(vpiImageLockData(imgGray, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
        auto &p = data.buffer.pitch;
        CV_Assert(data.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR);

        int cvtype = CV_8U;
        if (p.format == VPI_IMAGE_FORMAT_U8)
            cvtype = CV_8U;
        else if (p.format == VPI_IMAGE_FORMAT_S8)
            cvtype = CV_8S;
        else if (p.format == VPI_IMAGE_FORMAT_U16)
            cvtype = CV_16UC1;
        else if (p.format == VPI_IMAGE_FORMAT_S16)
            cvtype = CV_16SC1;
        else
            throw std::runtime_error("Unsupported image format");

        cv::Mat gray(p.planes[0].height, p.planes[0].width, cvtype, p.planes[0].data, p.planes[0].pitchBytes);
        if (gray.type() == CV_16U)
        {
            cv::Mat tmp8;
            gray.convertTo(tmp8, CV_8U);
            cv::cvtColor(tmp8, out, cv::COLOR_GRAY2BGR);
        }
        else
        {
            cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
        }
        CHECK_STATUS(vpiImageUnlock(imgGray));
    }

    VPIArrayData boxdata, preddata;
    CHECK_STATUS(vpiArrayLockData(boxes, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &boxdata));
    CHECK_STATUS(vpiArrayLockData(preds, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &preddata));

    auto *pboxes = reinterpret_cast<VPIKLTTrackedBoundingBox *>(boxdata.buffer.aos.data);
    auto *ppreds = reinterpret_cast<VPIHomographyTransform2D *>(preddata.buffer.aos.data);
    int N = *boxdata.buffer.aos.sizePointer;

    for (int i = 0; i < N; ++i)
    {
        if (pboxes[i].trackingStatus == 1)
            continue; // dropped

        // NOTE: visible box = template_size * (current_scale = xform.scale * pred.scale)
        float x = pboxes[i].bbox.xform.mat3[0][2] + ppreds[i].mat3[0][2];
        float y = pboxes[i].bbox.xform.mat3[1][2] + ppreds[i].mat3[1][2];
        float w = pboxes[i].bbox.width * pboxes[i].bbox.xform.mat3[0][0] * ppreds[i].mat3[0][0];
        float h = pboxes[i].bbox.height * pboxes[i].bbox.xform.mat3[1][1] * ppreds[i].mat3[1][1];

        cv::rectangle(out, cv::Rect2f(x, y, w, h),
                      cv::Scalar(0, 255 - ((i * 37) % 200), 50 + ((i * 53) % 205)), 2);
        cv::putText(out, std::to_string(i), cv::Point2f(x, y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    CHECK_STATUS(vpiArrayUnlock(preds));
    CHECK_STATUS(vpiArrayUnlock(boxes));
    return out;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <cpu|cuda|pva> <input_video> <yolov8_engine> [--conf=0.25] [--class=-1] [--show] [--reseed=0] [--pad=0.12] [--minwh=20] [--tmpl=64]\n";
        return 1;
    }

    std::string backendStr = argv[1];
    std::string videoPath = argv[2];
    std::string enginePath = argv[3];

    float confThr = 0.25f;
    int classFilter = -1; // -1 => all classes
    bool showWindow = false;
    int reseedEvery = 0; // 0=disabled
    float padFrac = 0.12f;
    int minwh = 20;
    int tmplMax = 64; // <-- critical: cap template size <= ~64

    for (int i = 4; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a.rfind("--conf=", 0) == 0)
            confThr = std::stof(a.substr(7));
        else if (a.rfind("--class=", 0) == 0)
            classFilter = std::stoi(a.substr(8));
        else if (a == "--show")
            showWindow = true;
        else if (a.rfind("--reseed=", 0) == 0)
            reseedEvery = std::stoi(a.substr(9));
        else if (a.rfind("--pad=", 0) == 0)
            padFrac = std::stof(a.substr(6));
        else if (a.rfind("--minwh=", 0) == 0)
            minwh = std::stoi(a.substr(8));
        else if (a.rfind("--tmpl=", 0) == 0)
            tmplMax = std::stoi(a.substr(7));
    }

    VPIBackend backend;
    if (backendStr == "cpu")
        backend = VPI_BACKEND_CPU;
    else if (backendStr == "cuda")
        backend = VPI_BACKEND_CUDA;
    else if (backendStr == "pva")
        backend = VPI_BACKEND_PVA;
    else
    {
        std::cerr << "Backend must be cpu|cuda|pva\n";
        return 1;
    }

    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Can't open video: " << videoPath << "\n";
        return 1;
    }

    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
        fps = 30.0;

    // Output writer
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // mp4-friendly
    cv::VideoWriter writer("klt_" + backendStr + ".mp4", fourcc, fps, cv::Size(W, H));

    // Init YOLO
    YOLOv8::Config ycfg;
    YOLOv8 yolo(enginePath, ycfg);

    // VPI core objects
    VPIStream stream = nullptr;
    VPIPayload klt = nullptr;
    VPIImage imgTemplate = nullptr, imgReference = nullptr;
    VPIArray inputBoxList = nullptr, inputPredList = nullptr;
    VPIArray outputBoxList = nullptr, outputEstimList = nullptr;

    // Wrapped storage
    std::vector<VPIKLTTrackedBoundingBox> bboxes;
    bboxes.reserve(128);
    std::vector<VPIHomographyTransform2D> preds;
    preds.reserve(128);
    int32_t bboxesSize = 0, predsSize = 0;

    auto seed_one = [&](const cv::Rect &rin, int indexTag)
    {
        // pad and clamp for good features
        cv::Rect r = padAndClamp(rin, W, H, padFrac, minwh);

        // choose a template size <= tmplMax while preserving aspect as much as possible
        float tw = std::min<float>(tmplMax, std::max(8, r.width));
        float th = std::min<float>(tmplMax, std::max(8, r.height));

        // Better: keep aspect. Fit within tmplMax box.
        float scaleToTmplW = (float)r.width / tw;
        float scaleToTmplH = (float)r.height / th;

        VPIKLTTrackedBoundingBox t{}; // zero
        t.bbox.width = tw;            // <-- template size (small)
        t.bbox.height = th;

        // identity *except scale encodes the mapping template->image
        t.bbox.xform.mat3[0][0] = scaleToTmplW; // sx
        t.bbox.xform.mat3[1][1] = scaleToTmplH; // sy
        t.bbox.xform.mat3[2][2] = 1.0f;

        // translation (top-left in image coords)
        t.bbox.xform.mat3[0][2] = (float)r.x;
        t.bbox.xform.mat3[1][2] = (float)r.y;

        t.trackingStatus = 0; // valid
        t.templateStatus = 1; // MUST UPDATE on first submit

        bboxes.push_back(t);

        VPIHomographyTransform2D xf{}; // identity
        xf.mat3[0][0] = xf.mat3[1][1] = xf.mat3[2][2] = 1.0f;
        preds.push_back(xf);

        std::cerr << "[Seed] #" << indexTag
                  << " imgBox=(" << r.x << "," << r.y << "," << r.width << "," << r.height << ")"
                  << " tmpl=(" << tw << "x" << th << ")"
                  << " scale=(" << scaleToTmplW << "," << scaleToTmplH << ")\n";
    };

    try
    {
        // Read first frame (color), run YOLO to seed trackers
        cv::Mat frameBGR;
        if (!cap.read(frameBGR) || frameBGR.empty())
            throw std::runtime_error("Empty first frame");

        // Run YOLO on color
        std::vector<Object> dets = yolo.detectObjects(frameBGR);

        // Filter detections
        std::vector<cv::Rect> seeds;
        seeds.reserve(dets.size());
        for (const auto &d : dets)
        {
            if (d.probability < confThr)
                continue;
            if (classFilter >= 0 && d.label != classFilter)
                continue;
            cv::Rect r = clampRect(d.rect, W, H);
            if (r.width > 4 && r.height > 4)
                seeds.push_back(r);
        }
        if (seeds.empty())
            throw std::runtime_error("YOLO produced no usable detections on first frame (adjust --conf/--class).");

        if ((int)seeds.size() > 128)
            seeds.resize(128); // PVA-friendly capacity

        // Prepare initial KLT boxes + identity predictions (with capped template)
        for (size_t i = 0; i < seeds.size(); ++i)
            seed_one(seeds[i], (int)i);

        std::cerr << "[Init] Seeded " << bboxes.size()
                  << " boxes (pad=" << padFrac << ", minwh=" << minwh
                  << ", tmplMax=" << tmplMax << ")\n";

        // Wrap input arrays
        {
            VPIArrayData data{};
            data.bufferType = VPI_ARRAY_BUFFER_HOST_AOS;
            data.buffer.aos.capacity = (int)bboxes.capacity();
            data.buffer.aos.sizePointer = &bboxesSize;

            // KLT boxes
            data.buffer.aos.type = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;
            data.buffer.aos.data = bboxes.data();
            CHECK_STATUS(vpiArrayCreateWrapper(&data, 0, &inputBoxList));

            // Preds
            data.buffer.aos.type = VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D;
            data.buffer.aos.data = preds.data();
            CHECK_STATUS(vpiArrayCreateWrapper(&data, 0, &inputPredList));

            // Set initial sizes
            bboxesSize = (int)bboxes.size();
            predsSize = (int)preds.size();
            CHECK_STATUS(vpiArraySetSize(inputBoxList, bboxesSize));
            CHECK_STATUS(vpiArraySetSize(inputPredList, predsSize));
        }

        // Create stream
        CHECK_STATUS(vpiStreamCreate(backend, &stream));

        // Prepare grayscale template image wrapper
        cv::Mat frameGray;
        if (frameBGR.channels() == 3)
            cv::cvtColor(frameBGR, frameGray, cv::COLOR_BGR2GRAY);
        else
            frameGray = frameBGR;
        if (backend == VPI_BACKEND_PVA)
        {
            cv::Mat u16;
            frameGray.convertTo(u16, CV_16U);
            frameGray = u16;
        }
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(frameGray, 0, &imgTemplate));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(frameGray, 0, &imgReference)); // updated per loop

        VPIImageFormat fmt;
        CHECK_STATUS(vpiImageGetFormat(imgTemplate, &fmt));

        // Create KLT payload
        CHECK_STATUS(vpiCreateKLTFeatureTracker(backend, frameGray.cols, frameGray.rows, fmt, nullptr, &klt));

        // Default KLT params
        VPIKLTFeatureTrackerParams params;
        CHECK_STATUS(vpiInitKLTFeatureTrackerParams(&params));
        // If your VPI version exposes more knobs, tweak here (numFeaturesPerBox, maxItersLK, etc).

        // Output arrays
        CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &outputBoxList));
        CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &outputEstimList));

        // -------- PRIME STEP --------
        // Build templates by submitting with template==reference on the first frame.
        CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgReference, frameGray));
        CHECK_STATUS(vpiSubmitKLTFeatureTracker(stream, backend, klt,
                                                imgTemplate, inputBoxList, inputPredList,
                                                imgReference, outputBoxList, outputEstimList, &params));
        CHECK_STATUS(vpiStreamSync(stream));

        // Pull results and clear templateStatus for boxes that succeeded
        {
            VPIArrayData outB0, outX0;
            CHECK_STATUS(vpiArrayLockData(outputBoxList, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outB0));
            CHECK_STATUS(vpiArrayLockData(outputEstimList, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outX0));
            CHECK_STATUS(vpiArrayLock(inputBoxList, VPI_LOCK_READ_WRITE));
            CHECK_STATUS(vpiArrayLock(inputPredList, VPI_LOCK_READ_WRITE));

            auto *u0 = reinterpret_cast<VPIKLTTrackedBoundingBox *>(outB0.buffer.aos.data);
            auto *ex0 = reinterpret_cast<VPIHomographyTransform2D *>(outX0.buffer.aos.data);
            int N0 = *outB0.buffer.aos.sizePointer;

            int primedOK = 0, primedDrop = 0;
            for (int i = 0; i < N0; ++i)
            {
                if (u0[i].trackingStatus == 0)
                {
                    bboxes[i] = u0[i];
                    bboxes[i].templateStatus = 0; // template is now built
                    preds[i] = ex0[i];            // often identity
                    primedOK++;
                }
                else
                {
                    primedDrop++;
                    std::cerr << "[Prime] dropped box " << i
                              << " (tmpl:" << bboxes[i].bbox.width << "x" << bboxes[i].bbox.height
                              << " scale:" << bboxes[i].bbox.xform.mat3[0][0] << "," << bboxes[i].bbox.xform.mat3[1][1]
                              << ")\n";
                }
            }
            CHECK_STATUS(vpiArrayUnlock(inputBoxList));
            CHECK_STATUS(vpiArrayUnlock(inputPredList));
            CHECK_STATUS(vpiArrayUnlock(outputBoxList));
            CHECK_STATUS(vpiArrayUnlock(outputEstimList));

            std::cerr << "[Prime] ok=" << primedOK << " drop=" << primedDrop << "\n";
        }
        // ----------------------------

        // MAIN LOOP
        size_t frameIdx = 0;
        for (;; ++frameIdx)
        {
            // Draw last template result and write/show
            {
                cv::Mat drawn = DrawKLT(imgTemplate, inputBoxList, inputPredList);
                writer << drawn;
                if (showWindow)
                {
                    cv::imshow("KLT+YOLO (tracked)", drawn);
                    if ((cv::waitKey(1) & 0xFF) == 27)
                        break; // ESC
                }
            }

            // Read next frame
            cv::Mat nextBGR;
            if (!cap.read(nextBGR) || nextBGR.empty())
                break;

            // Optional: periodic reseed (run YOLO and reset existing or add new boxes)
            if (reseedEvery > 0 && (frameIdx % (size_t)reseedEvery) == 0)
            {
                std::vector<Object> d2 = yolo.detectObjects(nextBGR);
                std::vector<cv::Rect> seeds2;
                for (const auto &d : d2)
                {
                    if (d.probability < confThr)
                        continue;
                    if (classFilter >= 0 && d.label != classFilter)
                        continue;
                    cv::Rect r = clampRect(d.rect, W, H);
                    if (r.width > 4 && r.height > 4)
                        seeds2.push_back(r);
                }
                if (!seeds2.empty())
                {
                    size_t K = std::min<size_t>(seeds2.size(), bboxes.capacity());
                    for (size_t i = 0; i < K; ++i)
                    {
                        // reseed uses same capped-template strategy
                        cv::Rect rin = seeds2[i];
                        cv::Rect r = padAndClamp(rin, W, H, padFrac, minwh);

                        float tw = std::min<float>(tmplMax, std::max(8, r.width));
                        float th = std::min<float>(tmplMax, std::max(8, r.height));
                        float sx = (float)r.width / tw;
                        float sy = (float)r.height / th;

                        VPIKLTTrackedBoundingBox t{};
                        t.bbox.width = tw;
                        t.bbox.height = th;
                        t.bbox.xform.mat3[0][0] = sx;
                        t.bbox.xform.mat3[1][1] = sy;
                        t.bbox.xform.mat3[2][2] = 1.0f;
                        t.bbox.xform.mat3[0][2] = (float)r.x;
                        t.bbox.xform.mat3[1][2] = (float)r.y;
                        t.trackingStatus = 0;
                        t.templateStatus = 1; // rebuild template on next submit

                        if (i < bboxes.size())
                            bboxes[i] = t;
                        else
                            bboxes.push_back(t);

                        VPIHomographyTransform2D xf{};
                        xf.mat3[0][0] = xf.mat3[1][1] = xf.mat3[2][2] = 1.0f;
                        if (i < preds.size())
                            preds[i] = xf;
                        else
                            preds.push_back(xf);

                        std::cerr << "[Reseed] #" << i
                                  << " imgBox=(" << r.x << "," << r.y << "," << r.width << "," << r.height << ")"
                                  << " tmpl=(" << tw << "x" << th << ")"
                                  << " scale=(" << sx << "," << sy << ")\n";
                    }
                    bboxesSize = (int)std::min(bboxes.size(), bboxes.capacity());
                    predsSize = (int)std::min(preds.size(), preds.capacity());
                    CHECK_STATUS(vpiArraySetSize(inputBoxList, bboxesSize));
                    CHECK_STATUS(vpiArraySetSize(inputPredList, predsSize));
                }
            }

            // Convert to gray for VPI
            cv::Mat nextGray;
            if (nextBGR.channels() == 3)
                cv::cvtColor(nextBGR, nextGray, cv::COLOR_BGR2GRAY);
            else
                nextGray = nextBGR;
            if (backend == VPI_BACKEND_PVA)
            {
                cv::Mat u16;
                nextGray.convertTo(u16, CV_16U);
                nextGray = u16;
            }
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgReference, nextGray));

            // Submit KLT tracking
            CHECK_STATUS(vpiSubmitKLTFeatureTracker(stream, backend, klt,
                                                    imgTemplate, inputBoxList, inputPredList,
                                                    imgReference, outputBoxList, outputEstimList, &params));
            CHECK_STATUS(vpiStreamSync(stream));

            // Update input arrays based on output for next iteration
            VPIArrayData outB, outX;
            CHECK_STATUS(vpiArrayLockData(outputBoxList, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outB));
            CHECK_STATUS(vpiArrayLockData(outputEstimList, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outX));
            CHECK_STATUS(vpiArrayLock(inputBoxList, VPI_LOCK_READ_WRITE));
            CHECK_STATUS(vpiArrayLock(inputPredList, VPI_LOCK_READ_WRITE));

            auto *updated = reinterpret_cast<VPIKLTTrackedBoundingBox *>(outB.buffer.aos.data);
            auto *est = reinterpret_cast<VPIHomographyTransform2D *>(outX.buffer.aos.data);
            int N = *outB.buffer.aos.sizePointer;

            int droppedThis = 0, templatedThis = 0;
            for (int i = 0; i < N; ++i)
            {
                if (updated[i].trackingStatus)
                {
                    // lost
                    if (bboxes[i].trackingStatus == 0)
                    {
                        droppedThis++;
                        bboxes[i].trackingStatus = 1;
                        bboxes[i].templateStatus = 0;
                    }
                    continue;
                }
                if (updated[i].templateStatus)
                {
                    // reset template with current box; identity prediction
                    templatedThis++;
                    bboxes[i] = updated[i];
                    bboxes[i].templateStatus = 1;
                    VPIHomographyTransform2D id{};
                    id.mat3[0][0] = id.mat3[1][1] = id.mat3[2][2] = 1.0f;
                    preds[i] = id;
                }
                else
                {
                    // keep template; update prediction with estimate
                    bboxes[i].templateStatus = 0;
                    preds[i] = est[i];
                    bboxes[i].trackingStatus = 0;
                }
            }

            if (droppedThis || templatedThis)
            {
                std::cerr << "[Frame " << frameIdx << "] dropped=" << droppedThis
                          << " templUpdated=" << templatedThis << "\n";
            }

            CHECK_STATUS(vpiArrayUnlock(inputBoxList));
            CHECK_STATUS(vpiArrayUnlock(inputPredList));
            CHECK_STATUS(vpiArrayUnlock(outputBoxList));
            CHECK_STATUS(vpiArrayUnlock(outputEstimList));

            // Next iteration: current reference becomes template
            std::swap(imgTemplate, imgReference);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << "\n";
        if (stream)
            vpiStreamSync(stream);
    }

    // Cleanup
    vpiImageDestroy(imgReference);
    vpiImageDestroy(imgTemplate);
    vpiArrayDestroy(outputEstimList);
    vpiArrayDestroy(outputBoxList);
    vpiArrayDestroy(inputPredList);
    vpiArrayDestroy(inputBoxList);
    vpiPayloadDestroy(klt);
    vpiStreamDestroy(stream);

    return 0;
}
