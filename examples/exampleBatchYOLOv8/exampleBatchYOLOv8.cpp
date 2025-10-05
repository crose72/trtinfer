#include "YOLOv8.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

int main()
{
    // --- init YOLO ---
    YOLOv8::Config cfg;
    YOLOv8 yolo("/workspace/examples/models/yolov8s_batch.engine", cfg);
    yolo.printEngineInfo();

    // --- open videos ---
    std::vector<std::string> paths = {
        "/workspace/examples/sampleData/dancing1.mp4",
        "/workspace/examples/sampleData/dancing2.mp4",
        "/workspace/examples/sampleData/soccer.mp4"};
    std::vector<cv::VideoCapture> caps(paths.size());
    for (size_t i = 0; i < paths.size(); ++i)
    {
        caps[i].open(paths[i]);
        if (!caps[i].isOpened())
        {
            std::cerr << "Failed to open: " << paths[i] << "\n";
        }
    }
    if (std::none_of(caps.begin(), caps.end(), [](auto &c)
                     { return c.isOpened(); }))
        return -1;

    std::vector<bool> done(paths.size(), false);

    while (true)
    {
        // Build batch from whatever frames are available this tick
        std::vector<cv::Mat> batch;
        std::vector<int> idx; // remember which cap each frame came from

        for (size_t i = 0; i < caps.size(); ++i)
        {
            if (done[i] || !caps[i].isOpened())
                continue;
            cv::Mat f;
            if (!caps[i].read(f) || f.empty())
            {
                done[i] = true;
                continue;
            }
            batch.push_back(f);
            idx.push_back((int)i);
        }

        // stop if nothing left
        if (batch.empty())
            break;

        // batched inference
        std::vector<std::vector<Object>> dets = yolo.detectObjects(batch);

        // draw & show (one window per stream)
        for (size_t k = 0; k < batch.size(); ++k)
        {
            yolo.drawObjectLabels(batch[k], dets[k]);
            std::string win = "Video " + std::to_string(idx[k] + 1);
            cv::imshow(win, batch[k]);
        }

        // ESC to quit
        int key = cv::waitKey(1);
        if (key == 27)
            break;

        // if all sources are done, exit
        if (std::all_of(done.begin(), done.end(), [](bool b)
                        { return b; }))
            break;
    }

    for (auto &c : caps)
        if (c.isOpened())
            c.release();
    cv::destroyAllWindows();
    return 0;
}
