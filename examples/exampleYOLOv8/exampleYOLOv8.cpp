#include "YOLOv8.h"
#include <chrono>
#include <thread>

// ... your YOLOv8 includes and using namespace ...

int main()
{
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/trtinfer/examples/exampleYOLOv8/yolov8s_pose_batch_fp32.engine", config);
    yolo.printEngineInfo();

    std::string video_path1 = "/workspace/trtinfer/examples/media/dancing1.mp4";
    std::string video_path2 = "/workspace/trtinfer/examples/media/dancing2.mp4";

    cv::VideoCapture cap1(video_path1);
    cv::VideoCapture cap2; // open after delay

    if (!cap1.isOpened())
    {
        std::cerr << "Error opening video: " << video_path1 << std::endl;
        return -1;
    }

    bool started2 = false;
    bool end1 = false;
    bool end2 = false;
    auto start_time = std::chrono::steady_clock::now();

    cv::Mat frame1;
    cv::Mat frame2;

    while (true)
    {
        // Read next frame from video 1
        if (!end1)
        {
            cap1 >> frame1;
            if (frame1.empty())
            {
                end1 = true;
            }
        }

        // Start video 2 after a 2-second delay
        if (!started2)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (elapsed_seconds >= 2)
            {
                cap2.open(video_path2);
                if (!cap2.isOpened())
                {
                    std::cerr << "Error opening video: " << video_path2 << std::endl;
                    break;
                }
                started2 = true;
            }
        }

        // Read next frame from video 2
        if (started2 && !end2)
        {
            cap2 >> frame2;
            if (frame2.empty())
            {
                end2 = true;
            }
        }

        // Build batch for available frames
        std::vector<cv::Mat> frames;
        std::vector<std::string> window_names;

        if (!end1)
        {
            frames.push_back(frame1);
            window_names.push_back("YOLOv8 Detection 1");
        }
        else
        {
            cap1.release();
            cv::destroyWindow("YOLOv8 Detection 1");
        }

        if (started2 && !end2)
        {
            frames.push_back(frame2);
            window_names.push_back("YOLOv8 Detection 2");
        }
        else if (started2 && end2)
        {
            cap2.release();
            cv::destroyWindow("YOLOv8 Detection 2");
        }

        if (frames.empty())
        {
            break; // both videos ended
        }

        // Batched inference (your API may differ)
        std::vector<std::vector<Object>> detections_batch = yolo.detectObjects(frames);

        // Draw results and show frames
        for (size_t i = 0; i < frames.size(); ++i)
        {
            yolo.drawObjectLabels(frames[i], detections_batch[i]);
            cv::imshow(window_names[i], frames[i]);
        }

        char key = static_cast<char>(cv::waitKey(1));

        if (key == 27)
        {
            break; // ESC to exit
        }
    }

    cv::destroyAllWindows();
    return 0;
}
