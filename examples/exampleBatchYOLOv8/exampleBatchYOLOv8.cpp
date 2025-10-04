#include "YOLOv8.h"
#include <chrono>
#include <thread>
#include <vector>
#include <string>

int main()
{
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/models/yolov8s_batch.engine", config);
    yolo.printEngineInfo();

    std::string video_path1 = "/workspace/examples/media/dancing1.mp4";
    std::string video_path2 = "/workspace/examples/media/dancing2.mp4";
    std::string video_path3 = "/workspace/examples/media/soccer.mp4";

    cv::VideoCapture cap1(video_path1);
    cv::VideoCapture cap2; // open after delay
    cv::VideoCapture cap3; // open after longer delay

    if (!cap1.isOpened())
    {
        std::cerr << "Error opening video: " << video_path1 << std::endl;
        return -1;
    }

    bool started2 = false;
    bool started3 = false;
    bool end1 = false;
    bool end2 = false;
    bool end3 = false;

    auto start_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point vid3_start_time;
    bool vid3_timer_started = false;

    cv::Mat frame1;
    cv::Mat frame2;
    cv::Mat frame3;

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

            if (frame1.empty())
            {
                end1 = true;
                cv::destroyWindow("YOLOv8 Detection 1");
            }
        }

        // Start video 2 after 2-second delay
        if (!started2)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;
            if (elapsed_seconds >= 2.0)
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

            if (frame2.empty())
            {
                end2 = true;
                cv::destroyWindow("YOLOv8 Detection 2");
            }
        }

        // Start video 3 after 3.5s delay, stop after 4s
        if (!started3)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;
            if (elapsed_seconds >= 3.5)
            {
                cap3.open(video_path3);
                if (!cap3.isOpened())
                {
                    std::cerr << "Error opening video: " << video_path3 << std::endl;
                    end3 = true;
                }
                else
                {
                    started3 = true;
                    vid3_start_time = std::chrono::steady_clock::now();
                    vid3_timer_started = true;
                }
            }
        }

        // Read next frame from video 3 and close after 4s of playback
        if (started3 && !end3)
        {
            cap3 >> frame3;
            bool expired = false;
            if (vid3_timer_started)
            {
                auto now = std::chrono::steady_clock::now();
                double vid3_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - vid3_start_time).count() / 1000.0;
                if (vid3_elapsed >= 4.0)
                {
                    expired = true;
                }
            }

            if (frame3.empty() || expired)
            {
                end3 = true;
                cv::destroyWindow("YOLOv8 Detection 3");
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
        if (started2 && !end2)
        {
            frames.push_back(frame2);
            window_names.push_back("YOLOv8 Detection 2");
        }
        if (started3 && !end3)
        {
            frames.push_back(frame3);
            window_names.push_back("YOLOv8 Detection 3");
        }

        if (frames.empty())
        {
            break; // all videos ended or closed
        }

        // Batched inference
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

    cap1.release();
    if (cap2.isOpened())
    {
        cap2.release();
    }
    if (cap3.isOpened())
    {
        cap3.release();
    }
    cv::destroyAllWindows();
    return 0;
}
