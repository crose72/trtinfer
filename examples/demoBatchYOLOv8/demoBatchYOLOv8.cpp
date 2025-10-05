#include "YOLOv8.h"
#include <chrono>
#include <thread>
#include <vector>
#include <string>

cv::VideoWriter writer;
bool writer_open = false;
const int PANEL_W = 640;
const int PANEL_H = 480;
const int FPS = 30;

// Demo timing (seconds)
const float T_START_2 = 2.0; // show 2nd stream at 2s
const float T_START_3 = 4.0; // show 3rd stream at 4s
const float T_END = 6.0;     // hard stop at 8s total

// Optional: taper off in the last 2s (3->2 at 7s, 2->1 at 8s)
const bool TAPER_DROPOFF = true;
const float T_DROP_3 = 7.0; // hide 3rd at 7s
const float T_DROP_2 = 8.0; // hide 2nd at 8s (loop ends)

int main()
{
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/models/yolov8s_seg_batch_fp32.engine", config);
    yolo.printEngineInfo();

    std::string video_path1 = "/workspace/examples/sampleData/dancing1.mp4";
    std::string video_path2 = "/workspace/examples/sampleData/dancing2.mp4";
    std::string video_path3 = "/workspace/examples/sampleData/soccer.mp4";

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

    cv::Mat frame1;
    cv::Mat frame2;
    cv::Mat frame3;

    // One elapsed timer drives the whole demo
    auto now = std::chrono::steady_clock::now();
    float elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;

    while (true)
    {
        // One elapsed timer drives the whole demo
        auto now = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;

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

        // Start video 2 at T_START_2
        if (!started2 && elapsed_seconds >= T_START_2)
        {
            cap2.open(video_path2);
            if (!cap2.isOpened())
            {
                std::cerr << "Error opening video: " << video_path2 << std::endl;
                end2 = true;
            }
            else
                started2 = true;
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

        // Start video 3 at T_START_3
        if (!started3 && elapsed_seconds >= T_START_3)
        {
            cap3.open(video_path3);
            if (!cap3.isOpened())
            {
                std::cerr << "Error opening video: " << video_path3 << std::endl;
                end3 = true;
            }
            else
                started3 = true;
        }

        // Read next frame from video 3
        if (started3 && !end3)
        {
            cap3 >> frame3;
            if (frame3.empty())
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
            // cv::imshow(window_names[i], frames[i]);
        }

        // ---- Side-by-side display (no padding) + combined video (no padding) ----
        const int DISPLAY_H = 480; // target panel height for display and video
        const int BASE_X = 200;    // where to place the first window on screen
        const int BASE_Y = 200;
        const int GAP_X = 0; // pixels between windows

        std::vector<cv::Mat> resized_for_concat;
        int curX = 0; // x position for next window

        for (size_t i = 0; i < frames.size(); ++i)
        {
            cv::Mat &f = frames[i];
            if (f.empty())
                continue;

            // Scale by height, preserve aspect (no letterbox)
            float scale = float(DISPLAY_H) / std::max(1, f.rows);
            int outW = std::max(1, int(std::round(f.cols * scale)));
            cv::Mat out;
            cv::resize(f, out, cv::Size(outW, DISPLAY_H),
                       0, 0, (scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR));

            // Show in its own window and place it right next to the previous one
            const std::string win = window_names[i]; // you already build this
            cv::namedWindow(win, cv::WINDOW_NORMAL);
            cv::imshow(win, out);
            cv::resizeWindow(win, outW, DISPLAY_H);
            cv::moveWindow(win, BASE_X + curX, BASE_Y);

            // Accumulate for combined video (no black bars)
            resized_for_concat.push_back(out);

            // Advance x for next window
            curX += outW + GAP_X;
        }

        // Build combined frame (no padding) and write it
        cv::Mat combined;
        if (!resized_for_concat.empty())
        {
            cv::hconcat(resized_for_concat, combined);

            if (!writer_open)
            {
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                writer.open("demo_batch.mp4", fourcc, FPS, combined.size());
                writer_open = writer.isOpened();
            }
            if (writer_open)
                writer.write(combined);

            // Optional: preview the combined strip in one window for OBS
            // cv::imshow("Combined Demo", combined);
        }

        // Optionally, show the combined window for live preview/OBS
        // cv::imshow("Combined Demo", combined);
        // cv::waitKey(0); // waits indefinitely for any key

        if (elapsed_seconds >= T_END)
        {
            // Write one last frame already done above; then exit cleanly
            break;
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
