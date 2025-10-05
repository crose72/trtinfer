#include "YOLOv8.h"

int main(void)
{
    YOLOv8::Config config;
    YOLOv8 yolo("/workspace/examples/models/yolov8s_batch.engine", config);
    yolo.printEngineInfo();

    // Open the video file or camera
    std::string video_path = "/workspace/examples/sampleData/dancing2.mp4"; // <--- YOUR VIDEO PATH
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video: " << video_path << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break; // End of video

        // Detect objects in the current frame
        std::vector<Object> detections = yolo.detectObjects(frame);

        // Draw results
        yolo.drawObjectLabels(frame, detections);

        // Show the frame
        cv::imshow("YOLOv8 Detection", frame);

        // Exit on ESC
        char key = (char)cv::waitKey(1);
        if (key == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}