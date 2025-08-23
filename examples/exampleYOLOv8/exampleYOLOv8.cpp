#include "engine.h"

int main(void)
{
    // Create engine directly from engine file path
    Engine<float> yoloEngine("/workspace/examples/exampleYOLOv8/yolov8s.engine");
    yoloEngine.printEngineInfo();
    return 0;
}