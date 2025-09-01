#include "TRTEngine.h"

#include <numeric>

cv::Mat preprocess(const std::string &img_path, int target_w = 224, int target_h = 224)
{
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR); // BGR
    if (img.empty())
        throw std::runtime_error("Failed to read image");

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize shortest side to 256 (maintain aspect)
    int new_w, new_h;
    if (img.cols < img.rows)
    {
        new_w = 256;
        new_h = static_cast<int>(img.rows * (256.0 / img.cols));
    }
    else
    {
        new_h = 256;
        new_w = static_cast<int>(img.cols * (256.0 / img.rows));
    }
    cv::resize(img, img, cv::Size(new_w, new_h));
    // Center-crop to 224x224
    int x = (img.cols - target_w) / 2;
    int y = (img.rows - target_h) / 2;
    img = img(cv::Rect(x, y, target_w, target_h)).clone();
    return img;
}

void postprocess(const std::vector<float> &scores)
{
    // Assuming size is 1000 for ImageNet
    int mOutputSize = scores.size();

    std::vector<size_t> idx(mOutputSize);
    std::iota(idx.begin(), idx.end(), size_t{0});
    size_t topk = std::min<size_t>(5, mOutputSize);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [&scores](size_t a, size_t b)
                      { return scores[a] > scores[b]; });

    for (size_t i = 0; i < topk; ++i)
    {
        size_t k = idx[i];
        std::cout << "#" << (i + 1) << ": class " << k
                  << " score " << scores[k] << "\n";
    }
}

int main(int argc, char **argv)
{
    TRTEngine<float> engine;
    engine.loadNetwork("/workspace/examples/exampleResnet50_v2/resnet_engine_intro.engine",
                       {0.485f, 0.456f, 0.406f}, // mean
                       {0.229f, 0.224f, 0.225f}, // stddev
                       true);                    // normalize to [0,1] before mean/std
    engine.printEngineInfo();
    cv::Mat img_cpu = preprocess("/workspace/examples/exampleResnet50_v2/elephant.jpg");
    cv::cuda::GpuMat img_gpu;
    img_gpu.upload(img_cpu);

    // Single input tensor, single batch
    std::vector<std::vector<cv::cuda::GpuMat>> engine_inputs = {{img_gpu}};

    std::vector<std::vector<std::vector<float>>> outputs;
    bool ok = engine.runInference(engine_inputs, outputs);

    postprocess(outputs[0][0]);

    if (!ok)
    {
        std::cerr << "Inference failed!" << std::endl;
        return 1;
    }

    return 0;
}
