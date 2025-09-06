#include "ResNet50.h"
#include <numeric>

ResNet50::ResNet50(const std::string &engine_path) : mEnginePath(engine_path) {}

cv::Mat ResNet50::preprocess(const cv::Mat &img_in, int target_w, int target_h)
{
    cv::Mat img = img_in.clone();
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize shortest side to 256
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
    // Center crop
    int x = (img.cols - target_w) / 2;
    int y = (img.rows - target_h) / 2;
    img = img(cv::Rect(x, y, target_w, target_h)).clone();
    return img;
}

void ResNet50::postprocess(const std::vector<float> &scores)
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

std::vector<float> ResNet50::classify(const cv::Mat &img_cpu)
{
    cv::Mat img_prep = preprocess(img_cpu);
    cv::cuda::GpuMat img_gpu;
    img_gpu.upload(img_prep);
    return classify(img_gpu);
}

std::vector<float> ResNet50::classify(const cv::cuda::GpuMat &img_gpu)
{
    std::vector<std::vector<cv::cuda::GpuMat>> inputs = {{img_gpu}};
    std::vector<std::vector<std::vector<float>>> outputs;
    bool ok = mEngine.runInference(inputs, outputs);
    if (!ok)
        throw std::runtime_error("ResNet50 inference failed!");

    postprocess(outputs[0][0]);
    return outputs[0][0]; // batch 0, output 0
}
