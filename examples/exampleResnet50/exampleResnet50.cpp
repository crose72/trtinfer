#include "tensorrt_infer.h"

// Resize shortest side to 256 (keep aspect), center-crop to HxW, convert to RGB,
// and save as binary PPM (P6).
inline void preprocess_to_ppm(const std::string &in_path,
                              const std::string &out_ppm_path,
                              int target_w, int target_h)
{
    cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR); // BGR uint8
    if (img.empty())
        throw std::runtime_error("Could not read image: " + in_path);

    // BGR -> RGB (PPM expects RGB; your reader assumes RGB)
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 1) Resize shortest side to 256 (keep aspect ratio)
    int new_w, new_h;
    if (img.cols < img.rows)
    {
        new_w = 256;
        new_h = static_cast<int>(std::round(img.rows * (256.0 / img.cols)));
    }
    else
    {
        new_h = 256;
        new_w = static_cast<int>(std::round(img.cols * (256.0 / img.rows)));
    }
    cv::resize(img, img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 2) Center-crop to target_w x target_h
    if (img.cols < target_w || img.rows < target_h)
    {
        throw std::runtime_error("Image too small after resize for requested crop.");
    }
    int x = (img.cols - target_w) / 2;
    int y = (img.rows - target_h) / 2;
    cv::Rect roi(x, y, target_w, target_h);
    img = img(roi).clone(); // now exactly HxW in RGB, uint8

    // 3) Save as PPM (P6). OpenCV picks P6 for 8-bit 3-channel PPM by default.
    if (!cv::imwrite(out_ppm_path, img))
    {
        throw std::runtime_error("Failed to write PPM: " + out_ppm_path);
    }
}

inline std::unique_ptr<float> process_input(const std::string &input_filename, nvinfer1::Dims dims)
{
    // Read image data from file and mean-normalize it
    const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    auto input_image{util::RGBImageReader(input_filename, dims, mean, stddev)};
    input_image.read();
    return input_image.process();
}

void process_output(std::unique_ptr<float> output)
{
    // ---- Top-K on FP32 scores from `output_buffer` ----
    float *scores = output.get(); // raw pointer for lambda
    int32_t mOutputSize = 1 * 3 * 224 * 224;

    std::vector<size_t> idx(mOutputSize);
    std::iota(idx.begin(), idx.end(), size_t{0});
    size_t topk = std::min<size_t>(5, mOutputSize);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [scores](size_t a, size_t b)
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
    int32_t width{224};
    int32_t height{224};
    nvinfer1::Dims input_dims = nvinfer1::Dims4{1, 3, 224, 224};

    TensorInfer sample("/workspace/examples/exampleResnet50/resnet_engine_intro.engine");

    std::string prepped_img = "/workspace/examples/exampleResnet50/elephant-out.ppm";

    preprocess_to_ppm(
        "/workspace/examples/exampleResnet50/elephant.jpg",
        prepped_img,
        width,
        height);

    std::unique_ptr<float> input = process_input(prepped_img, input_dims);

    std::unique_ptr<float> output = sample.infer(input);
    process_output(std::move(output));

    return 0;
}
