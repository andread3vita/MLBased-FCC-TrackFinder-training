// Minimal integration example: one event of native-endian float32 [N,10].
// Output is float32 [N,embed_dim+1], ending in the raw beta logit.
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::runtime_error("Usage: circe_cpu model.pt event.f32 prediction.f32 [threads=4]");
        int threads = argc == 5 ? std::stoi(argv[4]) : 4;
        if (threads < 1) throw std::runtime_error("threads must be positive");
        at::set_num_threads(threads);
        at::set_num_interop_threads(1);
        c10::InferenceMode inference;
        std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot open input event");
        const auto bytes = input.tellg();
        if (bytes <= 0 || bytes % (10 * sizeof(float)))
            throw std::runtime_error("Expected a nonempty float32 [N,10] event");
        std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(data.data()), bytes);
        if (!input) throw std::runtime_error("Cannot read input event");
        auto features = torch::from_blob(data.data(), {static_cast<int64_t>(data.size() / 10), 10}, torch::kFloat32);
        auto model = torch::jit::load(argv[1], torch::kCPU);
        model.eval();
        auto prediction = model.forward({features}).toTensor().contiguous();
        TORCH_CHECK(prediction.device().is_cpu() && prediction.scalar_type() == at::kFloat
                    && prediction.dim() == 2 && prediction.size(0) == features.size(0), "Invalid model output");
        std::ofstream output(argv[3], std::ios::binary);
        output.write(reinterpret_cast<const char*>(prediction.const_data_ptr<float>()), prediction.numel() * sizeof(float));
        if (!output) throw std::runtime_error("Cannot write predictions");
        std::cout << "Wrote " << prediction.size(0) << " x " << prediction.size(1)
                  << " FP32 predictions (last column: raw beta logit)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
