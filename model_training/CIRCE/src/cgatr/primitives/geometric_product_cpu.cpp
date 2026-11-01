// Inference-only CGA product. Keep sum_j first, then multiply by y_k and sum_k.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

static at::Tensor geometric_product(
    const at::Tensor& x, const at::Tensor& y, const at::Tensor& left,
    const at::Tensor& coefficient, const at::Tensor& count) {
    TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu(), "CPU inputs required");
    TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
                "FP32 inputs required");
    TORCH_CHECK(x.dim() > 0 && x.sizes() == y.sizes() && x.size(-1) == 32,
                "Equal shapes ending in 32 required");
    TORCH_CHECK(left.device().is_cpu() && coefficient.device().is_cpu()
                && count.device().is_cpu(), "CPU tables required");
    TORCH_CHECK(left.scalar_type() == at::kLong && count.scalar_type() == at::kLong
                && coefficient.scalar_type() == at::kFloat, "Invalid table types");
    TORCH_CHECK(left.is_contiguous() && coefficient.is_contiguous()
                && count.is_contiguous(), "Contiguous tables required");
    TORCH_CHECK(left.dim() == 3 && left.size(0) == 32 && left.size(1) == 32
                && left.size(2) > 0 && coefficient.sizes() == left.sizes()
                && count.sizes() == at::IntArrayRef({32, 32}), "Invalid table shapes");
    const auto width = left.size(2);
    const auto* ix = left.const_data_ptr<int64_t>();
    const auto* c = coefficient.const_data_ptr<float>();
    const auto* n = count.const_data_ptr<int64_t>();
    for (int64_t slot = 0; slot < 1024; ++slot) {
        TORCH_CHECK(n[slot] >= 0 && n[slot] <= width, "Invalid term count");
        for (int64_t j = 0; j < n[slot]; ++j) {
            TORCH_CHECK(ix[slot * width + j] >= 0 && ix[slot * width + j] < 32,
                        "Invalid blade index");
        }
    }
    auto a = x.contiguous(), b = y.contiguous(), out = at::empty_like(a);
    const auto* xp = a.const_data_ptr<float>();
    const auto* yp = b.const_data_ptr<float>();
    auto* dest = out.mutable_data_ptr<float>();
    const auto rows = a.numel() / 32;
    at::parallel_for(0, rows, 64, [&](int64_t start, int64_t end) {
        for (int64_t row = start; row < end; ++row) {
            const auto* xx = xp + row * 32;
            const auto* yy = yp + row * 32;
            auto* oo = dest + row * 32;
            for (int64_t i = 0; i < 32; ++i) {
                float result = 0;
                for (int64_t k = 0; k < 32; ++k) {
                    const int64_t slot = i * 32 + k, offset = slot * width;
                    float partial = 0;
                    for (int64_t j = 0; j < n[slot]; ++j)
                        partial += c[offset + j] * xx[ix[offset + j]];
                    result += partial * yy[k];
                }
                oo[i] = result;
            }
        }
    });
    return out;
}

TORCH_LIBRARY(circe_cpu, m) {
    m.def("geometric_product(Tensor x, Tensor y, Tensor left, Tensor coefficient, Tensor count) -> Tensor");
}
TORCH_LIBRARY_IMPL(circe_cpu, CPU, m) {
    m.impl("geometric_product", TORCH_FN(geometric_product));
}
