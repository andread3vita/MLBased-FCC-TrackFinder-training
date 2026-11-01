// Inference-only CGA product. Keep sum_j first, then multiply by y_k and sum_k.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

// Keep the dispatcher and fallback generic. Only this function may emit AVX-512.
#if (defined(__x86_64__) || defined(__i386__)) \
    && (defined(__GNUC__) || defined(__clang__)) \
    && !defined(_MSC_VER) && !defined(CIRCE_DISABLE_AVX512)
#include <immintrin.h>
#define CIRCE_HAS_AVX512_KERNEL 1

__attribute__((target("avx512f"), noinline))
static void avx512_rows(const float* x, const float* y, float* out,
                       int64_t start, int64_t end,
                       const int32_t index[32][32], const float coefficient[32][32]) {
    for (int64_t row = start; row < end; ++row) {
        const float* xx = x + row * 32;
        const float* yy = y + row * 32;
        const __m512 zero = _mm512_setzero_ps();
        const __m512 x0 = _mm512_loadu_ps(xx), x1 = _mm512_loadu_ps(xx + 16);
        __m512 r0 = zero, r1 = zero;
        for (int64_t k = 0; k < 32; ++k) {
            const __m512 right = _mm512_set1_ps(yy[k]);
            const __m512 a0 = _mm512_permutex2var_ps(x0, _mm512_load_si512(index[k]), x1);
            const __m512 a1 = _mm512_permutex2var_ps(x0, _mm512_load_si512(index[k] + 16), x1);
            const __m512 p0 = _mm512_add_ps(zero, _mm512_mul_ps(_mm512_load_ps(coefficient[k]), a0));
            const __m512 p1 = _mm512_add_ps(zero, _mm512_mul_ps(_mm512_load_ps(coefficient[k] + 16), a1));
            r0 = _mm512_add_ps(r0, _mm512_mul_ps(p0, right));
            r1 = _mm512_add_ps(r1, _mm512_mul_ps(p1, right));
        }
        _mm512_storeu_ps(out + row * 32, r0);
        _mm512_storeu_ps(out + row * 32 + 16, r1);
    }
}

static bool use_avx512() {
    // GCC/Clang's runtime feature detection also checks OS vector-state support.
    static const bool available = __builtin_cpu_supports("avx512f");
    return available;
}
#endif

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
    bool single_term = true;
    for (int64_t slot = 0; slot < 1024; ++slot) {
        TORCH_CHECK(n[slot] >= 0 && n[slot] <= width, "Invalid term count");
        single_term = single_term && n[slot] == 1;
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
#ifdef CIRCE_HAS_AVX512_KERNEL
    const bool vectorized = single_term && use_avx512();
    alignas(64) int32_t index[32][32];
    alignas(64) float packed_coefficient[32][32];
    if (vectorized) {
        for (int64_t k = 0; k < 32; ++k) for (int64_t i = 0; i < 32; ++i) {
            const auto offset = (i * 32 + k) * width;
            index[k][i] = static_cast<int32_t>(ix[offset]);
            packed_coefficient[k][i] = c[offset];
        }
    }
#endif
    at::parallel_for(0, rows, 64, [&](int64_t start, int64_t end) {
#ifdef CIRCE_HAS_AVX512_KERNEL
        if (vectorized) {
            avx512_rows(xp, yp, dest, start, end, index, packed_coefficient);
            return;
        }
#endif
        for (int64_t row = start; row < end; ++row) {
            const auto* xx = xp + row * 32;
            const auto* yy = yp + row * 32;
            auto* oo = dest + row * 32;
            if (single_term) {
                for (int64_t i = 0; i < 32; ++i) {
                    float result = 0;
                    for (int64_t k = 0; k < 32; ++k) {
                        const auto offset = (i * 32 + k) * width;
                        float partial = 0;
                        partial += c[offset] * xx[ix[offset]];
                        result += partial * yy[k];
                    }
                    oo[i] = result;
                }
                continue;
            }
            // Retain the general contraction for tables with zero/multiple terms.
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
