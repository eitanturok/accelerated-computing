// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector"}>

#include <cmath>
#include <cstdint>
#include <immintrin.h> //AVX2 header
#include <assert.h> // for assert statements
#include <iostream> // for printing

// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    int VECTOR_SIZE = 16; // todo: remove
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            // Get the plane coordinate X for the image pixel.
            float cx = (float(j) / float(img_size)) * 2.5f - 2.0f;
            float cy = (float(i) / float(img_size)) * 2.5f - 1.25f;
            std::cout << "i=" <<i <<"\t\tj=" << j << "\n" << "cx=" << cx << "\t\tcy=" << cy << "\n"; // todo: remove

            // Innermost loop: start the recursion from z = 0.
            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - x2 - y2 + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
                // todo: remove
                std::cout << "x=" << x << "\t\ty=" << y << "\n";
                std::cout << "x2=" << x2 << "\t\ty2=" << y2 << "\n";
                std::cout << "z=" << z << "\t\tw=" << w << "\n";
            }
            std::cout << "iters=" << iters << "\n";

            // if (j >= VECTOR_SIZE-1) { // todo: remove
            //     break;
            // }

            // Write result.
            out[i * img_size + j] = iters;

            std::cout << "\n\n";
        }
        if (i >= 0) { // todo: remove
            break;
        }
    }
}

/// <--- your code here --->
void print_m512(__m512 v) {
    float arr[16];
    _mm512_storeu_ps(arr, v);
    for (int i = 0; i < 16; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void print_m512i(__m512i v) {
    int32_t arr[16];
    _mm512_storeu_si512((__m512i*)arr, v);
    for (int i = 0; i < 16; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

__mmask16 is_less_than_four(__m512 x2_vec, __m512 y2_vec, __mmask16 mask, __m512 zeros, __m512 fours) {
    __m512 x_plus_y = _mm512_mask_add_ps(zeros, mask, x2_vec, y2_vec);
    __mmask16 less_than_four = _mm512_mask_cmple_ps_mask(mask, x_plus_y, fours);
    return less_than_four;
}

void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    // check image fits
    assert (img_size % 16 == 0);
    int VECTOR_SIZE = 16;

    // init values for outer loop
    // for some reason you must input the arguments to set_ps backwards
    __m512 range = _mm512_set_ps(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m512 img_size_vec = _mm512_set1_ps(img_size);
    __m512 scale = _mm512_set1_ps(2.5);
    __m512 x_shift = _mm512_set1_ps(2.0);
    __m512 y_shift = _mm512_set1_ps(1.25);

    __m512 i_vec;
    __m512 j_vec;
    __m512 cx_vec;
    __m512 cy_vec;

    // init values for inner loop
    __m512 zeros = _mm512_set1_ps(0.0);
    __m512 fours = _mm512_set1_ps(4.0);
    __m512i ones = _mm512_set1_epi32(1);
    __m512i max_iters_vec = _mm512_set1_epi32(max_iters);

    bool do_continue;
    __m512 x_vec;
    __m512 y_vec;
    __m512 x2_vec;
    __m512 y2_vec;
    __m512 w_vec;
    __m512 z_vec;
    __m512i iters_vec;

    __mmask16 mask = 0b1111111111111111; // initially everything is True

    // outer loop iterates over one 16-wide row of pixels at a time
    for (uint64_t i = 0; i < img_size; i+=1) {
        for (uint64_t j = 0; j < img_size; j+=VECTOR_SIZE) {
            std::cout << "i=" <<i <<"j=" << j << "\n";
            i_vec = _mm512_set1_ps(i);
            j_vec = _mm512_add_ps(_mm512_set1_ps(j), range);
            std::cout << "i_vec=";
            print_m512(i_vec);
            std::cout << "j_vec=";
            print_m512(j_vec);

            // get coordinate plane
            // cx contain different values; cy repeats the same value
            cx_vec = _mm512_sub_ps(_mm512_mul_ps(_mm512_div_ps(j_vec, img_size_vec), scale), x_shift);
            cy_vec = _mm512_sub_ps(_mm512_mul_ps(_mm512_div_ps(i_vec, img_size_vec), scale), y_shift);

            std::cout << "cx_vec=";
            print_m512(cx_vec);
            std::cout << "cy_vec=";
            print_m512(cy_vec);
            std::cout << "\n";

            // set values for inner loop
            x2_vec = _mm512_set1_ps(0.0);
            y2_vec = _mm512_set1_ps(0.0);
            w_vec = _mm512_set1_ps(0.0);
            z_vec = _mm512_set1_ps(0.0);
            zeros = _mm512_set1_ps(0.0);
            iters_vec = _mm512_set1_epi32(0);
            mask = 0b1111111111111111; // initially everything is True

            // while loop condition
            do_continue = (is_less_than_four(x2_vec, y2_vec, mask, zeros, fours) & _mm512_mask_cmplt_epi32_mask(mask, iters_vec, max_iters_vec)) != 0;
            std::cout << "do_continue=" << do_continue << "\n";

            while (do_continue) {
                // compute x_vec, y_vec
                x_vec = _mm512_mask_sub_ps(zeros, mask, x2_vec, y2_vec);
                x_vec = _mm512_mask_add_ps(zeros, mask, x_vec, cx_vec);

                y_vec = _mm512_mask_sub_ps(zeros, mask, w_vec, x2_vec);
                y_vec = _mm512_mask_sub_ps(zeros, mask, y_vec, y2_vec);
                y_vec = _mm512_mask_add_ps(zeros, mask, y_vec, cy_vec);

                std::cout << "x_vec=";
                print_m512(x_vec);
                std::cout << "y_vec=";
                print_m512(y_vec);

                // compute x2_vec, y2_vec
                x2_vec = _mm512_mask_mul_ps(zeros, mask, x_vec, x_vec);
                y2_vec = _mm512_mask_mul_ps(zeros, mask, y_vec, y_vec);

                std::cout << "x2_vec=";
                print_m512(x2_vec);
                std::cout << "y2_vec=";
                print_m512(y2_vec);

                // compute z_vec, w_vec
                z_vec = _mm512_mask_add_ps(zeros, mask, x_vec, y_vec);
                w_vec = _mm512_mask_mul_ps(zeros, mask, z_vec, z_vec);

                std::cout << "z_vec=";
                print_m512(z_vec);
                std::cout << "w_vec=";
                print_m512(w_vec);

                // update iters, mask, do_continue
                iters_vec = _mm512_mask_add_epi32(iters_vec, mask, iters_vec, ones);
                std::cout << "iters_vec=";
                print_m512i(iters_vec);
                std::cout << "mask=" << mask << "\n";

                mask = is_less_than_four(x2_vec, y2_vec, mask, zeros, fours);
                do_continue = (mask & _mm512_mask_cmplt_epi32_mask(mask, iters_vec, max_iters_vec)) != 0;
                std::cout << "new_mask=" << mask << "\t\tdo_continue=" << do_continue << "\n";
            }
            // break;
            // _mm512_storeu_si512(&out)
            std::cout << "\n\n";
        }
        break;
    }
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl { SCALAR, VECTOR, ALL };

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Testing with image size " << *img_size << "x" << *img_size << " and "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter));
                auto intensity = static_cast<uint8_t>(
                    log_iter * 222 / log2f(static_cast<float>(max_iters)));
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
// static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfOuterIterations = 1; // todo: remove
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::cout << "  Runtime: " << times[0] / 1'000'000 << " ms" << std::endl; \
    } while (0)

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 256;
    // uint32_t max_iters = 1000;
    uint32_t max_iters = 2; // todo: remove
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
