/* C ABI validation tests. Run through CTest with an exported model asset. */
#include "madmom_native.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Handle {
    void* value;
    explicit Handle(const char* asset) : value(mm_online_create(asset, 8, 0, 0)) {
        require(value != NULL, mm_online_error());
    }
    ~Handle() { mm_online_destroy(value); }
};

std::vector<double> state(void* handle) {
    const int count = mm_online_decoder_state(handle, NULL, 0);
    require(count > 0, "DBN state length must be positive");
    std::vector<double> result(count);
    require(mm_online_decoder_state(handle, result.data(), result.size()) == count,
            "DBN state copy failed");
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Pass an exported native model asset");
        const char* asset = argv[1];
        Handle tracker(asset);
        const std::vector<double> initial = state(tracker.value);
        const float sample = 0;
        double output[8];
        std::fill(output, output + 8, -123.);

        require(mm_online_push(tracker.value, NULL, 0, NULL, 0) == 0,
                "Empty push must not require buffers");
        require(mm_online_push(tracker.value, &sample, 1, output, 0) == -1,
                "Output capacity must be checked");
        require(state(tracker.value) == initial,
                "Invalid capacity must preserve DBN state");
        require(output[0] == -123., "Invalid capacity must preserve output");
        require(mm_online_push(tracker.value, &sample, 44101, output, 2) == -1,
                "PCM count limit must be checked before reading input");
        require(mm_online_push(tracker.value, NULL, 1, output, 2) == -1,
                "Missing PCM must be rejected");
        require(mm_online_push(tracker.value, &sample, 1, NULL, 2) == -1,
                "Missing output must be rejected");

        std::vector<float> invalid(441, 0);
        invalid.back() = std::numeric_limits<float>::quiet_NaN();
        require(mm_online_push(tracker.value, invalid.data(), invalid.size(),
                               output, 2) == -1,
                "Non-finite PCM must be rejected");
        require(state(tracker.value) == initial,
                "Invalid PCM must preserve DBN state");
        require(mm_online_push(tracker.value, &sample, 1, output, 1) == 1,
                "Exact single-row output capacity must work");
        Handle reference(asset);
        double expected[4];
        require(mm_online_push(reference.value, &sample, 1, expected, 1) == 1,
                "Reference first frame failed");
        require(std::equal(output, output + 4, expected),
                "Failed validation must preserve frontend and network state");

        require(mm_online_reset(tracker.value) == 0, "Reset failed");
        require(state(tracker.value) == initial, "Reset must restore initial DBN state");
        require(mm_online_decoder_state(tracker.value, output, 1) == -1,
                "DBN state capacity must be checked");
        require(mm_online_summary(tracker.value, NULL) == -1,
                "Missing summary output must be rejected");
        require(mm_online_decode(tracker.value, -.1f, output) == -1,
                "Negative activation must be rejected");
        require(mm_online_decode(tracker.value, 1.1f, output) == -1,
                "Activation above one must be rejected");
        require(mm_online_decode(tracker.value,
                                 std::numeric_limits<float>::infinity(), output) == -1,
                "Non-finite activation must be rejected");
        require(state(tracker.value) == initial,
                "Invalid activation must preserve DBN state");

        // Repeated exact zero observations eventually exclude every path.
        // Keep the last valid state and clock rather than divide by zero.
        bool impossible_observation = false;
        for (int frame = 0; frame < 1000; ++frame) {
            const std::vector<double> before = state(tracker.value);
            if (mm_online_decode(tracker.value, 0, output) == -1) {
                impossible_observation = true;
                require(state(tracker.value) == before,
                        "Impossible observation must preserve the posterior");
                require(mm_online_decode(tracker.value, .5f, output) == 0,
                        "A possible observation must allow recovery");
                require(output[0] == frame / 100.,
                        "Impossible observation must preserve the frame clock");
                break;
            }
            double total = 0;
            const std::vector<double> current = state(tracker.value);
            for (size_t i = 0; i < current.size(); ++i) {
                require(std::isfinite(current[i]), "DBN posterior must stay finite");
                total += current[i];
            }
            require(std::abs(total - 1) < 1e-12,
                    "DBN posterior must stay normalized");
        }
        require(impossible_observation,
                "All-zero observations must expose an impossible DBN path");

        require(mm_online_reset(NULL) == -1, "Null handle must be rejected");
        require(mm_online_create(NULL, 8, 0, 0) == NULL,
                "Missing asset path must be rejected");
        require(mm_online_create(asset, 0, 0, 0) == NULL,
                "Zero ensemble members must be rejected");
        require(mm_online_create(asset, 9, 0, 0) == NULL,
                "Ensemble size above eight must be rejected");
        require(mm_online_create(asset, 8, 3, 0) == NULL,
                "Unknown weight mode must be rejected");
        require(mm_online_create(asset, 8, 0, 2) == NULL,
                "Unknown SIMD option must be rejected");

        const char* truncated = "native-api-truncated-model.bin";
        {
            std::ofstream stream(truncated, std::ios::binary);
            stream.write("MMONL001", 8);
        }
        require(mm_online_create(truncated, 8, 0, 0) == NULL,
                "Truncated model must be rejected");
        std::remove(truncated);
        mm_online_destroy(NULL);
        std::cout << "Native C ABI validation passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
