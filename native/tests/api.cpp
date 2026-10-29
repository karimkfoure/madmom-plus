/* C ABI validation tests. Run through CTest with an exported model asset. */
#include "madmom_native.h"
#include "madmom_engine.h"

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

void test_frontend(const char* asset) {
    std::ifstream stream(asset, std::ios::binary);
    stream.seekg(8);
    std::vector<double> window(2048);
    std::vector<float> bank(1024 * 81);
    uint32_t lag = 0;
    stream.read(reinterpret_cast<char*>(window.data()), window.size() * sizeof(double));
    stream.read(reinterpret_cast<char*>(bank.data()), bank.size() * sizeof(float));
    stream.read(reinterpret_cast<char*>(&lag), sizeof(lag));
    require(stream.good(), "Cannot read test frontend asset");
    void* frontend = mm_frontend_create(window.data(), bank.data(), lag);
    require(frontend != NULL, mm_online_error());
    Handle tracker(asset);
    const size_t rows = 13;
    std::vector<float> input(rows * 2048), output(rows * 162), expected(rows * 162);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(std::sin(i * .013) * .3);
    }
    for (size_t i = 0; i < rows; ++i) {
        require(mm_online_features(tracker.value, input.data() + 2048 * i,
                                    expected.data() + 162 * i) == 0,
                "Reference frontend call failed");
    }
    require(mm_frontend_process(frontend, input.data(), rows, output.data()) == 0,
            "Batched frontend failed");
    require(output == expected, "Batch and streaming frontend kernels differ");
    require(mm_frontend_reset(frontend) == 0, "Frontend reset failed");
    require(mm_frontend_process(frontend, NULL, 0, NULL) == 0,
            "Empty frontend input must not require buffers");
    require(mm_frontend_process(frontend, NULL, 1, output.data()) == -1,
            "Null frontend input must fail");
    require(mm_frontend_process(frontend, input.data(), rows, output.data()) == 0,
            "Reset frontend processing failed");
    require(output == expected, "Frontend reset must restore difference history");
    std::vector<float> seeded(lag * 81);
    for (size_t i = 0; i < lag; ++i) {
        std::copy(expected.begin() + i * 162, expected.begin() + i * 162 + 81,
                  seeded.begin() + i * 81);
    }
    require(mm_frontend_history(frontend, seeded.data(), lag) == 0,
            "Frontend history seeding failed");
    require(mm_frontend_process(frontend, input.data() + lag * 2048, rows - lag,
                                output.data() + lag * 162) == 0,
            "Seeded frontend continuation failed");
    require(output == expected, "Seeded frontend history changed continuation");
    std::fill(seeded.begin(), seeded.end(), std::numeric_limits<float>::infinity());
    require(mm_frontend_history(frontend, seeded.data(), lag) == 0,
            "Frontend missing-history padding must be accepted");
    require(mm_frontend_process(frontend, input.data(), rows, output.data()) == 0,
            "Padded-history frontend processing failed");
    require(output == expected, "Missing-history padding must retain initial zero differences");
    require(mm_frontend_history(frontend, NULL, 0) == 0,
            "Empty history must reset the frontend difference state");
    require(mm_frontend_history(frontend, seeded.data(), lag + 1) == -1,
            "History beyond the configured lag must fail");
    seeded[0] = std::numeric_limits<float>::quiet_NaN();
    require(mm_frontend_history(frontend, seeded.data(), lag) == -1,
            "Invalid frontend history must fail");
    require(mm_frontend_process(frontend, input.data(), rows, output.data()) == 0,
            "Frontend processing after rejected history failed");
    require(output == expected, "Rejected history must preserve prior difference state");
    mm_frontend_destroy(frontend);
    require(mm_frontend_create(NULL, bank.data(), lag) == NULL,
            "Missing frontend window must fail");
    require(mm_frontend_create(window.data(), NULL, lag) == NULL,
            "Missing frontend bank must fail");
    require(mm_frontend_create(window.data(), bank.data(), 0) == NULL,
            "Zero frontend difference lag must fail");
    require(mm_frontend_reset(NULL) == -1, "Null frontend handle must fail");
    mm_frontend_destroy(NULL);
}

void test_engine(const char* asset) {
    require(mm_engine_version() == MM_ENGINE_ABI_V1, "Unexpected engine ABI version");
    mm_engine_config_v1 config = {
        sizeof(mm_engine_config_v1), MM_ENGINE_ABI_V1, 8, 0, 0, 1, {0, 0}};
    mm_engine* engine = mm_engine_create(asset, &config);
    require(engine != NULL, mm_engine_error());
    mm_engine_state_v1 status;
    require(mm_engine_state(engine, &status) == 0, "Initial engine state failed");
    require(status.sample_index == MM_ENGINE_NO_SAMPLE &&
            status.last_beat_sample == MM_ENGINE_NO_SAMPLE &&
            status.received_samples == 0 && status.pending_frames == 0,
            "Initial engine clocks must be empty");
    require(status.reliability > 0 && std::isfinite(status.reliability),
            "Initial engine reliability must be a finite raw probability");
    std::vector<float> audio(5000);
    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] = static_cast<float>(std::sin(i * .07) * .4);
    }
    size_t accepted = 123;
    require(mm_engine_push(engine, audio.data(), audio.size(), &accepted) == MM_ENGINE_FULL &&
            accepted == 0, "A full output queue must not consume input");
    require(mm_engine_state(engine, &status) == 0 && status.received_samples == 0,
            "Queue overflow must preserve engine state");
    require(mm_engine_push(engine, audio.data(), 1, &accepted) == 0 && accepted == 1,
            "Engine first sample failed");
    require(mm_engine_read(engine, NULL, 0) == 1,
            "Engine queue query must report pending frames");
    require(mm_engine_push(engine, audio.data() + 1, 440, &accepted) == 0 && accepted == 440,
            "Input without a new frame must fit a full queue");
    require(mm_engine_push(engine, audio.data() + 441, 1, &accepted) == MM_ENGINE_FULL &&
            accepted == 0, "Full queue must reject the next output frame");
    mm_engine_frame_v1 first;
    require(mm_engine_read(engine, &first, 1) == 1 && first.sample_index == 0,
            "The first engine frame must have sample index zero");
    require(mm_engine_push(engine, audio.data() + 441, 1, &accepted) == 0 && accepted == 1,
            "Input must be retryable after reading queued frames");
    mm_engine_frame_v1 second;
    require(mm_engine_read(engine, &second, 1) == 1 && second.sample_index == 441,
            "Second engine frame must use the integer sample clock");
    require(mm_engine_read(engine, NULL, 0) == 0, "Read must drain the output queue");
    require(mm_engine_state(engine, &status) == 0 && status.sample_index == 441 &&
            status.received_samples == 442, "Engine state clock mismatch");
    mm_engine_config_v1 invalid = config;
    invalid.abi_version += 1;
    require(mm_engine_configure(engine, &invalid) == -1,
            "Unsupported engine ABI must fail");
    require(mm_engine_state(engine, &status) == 0 && status.received_samples == 442,
            "Failed engine configuration must preserve state");
    config.queue_capacity = 64;
    require(mm_engine_configure(engine, &config) == 0, "Engine configuration failed");
    require(mm_engine_read(engine, NULL, 0) == 0, "Configure must clear the output queue");
    require(mm_engine_push(engine, audio.data(), audio.size(), &accepted) == 0 &&
            accepted == audio.size(), "Engine batch input failed");
    mm_engine_frame_v1 rows[64];
    const int count = mm_engine_read(engine, rows, 64);
    Handle reference(asset);
    double expected[64 * 4];
    require(mm_online_push(reference.value, audio.data(), audio.size(), expected, 64) == count,
            "Legacy and versioned engine frame counts differ");
    for (int i = 0; i < count; ++i) {
        require(rows[i].sample_index == static_cast<uint64_t>(i) * 441 &&
                rows[i].activation == expected[4 * i + 1] &&
                rows[i].beat == expected[4 * i + 2],
                "Legacy and versioned engine outputs differ");
    }
    require(mm_engine_reset(engine) == 0, "Engine reset failed");
    require(mm_engine_state(engine, &status) == 0 && status.received_samples == 0 &&
            status.sample_index == MM_ENGINE_NO_SAMPLE && status.pending_frames == 0,
            "Engine reset must clear sample clocks and output queue");
    require(mm_engine_push(engine, NULL, 0, &accepted) == 0 && accepted == 0,
            "Empty engine push failed");
    require(mm_engine_push(engine, NULL, 1, &accepted) == -1 && accepted == 0,
            "Missing engine audio must fail");
    require(mm_engine_push(engine, audio.data(), 1, NULL) == -1,
            "Missing accepted output must fail");
    require(mm_engine_read(engine, NULL, 1) == -1, "Missing frame output must fail");
    require(mm_engine_state(engine, NULL) == -1, "Missing state output must fail");
    require(mm_engine_configure(engine, NULL) == 0, "Default engine configuration failed");
    mm_engine_destroy(engine);
    mm_engine_destroy(NULL);
    require(mm_engine_create(NULL, NULL) == NULL, "Missing engine model must fail");
    require(mm_engine_reset(NULL) == -1, "Null engine handle must fail");
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Pass an exported native model asset");
        const char* asset = argv[1];
        test_frontend(asset);
        test_engine(asset);
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
