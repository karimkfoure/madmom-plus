#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "madmom_native.h"
#include "madmom_decoders.h"
#include "madmom_nn_bindings.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
const char* capsule_name = "madmom.frontend.v1";
const size_t frame_size = 2048;
const size_t bands = 81;

struct Buffer {
    Py_buffer value = {};
    bool active = false;
    ~Buffer() { if (active) PyBuffer_Release(&value); }

    bool get(PyObject* object, char format, bool writable = false) {
        const int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT |
                          (writable ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(object, &value, flags) < 0) return false;
        active = true;
        const size_t item_size = format == 'd' ? sizeof(double) :
                                 format == 'B' ? sizeof(uint8_t) : sizeof(float);
        if (!value.format || value.format[0] != format || value.format[1] ||
            value.itemsize != static_cast<Py_ssize_t>(item_size)) {
            PyErr_SetString(PyExc_ValueError, "Native frontend buffer dtype does not match");
            return false;
        }
        if (reinterpret_cast<uintptr_t>(value.buf) % item_size) {
            PyErr_SetString(PyExc_ValueError, "Native frontend buffer must be aligned");
            return false;
        }
        return true;
    }
};

struct Frontend {
    std::vector<double> window;
    std::vector<float> bank, history;
    size_t lag;
    void* handle = nullptr;

    Frontend(const double* fft_window, const float* filters, size_t difference_lag)
        : window(fft_window, fft_window + frame_size),
          bank(filters, filters + 1024 * bands),
          history(difference_lag * bands, std::numeric_limits<float>::infinity()),
          lag(difference_lag) {
        handle = mm_frontend_create(window.data(), bank.data(), lag);
        if (!handle) throw std::runtime_error(mm_online_error());
    }
    ~Frontend() { mm_frontend_destroy(handle); }

    bool matches(const double* fft_window, const float* filters, size_t difference_lag) {
        return lag == difference_lag &&
            std::memcmp(window.data(), fft_window, window.size() * sizeof(double)) == 0 &&
            std::memcmp(bank.data(), filters, bank.size() * sizeof(float)) == 0;
    }

    void remember(const float* output, size_t count) {
        if (!count) return;
        if (count < lag) {
            std::move(history.begin() + count * bands, history.end(), history.begin());
        }
        const size_t retained = std::min(count, lag);
        for (size_t row = 0; row < retained; ++row) {
            std::copy(output + (count - retained + row) * 2 * bands,
                      output + (count - retained + row) * 2 * bands + bands,
                      history.begin() + (lag - retained + row) * bands);
        }
    }
};

Frontend* frontend(PyObject* capsule) {
    return static_cast<Frontend*>(PyCapsule_GetPointer(capsule, capsule_name));
}

void destroy_frontend(PyObject* capsule) {
    delete frontend(capsule);
}

PyObject* configure_frontend(PyObject*, PyObject* args) {
    PyObject *capsule, *window, *bank;
    Py_ssize_t lag;
    if (!PyArg_ParseTuple(args, "OOOn", &capsule, &window, &bank, &lag)) return nullptr;
    if (lag < 1 || lag > 100000) {
        PyErr_SetString(PyExc_ValueError, "Invalid spectral difference lag");
        return nullptr;
    }
    Buffer fft_window, filters;
    if (!fft_window.get(window, 'd') || !filters.get(bank, 'f')) return nullptr;
    if (fft_window.value.len != static_cast<Py_ssize_t>(frame_size * sizeof(double)) ||
        filters.value.len != static_cast<Py_ssize_t>(1024 * bands * sizeof(float))) {
        PyErr_SetString(PyExc_ValueError, "Invalid native frontend configuration shape");
        return nullptr;
    }
    const double* window_values = static_cast<const double*>(fft_window.value.buf);
    const float* bank_values = static_cast<const float*>(filters.value.buf);
    if (capsule != Py_None) {
        Frontend* previous = frontend(capsule);
        if (!previous) return nullptr;
        if (previous->matches(window_values, bank_values, lag)) {
            Py_INCREF(capsule);
            return capsule;
        }
    }
    try {
        std::unique_ptr<Frontend> replacement(new Frontend(window_values, bank_values, lag));
        PyObject* result = PyCapsule_New(replacement.get(), capsule_name, destroy_frontend);
        if (result) replacement.release();
        return result;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* process_frontend(PyObject*, PyObject* args) {
    PyObject *capsule, *frames, *output, *history = Py_None;
    Py_ssize_t count;
    int reset = 0;
    if (!PyArg_ParseTuple(args, "OOnO|Op", &capsule, &frames, &count,
                          &output, &history, &reset)) return nullptr;
    Frontend* state = frontend(capsule);
    if (!state) return nullptr;
    Buffer input, result, prior;
    if (!input.get(frames, 'f') || !result.get(output, 'f', true)) return nullptr;
    if (count < 0 || static_cast<size_t>(count) >
            static_cast<size_t>(input.value.len) / (frame_size * sizeof(float)) ||
        static_cast<size_t>(count) >
            static_cast<size_t>(result.value.len) / (2 * bands * sizeof(float))) {
        PyErr_SetString(PyExc_ValueError, "Native frontend frame or output buffer is too small");
        return nullptr;
    }
    if (reset) {
        if (mm_frontend_reset(state->handle) < 0) {
            PyErr_SetString(PyExc_ValueError, mm_online_error());
            return nullptr;
        }
        std::fill(state->history.begin(), state->history.end(),
                  std::numeric_limits<float>::infinity());
    } else if (history != Py_None) {
        if (!prior.get(history, 'f')) return nullptr;
        const size_t row_bytes = bands * sizeof(float);
        if (prior.value.len < 0 || static_cast<size_t>(prior.value.len) % row_bytes ||
            static_cast<size_t>(prior.value.len) / row_bytes > state->lag) {
            PyErr_SetString(PyExc_ValueError, "Invalid native frontend history shape");
            return nullptr;
        }
        const size_t rows = static_cast<size_t>(prior.value.len) / row_bytes;
        const float* values = static_cast<const float*>(prior.value.buf);
        if (rows != state->lag ||
            std::memcmp(state->history.data() + (state->lag - rows) * bands,
                        values, rows * row_bytes) != 0) {
            if (mm_frontend_history(state->handle, values, rows) < 0) {
                PyErr_SetString(PyExc_ValueError, mm_online_error());
                return nullptr;
            }
            std::fill(state->history.begin(), state->history.end(),
                      std::numeric_limits<float>::infinity());
            std::copy(values, values + rows * bands,
                      state->history.begin() + (state->lag - rows) * bands);
        }
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = mm_frontend_process(state->handle,
        static_cast<const float*>(input.value.buf), count,
        static_cast<float*>(result.value.buf));
    Py_END_ALLOW_THREADS
    if (status < 0) {
        PyErr_SetString(PyExc_ValueError, mm_online_error());
        return nullptr;
    }
    state->remember(static_cast<const float*>(result.value.buf), count);
    Py_RETURN_NONE;
}

PyObject* beat_events(PyObject*, PyObject* args) {
    PyObject *flags, *output;
    long long first_frame;
    double fps, max_bpm, last_beat, tempo;
    if (!PyArg_ParseTuple(args, "OLddddO", &flags, &first_frame,
                          &fps, &max_bpm, &last_beat, &tempo, &output)) return nullptr;
    Buffer selected, events;
    if (!selected.get(flags, 'B') || !events.get(output, 'd', true)) return nullptr;
    if (selected.value.ndim != 1 || events.value.ndim != 1 ||
        selected.value.len > events.value.len / static_cast<Py_ssize_t>(sizeof(double))) {
        PyErr_SetString(PyExc_ValueError, "Invalid beat event buffer dimensions");
        return nullptr;
    }
    ptrdiff_t count;
    Py_BEGIN_ALLOW_THREADS
    count = mm_beat_events(static_cast<const uint8_t*>(selected.value.buf),
        selected.value.len, first_frame, fps, max_bpm, &last_beat, &tempo,
        static_cast<double*>(events.value.buf));
    Py_END_ALLOW_THREADS
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid beat event dimensions or timing parameters");
        return nullptr;
    }
    return Py_BuildValue("(ndd)", static_cast<Py_ssize_t>(count), last_beat, tempo);
}

PyMethodDef methods[] = {
    {"mm_beat_events", beat_events, METH_VARARGS,
     "Apply the shared beat-event rule to borrowed flag and output buffers."},
    {"mm_frontend_configure", configure_frontend, METH_VARARGS,
     "Configure an internal shared frontend capsule from contiguous buffers."},
    {"mm_frontend_process", process_frontend, METH_VARARGS,
     "Process internal frontend buffers with optional history synchronization."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native", "Native music-analysis kernels and beat engine.",
    -1, methods, nullptr, nullptr, nullptr, nullptr
};
} // namespace

PyMODINIT_FUNC PyInit__native() {
    PyObject* result = PyModule_Create(&module);
    if (result && mm_nn_register(result) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
