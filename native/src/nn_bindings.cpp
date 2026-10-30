/* Python buffer adapters for the shared neural kernels.
 * The madmom project code licence applies. No NumPy C ABI is required.
 */
#define PY_SSIZE_T_CLEAN
#include "madmom_nn_bindings.h"
#include "madmom_nn.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
const char *capsule_name = "madmom.neural.v1";

struct Object {
    PyObject *value = nullptr;
    Object() = default;
    explicit Object(PyObject *object, bool borrowed = false) : value(object) {
        if (borrowed) Py_XINCREF(value);
    }
    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&& other) noexcept : value(other.value) { other.value = nullptr; }
    Object& operator=(Object&& other) noexcept {
        if (this != &other) { Py_XDECREF(value); value = other.value; other.value = nullptr; }
        return *this;
    }
    ~Object() { Py_XDECREF(value); }
};

struct Buffer {
    Py_buffer value = {};
    bool active = false;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : value(other.value), active(other.active) {
        other.active = false;
    }
    ~Buffer() { if (active) PyBuffer_Release(&value); }
    bool get(PyObject *object, bool contiguous = false, bool writable = false) {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (contiguous) flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
        if (writable) flags |= PyBUF_WRITABLE;
        if (PyObject_GetBuffer(object, &value, flags) < 0) return false;
        active = true;
        return value.itemsize == sizeof(float) && value.format &&
               (std::strcmp(value.format, "f") == 0 ||
                std::strcmp(value.format, "=f") == 0 ||
                std::strcmp(value.format, "@f") == 0) && value.ndim > 0;
    }
};

PyObject *weak_object(PyObject *weak) {
    // All calls hold the GIL. The active processing chain keeps layers alive.
#if PY_VERSION_HEX >= 0x030D0000
    PyObject *object = nullptr;
    if (PyWeakref_GetRef(weak, &object) < 0) return nullptr;
    if (!object) return Py_None;
    Py_DECREF(object);
    return object;
#else
    return PyWeakref_GetObject(weak);
#endif
}

Object attr(PyObject *owner, const char *name) {
    Object result(PyObject_GetAttrString(owner, name));
    if (!result.value) throw std::runtime_error("Invalid neural descriptor");
    return result;
}

long number(PyObject *owner, const char *name) {
    Object value = attr(owner, name);
    const long result = PyLong_AsLong(value.value);
    if (result == -1 && PyErr_Occurred())
        throw std::runtime_error("Invalid neural descriptor dimension");
    return result;
}

Object sequence(PyObject *object) {
    Object result(PySequence_Fast(object, "Neural descriptor must be a sequence"));
    if (!result.value) throw std::runtime_error("Invalid neural descriptor sequence");
    return result;
}

struct ArrayCheck {
    Object owner, name, original, contiguous;
    std::vector<Py_ssize_t> shape, strides;
    const float **target = nullptr;
};
struct FunctionCheck { Object owner, function; };
struct GateCheck { Object owner, name, expected; };
struct State {
    Object owner, name, work;
    float **target = nullptr;
    size_t size = 0;
};

struct Model {
    std::vector<Object> layers;
    std::vector<PyTypeObject*> types;
    std::vector<mm_nn_layer> descriptors;
    std::vector<ArrayCheck> arrays;
    std::vector<FunctionCheck> functions;
    std::vector<GateCheck> gates;
    std::vector<State> states;
    Object array_type;

    Model(PyObject *source, PyObject *type) : array_type(type, true) {
        if (!PyType_Check(type)) throw std::runtime_error("Invalid neural array type");
        Object refs = attr(source, "_layers"), descriptor_source = attr(source, "descriptors");
        Object layer_refs = sequence(refs.value), source_layers = sequence(descriptor_source.value);
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(layer_refs.value);
        if (count < 1 || count != PySequence_Fast_GET_SIZE(source_layers.value))
            throw std::runtime_error("Invalid neural layer count");
        descriptors.resize(static_cast<size_t>(count));
        Object array_source = attr(source, "arrays"), state_source = attr(source, "states");
        Object source_arrays = sequence(array_source.value), source_states = sequence(state_source.value);
        Py_ssize_t array_index = 0, state_index = 0;
        size_t previous = 0;
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject *weak = PySequence_Fast_GET_ITEM(layer_refs.value, i);
            if (!PyWeakref_CheckRef(weak)) throw std::runtime_error("Invalid neural layer reference");
            PyObject *owner = weak_object(weak);
            if (!owner || owner == Py_None) throw std::runtime_error("Expired neural layer");
            layers.emplace_back(weak, true);
            types.push_back(Py_TYPE(owner));
            // A layer remains alive through the active chain, not this cache.
            PyObject *item = PySequence_Fast_GET_ITEM(source_layers.value, i);
            mm_nn_layer &layer = descriptors[i];
            layer.kind = static_cast<int>(number(item, "kind"));
            layer.activation = static_cast<int>(number(item, "activation"));
            const long inputs = number(item, "inputs"), outputs = number(item, "outputs");
            if (inputs < 1 || outputs < 1 || layer.kind < 0 || layer.kind > 2 ||
                layer.activation < 0 || layer.activation > 5 ||
                (previous && static_cast<size_t>(inputs) != previous))
                throw std::runtime_error("Invalid neural layer dimensions");
            layer.inputs = static_cast<size_t>(inputs);
            previous = layer.outputs = static_cast<size_t>(outputs);
            Object gate_source = attr(item, "gates");
            Object source_gates = sequence(gate_source.value);
            if (PySequence_Fast_GET_SIZE(source_gates.value) != 4)
                throw std::runtime_error("Invalid neural gate count");
            const int gate_count = layer.kind == 2 ? 4 : 1;
            for (int j = 0; j < gate_count; ++j) {
                mm_nn_gate &gate = layer.gates[j];
                gate.activation = static_cast<int>(number(
                    PySequence_Fast_GET_ITEM(source_gates.value, j), "activation"));
                add_array(source_arrays.value, array_index, {inputs, outputs}, &gate.weights);
                add_array(source_arrays.value, array_index, {outputs}, &gate.bias);
                if (layer.kind) add_array(source_arrays.value, array_index,
                                          {outputs, outputs}, &gate.recurrent);
                if (layer.kind == 2) add_array(source_arrays.value, array_index,
                                               {outputs}, &gate.peepholes, true);
            }
            if (layer.kind) add_state(source_states.value, state_index, layer.outputs, &layer.hidden);
            if (layer.kind == 2) {
                add_state(source_states.value, state_index, layer.outputs, &layer.cell);
                static const char *names[] = {"input_gate", "forget_gate", "cell", "output_gate"};
                for (const char *name : names) {
                    Object current = attr(owner, name);
                    Object expected(PyWeakref_NewRef(current.value, nullptr));
                    if (!expected.value) throw std::runtime_error("Invalid neural gate reference");
                    gates.push_back({Object(weak, true), Object(PyUnicode_InternFromString(name)),
                                     std::move(expected)});
                }
            }
        }
        if (array_index != PySequence_Fast_GET_SIZE(source_arrays.value) ||
            state_index != PySequence_Fast_GET_SIZE(source_states.value))
            throw std::runtime_error("Invalid neural descriptor buffer count");
        Object function_source = attr(source, "functions");
        Object source_functions = sequence(function_source.value);
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(source_functions.value); ++i) {
            Object check = sequence(PySequence_Fast_GET_ITEM(source_functions.value, i));
            if (PySequence_Fast_GET_SIZE(check.value) != 2 ||
                !PyWeakref_CheckRef(PySequence_Fast_GET_ITEM(check.value, 0)))
                throw std::runtime_error("Invalid neural function descriptor");
            functions.push_back({Object(PySequence_Fast_GET_ITEM(check.value, 0), true),
                                 Object(PySequence_Fast_GET_ITEM(check.value, 1), true)});
        }
    }

    void add_array(PyObject *source, Py_ssize_t &index,
                   std::vector<Py_ssize_t> shape, const float **target,
                   bool optional = false) {
        if (index >= PySequence_Fast_GET_SIZE(source))
            throw std::runtime_error("Missing neural array descriptor");
        Object item = sequence(PySequence_Fast_GET_ITEM(source, index++));
        if (PySequence_Fast_GET_SIZE(item.value) != 6)
            throw std::runtime_error("Invalid neural array descriptor");
        PyObject *weak = PySequence_Fast_GET_ITEM(item.value, 0),
                 *name = PySequence_Fast_GET_ITEM(item.value, 1),
                 *original = PySequence_Fast_GET_ITEM(item.value, 2),
                 *contiguous = PySequence_Fast_GET_ITEM(item.value, 5);
        if (!PyWeakref_CheckRef(weak) || !PyUnicode_Check(name) ||
            (original == Py_None && !optional))
            throw std::runtime_error("Invalid neural array reference");
        ArrayCheck check;
        check.owner = Object(weak, true);
        check.name = Object(name, true);
        check.original = Object(original, true);
        check.contiguous = Object(contiguous, true);
        check.shape = std::move(shape);
        check.target = target;
        if (original != Py_None) {
            Buffer view;
            if (!view.get(original) || !matches(view.value, check.shape))
                throw std::runtime_error("Invalid neural array shape or dtype");
            check.strides.assign(view.value.strides, view.value.strides + view.value.ndim);
            Buffer output;
            if (!output.get(contiguous, true) || !matches(output.value, check.shape))
                throw std::runtime_error("Invalid contiguous neural array");
        }
        arrays.push_back(std::move(check));
    }

    void add_state(PyObject *source, Py_ssize_t &index, size_t size, float **target) {
        if (index >= PySequence_Fast_GET_SIZE(source))
            throw std::runtime_error("Missing neural state descriptor");
        Object item = sequence(PySequence_Fast_GET_ITEM(source, index++));
        if (PySequence_Fast_GET_SIZE(item.value) != 3 ||
            !PyWeakref_CheckRef(PySequence_Fast_GET_ITEM(item.value, 0)) ||
            !PyUnicode_Check(PySequence_Fast_GET_ITEM(item.value, 1)))
            throw std::runtime_error("Invalid neural state descriptor");
        states.push_back({Object(PySequence_Fast_GET_ITEM(item.value, 0), true),
                          Object(PySequence_Fast_GET_ITEM(item.value, 1), true),
                          Object(PySequence_Fast_GET_ITEM(item.value, 2), true), target, size});
    }

    static bool matches(const Py_buffer &view, const std::vector<Py_ssize_t> &shape) {
        return view.ndim == static_cast<int>(shape.size()) &&
               std::equal(shape.begin(), shape.end(), view.shape);
    }
};

struct Work {
    Model &model;
    std::vector<Buffer> buffers;
    std::vector<Object> owners;
    Buffer result;
    explicit Work(Model &source) : model(source) {
        buffers.reserve(source.arrays.size() * 2 + source.states.size() * 2);
        owners.reserve(source.layers.size());
    }

    bool check(PyObject *layers) {
        Object current = sequence(layers);
        if (PySequence_Fast_GET_SIZE(current.value) != static_cast<Py_ssize_t>(model.layers.size()))
            return false;
        for (size_t i = 0; i < model.layers.size(); ++i) {
            PyObject *owner = weak_object(model.layers[i].value);
            if (!owner || owner != PySequence_Fast_GET_ITEM(current.value, i) ||
                Py_TYPE(owner) != model.types[i]) return false;
            Object dictionary = attr(owner, "__dict__");
            if (PyDict_GetItemString(dictionary.value, "activate") ||
                PyDict_GetItemString(dictionary.value, "reset")) return false;
            owners.emplace_back(owner, true);
        }
        for (const auto &check : model.functions) {
            PyObject *owner = weak_object(check.owner.value);
            if (!owner || owner == Py_None) return false;
            Object value(PyObject_GetAttrString(owner, "activation_fn"));
            if (!value.value) { PyErr_Clear(); return false; }
            if (value.value != check.function.value) return false;
        }
        for (const auto &check : model.gates) {
            PyObject *owner = weak_object(check.owner.value), *expected = weak_object(check.expected.value);
            if (!owner || owner == Py_None || !expected || expected == Py_None) return false;
            Object value(PyObject_GetAttr(owner, check.name.value));
            if (!value.value) { PyErr_Clear(); return false; }
            if (value.value != expected) return false;
        }
        for (const auto &check : model.arrays) {
            PyObject *owner = weak_object(check.owner.value);
            if (!owner || owner == Py_None) return false;
            Object value(PyObject_GetAttr(owner, check.name.value));
            if (!value.value) { PyErr_Clear(); return false; }
            if (value.value != check.original.value) return false;
            if (value.value == Py_None) { *check.target = nullptr; continue; }
            buffers.emplace_back();
            Buffer &source = buffers.back();
            if (!source.get(value.value) || !Model::matches(source.value, check.shape) ||
                !std::equal(check.strides.begin(), check.strides.end(), source.value.strides)) {
                PyErr_Clear(); return false;
            }
            if (check.original.value == check.contiguous.value) {
                if (reinterpret_cast<uintptr_t>(source.value.buf) % alignof(float)) return false;
                *check.target = static_cast<const float*>(source.value.buf);
            } else {
                buffers.emplace_back();
                Buffer &destination = buffers.back();
                if (!destination.get(check.contiguous.value, true, true) ||
                    !Model::matches(destination.value, check.shape)) {
                    PyErr_Clear(); return false;
                }
                if (PyBuffer_ToContiguous(destination.value.buf, &source.value,
                                          destination.value.len, 'C') < 0) return false;
                *check.target = static_cast<const float*>(destination.value.buf);
            }
        }
        return true;
    }

    bool prepare(PyObject *output, size_t frames, size_t inputs, bool reset) {
        if (model.descriptors.front().inputs != inputs) return false;
        if (!result.get(output, true, true) || result.value.ndim != 2 ||
            result.value.shape[0] != static_cast<Py_ssize_t>(frames) ||
            result.value.shape[1] != static_cast<Py_ssize_t>(model.descriptors.back().outputs) ||
            reinterpret_cast<uintptr_t>(result.value.buf) % alignof(float)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "Invalid neural output buffer");
            return false;
        }
        if (reset) {
            for (const auto &owner : owners) {
                Object value(PyObject_CallMethod(owner.value, "reset", nullptr));
                if (!value.value) return false;
            }
        }
        for (const auto &state : model.states) {
            PyObject *owner = weak_object(state.owner.value);
            if (!owner || owner == Py_None) return false;
            Object value(PyObject_GetAttr(owner, state.name.value));
            if (!value.value) return false;
            if (!PyObject_TypeCheck(value.value, reinterpret_cast<PyTypeObject*>(model.array_type.value)))
                return false;
            buffers.emplace_back();
            Buffer &source = buffers.back();
            if (!source.get(value.value) || source.value.ndim != 1 ||
                source.value.shape[0] != static_cast<Py_ssize_t>(state.size)) {
                PyErr_Clear(); return false;
            }
            buffers.emplace_back();
            Buffer &destination = buffers.back();
            if (!destination.get(state.work.value, true, true) ||
                destination.value.ndim != 1 || destination.value.shape[0] !=
                    static_cast<Py_ssize_t>(state.size) ||
                reinterpret_cast<uintptr_t>(destination.value.buf) % alignof(float)) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "Invalid neural state buffer");
                return false;
            }
            if (PyBuffer_ToContiguous(destination.value.buf, &source.value,
                                      destination.value.len, 'C') < 0) return false;
            *state.target = static_cast<float*>(destination.value.buf);
        }
        return true;
    }

    int run(const float *input, size_t frames, size_t inputs) {
        return mm_nn_network(input, frames, inputs, model.descriptors.data(),
                             model.descriptors.size(), static_cast<float*>(result.value.buf));
    }

    bool publish(size_t frames) {
        if (!frames) return true;
        for (const auto &state : model.states) {
            PyObject *owner = weak_object(state.owner.value);
            if (!owner || owner == Py_None) return false;
            Object copy(PyObject_CallMethod(state.work.value, "copy", nullptr));
            if (!copy.value || PyObject_SetAttr(owner, state.name.value, copy.value) < 0) return false;
        }
        return true;
    }
};

Model *model(PyObject *capsule) {
    return static_cast<Model*>(PyCapsule_GetPointer(capsule, capsule_name));
}
void destroy_model(PyObject *capsule) { delete model(capsule); }

PyObject *configure(PyObject*, PyObject *args) {
    PyObject *source, *array_type;
    if (!PyArg_ParseTuple(args, "OO", &source, &array_type)) return nullptr;
    try {
        std::unique_ptr<Model> value(new Model(source, array_type));
        PyObject *result = PyCapsule_New(value.get(), capsule_name, destroy_model);
        if (result) value.release();
        return result;
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
      catch (const std::exception &error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject *process_many(PyObject*, PyObject *args) {
    PyObject *capsules, *layers, *input, *outputs;
    int reset;
    if (!PyArg_ParseTuple(args, "OOOOp", &capsules, &layers, &input, &outputs, &reset)) return nullptr;
    try {
        Object models = sequence(capsules), chains = sequence(layers), results = sequence(outputs);
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(models.value);
        if (count < 1 || PySequence_Fast_GET_SIZE(chains.value) != count ||
            PySequence_Fast_GET_SIZE(results.value) != count) {
            PyErr_SetString(PyExc_ValueError, "Neural ensemble buffer counts differ"); return nullptr;
        }
        Buffer data;
        if (!data.get(input, true) || data.value.ndim != 2 || data.value.shape[1] < 1 ||
            reinterpret_cast<uintptr_t>(data.value.buf) % alignof(float)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "Invalid neural input buffer");
            return nullptr;
        }
        std::vector<std::unique_ptr<Work>> work;
        work.reserve(static_cast<size_t>(count));
        for (Py_ssize_t i = 0; i < count; ++i) {
            Model *state = model(PySequence_Fast_GET_ITEM(models.value, i));
            if (!state) return nullptr;
            work.emplace_back(new Work(*state));
            if (!work.back()->check(PySequence_Fast_GET_ITEM(chains.value, i))) {
                if (PyErr_Occurred()) return nullptr;
                return PyLong_FromSsize_t(i);
            }
        }
        // Validate every model before reset or arithmetic, so stale descriptors
        // can be rebuilt without consuming a frame in any ensemble member.
        for (Py_ssize_t i = 0; i < count; ++i) {
            if (!work[i]->prepare(PySequence_Fast_GET_ITEM(results.value, i),
                                  data.value.shape[0], data.value.shape[1], reset != 0)) {
                if (PyErr_Occurred()) return nullptr;
                return PyLong_FromSsize_t(i);
            }
        }
        int status = 0;
        Py_BEGIN_ALLOW_THREADS
        for (auto &item : work) {
            status = item->run(static_cast<const float*>(data.value.buf),
                               data.value.shape[0], data.value.shape[1]);
            if (status) break;
        }
        Py_END_ALLOW_THREADS
        if (status) { PyErr_SetString(PyExc_RuntimeError, "Native neural kernel failed"); return nullptr; }
        for (auto &item : work) if (!item->publish(data.value.shape[0])) return nullptr;
        return PyLong_FromLong(-1);
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
      catch (const std::exception &error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"mm_nn_configure", configure, METH_VARARGS,
     "Create a checked neural descriptor cache without copying model weights."},
    {"mm_nn_process_many", process_many, METH_VARARGS,
     "Validate buffers, run shared neural kernels, and publish independent states."},
    {nullptr, nullptr, 0, nullptr}
};
}

int mm_nn_register(PyObject *module) { return PyModule_AddFunctions(module, methods); }
