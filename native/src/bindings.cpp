#define PY_SSIZE_T_CLEAN
#include <Python.h>

// The public C ABI is also loaded by madmom.native through ctypes. ctypes
// releases the GIL during inference and needs no NumPy C ABI dependency.
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native", "Optional native streaming beat engine.",
    -1, nullptr, nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC PyInit__native() {
    return PyModule_Create(&module);
}
