#ifndef MADMOM_NN_BINDINGS_H
#define MADMOM_NN_BINDINGS_H
#include <Python.h>
/* Register Python buffer adapters. Inference stays in madmom_nn.h kernels. */
int mm_nn_register(PyObject *module);
#endif
