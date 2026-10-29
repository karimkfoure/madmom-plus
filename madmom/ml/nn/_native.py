"""Call native float32 kernels while retaining Python layer and model state.

The descriptor cache contains pointers, not a second copy of the model API.
It detects replaced model arrays. Writes to existing arrays remain visible.
Unsupported dtypes and activation functions use the corresponding Python
implementation because those operations do not have native kernels.
"""

import ctypes as ct
from functools import lru_cache
from weakref import WeakKeyDictionary, ref

import numpy as np

from . import activations


_FP = ct.POINTER(ct.c_float)
_FLOAT32 = np.dtype(np.float32)
_ACTIVATIONS = {None: 0, activations.linear: 0, activations.tanh: 1,
                np.tanh: 1, activations.sigmoid: 2, activations.relu: 3,
                activations.elu: 4, activations.softmax: 5}


class _Gate(ct.Structure):
    _fields_ = [('weights', _FP), ('bias', _FP), ('recurrent', _FP),
                ('peepholes', _FP), ('activation', ct.c_int)]


class _Layer(ct.Structure):
    _fields_ = [('kind', ct.c_int), ('inputs', ct.c_size_t),
                ('outputs', ct.c_size_t), ('activation', ct.c_int),
                ('gates', _Gate * 4), ('hidden', _FP), ('cell', _FP)]


def _pointer(value):
    return value.ctypes.data_as(_FP)


def _activation(value):
    try:
        return _ACTIVATIONS.get(value, -1)
    except TypeError:
        return -1


@lru_cache(maxsize=1)
def _library():
    from ... import _native

    lib = ct.CDLL(_native.__file__)
    lib.mm_nn_network.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t,
                                 ct.POINTER(_Layer), ct.c_size_t, ct.c_void_p]
    lib.mm_nn_network.restype = ct.c_int
    return lib


class _Unsupported(Exception):
    pass


class _Model:
    def __init__(self, layers):
        from .layers import FeedForwardLayer, LSTMLayer, RecurrentLayer

        self._layers = tuple(ref(layer) for layer in layers)
        self.arrays = []
        self.copies = []
        self.functions = []
        self.states = []
        self._gates = []
        self.descriptors = (_Layer * len(layers))()
        previous = None
        for index, layer in enumerate(layers):
            descriptor = self.descriptors[index]
            if type(layer) is LSTMLayer:
                descriptor.kind = 2
                gates = (layer.input_gate, layer.forget_gate,
                         layer.cell, layer.output_gate)
                shape = layer.cell.weights.shape
            elif type(layer) in (FeedForwardLayer, RecurrentLayer):
                descriptor.kind = 1 if type(layer) is RecurrentLayer else 0
                gates = (layer,)
                shape = layer.weights.shape
            else:
                raise _Unsupported
            self._gates.append(tuple(ref(gate) for gate in gates))
            if len(shape) != 2 or not all(shape):
                raise _Unsupported
            descriptor.inputs, descriptor.outputs = shape
            if previous is not None and previous != descriptor.inputs:
                raise _Unsupported
            previous = descriptor.outputs
            try:
                descriptor.activation = _ACTIVATIONS[layer.activation_fn]
            except (KeyError, TypeError):
                raise _Unsupported
            if descriptor.kind and descriptor.activation == 5:
                raise _Unsupported
            if descriptor.kind == 2 and layer.activation_fn is None:
                raise _Unsupported
            self.functions.append((ref(layer), layer.activation_fn))
            for gate_index, gate in enumerate(gates):
                target = descriptor.gates[gate_index]
                target.weights = self.array(gate, 'weights', shape)
                target.bias = self.array(gate, 'bias', (shape[1],))
                if descriptor.kind:
                    target.recurrent = self.array(
                        gate, 'recurrent_weights', (shape[1], shape[1]))
                if descriptor.kind == 2:
                    target.peepholes = self.array(
                        gate, 'peephole_weights', (shape[1],), optional=True)
                    try:
                        target.activation = _ACTIVATIONS[gate.activation_fn]
                    except (KeyError, TypeError):
                        raise _Unsupported
                    self.functions.append((ref(gate), gate.activation_fn))
                    # Gate.activate applies its function to a vector. Softmax
                    # requires a matrix, so its existing error must remain.
                    if target.activation == 5 or gate.activation_fn is None:
                        raise _Unsupported
            if descriptor.kind:
                self.state(layer, '_prev', 'hidden', descriptor, shape[1])
            if descriptor.kind == 2:
                self.state(layer, '_state', 'cell', descriptor, shape[1])

    def array(self, owner, name, shape, optional=False):
        original = getattr(owner, name, None)
        if original is None and optional:
            self.arrays.append((ref(owner), name, None, None, None, None))
            return _FP()
        if (not isinstance(original, np.ndarray) or original.dtype != np.float32
                or original.shape != shape):
            raise _Unsupported
        contiguous = np.ascontiguousarray(original)
        if contiguous is not original:
            self.copies.append((original, contiguous))
        self.arrays.append((ref(owner), name, original, original.shape,
                            original.strides, contiguous))
        return _pointer(contiguous)

    def state(self, owner, name, target, descriptor, size):
        original = getattr(owner, name)
        if (not isinstance(original, np.ndarray) or original.dtype != np.float32
                or original.shape != (size,)):
            raise _Unsupported
        value = np.array(original, copy=True, order='C')
        self.states.append((ref(owner), name, value))
        setattr(descriptor, target, _pointer(value))

    def valid(self, layers):
        if tuple(layers) != tuple(layer() for layer in self._layers):
            return False
        for owner, name, original, shape, strides, _ in self.arrays:
            value = getattr(owner(), name, None)
            if value is not original:
                return False
            if value is not None and (value.shape != shape or
                                      value.strides != strides or
                                      value.dtype != _FLOAT32):
                return False
        for owner, function in self.functions:
            if owner().activation_fn is not function:
                return False
        for owner, descriptor, original_gates in zip(
                self._layers, self.descriptors, self._gates):
            layer = owner()
            if descriptor.kind == 2:
                gates = (layer.input_gate, layer.forget_gate,
                         layer.cell, layer.output_gate)
                if gates != tuple(gate() for gate in original_gates):
                    return False
        return True

    def run(self, data, reset):
        if not self.descriptors or data.shape[1] != self.descriptors[0].inputs:
            raise _Unsupported
        if reset:
            for layer in self._layers:
                layer().reset()
        # Copies are needed only for strided model arrays or explicit resets.
        for original, contiguous in self.copies:
            np.copyto(contiguous, original)
        for owner, name, value in self.states:
            source = getattr(owner(), name)
            if (not isinstance(source, np.ndarray) or source.dtype != _FLOAT32
                    or source.shape != value.shape):
                raise _Unsupported
            if source is not value:
                np.copyto(value, source)
        result = np.empty((len(data), self.descriptors[-1].outputs), np.float32)
        if len(data):
            data = np.ascontiguousarray(data)
            status = _library().mm_nn_network(
                data.ctypes.data, len(data), data.shape[1],
                self.descriptors, len(self.descriptors), result.ctypes.data)
            if status:
                raise RuntimeError('Native neural kernel failed')
            for owner, name, value in self.states:
                # Previous state references must stay unchanged after a later
                # call, as they do with the Python layer implementation.
                setattr(owner(), name, value.copy())
        return result


_MODELS = WeakKeyDictionary()


def network(owner, layers, data, reset):
    """Return native output, or None when this model has no matching kernel."""
    if not isinstance(data, np.ndarray) or data.dtype != _FLOAT32:
        return None
    data = np.atleast_2d(data)
    if data.ndim != 2:
        return None
    try:
        model = _MODELS.get(owner)
        if model is None or not model.valid(layers):
            model = _Model(layers)
            _MODELS[owner] = model
        return model.run(data, reset)
    except _Unsupported:
        return None


def layer(owner, data, reset=False):
    """Process one layer through the same kernel used by whole networks."""
    if (isinstance(data, np.ndarray) and data.ndim == 1 and
            _activation(owner.activation_fn) == 5):
        return None
    result = network(owner, (owner,), data, reset)
    if result is not None and isinstance(data, np.ndarray) and data.ndim == 1:
        result = result[0]
    return result
