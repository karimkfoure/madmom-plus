"""Verify native neural kernels through the unchanged public layer API."""

import gc
import pickle
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from madmom.ml.nn import NeuralNetwork
from madmom.ml.nn import _native
from madmom.ml.nn.activations import linear, relu, sigmoid, softmax, tanh
from madmom.ml.nn.layers import (Cell, FeedForwardLayer, Gate, LSTMLayer,
                                 RecurrentLayer)


def _lstm(rng, inputs=5, outputs=7, peepholes=True):
    def weights(shape):
        return (rng.standard_normal(shape) * 0.15).astype(np.float32)

    def gate():
        return Gate(weights((inputs, outputs)), weights((outputs,)),
                    weights((outputs, outputs)),
                    weights((outputs,)) if peepholes else None)

    return LSTMLayer(gate(), gate(),
                     Cell(weights((inputs, outputs)), weights((outputs,)),
                          weights((outputs, outputs))), gate())


def _python(network, data, reset=True):
    # Disable dispatch, so this uses the retained upstream layer arithmetic.
    with patch.object(_native, 'network', return_value=None):
        return network(data, reset=reset)


@pytest.mark.parametrize('activation', [None, linear, tanh, sigmoid, relu, softmax])
def test_dense_native_parity_and_mutation(activation):
    rng = np.random.default_rng(5)
    layer = FeedForwardLayer(rng.normal(size=(5, 7)).astype(np.float32),
                             rng.normal(size=7).astype(np.float32), activation)
    network = NeuralNetwork([layer])
    data = rng.normal(size=(13, 5)).astype(np.float32)
    expected = _python(network, data)
    assert network in _native._MODELS or _native.network(network, [layer], data, True) is not None
    np.testing.assert_allclose(network(data), expected, atol=2e-6, rtol=2e-5)
    layer.weights[0] *= 2
    np.testing.assert_allclose(network(data), _python(network, data),
                               atol=2e-6, rtol=2e-5)
    layer.weights = layer.weights.copy(order='F')
    network(data)
    layer.weights[1] *= 2
    np.testing.assert_allclose(network(data), _python(network, data),
                               atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('peepholes', [False, True])
def test_lstm_network_native_parity_stream_reset_and_pickle(peepholes):
    rng = np.random.default_rng(17)
    layer = _lstm(rng, peepholes=peepholes)
    network = NeuralNetwork([layer, FeedForwardLayer(
        rng.normal(size=(7, 2)).astype(np.float32), np.zeros(2, np.float32), sigmoid)])
    data = rng.normal(size=(31, 5)).astype(np.float32)
    expected = _python(network, data)
    assert _native.network(network, network.layers, data, True) is not None
    np.testing.assert_allclose(network(data), expected, atol=2e-6, rtol=2e-5)
    states = (layer._prev.copy(), layer._state.copy())
    network.reset()
    streamed = np.vstack([network(frame, reset=False) for frame in data])
    np.testing.assert_allclose(streamed, expected, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(layer._prev, states[0])
    np.testing.assert_array_equal(layer._state, states[1])
    np.testing.assert_array_equal(layer.init, np.zeros(7, np.float32))
    np.testing.assert_array_equal(layer.cell_init, np.zeros(7, np.float32))
    restored = pickle.loads(pickle.dumps(network))
    np.testing.assert_array_equal(restored(data), network(data))
    assert network(data[:0]).shape == (0, 2)


def test_lstm_gate_replacement_and_strided_model_mutation():
    rng = np.random.default_rng(9)
    layer = _lstm(rng)
    network = NeuralNetwork([layer])
    data = rng.normal(size=(12, 10)).astype(np.float32)[:, ::2]
    network(data)
    layer.input_gate = _lstm(rng).input_gate
    layer.cell.weights = layer.cell.weights.copy(order='F')
    network(data)
    layer.cell.weights[:, 0] *= -2
    np.testing.assert_allclose(network(data), _python(network, data),
                               atol=2e-6, rtol=2e-5)


def test_recurrent_native_parity_and_custom_initial_state():
    rng = np.random.default_rng(3)
    layer = RecurrentLayer(rng.normal(size=(5, 7)).astype(np.float32) * 0.1,
                           np.zeros(7, np.float32),
                           rng.normal(size=(7, 7)).astype(np.float32) * 0.1)
    network = NeuralNetwork([layer])
    data = rng.normal(size=(19, 5)).astype(np.float32)
    np.testing.assert_allclose(network(data), _python(network, data),
                               atol=2e-6, rtol=2e-5)
    initial = np.ones(7, np.float32)
    layer.reset(initial)
    expected = _python(network, data, reset=False)
    layer.reset(initial)
    np.testing.assert_allclose(network(data, reset=False), expected,
                               atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(initial, np.ones(7, np.float32))


def test_unsupported_custom_activation_and_float64_keep_python_behavior():
    layer = FeedForwardLayer(np.eye(3, dtype=np.float32),
                             np.zeros(3, np.float32), lambda x, out: np.square(x, out=out))
    data = np.ones((2, 3), np.float32)
    assert _native.network(layer, (layer,), data, False) is None
    np.testing.assert_array_equal(layer(data), data)
    layer.activation_fn = linear
    assert _native.network(layer, (layer,), data.astype(np.float64), False) is None
    assert layer(data.astype(np.float64)).dtype == np.float64


def test_layer_cache_does_not_retain_unused_models():
    layer = FeedForwardLayer(np.eye(3, dtype=np.float32), np.zeros(3, np.float32))
    layer(np.ones((1, 3), np.float32))
    reference = weakref.ref(layer)
    del layer
    gc.collect()
    assert reference() is None


def test_direct_layer_api_uses_native_kernels_and_preserves_old_state():
    rng = np.random.default_rng(29)
    layer = _lstm(rng)
    data = rng.normal(size=(11, 5)).astype(np.float32)
    expected = _python(NeuralNetwork([layer]), data)
    with patch.object(_native, '_library', wraps=_native._library) as library:
        result = layer(data)
        assert library.called
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-5)
    previous = layer._prev
    previous_copy = previous.copy()
    state = layer._state
    state_copy = state.copy()
    layer(data, reset=False)
    np.testing.assert_array_equal(previous, previous_copy)
    np.testing.assert_array_equal(state, state_copy)


def test_single_frame_prediction_keeps_the_release_time_axis():
    layer = FeedForwardLayer(np.ones((3, 1), np.float32), np.zeros(1, np.float32))
    network = NeuralNetwork([layer])
    assert network(np.zeros((1, 3), np.float32)).shape == (1,)
    assert network(np.zeros((1, 3), np.float64)).shape == (1,)
    multi = NeuralNetwork([FeedForwardLayer(np.ones((3, 2), np.float32),
                                             np.zeros(2, np.float32))])
    assert multi(np.zeros((1, 3), np.float32)).shape == (1, 2)
