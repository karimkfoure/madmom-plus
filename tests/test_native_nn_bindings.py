"""Check the neural buffer adapter against public processor behavior."""

import gc
import pickle
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from madmom.ml.nn import NeuralNetwork, NeuralNetworkEnsemble, _native, average_predictions
from madmom.ml.nn.activations import sigmoid, tanh
from madmom.ml.nn.layers import FeedForwardLayer
from madmom.processors import Processor
from tests.test_native_nn import _lstm, _python


def _networks():
    rng = np.random.default_rng(41)
    return [NeuralNetwork([_lstm(rng), FeedForwardLayer(
        rng.normal(size=(7, 2)).astype(np.float32), np.zeros(2, np.float32), sigmoid)])
        for _ in range(3)]


def test_ensemble_uses_one_binding_call_and_keeps_individual_outputs():
    networks = _networks()
    ensemble = NeuralNetworkEnsemble(networks, ensemble_fn=None)
    data = np.random.default_rng(9).normal(size=(23, 5)).astype(np.float32)
    expected = [network(data) for network in networks]
    extension = _native._extension()
    with patch.object(extension, 'mm_nn_process_many',
                      wraps=extension.mm_nn_process_many) as binding:
        actual = ensemble(data)
        assert binding.call_count == 1
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    ensemble[1] = average_predictions
    np.testing.assert_array_equal(ensemble(data), average_predictions(expected))
    streamed = np.concatenate([ensemble(frame, reset=i == 0)
                               for i, frame in enumerate(data)])
    np.testing.assert_array_equal(streamed, average_predictions(expected))
    assert ensemble(data[:0]).shape == (0, 2)


def test_network_layer_replacement_and_ensemble_members_remain_mutable():
    networks = _networks()
    ensemble = NeuralNetworkEnsemble(networks)
    data = np.ones((3, 5), np.float32)
    ensemble(data)
    networks[0].layers[-1] = FeedForwardLayer(
        np.ones((7, 2), np.float32), np.zeros(2, np.float32), tanh)
    networks[1].layers[0].input_gate = _lstm(np.random.default_rng(1)).input_gate
    np.testing.assert_array_equal(ensemble(data), average_predictions([n(data) for n in networks]))
    replacement = _networks()[0]
    ensemble[0][1] = replacement
    np.testing.assert_array_equal(ensemble(data), average_predictions([n(data) for n in ensemble[0]]))


def test_strided_and_unaligned_models_inputs_and_state_edits_keep_parity():
    network = _networks()[0]
    gate = network.layers[0].cell
    gate.weights = gate.weights[::-1]
    original = network.layers[-1].weights.copy()
    storage = bytearray(original.nbytes + 1)
    unaligned = np.ndarray(original.shape, dtype=np.float32, buffer=storage, offset=1)
    unaligned[:] = original
    network.layers[-1].weights = unaligned
    source = np.arange(65, dtype=np.float32).reshape(13, 5) / 20
    audio_storage = bytearray(source.nbytes + 1)
    data = np.ndarray(source.shape, dtype=np.float32, buffer=audio_storage, offset=1)
    data[:] = source
    np.testing.assert_allclose(network(data), _python(network, data), atol=2e-6, rtol=2e-5)
    network(data)
    restored = pickle.loads(pickle.dumps(network))
    state = network.layers[0]
    state._prev[:] = .2
    state._state[:] = -.1
    restored.layers[0]._prev = state._prev.copy()
    restored.layers[0]._state = state._state.copy()
    previous, cell = state._prev, state._state
    expected = _python(restored, data, reset=False)
    np.testing.assert_allclose(network(data, reset=False), expected, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(previous, np.full(7, .2, np.float32))
    np.testing.assert_array_equal(cell, np.full(7, -.1, np.float32))


def test_custom_processors_map_postprocessor_and_extended_chain_are_called():
    class CustomNetwork(NeuralNetwork):
        def process(self, data, **kwargs):
            return np.full((len(data), 2), .75, np.float32)

    class PostProcessor(Processor):
        def process(self, data, offset=0, **kwargs):
            return average_predictions(data) + offset

    networks = _networks()
    ensemble = NeuralNetworkEnsemble(networks, ensemble_fn=PostProcessor())
    data = np.ones((3, 5), np.float32)
    expected = average_predictions([n(data) for n in networks])
    np.testing.assert_array_equal(ensemble(data, offset=2), expected + 2)
    calls = []

    def custom_map(function, items):
        calls.append(True)
        return list(map(function, items))

    ensemble[0].map = custom_map
    np.testing.assert_array_equal(ensemble(data), expected)
    assert calls == [True]
    ensemble[0].map = map
    ensemble[0][0] = CustomNetwork([])
    expected = average_predictions([n(data) for n in ensemble[0]])
    np.testing.assert_array_equal(ensemble(data), expected)
    ensemble.append(lambda values: values + 3)
    np.testing.assert_array_equal(ensemble(data), expected + 3)


def test_instance_processor_and_layer_overrides_keep_their_behavior():
    networks = _networks()
    ensemble = NeuralNetworkEnsemble(networks)
    data = np.ones((3, 5), np.float32)
    ensemble(data)
    ensemble[0].process = lambda values, **kwargs: [np.full((len(values), 2), .25, np.float32)]
    np.testing.assert_array_equal(ensemble(data), np.full((len(data), 2), .25, np.float32))
    del ensemble[0].process
    networks[0].process = lambda values, **kwargs: np.full((len(values), 2), .5, np.float32)
    np.testing.assert_array_equal(ensemble(data), average_predictions([n(data) for n in networks]))
    layer = networks[1].layers[0]
    original = layer.activate
    calls = []

    def activate(values, reset=True):
        calls.append(True)
        return original(values, reset=reset)

    layer.activate = activate
    networks[1](data)
    assert calls == [True]


def test_cached_ensemble_and_capsules_do_not_retain_unused_models():
    ensemble = NeuralNetworkEnsemble(_networks())
    references = [weakref.ref(network) for network in ensemble[0]]
    layers = [weakref.ref(network.layers[0]) for network in ensemble[0]]
    ensemble(np.ones((1, 5), np.float32))
    del ensemble
    gc.collect()
    assert all(reference() is None for reference in references + layers)


def test_invalid_binding_buffers_are_rejected_before_native_inference():
    network = _networks()[0]
    data = np.ones((1, 5), np.float32)
    network(data)
    capsule = _native._MODELS[network].capsule()
    binding = _native._extension().mm_nn_process_many
    with pytest.raises(ValueError):
        binding((capsule,), (network.layers,), data, (), False)
    with pytest.raises(ValueError):
        binding((capsule,), (network.layers,), data.astype(np.float64),
                (np.empty((1, 2), np.float32),), False)
    with pytest.raises(ValueError):
        binding((capsule,), (network.layers,), data,
                (np.empty((1, 3), np.float32),), False)
