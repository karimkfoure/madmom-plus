"""Optional native streaming beat tracker with the released online models.

Audio must be mono float PCM at 44100 Hz. Each output row contains frame time
in seconds, beat activation, beat event (0 or 1), and tempo in BPM. Frame times
refer to the last input sample used, starting at sample zero. The first frame
has zero padding on the left. Instances keep state until reset or closed.
"""

import ctypes as ct
import os
import sys
import tempfile

import numpy as np


def available():
    """Return whether the optional C++ extension can be imported."""
    try:
        from . import _native  # noqa: F401
    except ImportError:
        return False
    return True


def export_model(path):
    """Export installed madmom online weights and default DBN to a native asset.

    Export uses trusted installed pickle model files. The exported weights keep
    the model licence (CC BY-NC-SA 4.0); conversion does not change that licence.
    The asset has little-endian arrays and format magic ``MMONL001``.
    """
    from .features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

    processor = RNNBeatProcessor(online=True, origin='online', fps=100,
                                 num_threads=1)
    frontend = processor[0][1][0]
    signal = processor[0][0](np.zeros(4410, dtype=np.float32))
    spectrum = frontend[1](frontend[0](signal))
    filtered = frontend[2](spectrum)
    frontend[4](frontend[3](filtered), reset=True)
    if filtered.filterbank.shape != (1024, 81):
        raise ValueError('The native backend requires the released online frontend')
    decoder = DBNBeatTrackingProcessor(online=True, fps=100)
    transition = decoder.tm
    with open(path, 'wb') as stream:
        stream.write(b'MMONL001')

        def write(value, dtype='<f4'):
            stream.write(np.asarray(value, dtype=dtype).tobytes())

        write(spectrum.fft_window, '<f8')
        write(filtered.filterbank)
        write([frontend[4].diff_frames], '<u4')
        for network in processor[-1][0]:
            if len(network.layers) != 4:
                raise ValueError('Unsupported native network architecture')
            for layer in network.layers[:-1]:
                gates = [layer.input_gate, layer.output_gate,
                         layer.forget_gate, layer.cell]
                if layer.init.shape != (25,):
                    raise ValueError('Unsupported native recurrent layer width')
                write(np.concatenate([gate.weights.T for gate in gates]))
                write(np.concatenate([gate.recurrent_weights.T for gate in gates]))
                write(np.concatenate([gate.bias for gate in gates]))
                write(np.concatenate([gate.peephole_weights for gate in gates[:3]]))
                write(layer.init)
                write(layer.cell_init)
            write(network.layers[-1].weights.T)
            write(network.layers[-1].bias)
        write([transition.num_states, len(transition.states)], '<u4')
        write([decoder.om.observation_lambda, decoder.max_bpm], '<f8')
        write(transition.states, '<u4')
        write(transition.pointers, '<u4')
        write(transition.probabilities, '<f8')
        write(decoder.om.pointers, '<u4')
        write(decoder.hmm.initial_distribution, '<f8')
    return os.fspath(path)


def _library():
    try:
        from . import _native
    except ImportError as exc:
        raise ImportError('Native backend unavailable. Rebuild with '
                          'MADMOM_BUILD_NATIVE=1 to enable it.') from exc
    library = ct.CDLL(_native.__file__)
    fp = ct.POINTER(ct.c_float)
    dp = ct.POINTER(ct.c_double)
    handle = ct.c_void_p
    signatures = {
        'create': ([ct.c_char_p, ct.c_int, ct.c_int, ct.c_int], handle),
        'destroy': ([handle], None),
        'error': ([], ct.c_char_p),
        'reset': ([handle], ct.c_int),
        'features': ([handle, fp, fp], ct.c_int),
        'network': ([handle, fp, fp], ct.c_int),
        'decode': ([handle, ct.c_float, dp], ct.c_int),
        'push': ([handle, fp, ct.c_size_t, dp, ct.c_size_t], ct.c_int),
        'decoder_state': ([handle, dp, ct.c_size_t], ct.c_int),
        'summary': ([handle, dp], ct.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(library, 'mm_online_' + name)
        function.argtypes, function.restype = args, result
    return library


class NativeBeatTracker:
    """Stateful native frontend, LSTM ensemble, and forward DBN beat tracker.

    ``networks`` selects the first 1 to 8 released networks. ``precision`` is
    ``float32``, ``float16``, or ``int8``. Reduced precision rounds matrix weights
    at load time, then uses float32 arithmetic and storage. It is an accuracy
    experiment, not a packed low-precision kernel. ``simd=True`` enables SSE2 on
    x86 or NEON on AArch64; other targets use scalar arithmetic. Each instance
    must be used by one thread at a time. No resampling or capture is included.
    """

    def __init__(self, model_path=None, networks=8, precision='float32', simd=False):
        self._handle = None
        if isinstance(networks, bool) or not isinstance(networks, (int, np.integer)) or not 1 <= networks <= 8:
            raise ValueError('networks must be an integer from 1 to 8')
        modes = {'float32': 0, 'float16': 1, 'int8': 2}
        if precision not in modes:
            raise ValueError('precision must be float32, float16, or int8')
        if not isinstance(simd, (bool, np.bool_)):
            raise ValueError('simd must be a boolean')
        if sys.byteorder != 'little':
            raise RuntimeError('The native asset format requires a little-endian host')
        self._lib = _library()
        self.networks, self.precision, self.simd = int(networks), precision, bool(simd)
        if model_path is None:
            with tempfile.TemporaryDirectory(prefix='madmom-native-') as directory:
                self._create(export_model(os.path.join(directory, 'online.bin')), modes[precision])
        else:
            self._create(model_path, modes[precision])

    def _create(self, path, mode):
        self._handle = self._lib.mm_online_create(os.fsencode(path), self.networks, mode, self.simd)
        if not self._handle:
            raise RuntimeError(self._lib.mm_online_error().decode('utf-8', 'replace'))

    def _check(self, result):
        if result < 0:
            raise ValueError(self._lib.mm_online_error().decode('utf-8', 'replace'))
        return result

    def _open(self):
        if not self._handle:
            raise RuntimeError('The native tracker is closed')
        return self._handle

    @staticmethod
    def _input(values, size=None):
        values = np.asarray(values)
        if values.ndim != 1 or (size is not None and values.size != size):
            raise ValueError('Input must be a one-dimensional array' +
                             ('' if size is None else ' of length %d' % size))
        if values.dtype.kind not in 'f':
            raise ValueError('Input must use floating point PCM or features')
        with np.errstate(over='ignore', invalid='ignore'):
            values = np.ascontiguousarray(values, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError('Input must contain finite float32 values')
        return values

    @staticmethod
    def _float_pointer(values):
        return values.ctypes.data_as(ct.POINTER(ct.c_float))

    @staticmethod
    def _double_pointer(values):
        return values.ctypes.data_as(ct.POINTER(ct.c_double))

    def push(self, audio):
        """Push mono 44100 Hz float PCM and return rows (time, activation, beat, BPM).

        Empty input returns shape (0, 4) without advancing state. Sample values
        use the usual -1 to +1 PCM scale. Chunk boundaries do not change output.
        """
        handle = self._open()
        audio = self._input(audio)
        outputs = []
        for start in range(0, len(audio), 44100):
            chunk = audio[start:start + 44100]
            output = np.empty((len(chunk) // 441 + 2, 4), dtype=np.float64)
            count = self._check(self._lib.mm_online_push(
                handle, self._float_pointer(chunk), len(chunk),
                self._double_pointer(output), len(output)))
            outputs.append(output[:count])
        return np.concatenate(outputs) if outputs else np.empty((0, 4), dtype=np.float64)

    def features(self, frame):
        """Process one causal 2048-sample frame; update spectral difference state."""
        frame = self._input(frame, 2048)
        result = np.empty(162, dtype=np.float32)
        self._check(self._lib.mm_online_features(self._open(), self._float_pointer(frame), self._float_pointer(result)))
        return result

    def network(self, features):
        """Process one 162-value feature vector; update recurrent state."""
        features = self._input(features, 162)
        result = np.empty(1, dtype=np.float32)
        self._check(self._lib.mm_online_network(self._open(), self._float_pointer(features), self._float_pointer(result)))
        return result[0]

    def decode(self, activation):
        """Update the DBN with one activation and return a four-value output row."""
        if not np.isscalar(activation) or not np.isfinite(activation) or not 0 <= activation <= 1:
            raise ValueError('activation must be a finite scalar from 0 to 1')
        result = np.empty(4, dtype=np.float64)
        self._check(self._lib.mm_online_decode(self._open(), activation, self._double_pointer(result)))
        return result

    @property
    def decoder_state(self):
        """Return a copy of the normalized DBN forward probabilities."""
        count = self._check(self._lib.mm_online_decoder_state(self._open(), None, 0))
        result = np.empty(count, dtype=np.float64)
        self._check(self._lib.mm_online_decoder_state(self._open(), self._double_pointer(result), count))
        return result

    @property
    def summary(self):
        """Return BPM, last beat seconds, maximum posterior, and its state index."""
        result = np.empty(4, dtype=np.float64)
        self._check(self._lib.mm_online_summary(self._open(), self._double_pointer(result)))
        return result

    def reset(self):
        """Reset audio, frontend, networks, DBN, and frame counters."""
        self._check(self._lib.mm_online_reset(self._open()))

    def close(self):
        """Release native memory. Repeated calls have no effect."""
        if self._handle:
            self._lib.mm_online_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
