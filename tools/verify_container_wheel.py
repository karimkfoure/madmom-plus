#!/usr/bin/env python3
"""Verify a wheel in an OCI root filesystem with PRoot and optional QEMU.

PRoot provides filesystem virtualization. It does not provide kernel
namespace isolation. The supplied Python and dependency directories are
bound into the guest; shared system libraries come from the guest rootfs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


PROBE = r'''
import ctypes as ct
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import scipy

libc = ct.CDLL("libc.so.6")
libc.gnu_get_libc_version.restype = ct.c_char_p
release = {}
for line in Path("/etc/os-release").read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        release[key] = value.strip('"')
result = {
    "python": platform.python_version(), "architecture": platform.machine(),
    "glibc": libc.gnu_get_libc_version().decode(),
    "getconf": subprocess.check_output(["/usr/bin/getconf", "GNU_LIBC_VERSION"], text=True).strip(),
    "distribution": release, "numpy": np.__version__, "scipy": scipy.__version__,
}
if shutil.which("ffmpeg"):
    result["ffmpeg_version"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
if shutil.which("ffprobe"):
    result["ffprobe_version"] = subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0]
if sys.argv[1] == "wheel":
    import madmom
    from madmom import _native
    from madmom.native import export_model
    assert madmom.__file__.startswith("/wheel-site/"), madmom.__file__
    result["package_from_wheel"] = True
    result["native_filename"] = Path(_native.__file__).name
    result["model_count"] = len(list((Path(madmom.__file__).parent / "models").rglob("*.pkl")))

    class Frame(ct.Structure):
        _fields_ = [("sample_index", ct.c_uint64), ("activation", ct.c_float), ("beat", ct.c_uint32)]

    class State(ct.Structure):
        _fields_ = [("sample_index", ct.c_uint64), ("received_samples", ct.c_uint64),
                    ("last_beat_sample", ct.c_uint64), ("tempo_bpm", ct.c_double),
                    ("reliability", ct.c_double), ("state_index", ct.c_uint32),
                    ("pending_frames", ct.c_uint32)]

    lib = ct.CDLL(_native.__file__)
    lib.mm_engine_version.restype = ct.c_uint32
    signatures = {
        "create": ([ct.c_char_p, ct.c_void_p], ct.c_void_p),
        "destroy": ([ct.c_void_p], None),
        "reset": ([ct.c_void_p], ct.c_int),
        "configure": ([ct.c_void_p, ct.c_void_p], ct.c_int),
        "push": ([ct.c_void_p, ct.POINTER(ct.c_float), ct.c_size_t, ct.POINTER(ct.c_size_t)], ct.c_int),
        "read": ([ct.c_void_p, ct.POINTER(Frame), ct.c_size_t], ct.c_int),
        "state": ([ct.c_void_p, ct.POINTER(State)], ct.c_int),
    }
    for name, (args, result_type) in signatures.items():
        function = getattr(lib, "mm_engine_" + name)
        function.argtypes, function.restype = args, result_type
    result["native_abi_version"] = lib.mm_engine_version()
    assert result["native_abi_version"] == 0x00010000
    with tempfile.TemporaryDirectory(prefix="madmom-wheel-") as directory:
        asset = Path(directory) / "online.bin"
        export_model(asset)
        handle = lib.mm_engine_create(bytes(asset), None)
        assert handle
        try:
            pcm = np.zeros(442, dtype=np.float32)
            accepted = ct.c_size_t()
            assert lib.mm_engine_push(handle, pcm.ctypes.data_as(ct.POINTER(ct.c_float)), len(pcm), ct.byref(accepted)) == 0
            assert accepted.value == 442 and lib.mm_engine_read(handle, None, 0) == 2
            frames = (Frame * 2)()
            assert lib.mm_engine_read(handle, frames, 2) == 2
            indices = [frame.sample_index for frame in frames]
            assert indices == [0, 441]
            assert all(np.isfinite(frame.activation) and 0 <= frame.activation <= 1 for frame in frames)
            state = State()
            assert lib.mm_engine_state(handle, ct.byref(state)) == 0
            assert state.sample_index == 441 and state.received_samples == 442 and state.pending_frames == 0
            assert 0 <= state.reliability <= 1
            assert lib.mm_engine_reset(handle) == 0
            assert lib.mm_engine_state(handle, ct.byref(state)) == 0
            assert state.sample_index == (1 << 64) - 1 and state.received_samples == 0
            assert lib.mm_engine_configure(handle, None) == 0
        finally:
            lib.mm_engine_destroy(handle)
    result["engine_smoke"] = {"samples_accepted": 442, "frame_sample_indices": indices,
                              "queue_read_reset_and_configure": True}
print(json.dumps(result, sort_keys=True))
'''


def default_tests(source):
    names = [
        "test_audio_comb_filters.py", "test_ml_hmm.py", "test_ml_nn.py",
        "test_features_beats.py", "test_features_beats_online.py",
        "test_features_onsets.py", "test_features_downbeats.py",
        "test_features_downbeats_online.py", "test_features_downbeats_priors.py",
        "test_features_downbeats_window.py", "test_pickle_compat.py",
        "test_online_defaults.py",
    ]
    names.extend(path.name for path in sorted((source / "tests").glob("test_native*.py")))
    return ["tests/" + name for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootfs", required=True, type=Path)
    parser.add_argument("--proot", default="proot", type=Path)
    parser.add_argument("--qemu", type=Path)
    parser.add_argument("--python-prefix", required=True, type=Path)
    parser.add_argument("--dependencies", required=True, type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-glibc", default="2.36")
    parser.add_argument("--expected-id", default="debian")
    parser.add_argument("--test-file", action="append")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if not args.probe_only and args.wheel is None:
        parser.error("--wheel is required for tests")
    free = shutil.disk_usage("/").free
    if free < 10.1 * 2**30:
        parser.error("verification needs at least 10.1 GiB free before staging")
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    site, temporary = work / "wheel-site", work / "tmp"
    site.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    if args.wheel:
        shutil.rmtree(site)
        site.mkdir()
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(site)
        programs = work / "bin"
        if programs.exists():
            shutil.rmtree(programs)
        programs.mkdir(exist_ok=True)
        for scripts in site.glob("*.data/scripts"):
            for script in scripts.iterdir():
                shutil.copy2(script, programs / script.name)
    (work / "probe.py").write_text(PROBE)
    proot = Path(shutil.which(str(args.proot)) or args.proot).resolve()
    command = [str(proot), "-0", "-r", str(args.rootfs.resolve())]
    if args.qemu:
        qemu = Path(shutil.which(str(args.qemu)) or args.qemu).resolve()
        command.extend(["-q", str(qemu)])
    binds = [(Path("/dev"), "/dev"), (Path("/proc"), "/proc"),
             (args.python_prefix.resolve(), "/opt/python"),
             (args.dependencies.resolve(), "/deps"), (site, "/wheel-site"),
             (work, "/work"), (temporary, "/tmp")]
    if args.ffmpeg:
        binds.append((args.ffmpeg.resolve(), "/usr/local/bin/ffmpeg"))
    if args.ffprobe:
        binds.append((args.ffprobe.resolve(), "/usr/local/bin/ffprobe"))
    for host, guest in binds:
        command.extend(["-b", str(host) + ":" + guest])
    command.extend(["-w", "/work"])
    environment = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "QEMU_LD_PREFIX", "PYTHONHOME", "PYTHONPATH"):
        environment.pop(key, None)
    environment.update({"PYTHONHOME": "/opt/python", "PYTHONPATH": "/wheel-site:/deps",
                        "PYTHONNOUSERSITE": "1", "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "PATH": "/usr/local/bin:/opt/python/bin:/usr/bin:/bin"})
    python = "/opt/python/bin/python3"
    probe = subprocess.run(command + [python, "/work/probe.py", "wheel" if args.wheel else "deps"],
                           check=True, text=True, capture_output=True, env=environment)
    identity = json.loads(probe.stdout)
    assert identity["glibc"] == args.expected_glibc, identity
    assert identity["distribution"]["ID"] == args.expected_id, identity
    report = {"container_method": "OCI rootfs with PRoot filesystem virtualization",
              "kernel_namespace_isolation": False, "runtime": identity,
              "test_files": [], "test_exit_code": None}
    version_output = subprocess.check_output([str(proot), "--version"], text=True)
    version_match = re.search(r"v?(\d+\.\d+\.\d+(?:-[a-f0-9]+)?)", version_output)
    report["proot_version"] = version_match[1] if version_match else version_output.strip()
    if args.qemu:
        report["qemu_version"] = subprocess.check_output([str(qemu), "--version"], text=True).splitlines()[0]
    if args.wheel:
        report["wheel"] = args.wheel.name
        report["wheel_sha256"] = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    try:
        if not args.probe_only:
            if (work / "tests").exists():
                shutil.rmtree(work / "tests")
            shutil.copytree(args.source / "tests", work / "tests", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
            tests = args.test_file or default_tests(args.source)
            report["test_files"] = tests
            report["test_source_sha256"] = {
                name: hashlib.sha256((work / name).read_bytes()).hexdigest()
                for name in tests
            }
            with (work / "tests.log").open("w") as log:
                result = subprocess.run(command + [python, "-m", "pytest", "-q"] + tests + ["--tb=short"],
                                        stdout=log, stderr=subprocess.STDOUT, env=environment)
            report["test_exit_code"] = result.returncode
            report["test_log_sha256"] = hashlib.sha256((work / "tests.log").read_bytes()).hexdigest()
            for line in reversed((work / "tests.log").read_text().splitlines()):
                if " passed" in line or " failed" in line:
                    report["pytest_summary"] = line
                    break
    finally:
        audio = work / "tests/data/audio"
        if audio.exists():
            shutil.rmtree(audio)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report["test_exit_code"] or 0


if __name__ == "__main__":
    sys.exit(main())
