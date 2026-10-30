#!/usr/bin/env python3
"""Build a Linux aarch64 wheel with a matching target Python installation.

Run with an x86-64 Python of the same major and minor version as the target.
Install the normal build requirements in that host environment first. The
cross compiler and target Python headers produce ARM extension modules.
Run the result with the target Python under QEMU or on ARM before release.
"""

import argparse
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-python-prefix", required=True, type=Path)
    parser.add_argument("--cross-prefix", default="aarch64-linux-gnu-")
    parser.add_argument("--wheel-dir", type=Path, default=Path("dist"))
    parser.add_argument("--build-dir", type=Path,
                        default=Path("build/core-wheel-aarch64"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    prefix = args.target_python_prefix.resolve()
    version = "%d.%d" % sys.version_info[:2]
    config_files = list((prefix / "lib" / ("python" + version)).glob(
        "_sysconfigdata*linux*aarch64*.py"))
    if len(config_files) != 1:
        parser.error("target must contain one aarch64 sysconfig file for "
                     "Python " + version)
    target = runpy.run_path(str(config_files[0]))["build_time_vars"]
    include = prefix / "include" / ("python" + version)
    if not (include / "Python.h").is_file():
        parser.error("target Python headers are missing")
    if "aarch64" not in target["EXT_SUFFIX"]:
        parser.error("target extension suffix is not aarch64")

    os.environ.update({
        "_PYTHON_HOST_PLATFORM": "linux-aarch64",
        "CC": args.cross_prefix + "gcc",
        "CXX": args.cross_prefix + "g++",
        "AR": args.cross_prefix + "ar",
        "RANLIB": args.cross_prefix + "ranlib",
        "LDSHARED": args.cross_prefix + "gcc -shared",
    })
    changes = {name: target[name]
               for name in ("EXT_SUFFIX", "SOABI", "MULTIARCH")}
    changes.update({
        "INCLUDEPY": str(include), "CONFINCLUDEPY": str(include),
        "CFLAGS": "-O3 -fPIC -fno-strict-overflow -DNDEBUG",
    })
    sysconfig.get_config_vars().update(changes)
    # Setuptools keeps a separate copy of Python's build configuration.
    from setuptools._distutils import sysconfig as distutils_sysconfig
    distutils_sysconfig.get_config_vars().update(changes)

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.argv = [
        str(root / "setup.py"), "build", "--build-base", str(args.build_dir),
        "build_ext", "--parallel", str(args.workers),
        "bdist_wheel", "--plat-name", "linux_aarch64",
        "--dist-dir", str(args.wheel_dir),
    ]
    runpy.run_path(str(root / "setup.py"), run_name="__main__")


if __name__ == "__main__":
    main()
