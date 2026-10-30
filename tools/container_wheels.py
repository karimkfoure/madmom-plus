#!/usr/bin/env python3
"""Fetch pinned OCI roots and build wheels in PRoot filesystem containers.

PRoot uses ptrace to translate filesystem operations. It does not provide
kernel namespace isolation. Build only trusted source and image inputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def disk_check(path, extra_bytes=0, minimum_gib=10):
    info = os.statvfs(path)
    available = info.f_bavail * info.f_frsize
    if available - extra_bytes < minimum_gib * 2**30:
        raise RuntimeError("Operation would leave less than %s GiB free" % minimum_gib)


def proot_command(rootfs, proot="proot", binds=(), qemu=None, cwd="/", command=()):
    """Build a command with guest libraries and explicit host directory binds."""
    result = [str(proot), "-0", "-r", str(Path(rootfs).resolve()),
              "-b", "/dev", "-b", "/proc", "-b", "/etc/resolv.conf"]
    for source, target in binds:
        result.extend(["-b", str(Path(source).resolve()) + ":" + target])
    if qemu:
        result.extend(["-q", str(Path(qemu).resolve())])
    return result + ["-w", cwd] + list(command)


def run(command, log, cpus):
    subprocess.run(["nice", "-n", "10", "taskset", "-c", cpus] + command,
                   stdout=log, stderr=subprocess.STDOUT, check=True)


def fetch(args, log):
    if "@sha256:" not in args.image:
        raise ValueError("Use an image pinned by its platform manifest SHA-256 digest")
    destination = args.destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    disk_check(destination.parent)
    with tempfile.TemporaryDirectory(prefix="oci-wheel-", dir=destination.parent) as temp:
        temporary = Path(temp)
        auth = temporary / "auth.json"
        auth.write_text('{"auths": {}}\n')
        image = args.image if args.image.startswith("docker://") else "docker://" + args.image
        raw = subprocess.check_output([
            "nice", "-n", "10", "taskset", "-c", args.cpus,
            args.skopeo, "inspect", "--authfile", str(auth), "--raw", image])
        manifest = json.loads(raw)
        if "layers" not in manifest:
            raise ValueError("Pin a platform manifest, not a multi-platform image index")
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if digest != args.image.rsplit("@", 1)[1]:
            raise ValueError("The image manifest digest differs from the pinned value")
        # Minimal distribution roots expand to less than eight times their
        # compressed layer size. This reserve rejects large toolchain images.
        disk_check(destination.parent, sum(layer["size"] for layer in manifest["layers"]) * 8)
        oci = temporary / "image"
        run([args.skopeo, "--insecure-policy", "copy", "--authfile", str(auth),
             image, "oci:" + str(oci) + ":root"], log, args.cpus)
        config_path = oci / "blobs" / "sha256" / manifest["config"]["digest"].split(":")[1]
        config = json.loads(config_path.read_text())
        run([args.umoci, "unpack", "--rootless", "--image", str(oci) + ":root",
             str(destination)], log, args.cpus)
        provenance = {
            "image": args.image, "manifest_digest": digest,
            "architecture": config["architecture"], "os": config["os"],
            "config_digest": manifest["config"]["digest"],
            "layers": manifest["layers"],
            "runner": "PRoot filesystem translation; no kernel namespace isolation",
        }
        (destination / "image-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    disk_check(destination.parent)


def build(args, log):
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    disk_check(output, 256 * 2**20)
    link = source / ".build-deps"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise ValueError("The source .build-deps path must be absent or a symlink")
    link.symlink_to("/opt/build-deps", target_is_directory=True)
    environment = [
        "/usr/bin/env", "PYTHONPATH=/src/.build-deps", "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "CC=gcc", "CXX=g++",
        "LDSHARED=gcc -shared", "CFLAGS=-O3 -DNDEBUG -g0",
        "CXXFLAGS=-O3 -DNDEBUG -g0",
        "/opt/python/bin/" + args.python, "setup.py", "build_ext", "--parallel",
        str(args.workers), "bdist_wheel", "--dist-dir", "/wheels",
    ]
    command = proot_command(
        args.rootfs, args.proot,
        binds=[(source, "/src"), (output, "/wheels"),
               (args.python_prefix, "/opt/python"), (args.dependencies, "/opt/build-deps")],
        qemu=args.qemu, cwd="/src", command=environment)
    run(command, log, args.cpus)
    disk_check(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--cpus", default="0-1")
    subparsers = parser.add_subparsers(dest="action", required=True)
    pull = subparsers.add_parser("fetch")
    pull.add_argument("--image", required=True)
    pull.add_argument("--destination", required=True, type=Path)
    pull.add_argument("--skopeo", default="skopeo")
    pull.add_argument("--umoci", default="umoci")
    wheel = subparsers.add_parser("build")
    wheel.add_argument("--rootfs", required=True, type=Path)
    wheel.add_argument("--source", required=True, type=Path)
    wheel.add_argument("--output", required=True, type=Path)
    wheel.add_argument("--python-prefix", required=True, type=Path)
    wheel.add_argument("--dependencies", required=True, type=Path)
    wheel.add_argument("--python", required=True)
    wheel.add_argument("--proot", default="proot")
    wheel.add_argument("--qemu")
    wheel.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    forbidden = {10, 11, 22, 23}
    for group in args.cpus.split(","):
        ends = group.split("-")
        selected = range(int(ends[0]), int(ends[-1]) + 1)
        if forbidden.intersection(selected):
            parser.error("CPU affinity includes a reserved host CPU")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as log:
        (fetch if args.action == "fetch" else build)(args, log)


if __name__ == "__main__":
    main()
