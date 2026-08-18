"""Fast per-node health probe: GPUs AND filesystems.

GPU: touch every visible GCD with alloc + GEMM + sync — a wedged GPU
(LUMI 'GPU Hang', job 21143133) fails or hangs here in seconds.

FILESYSTEM: read from /flash and read+write+fsync on /scratch — CSC
diagnosis (2026-08-18): the recurring "hang class" was nodes with
Lustre ops left PENDING by previous jobs, not GPUs. Such a node wedges
any FS access; probing it here hangs the probe, the wrapper's timeout
fires, and NODE-FAIL names exactly the sick node — instead of a
mid-training watchdog kill condemning the whole allocation.

Run under `timeout` — a true hang never returns:

    timeout 180 python3 node_probe.py || echo "NODE-FAIL $(hostname)"

Exit 0 = devices and filesystems healthy.
"""
import os
import socket
import sys

import torch


def probe_fs():
    host = socket.gethostname()
    # read: hot inputs both tiers; write+fsync+unlink: scratch
    reads = ["/flash/project_465003284/SYNC_MANIFEST.json",
             "/scratch/project_465003284/data/cpp26ds-var/trainpack.json"]
    for f in reads:
        if os.path.exists(f):
            with open(f, "rb") as fh:
                fh.read(1 << 20)
    wpath = f"/scratch/project_465003284/.probe-{host}-{os.getpid()}"
    try:
        fd = os.open(wpath, os.O_CREAT | os.O_WRONLY, 0o644)
        os.write(fd, b"probe")
        os.fsync(fd)
        os.close(fd)
    finally:
        try:
            os.unlink(wpath)
        except OSError:
            pass


def main():
    probe_fs()
    print(f"NODE-PROBE {socket.gethostname()}: filesystems OK", flush=True)
    n = torch.cuda.device_count()
    if n == 0:
        print(f"NODE-PROBE {socket.gethostname()}: no CUDA devices", flush=True)
        return 1
    for d in range(n):
        torch.cuda.set_device(d)
        x = torch.randn(2048, 2048, dtype=torch.bfloat16, device=f"cuda:{d}")
        y = (x @ x).float().sum()
        torch.cuda.synchronize(d)
        if not torch.isfinite(y).item():
            print(f"NODE-PROBE {socket.gethostname()}: device {d} "
                  f"non-finite GEMM result", flush=True)
            return 1
    print(f"NODE-PROBE {socket.gethostname()}: {n} devices OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
