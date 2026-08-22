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
    """FLASH is the critical path (scratch left it 2026-08-18): flash
    read+write are FATAL checks. Scratch gets a bounded, NON-FATAL probe
    in a thread — a sick scratch is reported, never kills a job that no
    longer touches it."""
    host = socket.gethostname()
    f = "/flash/project_465003284/SYNC_MANIFEST.json"
    if os.path.exists(f):
        with open(f, "rb") as fh:
            fh.read(1 << 20)
    wpath = f"/flash/project_465003284/.probe-{host}-{os.getpid()}"
    fd = os.open(wpath, os.O_CREAT | os.O_WRONLY, 0o644)
    os.write(fd, b"probe")
    os.fsync(fd)
    os.close(fd)
    os.unlink(wpath)
    # scratch: informational only, 20s bound via daemon thread
    import threading

    def scratch_check():
        try:
            sp = f"/scratch/project_465003284/.probe-{host}-{os.getpid()}"
            sfd = os.open(sp, os.O_CREAT | os.O_WRONLY, 0o644)
            os.write(sfd, b"p")
            os.fsync(sfd)
            os.close(sfd)
            os.unlink(sp)
            scratch_check.ok = True
        except OSError:
            pass
    scratch_check.ok = False
    t = threading.Thread(target=scratch_check, daemon=True)
    t.start()
    t.join(20)
    if not scratch_check.ok:
        print(f"NODE-PROBE {host}: WARNING scratch unresponsive "
              f"(non-fatal — not in the critical path)", flush=True)


def main():
    probe_fs()
    print(f"NODE-PROBE {socket.gethostname()}: filesystems OK", flush=True)
    n = torch.cuda.device_count()
    if n == 0:
        print(f"NODE-PROBE {socket.gethostname()}: no CUDA devices", flush=True)
        return 1
    for d in range(n):
        torch.cuda.set_device(d)
        # LOAD-CLASS stress (2026-08-22, nid006946: passed the gentle
        # 2048^2 pat, hung at 14B load — the probe must reproduce real
        # pressure while the SPARE NODE is still held, which is the
        # entire point of the spare): ~14 GiB residency + three 8192^2
        # bf16 GEMMs per device. A hang parks this process; the
        # wrapper's timeout converts that into NODE-FAIL at probe time.
        big = torch.empty(7 * 1024**3, dtype=torch.bfloat16,
                          device=f"cuda:{d}")
        big.uniform_()
        x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=f"cuda:{d}")
        y = x
        for _ in range(3):
            y = y @ x
        r = y.float().sum()
        torch.cuda.synchronize(d)
        del big, x, y
        torch.cuda.empty_cache()
        if not torch.isfinite(r).item():
            print(f"NODE-PROBE {socket.gethostname()}: device {d} "
                  f"non-finite GEMM result", flush=True)
            return 1
    print(f"NODE-PROBE {socket.gethostname()}: {n} devices OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
