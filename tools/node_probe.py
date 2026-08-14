"""Fast per-node GPU health probe: touch every visible GCD with an
alloc + H2D copy + small GEMM + sync. A wedged GPU (LUMI 'GPU Hang',
job 21143133) fails or hangs here in seconds, BEFORE a job spends
minutes on staging and model load. Run under `timeout` — a true hang
never returns:

    timeout 180 python3 node_probe.py || echo "NODE-FAIL $(hostname)"

Exit 0 = all devices healthy.
"""
import socket
import sys

import torch


def main():
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
