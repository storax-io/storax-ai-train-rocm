"""Fit the dose-dynamics fixed-point curve and choose r*.

    python3 tools/dose_fit.py ~/storax-runs/lumi [--rate-min 0.5] [--guard-min 0.9]

Model (methodology-math §9): S*(r) = S_max / (1 + K·(1-r)/r), K = b/a.
Reads every runs/**/eval/eval.json whose run name encodes a dose
(gen 'dsweep', round 'rNN-sS') plus the standing anchors (v61s2 at
r=0.13, rel7 seg1 at r=0.055), least-squares the two parameters on a
grid (stdlib only — the data is 10 points, not a GPU problem), and
prints the fitted curve, r*, and the safety margin. Guard gets the
mirrored fit G*(r) = G_max / (1 + Kg·r/(1-r)).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


def collect(root):
    pts = []          # (r, rate, guard)
    for f in glob.glob(os.path.join(root, "**/eval/eval.json"),
                       recursive=True):
        name = f.split("/runs/")[-1] if "/runs/" in f else f
        m = re.search(r"dsweep/r(\d\d)-s\d", f)
        if not m:
            continue
        r = int(m.group(1)) / 100.0
        d = json.load(open(f))
        gf = os.path.join(os.path.dirname(f), "guard.json")
        g = json.load(open(gf)).get("rate") if os.path.exists(gf) else None
        pts.append((r, d.get("rate"), g))
    # standing anchors from the campaign record
    pts += [(0.13, 0.609, 0.953),      # v61s2 seg0 (realized mix)
            (0.055, 0.055, 0.953)]     # rel7 seg1 (fixed point reached)
    return pts


def fit_curve(pts, val=lambda p: p[1], mirror=False):
    best = None
    for smax_i in range(40, 96):
        smax = smax_i / 100.0
        for k_i in range(1, 400):
            k = k_i / 100.0
            err = 0.0
            for p in pts:
                r = p[0]
                x = (r / (1 - r)) if mirror else ((1 - r) / r)
                pred = smax / (1 + k * x)
                v = val(p)
                if v is None:
                    continue
                err += (pred - v) ** 2
            if best is None or err < best[0]:
                best = (err, smax, k)
    return best


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/storax-runs/lumi")
    rate_min = float(sys.argv[sys.argv.index("--rate-min") + 1]) \
        if "--rate-min" in sys.argv else 0.50
    guard_min = float(sys.argv[sys.argv.index("--guard-min") + 1]) \
        if "--guard-min" in sys.argv else 0.90
    pts = collect(root)
    print(f"{len(pts)} points:")
    for r, s, g in sorted(pts):
        print(f"  r={r:.3f}  rate={s:.3f}  guard={g if g is not None else '-'}")
    e1, smax, k = fit_curve(pts)
    print(f"\nrate fit:  S*(r) = {smax:.2f} / (1 + {k:.2f}·(1-r)/r)   "
          f"(sse {e1:.4f})")
    gpts = [p for p in pts if p[2] is not None]
    e2, gmax, kg = fit_curve(gpts, val=lambda p: p[2], mirror=True)
    print(f"guard fit: G*(r) = {gmax:.2f} / (1 + {kg:.2f}·r/(1-r))   "
          f"(sse {e2:.4f})")
    r_star = None
    for ri in range(1, 60):
        r = ri / 100.0
        s = smax / (1 + k * (1 - r) / r)
        g = gmax / (1 + kg * r / (1 - r))
        if s >= rate_min and g >= guard_min:
            r_star = (r, s, g)
            break
    if r_star:
        r, s, g = r_star
        print(f"\nr* = {r:.2f}  ->  predicted rate {s:.3f}, guard {g:.3f} "
              f"(gates: rate>={rate_min}, guard>={guard_min})")
    else:
        print("\nNO r satisfies both gates under this fit — the corpus, "
              "not the mix, is the binding constraint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
