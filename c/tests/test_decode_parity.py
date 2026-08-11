#!/usr/bin/env python3
"""Decode GEMV parity: the Metal persistent decode kernel (lg_decode_gemv)
must reproduce the CPU matmul_oq affine group factorisation for all four
formats (OQF32, OQBF16, F32, BF16) at S=1 and S=8.

Runs c/tests/decode_gemv_parity.mm (built against laguna_metal.o) and asserts
every reported max |gpu-cpu| diff stays under 2e-4 -- f32 rounding noise, not
a kernel-layout bug.

Run:  make -C c decode-parity
      python3 c/tests/test_decode_parity.py
"""

import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "c", "decode_parity_test")

RE_LINE = re.compile(
    r"^(OQF32|OQBF16|F32|BF16) bits=(\d+) gs=(\d+) S=(\d+): max\|gpu-cpu\| = ([\d.eE+-]+)$"
)
RE_SILU = re.compile(r"^SILU S=(\d+) n=(\d+): max\|gpu-cpu\| = ([\d.eE+-]+)$")


def main():
    if not os.path.exists(BIN):
        print("missing %s -- run: make -C c decode-parity" % BIN)
        sys.exit(1)
    p = subprocess.run([BIN], capture_output=True, text=True, timeout=600)
    if p.returncode != 0 and not p.stdout.startswith("no metal; skip"):
        print("harness failed rc=%d\n%s" % (p.returncode, p.stderr[-2000:]))
        sys.exit(1)
    if "no metal; skip" in p.stdout:
        print("SKIP: no Metal device on this host")
        sys.exit(0)

    lines = p.stdout.splitlines()
    n_checked = 0
    worst = 0.0
    for line in lines:
        m = RE_LINE.match(line)
        if m:
            fmt, bits, gs, S, diff = m.groups()
            diff = float(diff)
            worst = max(worst, diff)
            status = "PASS" if diff <= 2e-4 else "FAIL"
            print("  [%s] %s bits=%s gs=%s S=%s: max|diff|=%.3g" %
                  (status, fmt, bits, gs, S, diff))
            if diff > 2e-4:
                print("PARITY FAIL: %s S=%s diff %.3g > 2e-4" % (fmt, S, diff))
                sys.exit(1)
            n_checked += 1
        else:
            m2 = RE_SILU.match(line)
            if m2:
                S, n, diff = m2.groups()
                diff = float(diff)
                worst = max(worst, diff)
                status = "PASS" if diff <= 2e-4 else "FAIL"
                print("  [%s] SILU S=%s n=%s: max|diff|=%.3g" % (status, S, n, diff))
                if diff > 2e-4:
                    print("PARITY FAIL: SILU diff %.3g > 2e-4" % diff)
                    sys.exit(1)
                n_checked += 1

    if n_checked == 0:
        print("no parity lines parsed; full output:\n%s" % p.stdout)
        sys.exit(1)
    print("DECODE PARITY OK: %d checks, worst |diff| = %.3g (limit 2e-4)" % (n_checked, worst))
    if "DECODE PARITY FAIL" in p.stdout:
        sys.exit(1)


if __name__ == "__main__":
    main()
