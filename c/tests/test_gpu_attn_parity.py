#!/usr/bin/env python3
"""GPU attention parity: the Metal build must produce the SAME tokens as the
CPU build when the full-attention layers run on the GPU, because Phase 1's
tiled online-softmax path is a different summation order from both the CPU
reference and the old banded GPU path -- the regression bar is argmax parity,
same as the fixtures.

Both engines are otherwise identical (same checkpoint, same decode loop, same
expert path), so any token difference past position 0 of prefill means the GPU
attention diverged. Decode (S<64) never touches the GPU on either build.

Run:  python3 c/tests/test_gpu_attn_parity.py [--prompt-n 1900] [--gen 24]
"""

import argparse, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DINOSAUR_START = (
    "The coelacanth is a living fossil: no other living animal is more deeply "
    "tethered to a vanished world. For decades scientists assumed it had gone "
    "extinct with the dinosaurs, a 400-million-year-old dead end, until a "
    "fisherman hauled a blue-scaled, four-limbed creature from the Indian Ocean "
    "off the coast of South Africa. That discovery, in 1938, rewrote the story "
    "of evolution and our understanding of the deep sea. But the coelacanth's "
    "remarkable biology does not end with its ancient lineage. It has a "
    "hinged skull, a notochord instead of a spine, and a secret second lung. "
    "Its fins are fleshy and lobed, bearing the same basic structure as the "
    "limbs of tetrapods, the group that would ultimately walk onto land. "
    "Studying it is like opening a time capsule that has been sealed for "
    "hundreds of millions of years, its organs and tissues preserved in the "
    "darkness of the ocean floor.\n\n"
)

def make_prompt(n_tokens_seen_hint):
    # repeat a long prose block so the harness gets a multi-thousand-token
    # prompt cheaply; the engine dedups nothing and tokens differ per repeat.
    parts = []
    while True:
        parts.append(DINOSAUR_START)
        joined = "\n\n".join(parts)
        # rough: ~11 chars/token for this prose
        if len(joined) >= n_tokens_seen_hint * 11:
            return joined

def run(binary, prompt_path, gen, extra_env=None):
    env = dict(os.environ)
    env["SNAP"] = os.path.join(ROOT, "models", "Laguna-XS-2.1-oQ2")
    env["LG_SPEC"] = "0"
    if extra_env:
        env.update(extra_env)
    cmd = [binary, "0", "0", "--chat", "-n", str(gen), "-f", prompt_path]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)
    wall = time.time() - t0
    return p, wall

def token_lines(stdout, stderr):
    # the engine prints each generated token as text; the argmax loop is
    # deterministic so identical tokens -> identical stdout outside timing.
    lines = []
    for line in stdout.splitlines():
        if re.search(r"\[prefill|\[phases\]|RSS|reside|== |PEAK|prompt tokens", line):
            continue
        if line.startswith("== "):
            continue
        lines.append(line)
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-n", type=int, default=1900)
    ap.add_argument("--gen", type=int, default=24)
    a = ap.parse_args()

    prompt_path = os.path.join("/tmp", "gpu_attn_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(make_prompt(a.prompt_n))

    metal = os.path.join(ROOT, "c", "laguna_xs_metal")
    cpu   = os.path.join(ROOT, "c", "laguna_xs")

    pm, wm = run(metal, prompt_path, a.gen)
    if pm.returncode != 0:
        print("metal run failed rc=%d\n%s" % (pm.returncode, pm.stderr[-2000:]))
        sys.exit(1)
    pc, wc = run(cpu, prompt_path, a.gen)

    tm = token_lines(pm.stdout, pm.stderr)
    tc = token_lines(pc.stdout, pc.stderr)
    if len(tm) != len(tc):
        print("output line count differs: metal=%d cpu=%d" % (len(tm), len(tc)))
        sys.exit(1)
    bad = [i for i, (x, y) in enumerate(zip(tm, tc)) if x != y]
    if bad:
        print("PARITY MISMATCH at lines %s (metal %.1fs cpu %.1fs)" %
              (bad[:20], wm, wc))
        for i in bad[:5]:
            print("  metal: %r" % tm[i][:120])
            print("  cpu  : %r" % tc[i][:120])
        sys.exit(1)
    print("PARITY OK: %d generated text-identical (metal %.1fs, cpu %.1fs)"
          % (a.gen, wm, wc))
    # surface the GPU/dispatch evidence on the metal stderr
    for line in pm.stderr.splitlines():
        if re.search(r"gpuprof|phases|prefill", line):
            print("  " + line[:140])

if __name__ == "__main__":
    main()