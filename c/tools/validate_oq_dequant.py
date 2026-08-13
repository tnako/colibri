"""Validate an independent oQ unpacker against mlx.core.dequantize on real bytes.

Downloads ONE small quantized tensor (weight+scales+biases) from a real oQ
checkpoint via ranged GETs, unpacks it with a from-scratch numpy implementation
of the layout documented in docs/REFERENCE.md, and compares to MLX's own
dequantize. If these agree bit-for-bit the C implementation can be written
against the numpy version with confidence.
"""
import json, struct, urllib.request
import numpy as np
import mlx.core as mx

REPO = "mlx-community/Laguna-S-2.1-oQ2e-fast"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
SHARD = "model-00001-of-00007.safetensors"


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=120).read()


def header(url):
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    return json.loads(rng(url, 8, 8 + n - 1)), 8 + n


url = f"{BASE}/{SHARD}"
hdr, data0 = header(url)
cfg = json.loads(urllib.request.urlopen(
    f"https://huggingface.co/{REPO}/raw/main/config.json", timeout=60).read())
q = cfg["quantization"]
default = {"bits": q["bits"], "group_size": q["group_size"]}

DT = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}


def grab(name, nrows=None):
    """Fetch a tensor (optionally only its first nrows rows) as a numpy array."""
    t = hdr[name]
    shape = t["shape"]
    dt = DT[t["dtype"]]
    itemsize = np.dtype(dt).itemsize
    rowlen = shape[-1]
    rows = shape[-2] if len(shape) > 1 else 1
    if nrows is not None:
        rows = min(rows, nrows)
    nbytes = rows * rowlen * itemsize
    off = data0 + t["data_offsets"][0]
    raw = rng(url, off, off + nbytes - 1)
    a = np.frombuffer(raw, dtype=dt).reshape(rows, rowlen)
    return a


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def unpack_oq(words, bits, K):
    """Dense little-endian bitstream -> uint codes. words: [rows, K*bits/32] u32."""
    rows = words.shape[0]
    # expand each u32 to 32 bits, LSB-first within the word (MLX packs low bits
    # of the first value into the low bits of the first word)
    bitmat = np.unpackbits(
        words.view(np.uint8).reshape(rows, -1), axis=1, bitorder="little"
    )  # [rows, words*32], LSB-first, and little-endian bytes keep word order
    out = np.zeros((rows, K), dtype=np.uint32)
    for b in range(bits):
        out |= (bitmat[:, b::bits][:, :K].astype(np.uint32) << b)
    return out


ROWS = 8
results = []
for stem in [
    "language_model.model.layers.0.self_attn.q_proj",   # bits=3 gs=64
    "language_model.model.layers.0.mlp.down_proj",      # bits=6 gs=64
    "language_model.lm_head",                           # bits=8 gs=64
    "language_model.model.layers.1.mlp.shared_expert.gate_proj",  # bits=8 gs=128
]:
    c = q.get(stem, default)
    bits, gs = c["bits"], c["group_size"]
    w = grab(stem + ".weight", ROWS)
    sc = grab(stem + ".scales", ROWS)
    bi = grab(stem + ".biases", ROWS)
    K = sc.shape[-1] * gs

    # --- MLX reference ---
    ref = mx.dequantize(
        mx.array(w), mx.array(bf16_to_f32(sc)), mx.array(bf16_to_f32(bi)),
        group_size=gs, bits=bits, mode=c.get("mode", "affine"),
    )
    ref = np.array(ref, dtype=np.float32)

    # --- independent implementation ---
    codes = unpack_oq(w, bits, K)
    s32, b32 = bf16_to_f32(sc), bf16_to_f32(bi)
    mine = codes.astype(np.float32) * np.repeat(s32, gs, axis=1) \
           + np.repeat(b32, gs, axis=1)

    exact = np.array_equal(mine, ref)
    maxdiff = float(np.abs(mine - ref).max())
    results.append((stem.split("language_model.")[-1], bits, gs, exact, maxdiff))
    print(f"{'EXACT' if exact else 'DIFF '} bits={bits} gs={gs} K={K:>6}  "
          f"maxdiff={maxdiff:.3e}  {stem.split('language_model.')[-1]}")
    print(f"      code range [{codes.min()}, {codes.max()}] (expect 0..{2**bits - 1})")

print()
print("ALL EXACT" if all(r[3] for r in results) else "MISMATCH -- layout assumption wrong")
