# Model-free fixture v1

This fixture checks byte placement, page boundaries, cloning and external-device
canaries without loading a model. It is deterministic for an explicit unsigned
64-bit seed and logical coordinates. It does not claim model or inference
correctness.

## Frozen layout

Logical pages are exactly 4096 bytes, independent of CUDA/HIP allocation or
physical page sizes. At each page start, offsets 0 through 15 contain
`43 46 58 31`, the zero-based page number as eight little-endian bytes, then
`a5 5a 3c c3`. This marker is an intentional boundary sentinel. All remaining
bytes are counter-generated. Bytes are addressed with the tuple
`(seed, region, layer, token, channel, absolute_byte_offset)`; offsets do not
reset at a caller's transfer boundary or cell boundary. Canonical state uses
region 0 and traverses layer, then token, then channel, then bytes within a
cell. One monotonically increasing absolute byte offset spans that entire
flattened traversal; page markers occur only at offsets 0, 4096, 8192, and so
on. Callers should vary token/channel when the logical tensor varies those
dimensions.
Region `0xca` is reserved for external-device canary data and is never appended
to `canonical_state`.

For data bytes, `block = byte_offset // 8`. All arithmetic below wraps modulo
2^64. Define `M(x)` by:

1. `x ^= x >> 30; x *= 0xbf58476d1ce4e5b9`
2. `x ^= x >> 27; x *= 0x94d049bb133111eb`
3. `x ^= x >> 31`

Initialize `x = M(seed ^ 0x9e3779b97f4a7c15)`. In order, for
`(value,salt)` equal to region/`0xd1b54a32d192ed03`,
layer/`0xabc98388fb8fac03`, token/`0x8cb92ba72f3d8dd7`,
channel/`0xdb4f0b9175ae2165`, block/`0xbbe0563303a4615f`, set
`x = M(x ^ M(value + salt))`. The byte is bits `[8*(offset mod 8), +7]` of
`x`, least significant byte first. The Python oracle uses explicit scalar loops
and `int.from_bytes` semantics; the C++ header is independently implemented.

Finite-value fixtures use `q = counter(...) mod 257 - 128`, then `q / 16`.
These are exact dyadic values representable by FP16 and FP32, bounded by 8,
and include zero and both signs. They provide stable data for checking payload
movement and conversion. They do not establish numerical equivalence of a model.

## Known answers

For seed `0x0123456789abcdef`, tuple `(region=2, layer=3, token=17,
channel=5)`, data bytes at offsets 64, 65, 4095 are `8d db 46`. At offset
4096 the marker begins `43 46 58 31 01 00 00 00 00 00 00 00 a5 5a 3c c3`.
For seed `0x5eed`, dimensions `(layers=1,tokens=1,channels=1,bytes_per_cell=8192)`,
the SHA-256 of canonical bytes in flattened layer/token/channel/byte order is
`2eb34e7ed0da08b0012daa8bdb83c2555bf6e4d43d8c9c9b08a8e1c76287a7f3`.

GPU canaries must use separately selected canary seed material per device and
must remain outside the canonical state extent. A matching fixture digest is
evidence about data movement only. The scalar Python oracle is intended for
small independently checkable vectors; use a compiled streaming CPU helper for
large fixtures to avoid Python per-byte overhead.
