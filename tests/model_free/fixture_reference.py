"""Independent scalar oracle for the model-free COW fixture v1."""

LOGICAL_PAGE_BYTES = 4096
REGION_CANONICAL = 0
REGION_CANARY = 0xCA
_MASK = (1 << 64) - 1


def _mix64(x):
    x &= _MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK
    return (x ^ (x >> 31)) & _MASK


def _counter(seed, region, layer, token, channel, block):
    x = _mix64(seed ^ 0x9E3779B97F4A7C15)
    for value, salt in ((region, 0xD1B54A32D192ED03),
                        (layer, 0xABC98388FB8FAC03),
                        (token, 0x8CB92BA72F3D8DD7),
                        (channel, 0xDB4F0B9175AE2165),
                        (block, 0xBBE0563303A4615F)):
        x = _mix64(x ^ _mix64((value + salt) & _MASK))
    return x


def _marker(page):
    # A stable 16-byte sentinel at every logical page start, independent of GPU.
    return b"CFX1" + page.to_bytes(8, "little", signed=False) + b"\xA5\x5A\x3C\xC3"


def byte_at(seed, region, layer, token, channel, byte_offset):
    if min(seed, region, layer, token, channel, byte_offset) < 0:
        raise ValueError("fixture coordinates must be non-negative")
    local = byte_offset % LOGICAL_PAGE_BYTES
    if local < 16:
        return _marker(byte_offset // LOGICAL_PAGE_BYTES)[local]
    word = _counter(seed, region, layer, token, channel, byte_offset // 8)
    return (word >> (8 * (byte_offset % 8))) & 0xFF


def read_bytes(seed, region, layer, token, channel, start, length):
    if start < 0 or length < 0:
        raise ValueError("start and length must be non-negative")
    return bytes(byte_at(seed, region, layer, token, channel, start + i)
                 for i in range(length))


def canary_byte(seed, device_id, layer, token, channel, byte_offset):
    """Device-distinct canary stream, kept outside canonical state."""
    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    device_seed = _mix64(seed ^ _mix64(device_id + 0xA0761D6478BD642F))
    return byte_at(device_seed, REGION_CANARY, layer, token, channel, byte_offset)

def iter_canonical(seed, layers, tokens, channels, bytes_per_cell, chunk_size=1 << 20):
    """Yield flattened layer/token/channel/cell bytes with global page offsets."""
    if min(seed, layers, tokens, channels, bytes_per_cell) < 0 or chunk_size <= 0:
        raise ValueError("invalid canonical fixture dimensions")
    pending = bytearray()
    global_offset = 0
    for layer in range(layers):
        for token in range(tokens):
            for channel in range(channels):
                for _ in range(bytes_per_cell):
                    pending.append(byte_at(seed, REGION_CANONICAL, layer, token, channel,
                                           global_offset))
                    global_offset += 1
                    if len(pending) == chunk_size:
                        yield bytes(pending)
                        pending.clear()
    if pending:
        yield bytes(pending)


def canonical_state(seed, layers, tokens, channels, bytes_per_cell):
    return b"".join(iter_canonical(seed, layers, tokens, channels, bytes_per_cell))


def float_value(seed, region, layer, token, channel, precision="fp32"):
    """Return exact dyadic values, including zero and both signs, in [-8, 8]."""
    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be fp16 or fp32")
    selector = _counter(seed, region, layer, token, channel, 0)
    # Quantization in sixteenths is exactly representable in both supported formats.
    q = int(selector % 257) - 128
    return q / 16.0
