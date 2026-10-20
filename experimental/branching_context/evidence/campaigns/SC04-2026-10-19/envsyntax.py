import re

_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

def parse_entry(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        raise ValueError("missing =")
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError("empty key")
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError("invalid key")
    return key, value
