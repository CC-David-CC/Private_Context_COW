from envsyntax import parse_entry

def load(lines, *, reject_duplicates=False):
    config = {}
    for line in lines:
        entry = parse_entry(line)
        if entry is None:
            continue
        key, value = entry
        if reject_duplicates and key in config:
            raise ValueError(f"duplicate key: {key}")
        config[key] = value
    return config
