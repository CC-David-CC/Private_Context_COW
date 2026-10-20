from envsyntax import parse_entry

def load(lines):
    config = {}
    for line in lines:
        entry = parse_entry(line)
        if entry is None:
            continue
        key, value = entry
        config[key] = value
    return config
