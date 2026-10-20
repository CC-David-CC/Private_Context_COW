def summarize(lines):
    result = {}
    for line in lines:
        name, count = line.split(":")
        result[name] = int(count)
    return result
