def summarize(lines, *, casefold=False, min_count=0, exclude=()):
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 0:
        raise ValueError("min_count must be a nonnegative integer")
    excluded = set()
    for name in exclude:
        stripped = name.strip()
        if casefold:
            stripped = stripped.casefold()
        excluded.add(stripped)
    totals = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError("malformed record")
        name, count_text = stripped.split(":", 1)
        name = name.strip()
        count_text = count_text.strip()
        if not name:
            raise ValueError("empty name")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError("invalid count") from exc
        if count < 0:
            raise ValueError("negative count")
        key = name.casefold() if casefold else name
        totals[key] = totals.get(key, 0) + count
    result = totals
    if excluded:
        result = {name: total for name, total in result.items() if name not in excluded}
    if min_count == 0:
        return result
    return {name: total for name, total in result.items() if total >= min_count}
