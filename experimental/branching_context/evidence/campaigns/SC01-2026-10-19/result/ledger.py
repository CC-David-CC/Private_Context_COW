def summarize(lines):
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
        totals[name] = totals.get(name, 0) + count
    return totals
