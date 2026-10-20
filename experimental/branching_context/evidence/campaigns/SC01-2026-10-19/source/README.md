# Count ledger
Implement summarize(lines),which reads iterable text lines name:count and returns
a dict of summed counts for each name. Strip surrounding whitespace. Skip blank
lines and lines whose first non-space character is #. Names may contain Unicode;
preserve their text after trimming. Counts must be nonnegative integers; malformed
records,empty names,and negative counts raise ValueError. Repeated names add,
not replace. Do not modify tests. Run: python3 -m unittest discover -s tests -v
