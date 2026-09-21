# Compress text by collapsing runs of the same character
def compress_text(text: str) -> str:
    if not text:
        return ""

    # Track the current run
    prev = text[0]
    run_len = 1

    # Build pieces and join once at the end
    parts: list[str] = []

    # helper to flush a run to parts
    def _flush(char: str, length: int) -> None:
        # Keep singles readable, compress repeats
        parts.append(char if length == 1 else f"{char}{length}")

    # Walk the string and collapse consecutive identical chars
    for current in text[1:]:
        if current == prev:
            run_len += 1
            continue

        # Character changed, flush the previous run
        _flush(prev, run_len)
        prev = current
        run_len = 1

    # Flush the final run
    _flush(prev, run_len)
    return "".join(parts)
