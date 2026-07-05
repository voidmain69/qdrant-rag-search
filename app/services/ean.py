"""EAN-13 validation and checksum-aware error correction.

The mod-10 checksum lets us model the two dominant real-world barcode entry errors
precisely: a single wrong digit and a transposition of adjacent digits. For a
13-digit string that fails validation we generate every checksum-valid candidate
reachable by one such error — a handful of strings, each then looked up O(1) in the
in-memory code index.
"""

from __future__ import annotations


def ean13_check_digit(digits12: str) -> int:
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(digits12))
    return (10 - total % 10) % 10


def is_valid_ean13(s: str) -> bool:
    return len(s) == 13 and s.isdigit() and int(s[12]) == ean13_check_digit(s[:12])


def correction_candidates(s: str) -> list[str]:
    """All checksum-valid EAN-13 strings one digit-substitution or one adjacent
    transposition away from the (invalid) input."""
    if len(s) != 13 or not s.isdigit():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for i in range(13):
        for d in "0123456789":
            if d == s[i]:
                continue
            cand = s[:i] + d + s[i + 1 :]
            if cand not in seen and is_valid_ean13(cand):
                seen.add(cand)
                out.append(cand)
    for i in range(12):
        if s[i] == s[i + 1]:
            continue
        cand = s[:i] + s[i + 1] + s[i] + s[i + 2 :]
        if cand not in seen and is_valid_ean13(cand):
            seen.add(cand)
            out.append(cand)
    return out


def ean13_variants(token: str) -> tuple[list[str], bool]:
    """Candidate EAN-13 strings for a 12–14 digit token.

    Returns (candidates, was_exact): ``was_exact`` is True when the token itself is
    already a valid EAN-13 (candidates == [token]).
    """
    if not token.isdigit():
        return [], False
    n = len(token)
    if n == 13:
        if is_valid_ean13(token):
            return [token], True
        return correction_candidates(token), False
    if n == 12:
        cands: list[str] = []
        upc = "0" + token  # UPC-A zero-padded to EAN-13 keeps the same check digit
        if is_valid_ean13(upc):
            cands.append(upc)
        with_check = token + str(ean13_check_digit(token))  # check digit was omitted
        if with_check not in cands:
            cands.append(with_check)
        return cands, False
    if n == 14:
        cands = []
        if is_valid_ean13(token[1:]):  # GTIN-14: drop packaging indicator digit
            cands.append(token[1:])
        if is_valid_ean13(token[:13]) and token[:13] not in cands:  # extra trailing digit
            cands.append(token[:13])
        return cands, False
    return [], False
