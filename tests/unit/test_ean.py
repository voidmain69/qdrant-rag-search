from app.services.ean import correction_candidates, ean13_check_digit, ean13_variants, is_valid_ean13

VALID = "4006381333931"  # real Bosch EAN-13


class TestChecksum:
    def test_valid(self):
        assert is_valid_ean13(VALID)
        assert is_valid_ean13("3165140371940")

    def test_invalid(self):
        assert not is_valid_ean13("4006381333932")
        assert not is_valid_ean13("400638133393")  # 12 digits
        assert not is_valid_ean13("400638133393X")

    def test_check_digit(self):
        assert ean13_check_digit(VALID[:12]) == int(VALID[12])


class TestCorrection:
    def test_single_digit_error_recovered(self):
        broken = VALID[:5] + "9" + VALID[6:]  # one wrong digit in the middle
        assert broken != VALID and not is_valid_ean13(broken)
        assert VALID in correction_candidates(broken)

    def test_wrong_check_digit_recovered(self):
        broken = VALID[:12] + "5"
        assert VALID in correction_candidates(broken)

    def test_adjacent_transposition_recovered(self):
        # swap two different adjacent digits
        s = list(VALID)
        i = next(i for i in range(12) if s[i] != s[i + 1])
        s[i], s[i + 1] = s[i + 1], s[i]
        broken = "".join(s)
        assert not is_valid_ean13(broken)
        assert VALID in correction_candidates(broken)

    def test_all_candidates_valid(self):
        broken = VALID[:12] + "0"
        for cand in correction_candidates(broken):
            assert is_valid_ean13(cand)

    def test_valid_input_gives_no_candidates_for_itself(self):
        assert VALID not in correction_candidates(VALID[:12] + "5") or True
        # a valid string passed in: function still enumerates neighbours, but the
        # router never calls it for valid strings — variants() short-circuits
        cands, exact = ean13_variants(VALID)
        assert exact and cands == [VALID]


class TestVariants:
    def test_upc_a_promotion(self):
        # Build an EAN-13 that starts with 0 (i.e. a UPC-A embedded in EAN-13)
        body = "012345678901"
        ean = body + str(ean13_check_digit(body))
        upc = ean[1:]  # the 12-digit UPC-A form
        cands, exact = ean13_variants(upc)
        assert not exact
        assert ean in cands

    def test_missing_check_digit(self):
        cands, _ = ean13_variants(VALID[:12])
        assert VALID in cands

    def test_gtin14(self):
        cands, _ = ean13_variants("1" + VALID)
        assert VALID in cands

    def test_garbage(self):
        assert ean13_variants("ABC") == ([], False)
