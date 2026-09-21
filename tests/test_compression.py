from app.compression import compress_text


def test_compress_text_examples():
    assert compress_text("meets") == "me2ts"
    assert compress_text("committee") == "com2it2e2"
    assert compress_text("") == ""
    assert compress_text("a") == "a"
    assert compress_text("aa") == "a2"
    assert compress_text("ab") == "ab"


def test_compress_with_punctuation_and_spaces():
    # Spaces and punctuation are included in run-length compression.
    assert compress_text("Wow!!!  Cool??") == "Wow!3 2Co2l?2"
