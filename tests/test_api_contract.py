import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.compression import compress_text
from app.summarizer import SummarizationResult, Summarizer, InvalidInputError


class DummySummarizer:
    def __init__(self, *_args, **_kwargs):
        self.device = "cpu"
        self.max_source_tokens = 512

    def warmup(self):
        return None

    def summarize(self, *args, **kwargs):
        return SummarizationResult(summary="Dummy summary for testing.")


@pytest.fixture
def patch_summarizer(monkeypatch):
    monkeypatch.setattr(main, "Summarizer", DummySummarizer)
    yield


def test_summarize_contract(patch_summarizer):
    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"text": "This is some input text to summarize.", "summary_length": 50},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["summary"] == "Dummy summary for testing."


def test_compress_summary_contract(patch_summarizer):
    with TestClient(main.app) as client:
        response = client.post(
            "/compress_summary",
            json={"text": "This is some input text to summarize.", "summary_length": 50},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["summary"], str)
        assert len(body["summary"]) > 0


def test_rejects_oversized_input(patch_summarizer, monkeypatch):
    monkeypatch.setattr(main.settings, "max_input_chars", 10)
    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"text": "0123456789abcdef", "summary_length": 50},
        )
        assert response.status_code == 422
        assert "MAX_INPUT_CHARS" in response.json()["detail"]


def test_rejects_invalid_summary_length(patch_summarizer):
    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"text": "Valid input text", "summary_length": 1},
        )
        assert response.status_code == 422


def test_rejects_summary_length_above_schema(patch_summarizer):
    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"text": "Valid input text", "summary_length": 400},
        )
        assert response.status_code == 422


def test_whitespace_only_rejected(patch_summarizer):
    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"text": "   ", "summary_length": 50},
        )
        assert response.status_code == 422


def test_too_short_rejected(patch_summarizer):
    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"text": " short ", "summary_length": 50},
        )
        assert response.status_code == 422


def test_timeout_returns_504(monkeypatch):
    class SlowSummarizer(DummySummarizer):
        def summarize(self, *args, **kwargs):
            import time as _time

            _time.sleep(0.05)
            return SummarizationResult(summary="slow summary")

    monkeypatch.setattr(main, "Summarizer", SlowSummarizer)
    monkeypatch.setattr(main.settings, "request_timeout_seconds", 0.01)

    # Reset state so startup uses patched Summarizer
    main.summarizer = None
    main._is_ready = False
    main.semaphore = None

    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"text": "This is some input text to summarize.", "summary_length": 50},
        )
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_ready_and_summarize_return_503_when_model_not_ready(monkeypatch):
    class FailingSummarizer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("init failed")

    monkeypatch.setattr(main, "Summarizer", FailingSummarizer)

    # Reset readiness flags/state
    main.summarizer = None
    main._is_ready = False
    main.semaphore = None

    with TestClient(main.app) as client:
        resp_ready = client.get("/ready")
        assert resp_ready.status_code == 503

        resp_summarize = client.post(
            "/summarize",
            json={"text": "Valid input text", "summary_length": 50},
        )
        assert resp_summarize.status_code == 503
        assert "not ready" in resp_summarize.json()["detail"].lower()


def test_root_ui_served(patch_summarizer):
    with TestClient(main.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Summarization Service" in resp.text


class FakeSummarizer(Summarizer):
    def __init__(self):
        # Skip real model/tokenizer init
        pass


def test_low_entropy_spam_rejected():
    s = FakeSummarizer()
    text = "-" * 130
    with pytest.raises(InvalidInputError):
        s.summarize(text, max_new_tokens=50, enable_chunking=False)


def test_single_token_blob_rejected():
    s = FakeSummarizer()
    text = "a3" * 80
    with pytest.raises(InvalidInputError):
        s.summarize(text, max_new_tokens=50, enable_chunking=False)


def test_non_latin_bypasses_model():
    s = FakeSummarizer()
    text = "こんにちは世界" * 15
    result = s.summarize(text, max_new_tokens=50, enable_chunking=False)
    assert result.summary == text.strip()


def test_gibberish_rejected():
    s = FakeSummarizer()
    text = "qwrty plmnbv zxcvbnm qwrtyplm zxcvbnm qwrty plmnbv zxcvbnm qwrtyplm zxcvbnm"
    with pytest.raises(InvalidInputError):
        s.summarize(text, max_new_tokens=50, enable_chunking=False)


def test_invalid_input_error_becomes_422(monkeypatch):
    class RejectingSummarizer(DummySummarizer):
        def summarize(self, *args, **kwargs):
            raise InvalidInputError("Input rejected: looks like gibberish (not natural language).")

    monkeypatch.setattr(main, "Summarizer", RejectingSummarizer)

    # Reset state so startup uses the patched Summarizer
    main.summarizer = None
    main._is_ready = False
    main.semaphore = None

    with TestClient(main.app) as client:
        resp = client.post(
            "/summarize",
            json={"text": "This input is long enough.", "summary_length": 50},
        )
        assert resp.status_code == 422
        assert "gibberish" in resp.json()["detail"].lower()


def test_invalid_input_error_becomes_422_on_compress_summary(monkeypatch):
    class RejectingSummarizer(DummySummarizer):
        def summarize(self, *args, **kwargs):
            raise InvalidInputError("Input rejected: low-entropy repetitive text.")

    monkeypatch.setattr(main, "Summarizer", RejectingSummarizer)

    # Reset state so startup uses the patched Summarizer
    main.summarizer = None
    main._is_ready = False
    main.semaphore = None

    with TestClient(main.app) as client:
        resp = client.post(
            "/compress_summary",
            json={"text": "This input is long enough.", "summary_length": 50},
        )
        assert resp.status_code == 422
        assert "low-entropy" in resp.json()["detail"].lower()


def _decompress_text(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        i += 1
        num = []
        while i < len(s) and s[i].isdigit():
            num.append(s[i])
            i += 1
        count = int("".join(num)) if num else 1
        out.append(ch * count)
    return "".join(out)


def test_compress_round_trip():
    samples = [
        "",
        "a",
        "aa",
        "committee",
        "Wow!!!  Cool??",
        "aaabbbcccc",
        "abbbbccdddddd",
    ]

    for x in samples:
        assert _decompress_text(compress_text(x)) == x
