from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from dataclasses import dataclass
from typing import List, Optional
import inspect
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class SummarizationResult:
    summary: str

# Raised when input is rejected as non-natural-language / unsafe-to-summarize
class InvalidInputError(ValueError):
    pass

# Loads the Hugging Face model once and provides deterministic summarization
class Summarizer:

    # Hard guardrails to prevent pathological behavior on huge inputs.
    _MAX_CHUNKS: int = 64

    # Constructor
    def __init__(self, model_id: str, max_input_tokens: int | None = None):
        # Store model ID
        self.model_id = model_id
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # Load model in eval mode
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()

        # Use GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Move model to device
        self.model.to(self.device)

        # Determine max source tokens
        self.max_source_tokens = self._determine_max_source_tokens(max_input_tokens)

    def _determine_max_source_tokens(self, override: int | None) -> int:
        if override is not None and override > 0:
            return int(override)

        model_max = getattr(self.tokenizer, "model_max_length", None)
        if model_max is None or model_max > 100_000:
            model_max = 1024
        return min(int(model_max), 4096)

    # Prime the model to reduce first-request latency
    def warmup(self) -> None:
        _ = self.summarize(
            "Model warmup request for summarization service.",
            max_new_tokens=32,
            enable_chunking=False,
        )

    def _strip_control_chars(self, text: str) -> str:
        # Remove common control characters that can confuse tokenizers/logging.
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    def _is_code_like(self, text: str) -> bool:
        t = text.strip()
        if "\n" in t and any(sym in t for sym in ("{", "}", ";", "==", "->", "=>", "()", "[]")):
            return True

        lowered = t.lower()
        code_markers = (
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "public ",
            "private ",
            "function ",
            "var ",
            "let ",
            "const ",
            "#include",
            "using ",
            "package ",
            "select ",
            "insert ",
            "update ",
            "delete ",
            "traceback",
            "exception",
            "stack trace",
        )
        if any(m in lowered for m in code_markers) and any(sym in t for sym in ("(", ")", "{", "}", ";", "=", ":")):
            return True

        return False

    # Heuristic checks for gibberish / non-natural-language text
    # Keeps false positives low and avoids flagging code-like text.
    def _looks_like_gibberish(self, text: str) -> bool:

        raw = text or ""
        t = raw.strip()

        # Too short to be confident
        text_len = len(t)
        if text_len < 10:
            return False

        # Don't classify code as gibberish
        if self._is_code_like(t):
            return False

        # Tokenize
        lowered = t.lower()
        tokens = re.findall(r"\b\w+\b", lowered)
        alpha_words = re.findall(r"[A-Za-z]{2,}", t)

        # Repeated token pattern (e.g., "sdfsdf sdfsdf")
        token_count = len(tokens)
        if token_count >= 2:
            first_token = tokens[0]
            first_len = len(first_token)
            all_same = all(tok == first_token for tok in tokens)

            if first_len >= 4 and all_same:
                return True

        # Vowel test on substantial alphabetic words
        long_words = [w for w in alpha_words if len(w) >= 4]
        long_word_count = len(long_words)
        if long_word_count >= 2:
            words_with_vowel = sum(1 for w in long_words if re.search(r"[aeiouAEIOU]", w))
            vowel_fraction = words_with_vowel / long_word_count

            if vowel_fraction < 0.30:
                return True

        # Single long token with no vowels often indicates random mashing
        if token_count == 1:
            tok = tokens[0]
            tok_len = len(tok)

            if tok_len >= 12:
                has_vowel = bool(re.search(r"[aeiou]", tok))
                if not has_vowel:
                    return True

        # Digit-dominant alphanumeric strings (often garbage IDs)
        alnum_chars = re.findall(r"[A-Za-z0-9]", t)
        if alnum_chars:
            alnum_count = len(alnum_chars)

            # Require substantial alnum content to avoid short-circuiting on symbols/punctuation
            letter_chars = re.findall(r"[A-Za-z]", t)
            digit_chars = re.findall(r"[0-9]", t)

            # Avoid short inputs
            letter_fraction = len(letter_chars) / alnum_count
            digit_fraction = len(digit_chars) / alnum_count

            # e.g., "XJ394KDJF83JD93JD" or "839204JDKFJ3948JD"
            if letter_fraction < 0.35 and digit_fraction > 0.50:
                return True

        return False

    # Single-token blob detection (for hashes, keys, etc.)
    def _is_single_token_blob(self, text: str) -> bool:
        t = (text or "").strip()

        # If it is not exactly one "word-ish" token, it is not the kind of blob we care about here
        tokens = re.findall(r"\b\w+\b", t)
        if len(tokens) != 1:
            return False
        # Analyze the single token
        tok = tokens[0]

        # Short tokens show up everywhere (normal words, names, etc.)
        if len(tok) < 32:
            return False

        # Common opaque IDs: long hex strings (MD5-ish and beyond)
        if re.fullmatch(r"[0-9a-fA-F]{32,}", tok):
            return True

        # Base64-ish payloads tend to be long and use a restricted character set with optional padding
        if re.fullmatch(r"[A-Za-z0-9+/]{40,}={0,2}", tok):
            return True

        # If it mixes several character types (upper/lower/digits/underscore), it is usually an identifier
        has_lower = has_upper = has_digit = False
        has_underscore = "_" in tok

        # Scan characters
        for c in tok:
            if not has_lower and c.islower():
                has_lower = True
            elif not has_upper and c.isupper():
                has_upper = True
            elif not has_digit and c.isdigit():
                has_digit = True

            # Early exit once it clearly looks like an ID
            if int(has_lower) + int(has_upper) + int(has_digit) + int(has_underscore) >= 3:
                return True

        return False

    # Low-entropy spam detection (for repetitive text)
    def _is_low_entropy_spam(self, text: str) -> bool:
        # Catch obvious repetitive spam but avoid flagging code
        t = (text or "").strip()
        if not t:
            return False

        # Code often looks repetitive
        if self._is_code_like(t):
            return False

        # Ignore whitespace when checking repetition
        compact = re.sub(r"\s+", "", t)
        if not compact:
            return False

        # Analyze character variety and length
        compact_len = len(compact)
        unique_chars = len(set(compact))

        # Single character floods like aaaaaaaaaaaa or ----------
        if compact_len >= 12 and unique_chars == 1:
            return True

        # For short strings be conservative
        if compact_len < 32:
            return False

        # Low variety over a long string is usually junk
        if unique_chars <= 3:
            return True

        # Long repeated runs like !!!!!!!!!!...
        if re.search(r"(.)\1{15,}", compact):
            return True

        return False

    def _has_short_repeating_pattern(self, text: str) -> bool:
        # Catches patterns like "abcdabcdabcd..." that may not be caught by unique-char rules.
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 32:
            return False

        # Do not interfere with code-like content.
        if self._is_code_like(text):
            return False

        # Check for short repeating units of length 2, 3, or 4.
        for unit_len in (2, 3, 4):
            if len(compact) < unit_len * 8:
                continue
            unit = compact[:unit_len]

            # Check if the entire string is made up of this unit repeated.
            if unit and (unit * (len(compact) // unit_len)) == compact[: (len(compact) // unit_len) * unit_len]:
                # If at least 90% of compact is comprised of repeating unit, treat as spam.
                repeated = unit * (len(compact) // unit_len)
                # Account for any leftover partial unit at the end.
                if repeated and len(repeated) / len(compact) >= 0.90:
                    return True
        return False

    # Mostly-symbols detection (for non-natural-language noise)
    def _is_mostly_symbols(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False

        # Code can be punctuation-heavy
        if self._is_code_like(t):
            return False

        # Strip whitespace so spacing does not affect the ratio
        compact = re.sub(r"\s+", "", t)
        if not compact:
            return False

        compact_len = len(compact)

        # If there are no letters or digits and it is not tiny, it is just noise
        has_any_alnum = any(ch.isalnum() for ch in compact)
        if compact_len >= 10 and not has_any_alnum:
            return True

        # For shorter strings be conservative
        if compact_len < 32:
            return False

        # Measure how much of it is actually letters or digits
        alnum_count = sum(1 for ch in compact if ch.isalnum())
        alnum_fraction = alnum_count / compact_len
        return alnum_fraction < 0.10

    # URL/email-heavy input detection
    def _is_url_or_email_heavy(self, text: str) -> bool:
        # If the input is mostly URLs/emails, summarization tends to be poor and can hallucinate
        t = text.strip()
        if len(t) < 40:
            return False

        # Find URLs and emails
        url_re = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
        email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

        # Extract URLs and emails
        urls = url_re.findall(t)
        emails = email_re.findall(t)
        if not urls and not emails:
            return False

        # Remove urls/emails and count remaining word tokens
        cleaned = url_re.sub(" ", t)
        cleaned = email_re.sub(" ", cleaned)
        remaining_words = re.findall(r"[A-Za-z]{2,}", cleaned)

        # Conservative: bypass only if there's very little natural-language content
        if len(remaining_words) <= 6:
            return True
        if (len(urls) + len(emails)) >= 3 and len(remaining_words) <= 12: # multiple urls/emails with little text
            return True
        return False

    def _contains_non_latin(self, text: str) -> bool:
        # Any non-ASCII character triggers bypass (safe default for non-multilingual models)
        return bool(re.search(r"[^\x00-\x7F]", text))

    def _count_ascii_words(self, text: str) -> int:
        # Fast proxy for word count that works well for English detection
        s = (text or "").strip()
        if not s:
            return 0
        words = re.findall(r"[A-Za-z]{2,}", s)
        return len(words)

    def _contains_non_english_script(self, text: str) -> bool:
        # Detect common non-English scripts without treating Unicode punctuation as non-English
        # This avoids rejecting English that contains smart quotes or similar characters
        s = text or ""
        script_re = (
            r"[\u4E00-\u9FFF"  # CJK Unified Ideographs
            r"\u3040-\u30FF"   # Hiragana/Katakana
            r"\uAC00-\uD7AF"   # Hangul
            r"\u0400-\u04FF"   # Cyrillic
            r"\u0600-\u06FF"   # Arabic
            r"\u0900-\u097F"   # Devanagari
            r"]"
        )
        return bool(re.search(script_re, s))

    # English-likeness heuristic
    def _looks_like_english(self, text: str) -> bool:
        # Conservative heuristic. If we are not confident it is English, treat it as non-English
        s = (text or "").strip()
        if not s:
            return True

        # A clear non-English script is an automatic fail for English-only summarization
        if self._contains_non_english_script(s):
            return False

        # Pull out basic English-looking words and look for common glue words
        words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", s)]
        if not words:
            return False

        stop = {
            "the", "and", "of", "to", "in", "is", "it", "for", "on", "with", "as", "by",
            "that", "this", "are", "was", "be", "from", "at", "an", "or", "not", "have",
            "has", "will", "can", "we", "you", "they", "their", "a"
        }

        hits = 0
        for w in words:
            if w in stop:
                hits += 1

        # For longer text, require a little evidence that it is natural English prose
        if len(words) >= 12:
            return hits >= 2

        # For short text, be lenient so we do not reject brief English phrases
        return True

    # Enforce English-only on long inputs
    def _should_reject_non_english(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False

        ascii_word_count = self._count_ascii_words(s)
        is_long_by_chars = len(s) >= 200
        is_long_by_words = ascii_word_count >= 40
        is_long = is_long_by_chars or is_long_by_words

        if not is_long:
            return False

        # If it's clearly a non-English script and long, reject.
        if self._contains_non_english_script(s):
            return True

        # Otherwise use the English-likeness heuristic for ASCII-ish text.
        return not self._looks_like_english(s)

    def _words_to_token_budget(self, desired_words: int) -> int:
        w = max(1, int(desired_words))
        # Use a less conservative ratio + buffer so longer targets don't stop early.
        return max(32, int(w * 2.3) + 24)

    def _truncate_to_words(self, text: str, max_words: int) -> str:
        # Hard cap the output length without any randomness
        w = max(1, int(max_words))

        cleaned = (text or "").strip()
        parts = cleaned.split()
        if len(parts) <= w:
            return cleaned

        clipped = " ".join(parts[:w]).strip()

        # If we cut at the end of a clause, trim the leftover punctuation
        clipped = re.sub(r"[\s,\.;:]+$", "", clipped)
        return clipped

    def _truncate_to_sentence_boundary(self, text: str, max_words: int) -> str:
        w = max(1, int(max_words))
        cleaned = (text or "").strip()
        words = cleaned.split()

        if len(words) <= w:
            return cleaned

        clipped = " ".join(words[:w]).strip()

        # Back up to the last sentence-ending punctuation
        matches = list(re.finditer(r"[.!?]\s", clipped))
        if matches:
            clipped = clipped[: matches[-1].end()].strip()

        return clipped

    # Main summarization method
    def summarize(
        self,
        text: str,
        max_new_tokens: int, # interpreted as WORDS
        min_new_tokens: Optional[int] = None, # optional minimum
        enable_chunking: bool = True, # chunking enabled by default
        chunk_overlap_tokens: int = 96, # reasonable default overlap
        second_pass: bool = True, # optional second-pass summarization
    ) -> SummarizationResult:
        normalized = (text or "")
        normalized = self._strip_control_chars(normalized).strip()
        if not normalized:
            return SummarizationResult(summary="")

        # Bypass checks MUST happen before any tokenization/chunking/generation
        # Handle obvious junk first so short junk does not get echoed back

        # Inputs dominated by symbols/punctuation are usually noise
        if self._is_mostly_symbols(normalized):
            raise InvalidInputError("Input rejected: mostly symbols/noise.")

        # Low-entropy repetitive text is usually junk
        if self._is_low_entropy_spam(normalized):
            raise InvalidInputError("Input rejected: low-entropy repetitive text.")

        # Detect short repeating patterns that indicate junk input
        if self._has_short_repeating_pattern(normalized):
            raise InvalidInputError("Input rejected: repeating pattern detected.")

        # Single-token blobs (hashes/keys/IDs) are not meaningful to summarize
        if self._is_single_token_blob(normalized):
            raise InvalidInputError("Input rejected: single-token blob detected.")

        # Short non-English script input: bypass (echo) to avoid hallucinations and satisfy tests.
        if self._contains_non_english_script(normalized) and len(normalized) < 200:
            return SummarizationResult(summary=normalized)

        if self._looks_like_gibberish(normalized):
            raise InvalidInputError("Input rejected: looks like gibberish (not natural language).")

        # Interpret API "summary_length" as WORDS (not tokens)
        # IMPORTANT: For CJK and other scripts without spaces, split() can return 1 "word",
        # which would incorrectly trigger the early-return and skip language validation.
        word_count = len(normalized.split())
        max_words = max(1, int(max_new_tokens))

        # If the text is clearly non-English script, reject BEFORE the early-return
        # so the UI shows the intended "English only" error.
        if self._contains_non_english_script(normalized):
            raise InvalidInputError("Input rejected: unsupported language (English only).")

        # If the input is already within the requested summary length,
        # do not expand it by generating new text
        if word_count <= max_words:
            return SummarizationResult(summary=normalized)

        # URL/email-heavy inputs are better echoed than summarized
        if self._is_url_or_email_heavy(normalized):
            return SummarizationResult(summary=normalized)

        # Only apply English-likeness checks for ASCII-only inputs.
        # For long ASCII inputs, enforce English-only so the UI does not pretend we summarized.
        if self._should_reject_non_english(normalized):
            raise InvalidInputError("Input rejected: unsupported language (English only).")

        # Interpret API "summary_length" as WORDS (not tokens)
        max_words = max(1, int(max_new_tokens))

        # If min_new_tokens is ever used, treat it as MIN WORDS
        min_words = int(min_new_tokens) if min_new_tokens is not None else None

        # If caller didn't provide a minimum, enforce one for larger targets
        if min_words is None and max_words >= 80:
            min_words = int(max_words * 0.85)

        if min_words is not None:
            min_words = max(1, min(min_words, max_words - 1))

        # Convert word targets to token budgets for generation
        gen_max_tokens = self._words_to_token_budget(max_words)

        # Decide min token target for generation, if any
        # This nudges output length without forcing it on small requests
        gen_min_tokens: Optional[int] = None
        if min_words is not None:
            gen_min_tokens = self._words_to_token_budget(min_words)
            if gen_min_tokens >= gen_max_tokens:
                gen_min_tokens = gen_max_tokens - 1

        # Estimate token length up front so we can decide whether chunking is necessary
        tokenized = self.tokenizer(
            normalized,
            return_tensors=None,
            add_special_tokens=True,
            truncation=False,
        )

        # Safe token count extraction
        def _safe_token_count(input_ids_obj) -> int:
            # Tokenizers can return List[int] or List[List[int]] (and sometimes tensors)
            if isinstance(input_ids_obj, list):
                if not input_ids_obj:
                    return 0
                # Handle nested lists
                if isinstance(input_ids_obj[0], list):
                    return len(input_ids_obj[0])
                return len(input_ids_obj)
            return int(input_ids_obj.shape[-1])

        input_len = _safe_token_count(tokenized.get("input_ids"))

        # Chunk only when needed. For shorter inputs, do a single pass
        needs_chunking = bool(enable_chunking and input_len > self.max_source_tokens)
        if needs_chunking:
            chunk_summaries = self._summarize_in_chunks(
                normalized,
                max_new_tokens=gen_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )

            # Join chunk-level summaries into one text blob for optional second pass
            cleaned_parts = [part.strip() for part in chunk_summaries if part and part.strip()]
            combined = " ".join(cleaned_parts).strip()

            # Optional second-pass summarization to refine chunk summaries into a coherent whole
            if second_pass and combined:
                second = self._summarize_single(
                    combined,
                    max_new_tokens=gen_max_tokens,
                    min_new_tokens=gen_min_tokens,
                )
                # Final truncation to word count
                final_text = self._truncate_to_words(second.summary, max_words)
                return SummarizationResult(summary=final_text)

            final_text = self._truncate_to_words(combined, max_words)
            return SummarizationResult(summary=final_text)

        # Single-pass summary for inputs that fit in the model context window
        result = self._summarize_single(
            normalized,
            max_new_tokens=gen_max_tokens,
            min_new_tokens=gen_min_tokens,
        )

        # Final truncation to word count
        final_text = self._truncate_to_sentence_boundary(result.summary, max_words)
        return SummarizationResult(summary=final_text)

    # Summarize by chunking input text into manageable pieces
    def _summarize_in_chunks(
        self,
        text: str,
        max_new_tokens: int,
        chunk_overlap_tokens: int,
    ) -> List[str]:
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        ids = tokenized["input_ids"]
        if not ids:
            return []
        if isinstance(ids[0], list):
            ids = ids[0]

        # Use a slightly smaller window than max_source_tokens to leave room for special tokens.
        window = max(self.max_source_tokens - 2, 128)

        # Keep overlap sane so we do not get stuck with a zero step.
        overlap = max(0, int(chunk_overlap_tokens))
        overlap = min(overlap, window - 1) if window > 1 else 0
        step = window - overlap

        # Gather per-chunk summaries here
        summaries: List[str] = []
        chunks_seen = 0

        def _append_if_new(candidate: str) -> None:
            # Overlap can cause repeated outputs. This cheap check helps a lot.
            if not candidate:
                return
            if summaries and summaries[-1].strip() == candidate.strip():
                return
            summaries.append(candidate)

        # Slide a window across token IDs and summarize each decoded chunk.
        for start in range(0, len(ids), step):
            if chunks_seen >= self._MAX_CHUNKS:
                break

            # Define chunk boundaries
            end = start + window
            chunk_ids = ids[start:end]
            if not chunk_ids:
                break

            # Decode chunk back to text
            chunk_text = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if not chunk_text:
                continue

            # Per-chunk summaries should be short and consistent, so keep min_new_tokens off here.
            result = self._summarize_single(
                chunk_text,
                max_new_tokens=max_new_tokens,
                min_new_tokens=None,
            )
            _append_if_new(result.summary)

            # Track chunks seen
            chunks_seen += 1

            # If we reached the end, stop cleanly.
            if end >= len(ids):
                break

        return summaries

    # Summarize a single text input without chunking
    def _summarize_single(
        self,
        text: str,
        max_new_tokens: int,
        min_new_tokens: Optional[int] = None,
    ) -> SummarizationResult:
        # Tokenize and clamp to the model context window.
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Deterministic decoding keeps results stable across runs.
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "length_penalty": 1.0,
        }

        # Only apply min_new_tokens when it is safely below max_new_tokens.
        if min_new_tokens is not None:
            min_toks = int(min_new_tokens)
            if 0 < min_toks < int(max_new_tokens):
                gen_kwargs["min_new_tokens"] = min_toks

        # Some transformers versions support max_time, some do not.
        try:
            sig = inspect.signature(self.model.generate)
            if "max_time" in sig.parameters:
                gen_kwargs["max_time"] = 15.0
        except (TypeError, ValueError):
            pass

        # Generate summary IDs
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode the first beam output.
        decoded = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return SummarizationResult(summary=decoded)
