from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    model_id: str = Field(default="facebook/bart-large-cnn", alias="MODEL_ID")

    # Concurrency and safety
    max_concurrent_requests: int = Field(default=2, ge=1, alias="MAX_CONCURRENT_REQUESTS")

    # Chunking behavior
    enable_chunking: bool = Field(default=True, alias="ENABLE_CHUNKING")
    chunk_overlap_tokens: int = Field(default=96, ge=0, alias="CHUNK_OVERLAP_TOKENS")
    second_pass_summarization: bool = Field(default=True, alias="SECOND_PASS_SUMMARIZATION")
    max_input_tokens: int | None = Field(default=None, alias="MAX_INPUT_TOKENS")
    max_input_chars: int = Field(default=20000, ge=1, alias="MAX_INPUT_CHARS")

    # Output length limits (tokens)
    min_summary_tokens: int = Field(default=10, ge=1, alias="MIN_SUMMARY_TOKENS")
    max_summary_tokens: int = Field(default=300, ge=1, alias="MAX_SUMMARY_TOKENS")

    # Startup behavior
    warmup_enabled: bool = Field(default=True, alias="WARMUP_ENABLED")
    request_timeout_seconds: int = Field(default=120, ge=1, alias="REQUEST_TIMEOUT_SECONDS")

    @model_validator(mode="after")
    def _check_bounds(self) -> "Settings":
        if self.min_summary_tokens > self.max_summary_tokens:
            raise ValueError(
                f"MIN_SUMMARY_TOKENS ({self.min_summary_tokens}) must be <= "
                f"MAX_SUMMARY_TOKENS ({self.max_summary_tokens})."
            )
        return self


settings = Settings()
