from pydantic import BaseModel, ConfigDict, Field, field_validator

# Request schema for text summarization
class SummarizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "This is a long text input that should be summarized into something concise.",
                    "summary_length": 50, # Desired summary length in words
                }
            ]
        }
    )
    # Input text to be summarized
    text: str = Field(
        ...,
        min_length=1, # Minimum length enforced by validator
        description=(
            "Input text to summarize. Must contain at least 10 non-whitespace characters after trimming. Unicode supported."
        ),
        examples=["This is a long text input that should be summarized into something concise."],
    )
    # Desired length of the summary in words
    summary_length: int = Field(
        ...,
        ge=10, # Minimum summary length
        le=300, # Maximum summary length
        description=(
            "Desired summary length expressed as an approximate number of words in the returned summary. "
            "Internally converted to a token budget for generation and then truncated back to the requested word cap."
        ),
        examples=[50],
    )
    # Validator to ensure text is not just whitespace and meets minimum character requirement
    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, value: str) -> str:
        stripped = value.strip() # Remove leading/trailing whitespace
        if not stripped:
            raise ValueError("Text must not be empty or whitespace-only.")
        if len(stripped) < 10: # Minimum non-whitespace character requirement
            raise ValueError("Text must contain at least 10 non-whitespace characters.")
        return stripped

# Response schema for text summarization
class SummarizeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={ # Extra schema information for documentation
            "examples": [
                {"summary": "A concise, model-generated summary of the provided text."},
            ]
        }
    )

    summary: str
