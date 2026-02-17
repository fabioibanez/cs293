"""
Tests for model.py — validates provider detection, parsing, and live API queries.

Usage:
  pytest test_model.py -v                  # Run all tests (-v = verbose, shows each test name)
  pytest test_model.py -v -k unit          # Run only unit tests (no API calls)
  pytest test_model.py -v -k live_ollama   # Run only live Ollama tests
  pytest test_model.py -v -k live_openai    # Run only live OpenAI tests (needs OPENAI_API_KEY)
  pytest test_model.py -v -k live          # Run all live tests (Ollama + OpenAI)

Requires:
  - Ollama: ollama serve + qwen3:8b (or another model) pulled.
  - OpenAI: OPENAI_API_KEY set in environment.
"""

import pytest

from model import (
    SYSTEM_PROMPT,
    build_user_prompt,
    detect_provider,
    parse_annotation,
    query_model,
)

# ---------------------------------------------------------------------------
# Unit tests — no API calls, pure logic
# ---------------------------------------------------------------------------


class TestDetectProvider:
    """Test that model names route to the correct provider."""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("gpt-5-mini", "openai"),
            ("qwen3:8b", "ollama"),
        ],
    )
    def test_unit_detect_provider(self, model, expected):
        assert detect_provider(model) == expected


class TestParseAnnotation:
    """Test that model responses are correctly parsed into (offering, uptake, asking)."""

    def test_unit_clean_response(self):
        assert parse_annotation("0,0,1") == (0, 0, 1)
        assert parse_annotation("1,0,0") == (1, 0, 0)
        assert parse_annotation("1,1,1") == (1, 1, 1)
        assert parse_annotation("0,0,0") == (0, 0, 0)

    def test_unit_response_with_spaces(self):
        assert parse_annotation("0, 1, 0") == (0, 1, 0)
        assert parse_annotation("1 , 0 , 1") == (1, 0, 1)

    def test_unit_response_with_extra_text(self):
        assert parse_annotation("The answer is 1,0,1") == (1, 0, 1)
        assert parse_annotation("Based on the context: 0,1,0") == (0, 1, 0)

    def test_unit_fallback_digit_extraction(self):
        assert parse_annotation("Offering: 1, Uptake: 0, Asking: 1") == (1, 0, 1)

    def test_unit_unparseable_defaults_to_zeros(self):
        assert parse_annotation("I don't know") == (0, 0, 0)
        assert parse_annotation("") == (0, 0, 0)


class TestBuildUserPrompt:
    """Test that user prompts are correctly assembled."""

    def test_unit_prompt_contains_all_parts(self):
        prompt = build_user_prompt(
            pre_context="Teacher: What do you think?",
            utterance="I think the answer is 4.",
            post_context="Teacher: Can you explain why?",
        )
        assert "Teacher: What do you think?" in prompt
        assert "I think the answer is 4." in prompt
        assert "Teacher: Can you explain why?" in prompt
        assert "PRE-UTTERANCE CONTEXT" in prompt
        assert "FOCAL STUDENT UTTERANCE" in prompt
        assert "POST-UTTERANCE CONTEXT" in prompt

    def test_unit_prompt_with_empty_context(self):
        prompt = build_user_prompt("", "Hello", "")
        assert "Hello" in prompt


# ---------------------------------------------------------------------------
# Live tests — require Ollama running locally with a model pulled
# ---------------------------------------------------------------------------


class TestLiveOllama:
    """Live integration tests against a local Ollama instance."""

    MODEL = "qwen3:8b"

    def test_live_query_returns_string(self):
        """Verify query_model returns a non-empty string."""
        response = query_model(self.MODEL, "You are a helpful assistant.", "Say hello.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_live_annotation_format(self):
        """Verify the model returns a parseable annotation."""
        user_prompt = build_user_prompt(
            pre_context="Teacher: Who can help Maria with problem 5?",
            utterance="I can help. You need to multiply 3 times 4 first.",
            post_context="Maria: Oh okay, so it's 12?",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        offering, uptake, asking = parse_annotation(response)

        assert offering in (0, 1)
        assert uptake in (0, 1)
        assert asking in (0, 1)

    def test_live_asking_question(self):
        """A student clearly asking a question should be labeled as asking."""
        user_prompt = build_user_prompt(
            pre_context="Teacher: Open your books to page 42.",
            utterance="What page did you say?",
            post_context="Teacher: Page 42.",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        _, _, asking = parse_annotation(response)
        assert asking == 1, f"Expected asking=1 for a clear question, got response: {response}"

    def test_live_offering_help(self):
        """A student offering a math solution should be labeled as offering help."""
        user_prompt = build_user_prompt(
            pre_context="Teacher: Can someone help James with this fraction problem?",
            utterance="You need to find a common denominator. Try multiplying the bottom numbers.",
            post_context="James: Oh, so I multiply 3 and 4?",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        offering, _, _ = parse_annotation(response)
        assert offering == 1, f"Expected offering=1 for math help, got response: {response}"

    def test_live_successful_uptake(self):
        """A student responding to a classmate's idea should be labeled as uptake."""
        user_prompt = build_user_prompt(
            pre_context="Student B: I think the answer is 24 because 6 times 4 is 24.",
            utterance="I agree with her, 6 times 4 is definitely 24.",
            post_context="Teacher: Good, you're both right.",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        _, uptake, _ = parse_annotation(response)
        assert uptake == 1, f"Expected uptake=1 for responding to classmate, got response: {response}"


# ---------------------------------------------------------------------------
# Live tests — require OPENAI_API_KEY and bill to your account
# ---------------------------------------------------------------------------


class TestLiveOpenAI:
    """Live integration tests against the OpenAI API (gpt-5-mini)."""

    MODEL = "gpt-5-mini"

    def test_live_openai_query_returns_string(self):
        """Verify query_model returns a non-empty string."""
        response = query_model(self.MODEL, "You are a helpful assistant.", "Say hello.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_live_openai_annotation_format(self):
        """Verify the model returns a parseable annotation."""
        user_prompt = build_user_prompt(
            pre_context="Teacher: Who can help Maria with problem 5?",
            utterance="I can help. You need to multiply 3 times 4 first.",
            post_context="Maria: Oh okay, so it's 12?",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        offering, uptake, asking = parse_annotation(response)

        assert offering in (0, 1)
        assert uptake in (0, 1)
        assert asking in (0, 1)

    def test_live_openai_asking_question(self):
        """A student clearly asking a question should be labeled as asking."""
        user_prompt = build_user_prompt(
            pre_context="Teacher: Open your books to page 42.",
            utterance="What page did you say?",
            post_context="Teacher: Page 42.",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        _, _, asking = parse_annotation(response)
        assert asking == 1, f"Expected asking=1 for a clear question, got response: {response}"

    def test_live_openai_offering_help(self):
        """A student offering a math solution should be labeled as offering help."""
        user_prompt = build_user_prompt(
            pre_context="Teacher: Can someone help James with this fraction problem?",
            utterance="You need to find a common denominator. Try multiplying the bottom numbers.",
            post_context="James: Oh, so I multiply 3 and 4?",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        offering, _, _ = parse_annotation(response)
        assert offering == 1, f"Expected offering=1 for math help, got response: {response}"

    def test_live_openai_successful_uptake(self):
        """A student responding to a classmate's idea should be labeled as uptake."""
        user_prompt = build_user_prompt(
            pre_context="Student B: I think the answer is 24 because 6 times 4 is 24.",
            utterance="I agree with her, 6 times 4 is definitely 24.",
            post_context="Teacher: Good, you're both right.",
        )
        response = query_model(self.MODEL, SYSTEM_PROMPT, user_prompt)
        _, uptake, _ = parse_annotation(response)
        assert uptake == 1, f"Expected uptake=1 for responding to classmate, got response: {response}"
