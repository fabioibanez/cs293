"""
Annotates student utterances from math classroom transcripts.
Reads from annotated_single_utterances_new.csv, writes to annotations_{model}.csv.

Supported models:
  - gpt-5-mini    (OpenAI, requires OPENAI_API_KEY)
  - qwen3:8b      (Ollama, requires local ollama serve)

- Offering math help: Student A offering a new solution/reason to student B.
  Student A helping out student B, even if prompted by the teacher, still counts.

- Successful uptake: Student A responding to student B's existing plan/solution/idea.
  Agreeing/disagreeing with a classmate's solution (basically interacting with it in some way).

Usage:
  python model.py --model gpt-5-mini              # OpenAI
  python model.py --model qwen3:8b                # Ollama (local)
  python model.py --model gpt-5-mini --limit 5    # Test with 5 rows

Environment variables (set the ones you need):
  OPENAI_API_KEY      — for OpenAI models
  ANTHROPIC_API_KEY   — for Claude models
  GOOGLE_API_KEY      — for Gemini models

Requirements:
  pip install openai anthropic google-genai python-dotenv
"""

import argparse
import csv
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY, etc. from .env so you don't have to export manually

import openai
from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV = "annotated_single_utterances_new.csv"
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = """\
You are annotating student utterances from math classroom transcripts.
For each utterance, you must label it with exactly three binary values (0 or 1) for:

1. OFFERING MATH HELP: The student is offering mathematical help to someone else:
   - Student A offering a new solution or reason to student B
   - Student A helping out student B (even if the teacher prompted them)

2. SUCCESSFUL UPTAKE: The student is responding to a classmate's existing idea:
   - Student A responding to Student B's plan/solution/idea
   - Agreeing or disagreeing with a classmate's solution
   - Interacting with or building on what another student said

3. ASKING FOR MORE INFORMATION: The student is asking a question. This includes:
   - Questions about classroom procedures or management
   - Questions directed to the teacher OR another student
   - Any request for clarification or additional information
   - There must explicitly be a question asked by the student.

Consider the pre-utterance and post-utterance context to understand the \
significance of the focal utterance.
Multiple labels can be 1 for a single utterance (e.g., 1,1,0).
Respond with ONLY three digits separated by commas, in this exact order: \
Offering math help, Successful uptake, Asking for More Information.
Example: 1,0,0 or 0,1,1 or 0,0,1"""

USER_PROMPT_TEMPLATE = """\
PRE-UTTERANCE CONTEXT:
{pre_context}

FOCAL STUDENT UTTERANCE (the one to annotate):
{student_utterance}

POST-UTTERANCE CONTEXT:
{post_context}

Label this student utterance. Respond with only three digits (0 or 1) \
separated by commas: Offering math help, Successful uptake, Asking for More Information."""

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

PROVIDER_PREFIXES = {
    "gpt":    "openai",
    "o1":     "openai",
    "o3":     "openai",
    "o4":     "openai",
    "claude": "anthropic",
    "gemini": "google",
}


def detect_provider(model: str) -> str:
    """Detect the API provider from the model name."""
    for prefix, provider in PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    return "ollama"


# ---------------------------------------------------------------------------
# Provider-specific clients
# ---------------------------------------------------------------------------

def _query_openai(model: str, system: str, user: str) -> str:
    """Query OpenAI using the official client."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _query_anthropic(model: str, system: str, user: str) -> str:
    """Query Anthropic using the official client."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=32,
        system=system,
        messages=[
            {"role": "user", "content": user},
        ],
    )
    return response.content[0].text.strip()


def _query_google(model: str, system: str, user: str) -> str:
    """Query Google Gemini using the official client."""
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=genai.types.GenerateContentConfig(
            system_instruction=system,
        ),
    )
    return response.text.strip()


def _query_ollama(model: str, system: str, user: str) -> str:
    """Query a local Ollama instance via the OpenAI-compatible endpoint."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = openai.OpenAI(base_url=f"{host}/v1", api_key="ollama")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

_PROVIDER_DISPATCH = {
    "openai":    _query_openai,
    "anthropic": _query_anthropic,
    "google":    _query_google,
    "ollama":    _query_ollama,
}


def query_model(model: str, system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to any supported LLM and return the response text.

    The provider is auto-detected from the model name. Supported providers:
    OpenAI, Anthropic, Google (Gemini), and Ollama (local).
    """
    provider = detect_provider(model)
    return _PROVIDER_DISPATCH[provider](model, system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Annotation logic
# ---------------------------------------------------------------------------

def build_user_prompt(pre_context: str, utterance: str, post_context: str) -> str:
    """Build the user prompt from the three context segments."""
    return USER_PROMPT_TEMPLATE.format(
        pre_context=pre_context,
        student_utterance=utterance,
        post_context=post_context,
    )


def parse_annotation(response: str) -> tuple[int, int, int]:
    """Parse the model response into (offering, uptake, asking) — all 0 or 1."""
    match = re.search(r"\b([01])\s*,\s*([01])\s*,\s*([01])\b", response)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    digits = re.findall(r"[01]", response)
    if len(digits) >= 3:
        return int(digits[0]), int(digits[1]), int(digits[2])
    return 0, 0, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate classroom utterances using any LLM provider."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model name — provider is auto-detected (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (for testing)",
    )
    parser.add_argument(
        "--input", type=str, default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: annotations_{model}.csv)",
    )
    args = parser.parse_args()

    provider = detect_provider(args.model)
    safe_model_name = args.model.replace("/", "-").replace(":", "-")
    output_csv = args.output or f"annotations_{safe_model_name}.csv"

    # Load input rows
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            utterance = row.get("Student Utterance", "").strip()
            if not utterance and not row.get("Pre-Utterance Context", "").strip():
                continue
            rows.append(row)

    if args.limit:
        rows = rows[: args.limit]

    print(f"Provider : {provider}")
    print(f"Model    : {args.model}")
    print(f"Rows     : {len(rows)}")
    print(f"Output   : {output_csv}")
    print()

    # Write annotations
    fieldnames = [
        "Pre-Utterance Context",
        "Student Utterance",
        "Post-Utterance Context",
        "Offering math help",
        "Successful uptake",
        "Asking for More Information",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            pre = row.get("Pre-Utterance Context", "")
            utterance = row.get("Student Utterance", "")
            post = row.get("Post-Utterance Context", "")

            print(f"[{i + 1}/{len(rows)}] {utterance[:70]}...")

            try:
                user_prompt = build_user_prompt(pre, utterance, post)
                response = query_model(args.model, SYSTEM_PROMPT, user_prompt)
                offering, uptake, asking = parse_annotation(response)
                print(f"  -> {offering},{uptake},{asking}")
            except Exception as e:
                print(f"  ERROR: {e}")
                offering, uptake, asking = 0, 0, 0

            writer.writerow({
                "Pre-Utterance Context": pre,
                "Student Utterance": utterance,
                "Post-Utterance Context": post,
                "Offering math help": offering,
                "Successful uptake": uptake,
                "Asking for More Information": asking,
            })
            out_f.flush()

            if i < len(rows) - 1:
                time.sleep(0.3)

    print(f"\nDone. Saved to {output_csv}")


if __name__ == "__main__":
    main()
