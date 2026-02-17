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

Your task is to label the focal student utterance using EXACTLY THREE binary values (0 or 1), in this order:

1. OFFERING MATH HELP
2. SUCCESSFUL UPTAKE
3. ASKING FOR MORE INFORMATION

Each value must be either 0 or 1. Multiple labels may be 1 at the same time.

--------------------------------
LABEL DEFINITIONS
--------------------------------

1. OFFERING MATH HELP (1 or 0)

Label 1 if the student is offering mathematical assistance or contributing a solution step intended to help another student understand or proceed.

This includes:
- Offering a solution step or explanation
- Suggesting what to do next in a solution
- Explaining reasoning to help another student
- Offering to help or walk someone through the problem
- Helping even if prompted by the teacher

This does NOT include:
- Merely stating an answer without helping context
- Responding only to the teacher without helping peers
- Asking questions

Key idea: The student is helping someone move forward mathematically.


2. SUCCESSFUL UPTAKE (1 or 0)

Label 1 if the student directly engages with or responds to another student's prior idea, solution, or reasoning.

This includes:
- Agreeing or disagreeing with a classmate's idea
- Building on or modifying another student's solution
- Referencing or reacting to what another student just said
- Continuing a peer's reasoning or plan

This does NOT include:
- Starting a completely new idea unrelated to peers
- Only responding to the teacher
- Asking unrelated questions
- Simply stating an answer without helping context

Key idea: The student is interacting with or building on another student's contribution.


3. ASKING FOR MORE INFORMATION (1 or 0)

Label 1 if the focal utterance explicitly asks a question.

This includes:
- Questions about math content
- Questions about procedures or instructions
- Questions directed to teacher or peers
- Requests for clarification or explanation

This does NOT include:
- Statements that imply confusion without a question

Key idea: A question must be explicitly asked.


--------------------------------
EXAMPLE (OFFERING HELP)
--------------------------------

PRE-UTTERANCE CONTEXT:
[teacher] Okay, everyone should be solving for x after distributing the 3. What did you get so far?
[student] I got 3x plus 6 equals 18, but I don't know what to do after.
[teacher] Good, that's correct so far. What should we do after that?
[student] I'm not sure.

FOCAL STUDENT UTTERANCE:
[student] I think we can subtract 6 from each side.

POST-UTTERANCE CONTEXT:
[teacher] Go ahead and explain your thinking so everyone can follow.
[student] Subtract 6 from both sides to get 3x equals 12.
[teacher] And what happens after that?
[student] Then you divide both sides by 3, so x equals 4.
[multiple students] Ohhh, okay.

The focal student offers a solution step to help classmates proceed → label: 1,0,0.

--------------------------------
EXAMPLE (SUCCESSFUL UPTAKE)
--------------------------------

PRE-UTTERANCE CONTEXT:
[multiple students] [Crosstalk]
[teacher] Okay – someone tell me how they got started.
[student] I divided both sides by 5 cause I wanted to get x by itself.
[teacher] Okay, student A divided both sides by 5. Did anyone else try that?
[student] I think there's still a number being added, so dividing first might not work.

FOCAL STUDENT UTTERANCE:
[student] No you have to subtract 3 first before you divide by 5.

POST-UTTERANCE CONTEXT:
[teacher] So subtraction comes before division here?
[student] Yes, because we want x alone before dividing.
[teacher] Does that make sense to everyone?
[multiple students] Yeah.

The focal student builds directly on a classmate's idea and continues the reasoning → label: 0,1,0.

--------------------------------
EXAMPLE (ASKING FOR MORE INFORMATION)
--------------------------------

[teacher] Remember to combine like terms before solving the equation.
[student] I combined these two but I'm not sure what happens next.
[teacher] Think about what operations are left after combining terms.
[student] Umm I'm not sure.

FOCAL STUDENT UTTERANCE:
[student] Do we subtract 8 from both sides now?

POST-UTTERANCE CONTEXT:
[teacher] Exactly, Student J. We can subtract 8 from both sides at this point.
[student] And then we divide after that?
[teacher] Good, keep going.
[student] I got that too.

The focal student explicitly asks a question to clarify the next mathematical step → label: 0,0,1.


--------------------------------
ANNOTATION RULES
--------------------------------

• Use pre- and post-context to interpret the focal utterance.
• Focus on the communicative function of the utterance.
• Multiple labels can be 1 simultaneously.
• Do not output explanations.

--------------------------------
OUTPUT FORMAT
--------------------------------

Respond ONLY with three digits separated by commas:

Offering math help, Successful uptake, Asking for more information

Examples:
1,0,0
0,1,1
0,0,1"""

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
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
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
    "openai": _query_openai,
    "anthropic": _query_anthropic,
    "google": _query_google,
    "ollama": _query_ollama,
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
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name — provider is auto-detected (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for testing)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
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

            writer.writerow(
                {
                    "Pre-Utterance Context": pre,
                    "Student Utterance": utterance,
                    "Post-Utterance Context": post,
                    "Offering math help": offering,
                    "Successful uptake": uptake,
                    "Asking for More Information": asking,
                }
            )
            out_f.flush()

            if i < len(rows) - 1:
                time.sleep(0.3)

    print(f"\nDone. Saved to {output_csv}")


if __name__ == "__main__":
    main()
