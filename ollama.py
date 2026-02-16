"""
Makes API calls to Ollama to generate annotations for annotated_single_utterances_new.csv.
Outputs ollama_annotations.csv with the same context columns plus 3 annotation columns.

Constructs (with definitions for the model):
- Asking for More Information: Questions about classroom procedures or management count.
  The question may be directed to either the teacher or another student.

- Offering math help: Student A offering a new solution/reason to student B.
  Student A helping out student B, even if prompted by the teacher, still counts.

- Successful uptake: Student A responding to student B's existing plan/solution/idea.
  Agreeing/disagreeing with a classmate's solution (basically interacting with it in some way).

Usage:
  python ollama.py                    # Annotate all rows
  python ollama.py --limit 5         # Test with first 5 rows
  python ollama.py --model mistral    # Use a different model

Requires: Ollama running locally (ollama serve). Pull model first: ollama pull llama3.2
"""

import argparse
import csv
import json
import re
import time
import urllib.request
import urllib.error

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"  # NOTE: CAN CHANGE THIS TO BE A DIFFERENT MODEL IF YOU'D PREFER
INPUT_CSV = "annotated_single_utterances_new.csv"
OUTPUT_CSV = "ollama_annotations.csv"

SYSTEM_PROMPT = """You are annotating student utterances from math classroom transcripts.
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

Consider the pre-utterance and post-utterance context to understand the significance of the focal utterance.
Multiple labels can be 1 for a single utterance (e.g., 1,1,0).
Respond with ONLY three digits separated by commas, in this exact order: Offering math help, Successful uptake, Asking for More Information.
Example: 1,0,0 or 0,1,1 or 0,0,1"""


def call_ollama(
    pre_context: str, student_utterance: str, post_context: str, model: str
) -> str:
    """Call Ollama API and return the raw response content."""
    user_content = f"""PRE-UTTERANCE CONTEXT:
{pre_context}

FOCAL STUDENT UTTERANCE (the one to annotate):
{student_utterance}

POST-UTTERANCE CONTEXT:
{post_context}

Label this student utterance. Respond with only three digits (0 or 1) separated by commas: Offering math help, Successful uptake, Asking for More Information."""

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode())
        return result.get("message", {}).get("content", "").strip()


def parse_annotation(response: str) -> tuple[int, int, int]:
    """Parse the model response into (offering, uptake, asking) — all 0 or 1."""
    match = re.search(r"\b([01])\s*[,]\s*([01])\s*[,]\s*([01])\b", response)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    # Fallback: look for any three 0/1 digits
    digits = re.findall(r"[01]", response)
    if len(digits) >= 3:
        return (int(digits[0]), int(digits[1]), int(digits[2]))
    # Default to 0,0,0 if unparseable
    return (0, 0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate classroom utterances via Ollama")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N rows (for testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utterance = row.get("Student Utterance", "").strip()
            if not utterance and not row.get("Pre-Utterance Context", "").strip():
                continue
            rows.append(row)

    if args.limit:
        rows = rows[: args.limit]
        print(f"Limited to first {args.limit} rows")

    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")
    print(f"Using model: {args.model}")
    print(f"Writing to: {OUTPUT_CSV}")
    print()

    fieldnames = [
        "Pre-Utterance Context",
        "Student Utterance",
        "Post-Utterance Context",
        "Offering math help",
        "Successful uptake",
        "Asking for More Information",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            pre = row.get("Pre-Utterance Context", "")
            utterance = row.get("Student Utterance", "")
            post = row.get("Post-Utterance Context", "")

            print(f"[{i + 1}/{len(rows)}] Annotating: {utterance[:60]}...")

            try:
                response = call_ollama(pre, utterance, post, args.model)
                offering, uptake, asking = parse_annotation(response)
                print(f"  -> {response} -> parsed as offering={offering}, uptake={uptake}, asking={asking}")
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
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

            # Brief pause to avoid overwhelming the API
            if i < len(rows) - 1:
                time.sleep(0.5)

    print(f"\nDone. Output saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
