# CS293

# Requirements:
You should have the requirements in `requirements.txt` isntalled.
This repo expects that you have a `data` folder like:

```
data
├── ICPSR_36095
├── ncte_single_utterances.csv
├── paired_annotations.csv
├── student_reasoning.csv
└── transcript_metadata.csv
```

## Running Tests

This project uses [pytest](https://docs.pytest.org/) for testing.

To run all tests:
```bash
pytest -v
```

You can run specific types of tests using `-k` to match test names, for example:
```bash
pytest -v -k unit          # Run only unit tests (no API calls)
pytest -v -k live_ollama   # Run only live Ollama tests (local inference)
pytest -v -k live_openai   # Run only live OpenAI tests (requires OPENAI_API_KEY)
pytest -v -k live          # Run all live model tests (Ollama and OpenAI)
```

**Requirements:**

- Ollama live tests require `ollama serve` running and the appropriate model(s) pulled.
- OpenAI live tests require you to set your `OPENAI_API_KEY` in the environment.

All tests are in `test_model.py`.


[Edu-Convokit annotations](https://drive.google.com/drive/folders/1dw6sUdXGnXsGCU5HUjJQKhjUrcBQF_wk?usp=drive_link)
