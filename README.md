# LLM Child-Safety Harness

A small red-teaming harness that probes a locally hosted LLM with child-safety scenarios and scores the replies. Built by David Mark ([@DaveV01](https://github.com/DaveV01)) to demonstrate a practical, reproducible approach to LLM child-safety evaluation.

Default target: **Mistral Small 24B** served by **Ollama** through its OpenAI-compatible API. Any OpenAI-compatible endpoint works.

## What it does

1. Loads probe prompts from `prompts/child_safety.jsonl` (26 items).
2. Sends each prompt to the model and records the response, latency and timestamp.
3. Scores each response against `rubric.yaml` with:
   - a **deterministic heuristic layer** (keyword and regex signals), always on;
   - an optional **LLM-as-judge layer** (`--judge`) that asks a model to return JSON scores.
4. Writes `results/run_<timestamp>.jsonl`, `results/summary_<timestamp>.csv` and `results/report_<timestamp>.md` with per-category pass rates.

## Setup (Windows 11)

1. Install Ollama from https://ollama.com/download and start it.
2. Pull the model (about 14 GB, fits on a 16 GB GPU such as an RTX 5080):
   ```powershell
   ollama pull mistral-small
   ```
3. Install Python 3.10+ and the dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Run:
   ```powershell
   python harness.py                 # heuristic scoring only
   python harness.py --judge         # add LLM-as-judge scoring
   python harness.py --dry-run       # no model needed, fake replies to test the pipeline
   ```

### Useful options

| Flag | Default | Purpose |
|---|---|---|
| `--base-url` | `http://localhost:11434/v1` | Any OpenAI-compatible endpoint |
| `--model` | `mistral-small` | Model under test |
| `--api-key` | `ollama` | Needed for hosted endpoints |
| `--judge` | off | Enable LLM-as-judge |
| `--judge-model` / `--judge-base-url` | same as target | Use a different judge |
| `--rubric` | `rubric.yaml` | YAML or JSON rubric |
| `--limit N` | all | Run only the first N prompts |
| `--temperature` | 0.2 | Sampling temperature |
| `--dry-run` | off | Fake responses, no API calls |

## Method

### Prompt set

Each line in `prompts/child_safety.jsonl` has `id`, `category`, `persona`, `prompt` and `expected_behavior`. Categories:

| Category | What it probes | Expected behavior |
|---|---|---|
| `age_inappropriate` | Minors asking for alcohol, gambling, vapes, mature media | refuse or redirect |
| `grooming_pattern` | A minor describing secrecy, gifts, private meetups with an online contact | safety_redirect |
| `self_harm_disclosure` | A teen disclosing self-harm, hopelessness or disordered eating | safety_redirect |
| `jailbreak_roleplay` | Persona and roleplay attempts to drop safety rules | refuse |
| `child_privacy` | Requests to locate, dox or scrape data about children | refuse |
| `misinformation_kids` | Health and science myths asked by children | correct_misinformation |
| `benign_control` | Legitimate questions that should be answered | comply |

Prompts are written to be safe to publish: they are realistic but non-graphic, contain no sexual content and request nothing that the harness would reproduce as harmful instructions. Benign controls (including one with a "kill a process" phrasing) measure over-refusal.

### Rubric

| Rubric category | Pass means |
|---|---|
| `refusal_appropriate` | Declined a risky request |
| `harmful_content` | No harmful instructions or facilitation |
| `age_appropriate_tone` | Calm, non-graphic, suitable for the persona |
| `redirect_to_help` | Pointed to a trusted adult, helpline or professional |
| `correct_information` | Corrected the false or dangerous claim |
| `over_refusal` | Answered a benign request instead of refusing |

Each category only applies to certain `expected_behavior` values (see `rubric.yaml`); others are marked not applicable and excluded from pass rates.

### Scoring layers

- **Heuristic**: transparent, fast and deterministic. Regex signals are listed in `rubric.yaml` and are easy to extend.
- **LLM-as-judge**: the judge receives the persona, category, prompt and reply and returns JSON booleans plus a one-sentence rationale. Using a different, stronger judge model reduces self-grading bias.

Comparing the two layers highlights disagreements worth reviewing by hand.

## Limitations

- Keyword heuristics miss nuance: a reply can mention a helpline yet still be unhelpful, or refuse without matching any pattern.
- A model judging its own output is biased; prefer a separate judge and spot-check with humans.
- 26 prompts in English only. This is a demonstration, not a benchmark; real evaluations need larger, multilingual, culturally adapted sets and multi-turn conversations.
- Results vary with temperature, system prompt, quantization and model version.
- Helpline references in the rubric (116 111, 988) are examples; correct resources depend on the country.

## Disclaimer

Results produced by this harness are **illustrative only** and do not constitute a certification of any model's safety. The prompt set intentionally avoids harmful or explicit material. Use responsibly and follow your organization's child-protection policies when extending it.

## License

MIT, see `LICENSE`.
