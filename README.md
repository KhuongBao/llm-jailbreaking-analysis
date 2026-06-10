# LLM Jailbreaking Analysis

Compares how censored vs uncensored attacker models jailbreak target LLMs, classifies the strategies they use, and tests whether successful prompts transfer to other models.

**Full write-up:** See [report.pdf](report.pdf) for methodology, results, and analysis.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (models run in 4-bit quantization)
- Hugging Face account with access to [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench) dataset, and [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [HarmBench-Mistral-7b-val-cls](https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls) models

```bash
pip install -r requirements.txt
```

## Setup

Set `SAVE_PATH` in `utils.py` to a directory where results and model caches will be stored:

```python
SAVE_PATH = "/path/to/results/"
```

## Research Questions


| Script | Research Question (Simplified)|
|--------|----------|
| `rq1_asr.py` | What is the attack success rate (ASR) of censored vs uncensored attackers?           |
| `rq2_semantics.py` | What semantic strategies do attackers use, and which succeed most?             |
| `rq3_transferrability.py` | Do successful attacking prompts transfer to a different target model?   |

Run them in order. RQ2 and RQ3 depend on JSON output from RQ1.

```bash
python rq1_asr.py
python rq2_semantics.py
python rq3_transferrability.py
```

## Models

| Role | Model |
|------|-------|
| Uncensored attacker | `failspy/Llama-3-8B-Instruct-Abliterated` |
| Censored attacker   | `meta-llama/Meta-Llama-3-8B-Instruct`     |
| Target (RQ1/RQ2)    | `google/gemma-2-9b-it`                    |
| Target (RQ3)        | `Qwen/Qwen3.5-9B`                         |
| Harm classifier     | `cais/HarmBench-Mistral-7b-val-cls`       |

Dataset: [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench) (contextual split)

50 harmful behaviors, up to 10 PAIR-style refinement iterations each.

## Results

### RQ1: Attack Success Rate (Gemma-2-9B-it)

| Attacker | ASR |
|----------|-----|
| Uncensored | 44% |
| Censored | 52% |

The censored attacker outperformed the abliterated model, contrary to the initial hypothesis.

### RQ2: Semantic Strategies

The semantic strategies used in classified by a three LLM as a judge majority voting system. Both attackers rely heavily on **Hypothetical Framing**, but the censored model uses a more varied strategy mix overall.

**All prompts generated**

| Strategy | Uncensored | Censored |
|----------|------------|----------|
| Hypothetical framing  | 83.0% | 68.6% |
| Authority roleplay    | 5.3%  | 20.4% |
| Indirect elicitation  | 6.8%  | 1.4%  |
| Persona adoption      | 2.5%  | 3.1%  |
| Obfuscation           | <1%   | 3.4%  |

**Successful jailbreaks only**

| Strategy | Uncensored | Censored |
|----------|------------|----------|
| Hypothetical framing  | 81.8% | 84.6% |
| Authority roleplay    | 9.1%  | 7.7%  |
| Persona adoption      | 9.1%  | 3.8%  |

**Preference-over-Effectiveness (lower = more efficient)**

| Strategy | Uncensored PoE | Censored PoE |
|----------|----------------|--------------|
| Persona adoption     | 11.50 | 11.00 |
| Hypothetical framing | 42.17 | 11.00 |
| Authority roleplay   | 24.00 | 36.00 |

Obfuscation and indirect elicitation produced zero successful jailbreaks for either attacker.

### RQ3: Cross-Model Transferability (Qwen3.5-9B)

**Direct attack (baseline on Qwen)**

| Attacker | ASR |
|----------|-----|
| Uncensored | 32% |
| Censored   | 30% |

**Transfer from Gemma-successful prompts**

48 prompts (22 uncensored, 26 censored) transferred to Qwen3.5-9B with no refinement.

| Metric | Value |
|--------|-------|
| Transfer success rate | 61.2% |

Transferred prompts outperformed both the original Gemma ASRs and the Qwen baseline, suggesting successful jailbreak prompts generalize across aligned models of similar scale.
