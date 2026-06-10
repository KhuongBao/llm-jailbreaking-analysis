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
