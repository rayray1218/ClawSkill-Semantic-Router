---
name: semantic-model-router
description: Smart LLM Router — routes every query to the cheapest capable model. Supports 17 models across Anthropic, OpenAI, Google, DeepSeek & xAI (Grok). Uses a pre-trained ML classifier. No extra API keys required.
version: 2.0.0
author: Ray
tags: [llm-ops, routing, cost-saving, openclaw, semantic-router, multi-model]
homepage: https://github.com/rayray1218/ClawSkill-Semantic-Router
files: ["scripts/model_router.py", "scripts/model_weights.py", "scripts/requirements.txt"]
dependencies:
  - sentence-transformers>=2.2.2
  - numpy>=1.24.0
---

# Semantic Model Router

Smart LLM router that saves up to **99%** on inference costs by routing each request to the cheapest model that can handle it. Powered by a pre-trained ML classifier and semantic embeddings — no external calls, no API keys needed.

## Install

```bash
openclaw plugins install @rayray1218/semantic-model-router
```

## Quick Start

```python
from scripts.model_router import ModelRouter

router = ModelRouter()
res = router.route("Design a distributed caching layer for a fintech platform.")
print(res["report"])
# [ClawRouter] anthropic/claude-sonnet-4-6 (ELITE, ml, conf=0.97)
#              Cost: $3.0/M | Baseline: $10.0/M | Saved: 70.0%
```

## How Routing Works

Queries are classified into three tiers through a **3-stage pipeline**:

1. **ML Classifier** (primary): A Logistic Regression model trained on 6,000+ labeled queries. Runs in <1ms from embedded weights in `model_weights.py`.
2. **Semantic Embeddings** (fallback): Cosine similarity to tier intent vectors via `sentence-transformers`.
3. **Keyword Rules** (last resort): Pattern matching with no dependencies.

| Tier | Default Model | Typical Workload | Cost/1M | vs Baseline |
|---|---|---|---|---|
| **BASIC** | `deepseek/deepseek-chat` | Greetings, simple Q&A, chit-chat | $0.14 | **99% saved** |
| **BALANCED** | `openai/gpt-4o-mini` | Summaries, translations, explanations | $0.15 | **99% saved** |
| **ELITE** | `anthropic/claude-sonnet-4-6` | Complex coding, architecture, security | $3.00 | **70% saved** |

## Supported Models (17 total)

### Anthropic
| Model | $/1M tokens |
|---|---|
| `anthropic/claude-sonnet-4-6` | $3.00 ★ ELITE default |
| `anthropic/claude-opus-4-5` | $15.00 |
| `anthropic/claude-haiku-4-5` | $0.80 |

### OpenAI
| Model | $/1M tokens |
|---|---|
| `openai/gpt-5.2` | $15.00 |
| `openai/gpt-4o` | $5.00 |
| `openai/gpt-4o-mini` | $0.15 ★ BALANCED default |
| `openai/o3` | $8.00 |
| `openai/o4-mini` | $1.10 |

### Google
| Model | $/1M tokens |
|---|---|
| `google/gemini-3.0-pro` | $10.00 |
| `google/gemini-2.5-pro` | $7.00 |
| `google/gemini-2.5-flash` | $0.60 |
| `google/gemini-2.5-flash-lite` | $0.10 |

### DeepSeek
| Model | $/1M tokens |
|---|---|
| `deepseek/deepseek-chat` | $0.14 ★ BASIC default |
| `deepseek/deepseek-reasoner` | $0.55 |

### xAI (Grok)
| Model | $/1M tokens |
|---|---|
| `xai/grok-3` | $3.00 |
| `xai/grok-3-mini` | $0.30 |

## Override Models at Runtime

```python
# Use GPT-5.2 for ELITE, Gemini Flash Lite for BASIC
router = ModelRouter(
    elite_model="openai/gpt-5.2",
    balanced_model="google/gemini-2.5-flash",
    basic_model="google/gemini-2.5-flash-lite",
)
```

```python
# Swap a tier's model without recreating the router
router.set_model("ELITE", "anthropic/claude-opus-4-5")
```

## List All Available Models (CLI)

```bash
python3 scripts/model_router.py --list-models
```

## CLI Usage

```bash
# Route a single query
python3 scripts/model_router.py "Implement AES encryption from scratch"

# Override ELITE model
python3 scripts/model_router.py --elite openai/gpt-5.2 "Write a compiler"

# Run full smoke-test
python3 scripts/model_router.py
```

## Dynamic Keyword Expansion

```python
router.add_keywords("ELITE", ["cryptographic proof", "zero-knowledge"])
```

## Example Output

```
Query                                              Predicted  Expected   ✓  Cost Info
────────────────────────────────────────────────────────────────────────────────────
How are you doing today?                           BASIC      BASIC      ✓  $0.14/M  saved 98.6%
Summarize this article in three bullet points.     BALANCED   BALANCED   ✓  $0.15/M  saved 98.5%
Implement a thread-safe LRU cache in Python.       ELITE      ELITE      ✓  $3.0/M   saved 70.0%
```

## Security & Privacy

- **Zero external calls**: All classification runs locally.
- **No API keys**: The router itself needs none.
- **Transparent weights**: All model parameters live in `scripts/model_weights.py` — fully auditable.

---
*Save costs, route smarter. Built for the OpenClaw community.*
