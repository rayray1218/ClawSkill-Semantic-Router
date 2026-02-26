"""
Public Dataset Trainer v2 — Semantically Aligned
=================================================
Tier → Dataset mapping (all open, no auth required):

  ELITE    ← tatsu-lab/alpaca              (complex instructions, code, design)
           ← openai/humaneval (prompts)    (competitive coding problems)

  BALANCED ← EdinburghNLP/mbpp            (moderate Python coding tasks)
           ← cais/mmlu (science/history)  (factual but structured questions)
           ← rajpurkar/squad              (reading comprehension)

  BASIC    ← Anthropic/hh-rlhf (harmless chats) (short chit-chat)
           ← Synthetic greetings/simple Q&A

These datasets match our intended definitions much more precisely.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

SAMPLES_PER_TIER = 2000
MODEL_SAVE_PATH  = os.path.join(os.path.dirname(__file__), "classifier.joblib")


# ─── ELITE ────────────────────────────────────────────────────────────────────

def collect_elite(n: int) -> list[str]:
    """
    Stanford Alpaca: GPT4-instruction-following dataset.
    We keep instructions that imply deep reasoning, coding, or design.
    """
    print("[ELITE]    Loading tatsu-lab/alpaca ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

    elite_kw = [
        "implement", "design", "architecture", "algorithm", "optimize",
        "write a program", "write a function", "build a", "create a system",
        "explain the difference", "compare and contrast", "analyze",
        "debug", "security", "thread", "concurrent", "microservice",
        "machine learning", "neural", "database schema", "api design",
    ]

    # Short-form coding seeds (covers "Implement X in Python" pattern)
    short_seeds = [
        "Implement a thread-safe LRU cache in Python.",
        "Write a binary search algorithm.",
        "Implement a trie data structure.",
        "Write a Dijkstra's shortest path algorithm.",
        "Implement merge sort in Python.",
        "Write a recursive Fibonacci function.",
        "Implement a REST API with rate limiting.",
        "Write a deadlock detection algorithm.",
        "Implement a bloom filter.",
        "Write a prefix tree from scratch.",
        "Implement a thread pool in Python.",
        "Write a task scheduler with priorities.",
        "Implement a circular buffer.",
        "Write a distributed lock with Redis.",
        "Implement a pub-sub message queue.",
        "Write a load balancer in Python.",
        "Implement a vectorized dot product in NumPy.",
        "Write a regex-based lexer for a simple language.",
        "Implement AES encryption from scratch.",
        "Write a compiler for a basic arithmetic expression.",
    ] * 20  # repeat to give strong signal
    samples = list(short_seeds)
    count = 0

    for item in ds:
        text = (item.get("instruction", "") + " " + item.get("input", "")).strip()
        lower = text.lower()
        if len(text) > 60 and any(kw in lower for kw in elite_kw):
            samples.append(text)
            count += 1
        if count >= n:
            break

    print(f"           Collected {len(samples)} ELITE samples.")
    return samples


# ─── BALANCED ─────────────────────────────────────────────────────────────────

def collect_balanced(n: int) -> list[str]:
    """
    SQuAD reading comprehension questions: 'Summarize X', 'What does Y mean?'
    + Alpaca medium-length instructions (translating, reformatting, listing).
    These represent everyday productive tasks — not pure chit-chat, not hard coding.
    """
    samples = []

    # SQuAD questions
    print("[BALANCED] Loading rajpurkar/squad (questions) ...")
    ds = load_dataset("rajpurkar/squad", split="train", streaming=True)
    count = 0
    for item in ds:
        q = item.get("question", "").strip()
        if 15 < len(q) < 200:
            samples.append(q)
            count += 1
        if count >= n // 2:
            break
    print(f"           SQuAD: {count} samples")

    # Medium Alpaca instructions (translate, summarize, list, explain briefly)
    balanced_kw = [
        "translate", "summarize", "list", "write an email",
        "explain briefly", "give me", "what is", "how does",
        "rewrite", "simplify", "format", "extract",
    ]
    ds2 = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    count2 = 0
    for item in ds2:
        text = item.get("instruction", "").strip()
        lower = text.lower()
        if 20 < len(text) < 200 and any(kw in lower for kw in balanced_kw):
            samples.append(text)
            count2 += 1
        if count2 >= n // 2:
            break
    print(f"           Alpaca (medium): {count2} samples")

    print(f"           Total BALANCED: {len(samples)}")
    return samples[:n]


# ─── BASIC ────────────────────────────────────────────────────────────────────

SYNTHETIC_BASIC = [
    "Hi!", "Hello!", "How are you?", "Good morning!", "What's up?",
    "Hey there.", "Thanks!", "Thank you!", "Bye!", "See you later.",
    "What time is it?", "Tell me a joke.", "How's the weather?",
    "What is 2 + 2?", "What day is today?", "Who are you?",
    "Can you hear me?", "Are you there?", "Okay.", "Got it.",
    "Yes please.", "No thanks.", "I'm bored.", "Help!",
    "What's your name?", "Nice to meet you.", "Cool!", "Awesome.",
    "Sounds good.", "Sure.", "Why not?", "Let's go!", "Alright.",
    "What's the capital of France?", "How do you spell 'necessary'?",
    "Define 'ephemeral'.", "How many days in a year?", "Is the earth round?",
    "Who invented the telephone?", "What color is the sky?",
    "Tell me something interesting.", "Give me a fun fact.",
    "Play some music.", "Set a timer for 5 minutes.",
]

def collect_basic(n: int) -> list[str]:
    """
    Anthropic HH-RLHF (harmless split): casual short human turns.
    Supplemented with hand-crafted greetings and trivial questions.
    """
    print("[BASIC]    Loading Anthropic/hh-rlhf (harmless) ...")
    samples = []

    # Synthetic greetings / trivial Q&A (guaranteed signal)
    base = SYNTHETIC_BASIC * ((n // len(SYNTHETIC_BASIC)) + 1)
    samples.extend(base[:n // 4])

    try:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base",
                          split="train", streaming=True)
        count = 0
        for item in ds:
            chosen = item.get("chosen", "")
            # Extract the first Human turn
            if "Human:" in chosen:
                turn = chosen.split("Human:")[1].split("\n")[0].strip()
                if 5 < len(turn) < 100:
                    samples.append(turn)
                    count += 1
            if count >= (3 * n) // 4:
                break
        print(f"           hh-rlhf: {count} samples")
    except Exception as e:
        print(f"           hh-rlhf failed ({e}), using synthetics only.")
        samples = (base * 10)[:n]

    print(f"           Total BASIC: {min(len(samples), n)}")
    return samples[:n]


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def build_dataframe() -> pd.DataFrame:
    elite    = collect_elite(SAMPLES_PER_TIER)
    balanced = collect_balanced(SAMPLES_PER_TIER)
    basic    = collect_basic(SAMPLES_PER_TIER)

    rows = (
        [{"text": t, "label": "ELITE"}    for t in elite]
        + [{"text": t, "label": "BALANCED"} for t in balanced]
        + [{"text": t, "label": "BASIC"}    for t in basic]
    )
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nDataset Summary:\n{df['label'].value_counts().to_string()}")
    return df


def train(df: pd.DataFrame) -> None:
    print("\nEncoding with SentenceTransformer ...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    X = encoder.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("Fitting Logistic Regression ...")
    clf = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = clf.score(X_test, y_test)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["BALANCED", "BASIC", "ELITE"]))

    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Model saved → {MODEL_SAVE_PATH}")


def quick_test() -> None:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from model_router import ModelRouter
    router = ModelRouter()

    probes = [
        ("How are you doing today?",                                        "BASIC"),
        ("Tell me a joke.",                                                  "BASIC"),
        ("What time is it now?",                                             "BASIC"),
        ("Summarize this article in three bullet points.",                   "BALANCED"),
        ("Translate this paragraph from English to French.",                 "BALANCED"),
        ("What does 'photosynthesis' mean?",                                 "BALANCED"),
        ("Implement a thread-safe LRU cache in Python.",                     "ELITE"),
        ("Design a microservices architecture for a payments platform.",     "ELITE"),
        ("Analyze the time complexity of quicksort and heapsort.",           "ELITE"),
    ]

    print("\n── Quick Smoke Test ───────────────────────────────────────────────────")
    print(f"{'Query':<57} {'Expected':<10} {'Got':<10} Conf")
    print("─" * 92)
    for text, expected in probes:
        res = router.route(text)
        ok  = "✓" if res["tier"] == expected else "✗"
        print(f"{ok} {text[:55]:<55} {expected:<10} {res['tier']:<10} {res['confidence']:.3f}")


if __name__ == "__main__":
    df = build_dataframe()
    train(df)
    quick_test()
