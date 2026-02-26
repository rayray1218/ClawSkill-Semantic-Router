"""
Semantic Model Orchestrator — model_router.py
=============================================
Entry point for the ClawHub Skill.
Routes incoming queries to the optimal LLM tier based on semantic similarity.

Tiers:
  ELITE    → anthropic/claude-3-5-sonnet-latest
  BALANCED → openai/gpt-4o-mini
  BASIC    → deepseek/deepseek-chat

Dependencies (all optional — falls back to keyword matching if missing):
  sentence-transformers, numpy
"""

import re
import os
import json

# ── Optional heavy imports ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False


class ModelRouter:
    """
    Intelligent LLM Model Router for OpenClaw / ClawHub.

    Classifies a natural-language query into ELITE / BALANCED / BASIC
    using cosine similarity over sentence embeddings, with a keyword
    fallback when the embedding library is not installed.
    """

    def __init__(
        self,
        elite_model:    str = "anthropic/claude-3-5-sonnet-latest",
        balanced_model: str = "openai/gpt-4o-mini",
        basic_model:    str = "deepseek/deepseek-chat",
        history_file:   str = "query_history.json",
    ):
        self.mapping = {
            "ELITE":    elite_model,
            "BALANCED": balanced_model,
            "BASIC":    basic_model,
        }
        self.history_file = history_file
        self.encoder = None
        self.intent_embeddings: dict = {}

        # ── Intent Matrix ──────────────────────────────────────────────────────
        # ELITE   : deep technical, architectural, or security-critical tasks
        # BALANCED: everyday productive tasks — summaries, translations, Q&A
        # BASIC   : chit-chat, trivial facts, status checks
        self.intent_matrix: dict[str, list[str]] = {
            "ELITE": [
                # English
                "architecture design", "complex algorithm", "system optimization",
                "precision reasoning", "heavy coding", "security audit",
                "high-frequency trading", "latency optimization", "financial modeling",
                "distributed systems", "database internals", "kernel programming",
                "implement a", "write a program", "write a function",
                "build a system", "design a", "analyze the complexity",
                "debug this code", "refactor", "microservice",
                # Chinese
                "精密推理", "架構設計", "演算法開發", "安全審查",
                "複雜邏輯開發", "深度性能優化", "分散式架構",
            ],
            "BALANCED": [
                # English
                "data extraction", "summarization", "summarize", "translation",
                "translate", "creative writing", "email drafting", "bug fixing",
                "code explanation", "test case generation", "web scraping",
                "text classification", "format conversion", "explain",
                "what is", "how does", "list the", "give me an example",
                "rewrite", "simplify", "extract",
                # Chinese
                "摘要", "內容整理", "翻譯", "創意寫作",
                "日常郵件", "程式碼解釋", "格式轉換",
            ],
            "BASIC": [
                # English
                "hello", "hi", "hey", "good morning", "how are you",
                "small talk", "weather", "simple math", "what time",
                "tell me a joke", "fun fact", "status check", "who are you",
                "thank you", "thanks", "bye", "goodbye", "set a timer",
                # Chinese
                "日常對話", "閒聊", "天氣查詢", "簡單運算",
                "狀態檢查", "問候", "講笑話",
            ],
        }

        # ── Load encoder (optional) ────────────────────────────────────────────
        if _HAS_SEMANTIC:
            try:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self._prepare_embeddings()
            except Exception as exc:
                # Non-fatal: we fall back to keyword matching
                print(f"[ModelRouter] Encoder unavailable: {exc}")

    # ── Embedding helpers ──────────────────────────────────────────────────────

    def _prepare_embeddings(self) -> None:
        """Pre-compute per-tier embedding matrices from the intent matrix."""
        self.intent_embeddings = {}
        for tier, phrases in self.intent_matrix.items():
            self.intent_embeddings[tier] = self.encoder.encode(
                phrases, convert_to_tensor=True
            )

    # ── Core routing ───────────────────────────────────────────────────────────

    def route(self, query: str) -> dict:
        """
        Classify *query* and return the recommended model.

        Returns
        -------
        {
            "tier":       "ELITE" | "BALANCED" | "BASIC",
            "model":      "<provider>/<model-id>",
            "confidence": float,   # 0–1
        }
        """
        best_tier = "BASIC"
        confidence = 1.0

        if self.encoder and self.intent_embeddings:
            best_tier, confidence = self._semantic_route(query)
        else:
            best_tier = self._keyword_route(query)

        self._log_query(query, best_tier)

        return {
            "tier":       best_tier,
            "model":      self.mapping[best_tier],
            "confidence": confidence,
        }

    def _semantic_route(self, query: str) -> tuple[str, float]:
        """Return (tier, confidence) using max cosine-similarity per tier."""
        q_emb = self.encoder.encode(query, convert_to_tensor=True)
        best_tier = "BASIC"
        max_sim = 0.0

        for tier, cluster in self.intent_embeddings.items():
            sims = util.cos_sim(q_emb, cluster)[0]
            peak = float(sims.max())
            if peak > max_sim:
                max_sim = peak
                best_tier = tier

        # Low-confidence → BASIC (safe default)
        if max_sim < 0.40:
            best_tier = "BASIC"

        return best_tier, round(max_sim, 4)

    def _keyword_route(self, query: str) -> str:
        """Rule-based fallback when sentence-transformers is not available."""
        lower = query.lower()
        for tier in ("ELITE", "BALANCED"):          # BASIC is the catch-all
            for kw in self.intent_matrix[tier]:
                if re.search(re.escape(kw.lower()), lower):
                    return tier
        return "BASIC"

    # ── Rolling Adjustment API ─────────────────────────────────────────────────

    def add_keywords(self, tier: str, keywords: list[str]) -> list[str]:
        """
        Add new routing signals to *tier* and refresh embeddings.

        Parameters
        ----------
        tier     : "ELITE", "BALANCED", or "BASIC"
        keywords : list of new keyword phrases

        Returns
        -------
        List of keywords that were actually added (skips duplicates).
        """
        if tier not in self.intent_matrix:
            raise ValueError(f"Unknown tier '{tier}'. Must be ELITE, BALANCED, or BASIC.")

        added = []
        for kw in keywords:
            if kw not in self.intent_matrix[tier]:
                self.intent_matrix[tier].append(kw)
                added.append(kw)

        if added and self.encoder:
            self._prepare_embeddings()

        return added

    def refine_keywords(self) -> str:
        """
        Read collected query history and return a status string.
        In a live skill, this is the hook for an LLM to suggest new keywords.
        """
        if not os.path.exists(self.history_file):
            return "No history found."
        with open(self.history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        return f"History contains {len(history)} queries ready for refinement."

    # ── Logging (internal) ─────────────────────────────────────────────────────

    def _log_query(self, query: str, tier: str) -> None:
        """Append the query + tier to the history file for offline analysis."""
        try:
            history: list = []
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            history.append({"query": query, "tier": tier})
            if len(history) > 1000:
                history = history[-1000:]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Non-fatal


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = ModelRouter()

    probes = [
        ("How are you doing today?",                                       "BASIC"),
        ("Tell me a joke.",                                                "BASIC"),
        ("What time is it now?",                                           "BASIC"),
        ("Summarize this article in three bullet points.",                 "BALANCED"),
        ("Translate this paragraph from English to French.",               "BALANCED"),
        ("What does 'photosynthesis' mean?",                               "BALANCED"),
        ("Implement a thread-safe LRU cache in Python.",                   "ELITE"),
        ("Design a microservices architecture for a payments platform.",   "ELITE"),
        ("Analyze the time complexity of quicksort and heapsort.",         "ELITE"),
    ]

    print(f"\n{'Query':<55} {'Tier':<10} {'Conf'}")
    print("─" * 75)
    for text, _ in probes:
        res = router.route(text)
        print(f"{text[:53]:<55} {res['tier']:<10} {res['confidence']:.3f}")
