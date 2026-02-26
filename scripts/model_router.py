import re
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False

class ModelRouter:
    """
    Standalone Model Router for OpenClaw Skills.
    Categorizes inputs into Elite, Balanced, or Basic tiers.
    """
    def __init__(self, elite_model="anthropic/claude-3-5-sonnet-latest", 
                 balanced_model="gemini/gemini-1.5-flash-latest", 
                 basic_model="openai/gpt-4o-mini"):
        self.mapping = {
            "ELITE": elite_model,
            "BALANCED": balanced_model,
            "BASIC": basic_model
        }
        self.encoder = None
        self.intent_embeddings = {}
        
        # Default Intent Matrix
        self.intent_matrix = {
            "ELITE": ["architecture design", "complex algorithm", "system optimization", "精密推理", "架構設計"],
            "BALANCED": ["data extraction", "summarization", "translation", "摘要", "內容整理"],
            "BASIC": ["greeting", "small talk", "weather", "日常對話", "閒聊"]
        }

        if _HAS_SEMANTIC:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                self._prepare_embeddings()
            except:
                pass

    def _prepare_embeddings(self):
        for tier, examples in self.intent_matrix.items():
            embeddings = self.encoder.encode(examples, convert_to_tensor=True)
            self.intent_embeddings[tier] = embeddings.mean(axis=0)

    def route(self, query: str):
        """
        Input: User query string
        Output: Dict containing predicted tier and the corresponding model name
        """
        # 1. Semantic Check
        if self.encoder:
            query_emb = self.encoder.encode(query, convert_to_tensor=True)
            import torch
            best_tier = "BASIC"
            max_similarity = -1.0
            for tier, centroid in self.intent_embeddings.items():
                sim = util.cos_sim(query_emb, centroid).item()
                if sim > max_similarity:
                    max_similarity = sim
                    best_tier = tier
            
            # Simple threshold check
            if max_similarity < 0.3: best_tier = "BASIC"
        else:
            # 2. Simple Keyword Fallback
            best_tier = "BASIC"
            if any(re.search(kw, query, re.IGNORECASE) for kw in self.intent_matrix["ELITE"]):
                best_tier = "ELITE"
            elif any(re.search(kw, query, re.IGNORECASE) for kw in self.intent_matrix["BALANCED"]):
                best_tier = "BALANCED"

        return {
            "tier": best_tier,
            "model": self.mapping.get(best_tier),
            "confidence": max_similarity if self.encoder else 1.0
        }

# Usage Example for the Skill
if __name__ == "__main__":
    router = ModelRouter()
    test_query = "Please design a high-performance database schema for a global e-commerce platform."
    decision = router.route(test_query)
    print(f"Decision: {decision}")
