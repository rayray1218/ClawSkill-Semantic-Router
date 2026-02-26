import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import joblib

class CLAWMLTrainer:
    """
    Cold Start ML Trainer for Model Router.
    Uses 'Synthetic Augmentation' from seed keywords to train a robust classifier.
    """
    def __init__(self, model_save_path="classifier.joblib"):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.model_save_path = model_save_path
        self.clf = LogisticRegression(max_iter=1000)
        
    def generate_synthetic_data(self, intent_matrix):
        """
        In the absence of real user data, we 'augment' the seeds.
        For a real ML project, you'd use an LLM API here to generate 100 variations per keyword.
        Here we simulate it by adding common prefixes/suffixes to create a larger X.
        """
        X_text = []
        y = []
        
        prefixes = ["Please ", "Can you ", "I need to ", "Help me with ", "How to ", "Give me ", ""]
        suffixes = ["", " for my project", " quickly", " please", " accurately", " in detail"]
        
        for tier, seeds in intent_matrix.items():
            for seed in seeds:
                # Basic augmentation: combine variations
                for p in prefixes:
                    for s in suffixes:
                        synthetic_query = f"{p}{seed}{s}"
                        X_text.append(synthetic_query)
                        y.append(tier)
                        
        print(f"Generated {len(X_text)} synthetic training samples.")
        return X_text, y

    def train(self, intent_matrix):
        X_text, y = self.generate_synthetic_data(intent_matrix)
        
        print("Encoding datasets... (This may take a minute)")
        X_embeddings = self.encoder.encode(X_text, show_progress_bar=True)
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)
        
        print("Fitting Logistic Regression Classifier...")
        self.clf.fit(X_train, y_train)
        
        # Check Accuracy
        score = self.clf.score(X_test, y_test)
        print(f"Training Complete. Validation Accuracy: {score:.4f}")
        
        # Save model
        joblib.dump(self.clf, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        return score

if __name__ == "__main__":
    # Test with current intent matrix
    from model_router import ModelRouter
    router = ModelRouter()
    
    trainer = CLAWMLTrainer()
    trainer.train(router.intent_matrix)
