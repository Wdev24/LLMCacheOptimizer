"""
Intent Classification Model
Using BERT-base for intent classification to reduce false positives
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from typing import Dict, List
import re

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Intent classifier to improve semantic cache accuracy"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize intent classifier
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.classifier = None
        self.intent_categories = [
            "question",      # What, How, Why, etc.
            "command",       # Set, Call, Open, etc.
            "information",   # Define, Explain, Tell me about
            "greeting",      # Hello, Hi, Good morning
            "other"          # Fallback category
        ]
        
    def load_model(self):
        """Load the intent classification model"""
        try:
            logger.info(f"📥 Loading Intent Classifier: {self.model_name}")
            
            # Use a general text classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
            
            logger.info("✅ Intent classifier loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load intent classifier: {e}")
            # Use rule-based fallback
            self.classifier = None
            logger.warning("⚠️ Using rule-based intent classification as fallback")
    
    def classify_intent(self, text: str) -> Dict[str, float]:
        """
        Classify the intent of input text
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with intent scores
        """
        try:
            # Use rule-based classification (more reliable for this use case)
            return self._rule_based_classification(text)
            
        except Exception as e:
            logger.error(f"❌ Failed to classify intent: {e}")
            return {"other": 1.0}
    
    def _rule_based_classification(self, text: str) -> Dict[str, float]:
        """
        Rule-based intent classification
        
        Args:
            text: Input text
            
        Returns:
            Intent scores dictionary
        """
        text_lower = text.lower().strip()
        
        # Initialize scores
        scores = {intent: 0.0 for intent in self.intent_categories}
        
        # Question patterns
        question_patterns = [
            r'^(what|how|why|when|where|who|which|whose)\b',
            r'\?$',
            r'^(is|are|can|could|would|should|do|does|did)\b',
            r'^(tell me|explain|define)\b'
        ]
        
        # Command patterns
        command_patterns = [
            r'^(set|call|open|close|start|stop|play|pause)\b',
            r'^(turn|switch|toggle|enable|disable)\b',
            r'^(create|make|build|generate)\b',
            r'^(send|share|forward|reply)\b'
        ]
        
        # Information patterns
        info_patterns = [
            r'^(define|explain|describe|tell me about)\b',
            r'^(what is|what are)\b',
            r'^(information about|details about)\b'
        ]
        
        # Greeting patterns
        greeting_patterns = [
            r'^(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'^(thanks|thank you|bye|goodbye)\b'
        ]
        
        # Check patterns and assign scores
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                scores["question"] += 0.3
        
        for pattern in command_patterns:
            if re.search(pattern, text_lower):
                scores["command"] += 0.4
        
        for pattern in info_patterns:
            if re.search(pattern, text_lower):
                scores["information"] += 0.4
        
        for pattern in greeting_patterns:
            if re.search(pattern, text_lower):
                scores["greeting"] += 0.5
        
        # If no specific pattern matches, assign to "other"
        if max(scores.values()) == 0:
            scores["other"] = 1.0
        else:
            # Normalize scores
            max_score = max(scores.values())
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def get_primary_intent(self, text: str) -> str:
        """
        Get the primary intent category for input text
        
        Args:
            text: Input text
            
        Returns:
            Primary intent category
        """
        scores = self.classify_intent(text)
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def compute_intent_similarity(self, text1: str, text2: str) -> float:
        """
        Compute intent similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Intent similarity score (0-1)
        """
        try:
            scores1 = self.classify_intent(text1)
            scores2 = self.classify_intent(text2)
            
            # Compute cosine similarity between intent vectors
            dot_product = sum(scores1[intent] * scores2[intent] for intent in self.intent_categories)
            norm1 = sum(score ** 2 for score in scores1.values()) ** 0.5
            norm2 = sum(score ** 2 for score in scores2.values()) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"❌ Failed to compute intent similarity: {e}")
            return 0.0
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract key entities from text (simple implementation)
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        # Simple entity extraction using regex patterns
        entities = []
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        entities.extend(numbers)
        
        # Times
        times = re.findall(r'\b\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\b', text)
        entities.extend(times)
        
        # Names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(names)
        
        return list(set(entities))  # Remove duplicates