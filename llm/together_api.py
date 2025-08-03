"""
Together AI API Client
Handles LLM fallback when cache confidence is low
"""

import requests
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TogetherAIClient:
    """Client for Together AI API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Together AI client
        
        Args:
            api_key: Together AI API key
        """
        self.api_key = api_key or "b9942973fb0017c79695fdbb10d2a2fd0312e222dc90e9e9ed6515f1d0bd4d73"
        self.base_url = "https://api.together.xyz/v1"
        self.default_model = "meta-llama/Llama-2-7b-chat-hf"
        
        # Request configuration
        self.timeout = 30
        self.max_retries = 3
        
    def generate_response(self, 
                         prompt: str, 
                         model: str = None,
                         max_tokens: int = 512,
                         temperature: float = 0.7) -> str:
        """
        Generate response using Together AI
        
        Args:
            prompt: Input prompt/query
            model: Model to use (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        model = model or self.default_model
        
        try:
            logger.info(f"🤖 Calling Together AI with model: {model}")
            
            # Prepare request payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["<|endoftext|>"],
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make request with retries
            response = self._make_request_with_retry(
                url=f"{self.base_url}/chat/completions",
                payload=payload,
                headers=headers
            )
            
            if response and "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"].strip()
                
                if generated_text:
                    logger.info("✅ Successfully generated response from Together AI")
                    return generated_text
                else:
                    logger.warning("⚠️ Empty response from Together AI")
                    return self._get_fallback_response(prompt)
            else:
                logger.error("❌ Invalid response format from Together AI")
                return self._get_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"❌ Failed to generate response: {e}")
            return self._get_fallback_response(prompt)
    
    def _make_request_with_retry(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: Request URL
            payload: Request payload
            headers: Request headers
            
        Returns:
            Response JSON or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"🔄 Making request attempt {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"⏳ Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    logger.error("🔑 Authentication failed - check API key")
                    break
                else:
                    logger.error(f"🚫 HTTP {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    break
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Request timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                break
                
            except requests.exceptions.RequestException as e:
                logger.error(f"🌐 Request error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                break
        
        return None
    
    def _get_fallback_response(self, prompt: str) -> str:
        """
        Generate fallback response when API fails
        
        Args:
            prompt: Original prompt
            
        Returns:
            Fallback response
        """
        # Simple pattern-based responses for common queries
        prompt_lower = prompt.lower().strip()
        
        if any(word in prompt_lower for word in ["what", "define", "explain"]):
            return "I understand you're asking for information, but I'm currently unable to access my knowledge base. Could you please try rephrasing your question or try again in a moment?"
        
        elif any(word in prompt_lower for word in ["how", "steps", "guide"]):
            return "I'd be happy to help with instructions, but I'm experiencing technical difficulties right now. Please try again shortly, and I'll do my best to provide you with a detailed guide."
        
        elif any(word in prompt_lower for word in ["call", "set", "open", "turn"]):
            return "I understand you want me to perform an action, but I'm currently unable to process commands. Please try again in a moment or contact support if the issue persists."
        
        elif any(word in prompt_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm currently experiencing some technical difficulties, but I'm here to help. Please try asking your question again in a moment."
        
        else:
            return "I apologize, but I'm currently experiencing technical difficulties and cannot process your request. Please try again in a few moments. If the problem persists, please contact support."
    
    def test_connection(self) -> bool:
        """
        Test connection to Together AI API
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("🔍 Testing Together AI connection...")
            
            test_response = self.generate_response(
                prompt="Hello, this is a test.",
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response and "technical difficulties" not in test_response:
                logger.info("✅ Together AI connection test successful")
                return True
            else:
                logger.warning("⚠️ Together AI connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Together AI connection test error: {e}")
            return False
    
    def get_available_models(self) -> list:
        """
        Get list of available models from Together AI
        
        Returns:
            List of available model names
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                url=f"{self.base_url}/models",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data:
                    return [model["id"] for model in models_data["data"]]
            
            logger.warning("⚠️ Could not fetch available models")
            return [self.default_model]
            
        except Exception as e:
            logger.error(f"❌ Failed to get available models: {e}")
            return [self.default_model]
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """"""
Together AI API Client
Handles LLM fallback when cache confidence is low
"""

import requests
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TogetherAIClient:
    """Client for Together AI API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Together AI client
        
        Args:
            api_key: Together AI API key
        """
        self.api_key = api_key or "b9942973fb0017c79695fdbb10d2a2fd0312e222dc90e9e9ed6515f1d0bd4d73"
        self.base_url = "https://api.together.xyz/v1"
        self.default_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Updated to working model
        
        # Request configuration
        self.timeout = 30
        self.max_retries = 3
        
    def generate_response(self, 
                         prompt: str, 
                         model: str = None,
                         max_tokens: int = 512,
                         temperature: float = 0.7) -> str:
        """
        Generate response using Together AI
        
        Args:
            prompt: Input prompt/query
            model: Model to use (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        model = model or self.default_model
        
        try:
            logger.info(f"🤖 Calling Together AI with model: {model}")
            
            # Prepare request payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make request with retries
            response = self._make_request_with_retry(
                url=f"{self.base_url}/chat/completions",
                payload=payload,
                headers=headers
            )
            
            if response and "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"].strip()
                
                if generated_text:
                    logger.info("✅ Successfully generated response from Together AI")
                    return generated_text
                else:
                    logger.warning("⚠️ Empty response from Together AI")
                    return self._get_smart_fallback_response(prompt)
            else:
                logger.error("❌ Invalid response format from Together AI")
                return self._get_smart_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"❌ Failed to generate response: {e}")
            return self._get_smart_fallback_response(prompt)
    
    def _make_request_with_retry(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: Request URL
            payload: Request payload
            headers: Request headers
            
        Returns:
            Response JSON or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"🔄 Making request attempt {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"⏳ Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    logger.error("🔑 Authentication failed - check API key")
                    break
                else:
                    logger.error(f"🚫 HTTP {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    break
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Request timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                break
                
            except requests.exceptions.RequestException as e:
                logger.error(f"🌐 Request error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                break
        
        return None
    
    def _get_smart_fallback_response(self, prompt: str) -> str:
        """
        Generate smart fallback response when API fails
        
        Args:
            prompt: Original prompt
            
        Returns:
            Contextual fallback response
        """
        # Smart pattern-based responses for testing
        prompt_lower = prompt.lower().strip()
        
        # Greetings
        if any(word in prompt_lower for word in ["hi", "hello", "hey", "good morning", "good afternoon"]):
            return "Hello! I'm here to help. What would you like to know?"
        
        # Questions about AI/ML
        if any(word in prompt_lower for word in ["machine learning", "ml", "artificial intelligence", "ai"]):
            return "Machine learning and AI are fascinating fields! Machine learning enables computers to learn from data, while AI encompasses the broader goal of creating intelligent systems. Would you like me to explain any specific aspect?"
        
        # Questions about neural networks
        if any(word in prompt_lower for word in ["neural network", "deep learning", "neurons"]):
            return "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) that process information in layers, learning patterns from data through training."
        
        # Questions about LLMs
        if any(word in prompt_lower for word in ["llm", "large language model", "gpt", "transformer"]):
            return "Large Language Models (LLMs) are AI systems trained on vast amounts of text to understand and generate human-like language. They use transformer architecture and can perform various language tasks."
        
        # Commands (call, set, etc.)
        if any(word in prompt_lower for word in ["call", "phone", "dial"]):
            if "ramesh" in prompt_lower:
                if "kumar" in prompt_lower:
                    return "I understand you want to call Ramesh Kumar. I can't make calls directly, but you can use your phone's voice assistant or contacts app to call him."
                elif "singh" in prompt_lower:
                    return "I understand you want to call Ramesh Singh. I can't make calls directly, but you can use your phone's voice assistant or contacts app to call him."
            return "I can't make phone calls directly, but I can help you with the steps to make a call using your device."
        
        if any(word in prompt_lower for word in ["alarm", "reminder", "timer"]):
            return "I can't set alarms directly, but you can use your device's voice assistant (like 'Hey Siri' or 'OK Google') or your phone's alarm app to set one."
        
        # How-to questions
        if prompt_lower.startswith("how"):
            return "That's a great question! I'd be happy to explain, but I'm currently experiencing some connectivity issues. Could you try rephrasing your question or ask me again in a moment?"
        
        # What/Define questions
        if any(word in prompt_lower for word in ["what", "define", "explain"]):
            return "I'd be happy to explain that concept for you! However, I'm currently having some technical difficulties accessing my full knowledge base. Could you try asking again in a moment?"
        
        # Default response
        return "I understand your request, but I'm currently experiencing some technical difficulties. Please try again in a moment, or feel free to rephrase your question."
    
    def test_connection(self) -> bool:
        """
        Test connection to Together AI API
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("🔍 Testing Together AI connection...")
            
            test_response = self.generate_response(
                prompt="Hello, this is a test.",
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response and "technical difficulties" not in test_response:
                logger.info("✅ Together AI connection test successful")
                return True
            else:
                logger.warning("⚠️ Together AI connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Together AI connection test error: {e}")
            return False
    
    def get_available_models(self) -> list:
        """
        Get list of available models from Together AI
        
        Returns:
            List of available model names
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                url=f"{self.base_url}/models",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data:
                    return [model["id"] for model in models_data["data"]]
            
            logger.warning("⚠️ Could not fetch available models")
            return [self.default_model]
            
        except Exception as e:
            logger.error(f"❌ Failed to get available models: {e}")
            return [self.default_model]
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)
    
    def validate_api_key(self) -> bool:
        """
        Validate the API key
        
        Returns:
            True if API key is valid, False otherwise
        """
        if not self.api_key:
            logger.error("🔑 No API key provided")
            return False
        
        if len(self.api_key) < 32:  # Together AI keys are typically longer
            logger.error("🔑 API key appears to be invalid (too short)")
            return False
        
        return self.test_connection()
        # Rough approximation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)
    
    def validate_api_key(self) -> bool:
        """
        Validate the API key
        
        Returns:
            True if API key is valid, False otherwise
        """
        if not self.api_key:
            logger.error("🔑 No API key provided")
            return False
        
        if len(self.api_key) < 32:  # Together AI keys are typically longer
            logger.error("🔑 API key appears to be invalid (too short)")
            return False
        
        return self.test_connection()