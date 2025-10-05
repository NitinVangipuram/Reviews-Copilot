import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from functools import lru_cache
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from openai import OpenAI

from config import get_settings
from database import get_db_manager

logger = logging.getLogger(__name__)
settings = get_settings()
from dotenv import load_dotenv
load_dotenv()

class AIService:
    """AI service using Hugging Face Inference API + DeepSeek for text generation"""
    
    def __init__(self):
        # Hugging Face API configuration
        self.hf_api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        self.hf_api_url = "https://api-inference.huggingface.co/models"
        
        # Initialize OpenAI client with HF Router for DeepSeek
        self.hf_token = os.environ.get("HF_TOKEN", self.hf_api_token)
        self.openai_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.hf_token,
        )
        
        # Model endpoints
        self.models = {
            'sentiment': "cardiffnlp/twitter-roberta-base-sentiment-latest",
            'summarization': "facebook/bart-large-cnn",
            'text_generation': "deepseek-ai/DeepSeek-V3.2-Exp:novita",  # Using DeepSeek
            'emotion': "cardiffnlp/twitter-roberta-base-emotion"
        }
        
        # Local components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.review_texts = []
        self.review_ids = []
        self._db_manager = get_db_manager()
        
        # Performance tracking
        self.performance_metrics = {
            'sentiment_analysis_time': [],
            'summarization_time': [],
            'reply_generation_time': [],
            'search_time': []
        }
        
        # Initialize search index
        self._initialize_search_index()
        
        logger.info("AI Service initialized with Hugging Face models + DeepSeek for text generation")
    
    def _query_huggingface(self, model_name: str, payload: Dict, retry: int = 3) -> Optional[Dict]:
        """Query Hugging Face Inference API"""
        headers = {"Content-Type": "application/json"}
        
        # Only add auth if token is valid
        if self.hf_api_token and len(self.hf_api_token) > 10 and self.hf_api_token.startswith('hf_'):
            headers["Authorization"] = f"Bearer {self.hf_api_token}"
        else:
            logger.info("Using Hugging Face API without authentication (rate limited)")
        
        url = f"{self.hf_api_url}/{model_name}"
        
        for attempt in range(retry):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    logger.warning(f"Model loading, attempt {attempt + 1}/{retry}")
                    time.sleep(3)
                elif response.status_code == 403:
                    # Auth error - fall back to no auth or use fallback methods
                    logger.warning(f"Authentication error, using fallback methods")
                    return None
                elif response.status_code == 429:
                    # Rate limit - wait longer
                    logger.warning(f"Rate limited, waiting...")
                    time.sleep(5)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error querying Hugging Face API: {str(e)}")
                if attempt < retry - 1:
                    time.sleep(2)
        
        return None
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using remote Hugging Face API"""
        try:
            start_time = time.time()
            
            # Truncate text if too long
            text = text[:512]
            
            result = self._query_huggingface(
                self.models['sentiment'],
                {"inputs": text}
            )
            
            if result and isinstance(result, list) and len(result) > 0:
                # Get the label with highest score
                best_score = max(result[0], key=lambda x: x['score'])
                sentiment = best_score['label'].lower()
            else:
                sentiment = self._analyze_sentiment_fallback(text)
            
            processing_time = time.time() - start_time
            self.performance_metrics['sentiment_analysis_time'].append(processing_time)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self._analyze_sentiment_fallback(text)
    
    def _analyze_sentiment_fallback(self, text: str) -> str:
        """Fallback sentiment analysis using keyword matching"""
        text_lower = text.lower()
        
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'perfect', 'delicious', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disgusting', 'disappointing', 'poor', 'hate']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def extract_topic(self, text: str) -> str:
        """Extract topic using keyword-based approach"""
        text_lower = text.lower()
        
        topic_keywords = {
            'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'delicious', 'tasty', 'cooking', 'chef', 'menu'],
            'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'friendly', 'helpful', 'attentive'],
            'atmosphere': ['atmosphere', 'ambiance', 'decor', 'music', 'lighting', 'cozy', 'romantic'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money'],
            'location': ['location', 'parking', 'convenient', 'accessible'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'tidy']
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            if topic_scores[best_topic] > 0:
                return best_topic
        
        return 'service'
    
    def summarize_text(self, text: str) -> str:
        """Summarize text using remote Hugging Face API"""
        try:
            start_time = time.time()
            
            # Summarization works best with longer text
            if len(text) < 50:
                return text
            
            # Truncate if too long
            text = text[:1024]
            
            result = self._query_huggingface(
                self.models['summarization'],
                {
                    "inputs": text,
                    "parameters": {
                        "max_length": 100,
                        "min_length": 20
                    }
                }
            )
            
            if result and isinstance(result, list) and len(result) > 0:
                summary = result[0].get('summary_text', text[:100] + "...")
            else:
                summary = self._summarize_fallback(text)
            
            processing_time = time.time() - start_time
            self.performance_metrics['summarization_time'].append(processing_time)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return self._summarize_fallback(text)
    
    def _summarize_fallback(self, text: str) -> str:
        """Fallback summarization"""
        if len(text) <= 100:
            return text
        return text[:100] + "..."
    
    def generate_reply(self, review_text: str, rating: int, sentiment: str, summary: str = None) -> Dict[str, Any]:
        """Generate dynamic reply using DeepSeek"""
        try:
            start_time = time.time()
            
            # Analyze the review
            sentiment_analysis = self.analyze_sentiment(review_text)
            topic = self.extract_topic(review_text)
            
            if summary is None:
                summary = self.summarize_text(review_text)
            
            # Try to generate reply using DeepSeek
            reply = self._generate_with_deepseek(review_text, rating, sentiment_analysis, topic)
            print("ai reply:", reply)
            processing_time = time.time() - start_time
            self.performance_metrics['reply_generation_time'].append(processing_time)
            
            reasoning_log = f"AI Analysis: {sentiment_analysis} sentiment | Topic: {topic} | Rating: {rating}/5 | Method: DeepSeek V3.2"
            
            return {
                "reply": reply,
                "reasoning_log": reasoning_log
            }
            
        except Exception as e:
            logger.error(f"Error in reply generation: {str(e)}")
            fallback_reply = self._generate_contextual_reply(review_text, rating, sentiment, "service")
            return {
                "reply": fallback_reply,
                "reasoning_log": f"Fallback reply generated"
            }
    
    def _generate_with_deepseek(self, review_text: str, rating: int, sentiment: str, topic: str) -> str:
        """Generate reply using DeepSeek V3.2 via OpenAI-compatible API"""
        try:
            # Create system prompt
            system_prompt = """You are a professional restaurant manager responding to customer reviews. 
Your responses should be:
- Brief and concise (2-3 sentences maximum)
- Professional and courteous
- Address the specific points mentioned in the review
- For positive reviews: express gratitude and welcome them back
- For negative reviews: apologize sincerely and offer to make things right
- Never make excuses, just acknowledge and commit to improvement
- Don't mention any contact details like phone and email"""

            # Create user prompt based on sentiment
            if sentiment == "positive" and rating >= 4:
                user_prompt = f"Write a brief, grateful response (2-3 sentences) to this positive customer review:\n\n{review_text}"
            elif sentiment == "negative" and rating <= 2:
                user_prompt = f"Write a brief, apologetic response (2-3 sentences) to this negative customer complaint, offering to help:\n\n{review_text}"
            else:
                user_prompt = f"Write a brief, professional response (2-3 sentences) to this customer feedback:\n\n{review_text}"
            
            print(f"DeepSeek prompt - Sentiment: {sentiment}, Rating: {rating}")
            
            # Call DeepSeek API via OpenAI client
            completion = self.openai_client.chat.completions.create(
                model=self.models['text_generation'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=150,
                top_p=0.9,
            )
            
            # Extract reply
            reply = completion.choices[0].message.content.strip()
            
            print(f"DeepSeek raw response: {reply}")
            
            # Clean and validate the reply
            cleaned_reply = self._clean_deepseek_reply(reply, sentiment)
            
            if cleaned_reply and len(cleaned_reply) >= 30:
                print("✓ Using DeepSeek-generated reply")
                return cleaned_reply
            else:
                print("✗ DeepSeek reply too short, using fallback")
                return self._generate_contextual_reply(review_text, rating, sentiment, topic)
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            print(f"✗ DeepSeek API error: {str(e)}")
            return self._generate_contextual_reply(review_text, rating, sentiment, topic)
    
    def _clean_deepseek_reply(self, reply: str, sentiment: str) -> str:
        """Clean and validate DeepSeek-generated reply"""
        if not reply:
            return ""
        
        # Remove any quotes if wrapped
        reply = reply.strip()
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1].strip()
        if reply.startswith("'") and reply.endswith("'"):
            reply = reply[1:-1].strip()
        
        # Remove any meta-text
        unwanted_prefixes = [
            "Here's a response:", "Here is a response:", "Response:",
            "Dear customer,", "Dear Customer,",
            "Restaurant Manager:", "Manager:"
        ]
        
        for prefix in unwanted_prefixes:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()
        
        # Ensure proper capitalization
        if reply:
            reply = reply[0].upper() + reply[1:] if len(reply) > 1 else reply.upper()
        
        # Ensure proper ending punctuation
        if reply and not reply.endswith(('.', '!', '?')):
            reply += '.'
        
        # Limit to 3 sentences max
        sentences = reply.split('. ')
        if len(sentences) > 3:
            reply = '. '.join(sentences[:3]) + '.'
        
        # Validate length (should be concise)
        if len(reply) > 350:
            reply = reply[:350]
            last_period = reply.rfind('.')
            if last_period > 100:
                reply = reply[:last_period + 1]
        
        return reply
    
    def _generate_contextual_reply(self, review_text: str, rating: int, sentiment: str, topic: str) -> str:
        """Generate contextual reply based on analysis (fallback method)"""
        
        if sentiment == "positive":
            if rating >= 4:
                responses = [
                    f"Thank you for your wonderful feedback! We're thrilled that you enjoyed our {topic} and look forward to serving you again!",
                    f"We're delighted to hear about your positive experience with our {topic}! Thank you for taking the time to share your feedback.",
                    f"Thank you for your amazing review! We're so happy that you loved our {topic} and we can't wait to welcome you back!"
                ]
            else:
                responses = [
                    f"Thank you for your positive feedback about our {topic}! We appreciate your support and hope to see you again soon!",
                    f"We're glad you had a good experience with our {topic}! Thank you for sharing your thoughts with us."
                ]
        elif sentiment == "negative":
            if rating <= 2:
                responses = [
                    f"Thank you for bringing this to our attention. We sincerely apologize for not meeting your expectations with our {topic}. Please contact us directly so we can address your concerns.",
                    f"We're sorry to hear about your disappointing experience with our {topic}. We take all feedback seriously and would like to make this right.",
                    f"We apologize for falling short of your expectations with our {topic}. Your feedback is important to us, and we'd like to address your concerns personally."
                ]
            else:
                responses = [
                    f"Thank you for your feedback about our {topic}. We'd like to discuss your concerns and ensure you have a better experience next time.",
                    f"We appreciate you sharing your experience with our {topic} and will work to address your concerns."
                ]
        else:
            responses = [
                f"Thank you for your feedback about our {topic}! We appreciate you taking the time to share your experience.",
                f"We value your input about our {topic}! Thank you for sharing your thoughts with us.",
                f"Thank you for reviewing our {topic}! Your feedback helps us continue improving."
            ]
        
        return random.choice(responses)
    
    def search_similar_reviews(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar reviews using TF-IDF"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            start_time = time.time()
            
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.review_ids):
                    results.append({
                        'id': self.review_ids[idx],
                        'similarity': float(similarities[idx]),
                        'text': self.review_texts[idx]
                    })
            
            processing_time = time.time() - start_time
            self.performance_metrics['search_time'].append(processing_time)
            
            return results
        
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def update_tfidf_matrix(self, reviews: List[Dict[str, Any]]):
        """Update TF-IDF matrix with new reviews"""
        try:
            texts = [review.get('text', '') for review in reviews]
            self.review_texts = texts
            self.review_ids = [review.get('id') for review in reviews]
            
            if texts:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                logger.info(f"TF-IDF matrix updated with {len(texts)} reviews")
            
        except Exception as e:
            logger.error(f"Error updating TF-IDF matrix: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}
        for key, times in self.performance_metrics.items():
            if times:
                metrics[key] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'count': len(times)
                }
            else:
                metrics[key] = {'avg_time': 0, 'min_time': 0, 'max_time': 0, 'count': 0}
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        health = {
            'status': 'healthy',
            'using_remote_models': True,
            'text_generation_model': 'DeepSeek V3.2',
            'api_token_configured': bool(self.hf_api_token),
            'performance_metrics': self.get_performance_metrics()
        }
        
        return health
    
    def _initialize_search_index(self):
        """Initialize search index with existing reviews"""
        try:
            reviews = self._db_manager.execute_query(
                "SELECT id, text FROM reviews WHERE text IS NOT NULL"
            )
            
            if reviews:
                reviews_data = [{'id': row[0], 'text': row[1]} for row in reviews]
                self.update_tfidf_matrix(reviews_data)
                logger.info(f"Search index initialized with {len(reviews)} reviews")
            
        except Exception as e:
            logger.error(f"Error initializing search index: {str(e)}")
    
    def refresh_search_index(self):
        """Refresh the search index with current reviews"""
        try:
            reviews = self._db_manager.execute_query(
                "SELECT id, text FROM reviews WHERE text IS NOT NULL"
            )
            
            if reviews:
                reviews_data = [{'id': row[0], 'text': row[1]} for row in reviews]
                self.update_tfidf_matrix(reviews_data)
                logger.info(f"Search index refreshed with {len(reviews)} reviews")
            else:
                logger.warning("No reviews found to index")
                
        except Exception as e:
            logger.error(f"Error refreshing search index: {str(e)}")
    
    def cleanup_cache(self):
        """Clean up AI service cache and resources"""
        try:
            # Clear performance metrics
            for key in self.performance_metrics:
                self.performance_metrics[key].clear()
            
            # Clear TF-IDF data
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.review_texts = []
            self.review_ids = []
            
            logger.info("AI service cache cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up AI service cache: {str(e)}")

# Create global instance
ai_service = AIService()