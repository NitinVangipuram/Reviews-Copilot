from fastapi import FastAPI, HTTPException, Depends, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
import sqlite3
import json
import os
import sys
from pathlib import Path
import logging
import asyncio
from contextlib import asynccontextmanager
import traceback
import time

from config import get_settings, setup_logging
from database import get_db_manager
from ai_service import ai_service

# Initialize settings and logging
settings = get_settings()
setup_logging()
logger = logging.getLogger(__name__)

# Initialize database manager
db_manager = get_db_manager()

# Create FastAPI app with configuration
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug
)

# Add security middleware
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure properly for production
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Security
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key with proper error handling"""
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Pydantic models
class Review(BaseModel):
    id: int
    location: str
    rating: int = Field(..., ge=1, le=5)
    text: str
    date: str

class ReviewResponse(BaseModel):
    id: int
    location: str
    rating: int
    text: str
    date: str
    sentiment: Optional[str] = None
    topic: Optional[str] = None
    created_at: Optional[str] = None

class IngestRequest(BaseModel):
    reviews: List[Review]

class SuggestReplyRequest(BaseModel):
    review_id: int

class SuggestReplyResponse(BaseModel):
    reply: str
    tags: Dict[str, str]
    reasoning_log: str

class AnalyticsResponse(BaseModel):
    sentiment_counts: Dict[str, int] = Field(..., description="Sentiment distribution counts")
    topic_counts: Dict[str, int] = Field(..., description="Topic distribution counts")
    location_counts: Dict[str, int] = Field(..., description="Location distribution counts")
    rating_distribution: Dict[str, int] = Field(..., description="Rating distribution counts")
    total_reviews: int = Field(..., description="Total number of reviews", ge=0)
    
    class Config:
        json_encoders = {
            int: lambda v: v
        }

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total: int

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        logger.info("Starting Reviews Copilot API...")
        
        # Initialize AI service
        ai_health = ai_service.health_check()
        logger.info(f"AI Service Status: {ai_health}")
        
        # Clean up old cache entries
        db_manager.cleanup_old_cache()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    try:
        logger.info("Shutting down Reviews Copilot API...")
        
        # Cleanup AI service
        ai_service.cleanup_cache()
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check"""
    try:
        # Check database connectivity
        db_health = "healthy"
        try:
            db_manager.execute_query("SELECT 1", fetch_one=True)
        except Exception as e:
            db_health = f"unhealthy: {str(e)}"
        
        # Check AI service health
        ai_health = ai_service.health_check()
        
        return {
            "status": "healthy" if db_health == "healthy" else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.api_version,
            "database": db_health,
            "ai_service": ai_health,
            "uptime": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

# Ingest reviews
@app.post("/ingest", response_model=Dict[str, str])
async def ingest_reviews(request: IngestRequest, api_key: str = Depends(verify_api_key)):
    """Ingest a batch of reviews with enhanced validation and error handling"""
    try:
        start_time = time.time()
        
        # Validate input data
        if not request.reviews:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No reviews provided"
            )
        
        # Convert to dictionary format for database
        reviews_data = []
        for review in request.reviews:
            reviews_data.append({
                'id': review.id,
                'location': review.location,
                'rating': review.rating,
                'text': review.text,
                'date': review.date
            })
        
        # Insert reviews using database manager
        inserted_count = db_manager.insert_reviews(reviews_data)
        
        # Refresh search index if AI is enabled
        if settings.ai_enabled:
            ai_service.refresh_search_index()
        
        processing_time = time.time() - start_time
        logger.info(f"Ingested {inserted_count} reviews in {processing_time:.3f}s")
        
        return {
            "message": f"Successfully ingested {inserted_count} reviews",
            "processing_time": f"{processing_time:.3f}s"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting reviews: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting reviews: {str(e)}"
        )

# Get reviews with filtering and pagination
@app.get("/reviews", response_model=Dict[str, Any])
async def get_reviews(
    location: Optional[str] = Query(None, description="Filter by location"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    q: Optional[str] = Query(None, description="Search in review text"),
    rating_min: Optional[int] = Query(None, ge=1, le=5, description="Minimum rating"),
    rating_max: Optional[int] = Query(None, ge=1, le=5, description="Maximum rating"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(None, ge=1, le=100, description="Items per page"),
    api_key: str = Depends(verify_api_key)
):
    """Get reviews with advanced filtering and pagination"""
    try:
        # Use default page size if not specified
        if page_size is None:
            page_size = settings.default_page_size
        
        # Build filters dictionary
        filters = {
            'location': location,
            'sentiment': sentiment,
            'q': q,
            'rating_min': rating_min,
            'rating_max': rating_max,
            'date_from': date_from,
            'date_to': date_to
        }
        
        # Remove None values
        filters = {k: v for k, v in filters.items() if v is not None}
        
        # Get paginated results using database manager
        reviews_data, total = db_manager.get_reviews_paginated(filters, page, page_size)
        
        # Convert to response format
        reviews = []
        for review_data in reviews_data:
            reviews.append(ReviewResponse(
                id=review_data['id'],
                location=review_data['location'],
                rating=review_data['rating'],
                text=review_data['text'],
                date=review_data['date'],
                sentiment=review_data.get('sentiment'),
                topic=review_data.get('topic'),
                created_at=review_data.get('created_at')
            ))
        
        total_pages = (total + page_size - 1) // page_size
        
        return {
            "reviews": reviews,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    
    except Exception as e:
        logger.error(f"Error fetching reviews: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching reviews: {str(e)}"
        )

# Get single review
@app.get("/reviews/{review_id}", response_model=ReviewResponse)
async def get_review(review_id: int, api_key: str = Depends(verify_api_key)):
    """Get a single review by ID with enhanced error handling"""
    try:
        # Get review using database manager
        review_data = db_manager.get_review_by_id(review_id)
        
        if not review_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review with ID {review_id} not found"
            )
        
        return ReviewResponse(
            id=review_data['id'],
            location=review_data['location'],
            rating=review_data['rating'],
            text=review_data['text'],
            date=review_data['date'],
            sentiment=review_data.get('sentiment'),
            topic=review_data.get('topic'),
            created_at=review_data.get('created_at')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching review {review_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching review: {str(e)}"
        )

# Suggest reply for a review
@app.post("/reviews/{review_id}/suggest-reply", response_model=SuggestReplyResponse)
async def suggest_reply(review_id: int, api_key: str = Depends(verify_api_key)):
    """Generate a suggested reply for a review with AI analysis"""
    try:
        if not settings.enable_ai_replies:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI reply generation is currently disabled"
            )
        
        # Get the review using database manager
        review_data = db_manager.get_review_by_id(review_id)
        
        if not review_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review with ID {review_id} not found"
            )
        
        review_text = review_data['text']
        rating = review_data['rating']
        
        # Analyze sentiment and topic if not already done
        sentiment = review_data.get('sentiment') or ai_service.analyze_sentiment(review_text)
        topic = review_data.get('topic') or ai_service.extract_topic(review_text)
        
        # Update the review with AI analysis results
        if not review_data.get('sentiment') or not review_data.get('topic'):
            db_manager.update_review_ai_data(review_id, sentiment, topic)
        
        # Generate reply using AI service
        reply_data = ai_service.generate_reply(review_text, rating, sentiment, None)
        
        return SuggestReplyResponse(
            reply=reply_data["reply"],
            tags={"sentiment": sentiment, "topic": topic},
            reasoning_log=reply_data["reasoning_log"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating reply for review {review_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating reply: {str(e)}"
        )

# Analytics endpoint
@app.get("/analytics")
async def get_analytics(api_key: str = Depends(verify_api_key)):
    """Get comprehensive analytics data for reviews with caching"""
    try:
        if not settings.enable_analytics:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Analytics are currently disabled"
            )
        
        # Get analytics data using database manager (with caching)
        analytics_data = db_manager.get_analytics_data()
        
        # Get total reviews count
        total_reviews_query = "SELECT COUNT(*) FROM reviews"
        total_reviews_result = db_manager.execute_query(total_reviews_query, fetch_one=True)
        total_reviews = total_reviews_result[0] if total_reviews_result else 0
        
        # Convert rating distribution keys to strings
        rating_distribution = analytics_data.get('rating_distribution', {})
        rating_distribution_str = {str(k): v for k, v in rating_distribution.items()}
        
        return {
            "sentiment_counts": analytics_data.get('sentiment_counts', {}),
            "topic_counts": analytics_data.get('topic_counts', {}),
            "location_counts": analytics_data.get('location_counts', {}),
            "rating_distribution": rating_distribution_str,
            "total_reviews": int(total_reviews)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching analytics: {str(e)}"
        )

# Search similar reviews
@app.get("/search", response_model=SearchResponse)
async def search_similar_reviews(
    q: str = Query(..., description="Search query"),
    k: int = Query(None, ge=1, le=20, description="Number of results to return"),
    api_key: str = Depends(verify_api_key)
):
    """Search for similar reviews using TF-IDF and cosine similarity"""
    try:
        if not settings.enable_search:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search functionality is currently disabled"
            )
        
        # Use default k if not specified
        if k is None:
            k = settings.search_max_results
        
        # Search using AI service
        results = ai_service.search_similar_reviews(q, k)
        
        return SearchResponse(
            query=q,
            results=results,
            total=len(results)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching reviews: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching reviews: {str(e)}"
        )

# Batch process reviews for sentiment and topic analysis
@app.post("/process-reviews", response_model=Dict[str, str])
async def process_reviews(api_key: str = Depends(verify_api_key)):
    """Process all reviews to add sentiment and topic analysis"""
    try:
        if not settings.enable_batch_processing:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processing is currently disabled"
            )
        
        if not settings.ai_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI processing is currently disabled"
            )
        
        start_time = time.time()
        
        # Get reviews without sentiment or topic using database manager
        query = "SELECT id, text FROM reviews WHERE sentiment IS NULL OR topic IS NULL"
        reviews = db_manager.execute_query(query)
        
        processed_count = 0
        for review_id, text in reviews:
            try:
                sentiment = ai_service.analyze_sentiment(text)
                topic = ai_service.extract_topic(text)
                
                # Update using database manager
                success = db_manager.update_review_ai_data(review_id, sentiment, topic)
                if success:
                    processed_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process review {review_id}: {str(e)}")
                continue
        
        processing_time = time.time() - start_time
        
        # Refresh search index after processing
        ai_service.refresh_search_index()
        
        return {
            "message": f"Successfully processed {processed_count} reviews",
            "processing_time": f"{processing_time:.3f}s",
            "total_reviews": str(len(reviews))  # Convert to string to match response model
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing reviews: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing reviews: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Render requirement)
    port = int(os.environ.get("PORT", settings.port))
    
    # Configure uvicorn based on settings
    uvicorn_config = {
        "app": app,
        "host": "0.0.0.0",  # Must be 0.0.0.0 for Render
        "port": port,       # Use PORT env variable
        "log_level": settings.log_level.lower(),
        "reload": False,    # Disable reload in production
        "access_log": settings.debug
    }
    
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(**uvicorn_config)
