import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Star, MapPin, Calendar, MessageSquare, Copy, RefreshCw } from 'lucide-react';
import { reviewsAPI } from '../services/api';
import { ReviewResponse, SuggestReplyResponse, SearchResponse } from '../types';

const ReviewDetail = () => {
  const { id } = useParams<{ id: string }>();
  const [review, setReview] = useState<ReviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [suggestedReply, setSuggestedReply] = useState<SuggestReplyResponse | null>(null);
  const [loadingReply, setLoadingReply] = useState<boolean>(false);
  const [replyError, setReplyError] = useState<string | null>(null);
  const [similarReviews, setSimilarReviews] = useState<SearchResponse['results']>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searching, setSearching] = useState<boolean>(false);

  useEffect(() => {
    loadReview();
  }, [id]);

  const loadReview = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await reviewsAPI.getReview(Number(id));
      setReview(data);
    } catch (err) {
      setError('Failed to load review. Please try again.');
      console.error('Error loading review:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestReply = async () => {
    try {
      setLoadingReply(true);
      setReplyError(null);
      const data = await reviewsAPI.suggestReply(Number(id));
      setSuggestedReply(data);
    } catch (err) {
      setReplyError('Failed to generate reply suggestion. Please try again.');
      console.error('Error generating reply:', err);
    } finally {
      setLoadingReply(false);
    }
  };

  const handleSearchSimilar = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setSearching(true);
      const data = await reviewsAPI.searchSimilar(searchQuery, 5);
      console.log(data);
      setSimilarReviews(data.results);
    } catch (err) {
      console.error('Error searching similar reviews:', err);
    } finally {
      setSearching(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      // You could add a toast notification here
      alert('Reply copied to clipboard!');
    });
  };

  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <span
        key={i}
        className={`rating-star ${i < rating ? '' : 'empty'}`}
        style={{ fontSize: '1.2rem' }}
      >
        ★
      </span>
    ));
  };

  const getSentimentClass = (sentiment: string | undefined) => {
    if (!sentiment) return '';
    return `sentiment ${sentiment.toLowerCase()}`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  if (loading) {
    return (
      <div className="loading">
        <div>Loading review...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error">
        {error}
        <div style={{ marginTop: '1rem' }}>
          <Link to="/" className="btn btn-primary">
            <ArrowLeft size={16} style={{ marginRight: '0.5rem' }} />
            Back to Reviews
          </Link>
        </div>
      </div>
    );
  }

  if (!review) {
    return (
      <div className="error">
        Review not found
        <div style={{ marginTop: '1rem' }}>
          <Link to="/" className="btn btn-primary">
            <ArrowLeft size={16} style={{ marginRight: '0.5rem' }} />
            Back to Reviews
          </Link>
        </div>
      </div>
    );
  }
console.log(similarReviews)
  return (
    <div>
      <div style={{ marginBottom: '1rem' }}>
        <Link to="/" className="btn btn-secondary">
          <ArrowLeft size={16} style={{ marginRight: '0.5rem' }} />
          Back to Reviews
        </Link>
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
          <h2>Review #{review.id}</h2>
          <button
            className="btn btn-primary"
            onClick={handleSuggestReply}
            disabled={loadingReply}
          >
            {loadingReply ? (
              <>
                <RefreshCw size={16} style={{ marginRight: '0.5rem' }} />
                Generating...
              </>
            ) : (
              <>
                <MessageSquare size={16} style={{ marginRight: '0.5rem' }} />
                Suggest Reply
              </>
            )}
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
          <div>
            <strong>Location:</strong>
            <div style={{ display: 'flex', alignItems: 'center', marginTop: '0.25rem' }}>
              <MapPin size={16} style={{ marginRight: '0.5rem' }} />
              {review.location}
            </div>
          </div>
          
          <div>
            <strong>Rating:</strong>
            <div className="rating" style={{ marginTop: '0.25rem' }}>
              {renderStars(review.rating)}
              <span style={{ marginLeft: '0.5rem' }}>({review.rating}/5)</span>
            </div>
          </div>
          
          <div>
            <strong>Date:</strong>
            <div style={{ display: 'flex', alignItems: 'center', marginTop: '0.25rem' }}>
              <Calendar size={16} style={{ marginRight: '0.5rem' }} />
              {formatDate(review.date)}
            </div>
          </div>
          
          {review.sentiment && (
            <div>
              <strong>Sentiment:</strong>
              <div style={{ marginTop: '0.25rem' }}>
                <span className={getSentimentClass(review.sentiment?.replace(/^"|"$/g, ''))}>
  {review.sentiment?.replace(/^"|"$/g, '')}
</span>
              </div>
            </div>
          )}
          
          {review.topic && (
            <div>
              <strong>Topic:</strong>
              <div style={{ marginTop: '0.25rem' }}>
                <span className="topic">
                  {review.topic}
                </span>
              </div>
            </div>
          )}
        </div>

        <div>
          <strong>Review Text:</strong>
          <div style={{ 
            background: '#f8fafc', 
            border: '1px solid #e2e8f0', 
            borderRadius: '6px', 
            padding: '1rem', 
            marginTop: '0.5rem',
            whiteSpace: 'pre-wrap'
          }}>
            {review.text}
          </div>
        </div>
      </div>

      {replyError && (
        <div className="error">
          {replyError}
        </div>
      )}

      {suggestedReply && (
        <div className="reply-suggestion">
          <h4>Suggested Reply</h4>
          <div className="reply-text">{suggestedReply.reply}</div>
          <div style={{ margin: '0.5rem 0', fontSize: '0.875rem', color: '#6b7280' }}>
            <strong>AI Analysis:</strong> {suggestedReply.reasoning_log}
          </div>
          <div style={{ margin: '0.5rem 0', fontSize: '0.875rem' }}>
            <strong>Tags:</strong> 
            {suggestedReply.tags?.sentiment && (
              <span className={getSentimentClass(suggestedReply.tags.sentiment)} style={{ marginLeft: '0.5rem' }}>
                {suggestedReply.tags.sentiment}
              </span>
            )}
            {suggestedReply.tags?.topic && (
              <span className="topic" style={{ marginLeft: '0.5rem' }}>
                {suggestedReply.tags.topic}
              </span>
            )}
          </div>
          <div className="reply-actions">
            <button
              className="btn btn-success"
              onClick={() => copyToClipboard(suggestedReply.reply)}
            >
              <Copy size={16} style={{ marginRight: '0.5rem' }} />
              Copy Reply
            </button>
          </div>
        </div>
      )}

      <div className="card">
        <h3>Find Similar Reviews</h3>
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
          <input
            type="text"
            className="form-input"
            placeholder="Search for similar reviews..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearchSimilar()}
          />
          <button
            className="btn btn-primary"
            onClick={handleSearchSimilar}
            disabled={searching || !searchQuery.trim()}
          >
            {searching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {similarReviews.length > 0 && (
          <div className="search-results">
            <h4>Similar Reviews ({similarReviews.length})</h4>
            {similarReviews.map((result, index) => (
              <div key={index} className="search-result">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                  
                  <span className="similarity-score">
                    {(result.similarity * 100).toFixed(1)}% similar
                  </span>
                </div>
                <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>
                  {result.text}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ReviewDetail;
