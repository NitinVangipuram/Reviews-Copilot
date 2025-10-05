import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Search, Star, MapPin, Calendar } from 'lucide-react';
import { reviewsAPI } from '../services/api';
import { ReviewResponse, ReviewFilters, Pagination } from '../types';

const ReviewsList = () => {
  const [reviews, setReviews] = useState<ReviewResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<ReviewFilters>({
    location: '',
    sentiment: '',
    search: '',
  });
  const [pagination, setPagination] = useState<Pagination & { total: number; total_pages: number }>({
    page: 1,
    size: 10,
    total: 0,
    total_pages: 0,
  });

  const loadReviews = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = {
        ...filters,
        page: pagination.page,
        size: pagination.size,
      };
      
      const data = await reviewsAPI.getReviews(params);
      setReviews(data.reviews);
      setPagination(prev => ({
        ...prev,
        total: data.total,
        total_pages: data.total_pages,
      }));
    } catch (err) {
      setError('Failed to load reviews. Please try again.');
      console.error('Error loading reviews:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReviews();
  }, [filters, pagination.page, pagination.size]);

  const handleFilterChange = (key: keyof ReviewFilters, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setPagination(prev => ({ ...prev, page: 1 })); // Reset to first page
  };

  const handlePageChange = (newPage: number) => {
    setPagination(prev => ({ ...prev, page: newPage }));
  };

  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <span
        key={i}
        className={`rating-star ${i < rating ? '' : 'empty'}`}
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
    return new Date(dateString).toLocaleDateString();
  };

  if (loading && reviews.length === 0) {
    return (
      <div className="loading">
        <div>Loading reviews...</div>
      </div>
    );
  }
console.log(reviews);
  return (
    <div>
      <div className="card">
        <h2 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center' }}>
          <Search size={20} style={{ marginRight: '0.5rem' }} />
          Reviews Management
        </h2>
        
        {error && <div className="error">{error}</div>}
        
        <div className="filters">
          <div className="form-group">
            <label className="form-label">Search Text</label>
            <input
              type="text"
              className="form-input"
              placeholder="Search in review text..."
              value={filters.search || ''}
              onChange={(e) => handleFilterChange('search', e.target.value)}
            />
          </div>
          
          <div className="form-group">
            <label className="form-label">Location</label>
            <select
              className="form-select"
              value={filters.location}
              onChange={(e) => handleFilterChange('location', e.target.value)}
            >
              <option value="">All Locations</option>
              <option value="NYC">NYC</option>
              <option value="SF">SF</option>
              <option value="LA">LA</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">Sentiment</label>
            <select
              className="form-select"
              value={filters.sentiment}
              onChange={(e) => handleFilterChange('sentiment', e.target.value)}
            >
              <option value="">All Sentiments</option>
              <option value="positive">Positive</option>
              <option value="negative">Negative</option>
              <option value="neutral">Neutral</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3>Reviews ({pagination.total})</h3>
          <button
            className="btn btn-primary btn-sm"
            onClick={loadReviews}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {reviews.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#6b7280' }}>
            No reviews found matching your criteria.
          </div>
        ) : (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Location</th>
                  <th>Rating</th>
                  <th>Text</th>
                  <th>Sentiment</th>
                  <th>Topic</th>
                  <th>Date</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {reviews.map((review) => (
                  <tr key={review.id}>
                    <td>{review.id}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <MapPin size={14} style={{ marginRight: '0.25rem' }} />
                        {review.location}
                      </div>
                    </td>
                    <td>
                      <div className="rating">
                        {renderStars(review.rating)}
                        <span style={{ marginLeft: '0.5rem' }}>({review.rating})</span>
                      </div>
                    </td>
                    <td style={{ maxWidth: '300px', wordWrap: 'break-word' }}>
                      {review.text.length > 100 
                        ? `${review.text.substring(0, 100)}...` 
                        : review.text
                      }
                    </td>
                    <td>
                      {review.sentiment && (
                            <span className={getSentimentClass(review.sentiment?.replace(/^"|"$/g, ''))}>
  {review.sentiment?.replace(/^"|"$/g, '')}
</span>
                      )}
                    </td>
                    <td>
                      {review.topic && (
                        <span className="topic">
                          {review.topic}
                        </span>
                      )}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <Calendar size={14} style={{ marginRight: '0.25rem' }} />
                        {formatDate(review.date)}
                      </div>
                    </td>
                    <td>
                      <Link
                        to={`/reviews/${review.id}`}
                        className="btn btn-primary btn-sm"
                      >
                        View Details
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {pagination.total_pages > 1 && (
              <div className="pagination">
                <button
                  onClick={() => handlePageChange((pagination.page || 1) - 1)}
                  disabled={(pagination.page || 1) === 1}
                >
                  Previous
                </button>
                
                {Array.from({ length: Math.min(5, pagination.total_pages) }, (_, i) => {
                  const pageNum = i + 1;
                  return (
                    <button
                      key={pageNum}
                      onClick={() => handlePageChange(pageNum)}
                      className={(pagination.page || 1) === pageNum ? 'active' : ''}
                    >
                      {pageNum}
                    </button>
                  );
                })}
                
                <button
                  onClick={() => handlePageChange((pagination.page || 1) + 1)}
                  disabled={(pagination.page || 1) === pagination.total_pages}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ReviewsList;
