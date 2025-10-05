// Basic types for the Reviews Copilot application

export interface Review {
  id: number;
  location: string;
  rating: number;
  text: string;
  date: string;
  sentiment?: string;
  topic?: string;
  created_at?: string;
  updated_at?: string;
  processed_at?: string;
  metadata?: string;
}

export interface ReviewResponse extends Review {
  // Additional fields that might be returned by the API
}

export interface IngestRequest {
  reviews: Review[];
}

export interface IngestResponse {
  message: string;
  count: number;
  processing_time: number;
}

export interface ReviewsListResponse {
  reviews: ReviewResponse[];
  total: number;
  page: number;
  size: number;
  total_pages: number;
}

export interface SuggestReplyResponse {
  reply: string;
  reasoning_log: string;
  tags: {
    sentiment: string;
    topic: string;
  };
}

export interface AnalyticsResponse {
  sentiment_counts: Record<string, number>;
  topic_counts: Record<string, number>;
  location_counts: Record<string, number>;
  rating_distribution: Record<string, number>;
  total_reviews: number;
}

export interface SearchResponse {
  query: string;
  results: Array<{
    review: ReviewResponse;
    text: string;
    similarity: number;
  }>;
  total: number;
}

export interface ProcessReviewsResponse {
  message: string;
  processed_count: number;
  processing_time: number;
}

export interface HealthCheckResponse {
  status: string;
  timestamp: string;
  version: string;
  database: string;
  ai_service: {
    ai_enabled: boolean;
    sentiment_model_loaded: boolean;
    summarization_model_loaded: boolean;
    search_index_ready: boolean;
    cached_entries: number;
    reviews_indexed: number;
    performance_stats: Record<string, any>;
  };
  uptime?: number;
  error?: string;
}

export interface ReviewFilters {
  location?: string;
  sentiment?: string;
  rating?: number;
  topic?: string;
  search?: string;
}

export interface Pagination {
  page?: number;
  size?: number;
}

export interface ConnectionStatus {
  isOnline: boolean;
  apiStatus: 'healthy' | 'unhealthy' | 'unknown' | 'checking';
  lastChecked: Date;
}
