from typing import List, Dict, Tuple, Optional, Generator
import numpy as np
from sqlalchemy.orm import Session, joinedload
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import logging

from .database import SessionLocal, Post, UserInteraction, User, Category, Topic, Tag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, db: Session):
        self.db = db
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.user_item_matrix = None
        self.svd_model = None
        self.user_to_index = {}
        # Separate mappings for content vs collaborative indices
        self.content_post_to_index = {}
        self.content_index_to_post = {}
        self.collab_post_to_index = {}
        self.collab_index_to_post = {}
        
    def _get_post_features(self, post: Post) -> str:
        """Extract features from a post for content-based filtering."""
        # Eager load relationships to avoid N+1 queries
        post = self.db.query(Post).options(
            joinedload(Post.category),
            joinedload(Post.topic),
            joinedload(Post.tags)
        ).filter(Post.id == post.id).first()
        
        if not post:
            return ""
            
        features = [
            post.title or "",
            post.category.name if post.category else "",
            post.topic.name if post.topic else "",
            " ".join([tag.name for tag in post.tags]) if post.tags else "",
            post.owner_username or ""
        ]
        return " ".join(filter(None, features))
    
    def build_content_similarity_matrix(self):
        """Build a content similarity matrix using TF-IDF and cosine similarity."""
        logger.info("Building content similarity matrix...")
        
        try:
            # Fetch all posts with necessary relationships
            posts = self.db.query(Post).options(
                joinedload(Post.category),
                joinedload(Post.topic),
                joinedload(Post.tags)
            ).all()
            
            if not posts:
                logger.warning("No posts found in the database.")
                return
                
            # Create content mappings
            self.content_post_to_index = {post.id: idx for idx, post in enumerate(posts)}
            self.content_index_to_post = {idx: post.id for idx, post in enumerate(posts)}
            
            # Extract features
            post_features = [self._get_post_features(post) for post in posts]
            
            if not post_features or not any(post_features):
                logger.warning("No valid post features found.")
                return
            
            # Create TF-IDF vectors
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.8  # Ignore terms that appear in more than 80% of documents
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(post_features)
            
            if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
                logger.warning("TF-IDF matrix is empty.")
                return
            
            # Calculate cosine similarity
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            logger.info(f"Content similarity matrix built successfully. Shape: {self.content_similarity_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error building content similarity matrix: {str(e)}", exc_info=True)
            self.content_similarity_matrix = None
    
    def build_user_item_matrix(self):
        """Build user-item interaction matrix for collaborative filtering."""
        logger.info("Building user-item matrix...")
        
        try:
            # Fetch all interactions with related data
            interactions = self.db.query(UserInteraction).options(
                joinedload(UserInteraction.user),
                joinedload(UserInteraction.post)
            ).all()
            
            if not interactions:
                logger.warning("No user interactions found in the database.")
                return
            
            # Get unique users and posts
            users = list({interaction.username for interaction in interactions})
            posts = list({interaction.post_id for interaction in interactions if interaction.post_id is not None})
            
            if not users or not posts:
                logger.warning("Not enough data to build user-item matrix.")
                return
            
            # Create collaborative mappings
            self.user_to_index = {user: idx for idx, user in enumerate(users)}
            self.collab_post_to_index = {post: idx for idx, post in enumerate(posts)}
            self.collab_index_to_post = {idx: post for post, idx in self.collab_post_to_index.items()}
            
            # Initialize user-item matrix
            user_item_matrix = np.zeros((len(users), len(posts)))
            
            # Fill the matrix with interaction weights
            for interaction in interactions:
                if interaction.post_id is None:
                    continue
                    
                user_idx = self.user_to_index.get(interaction.username)
                post_idx = self.collab_post_to_index.get(interaction.post_id)
                
                if user_idx is not None and post_idx is not None:
                    # Weight different types of interactions
                    if interaction.interaction_type == "like":
                        user_item_matrix[user_idx, post_idx] = 1.0
                    elif interaction.interaction_type == "inspire":
                        user_item_matrix[user_idx, post_idx] = 1.5
                    elif interaction.interaction_type == "rating" and interaction.rating:
                        # Normalize rating to 0-1 range
                        user_item_matrix[user_idx, post_idx] = interaction.rating / 5.0
                    elif interaction.interaction_type == "view":
                        # Lower weight for views to not dominate other interactions
                        user_item_matrix[user_idx, post_idx] = 0.3
            
            # Convert to sparse matrix to save memory
            self.user_item_matrix = csr_matrix(user_item_matrix)
            logger.info(f"User-item matrix built successfully. Shape: {self.user_item_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error building user-item matrix: {str(e)}", exc_info=True)
            self.user_item_matrix = None
    
    def train_collaborative_filtering(self, n_components=50):
        """
        Train SVD model for collaborative filtering.
        
        Args:
            n_components: Number of components for SVD (dimensionality reduction)
        """
        try:
            if self.user_item_matrix is None:
                logger.warning("User-item matrix not built. Building it first...")
                self.build_user_item_matrix()
            
            if self.user_item_matrix is None or self.user_item_matrix.shape[1] < 2:
                logger.error("Insufficient data to train SVD model. Need at least 2 items.")
                return False
                
            # Ensure n_components is less than both the number of users and items
            n_users, n_items = self.user_item_matrix.shape
            max_components = min(n_components, n_users - 1, n_items - 1)
            if max_components < 1:
                logger.error("Not enough items to perform SVD.")
                return False
                
            logger.info(f"Training SVD model with {max_components} components...")
            
            # Initialize and train the SVD model
            self.svd_model = TruncatedSVD(
                n_components=max_components,
                algorithm='randomized',  # More robust for small/ill-conditioned matrices
                random_state=42
            )
            
            self.svd_model.fit(self.user_item_matrix)
            
            # Log explained variance ratio
            explained_variance = sum(self.svd_model.explained_variance_ratio_)
            logger.info(
                f"SVD model trained successfully. "
                f"Explained variance: {explained_variance:.2%} with {max_components} components"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error training SVD model: {str(e)}", exc_info=True)
            self.svd_model = None
            return False
    
    def get_content_based_recommendations(self, post_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get content-based recommendations for a given post.
        
        Args:
            post_id: ID of the post to get recommendations for
            top_n: Number of recommendations to return
            
        Returns:
            List of tuples containing (post_id, similarity_score) sorted by score descending
        """
        try:
            # Validate input
            if not isinstance(post_id, int) or post_id <= 0:
                logger.error(f"Invalid post_id: {post_id}")
                return []
                
            if top_n <= 0:
                logger.warning(f"top_n must be positive, got {top_n}. Using default value 10.")
                top_n = 10
            
            # Build similarity matrix if not already built
            if self.content_similarity_matrix is None:
                logger.info("Content similarity matrix not built. Building it first...")
                self.build_content_similarity_matrix()
            
            if self.content_similarity_matrix is None:
                logger.error("Failed to build content similarity matrix.")
                return []
                
            # Check if post exists in our matrix
            if post_id not in self.content_post_to_index:
                logger.warning(f"Post ID {post_id} not found in the similarity matrix.")
                return []
                
            # Get the index of the input post
            idx = self.content_post_to_index[post_id]
            
            # Get similarity scores for the post
            sim_scores = list(enumerate(self.content_similarity_matrix[idx]))
            
            # Sort posts by similarity score in descending order
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the top N most similar posts (excluding the input post itself)
            sim_scores = sim_scores[1:top_n+1]  # +1 because we're skipping the first one
            
            # Map indices back to post IDs and return with scores
            recommendations = []
            for i, score in sim_scores:
                if i in self.content_index_to_post:  # Extra safety check
                    recommendations.append((self.content_index_to_post[i], float(score)))
            
            logger.info(f"Generated {len(recommendations)} content-based recommendations for post {post_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating collaborative recommendations: {str(e)}", exc_info=True)
            return []

    def get_hybrid_recommendations(
        self, 
        username: str, 
        project_code: str = None, 
        num_recommendations: int = 10
    ) -> List[Dict]:
        """
        Get hybrid recommendations combining:
        1. Content-Based Filtering (40%)
        2. Collaborative Filtering (40%)
        3. Popularity-Based (20%)
        
        Args:
            username: Username to get recommendations for
            project_code: Optional project code to filter recommendations
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of dictionaries containing post_id and score
        """
        try:
            logger.info(f"Generating hybrid recommendations for user: {username}")
            
            # Initialize models if needed
            if self.content_similarity_matrix is None:
                logger.info("Building content similarity matrix...")
                self.build_content_similarity_matrix()
                
            if self.user_item_matrix is None:
                logger.info("Building user-item matrix...")
                self.build_user_item_matrix()
                
            if self.svd_model is None:
                logger.info("Training collaborative filtering model...")
                success = self.train_collaborative_filtering()
                if not success:
                    logger.error("Failed to train collaborative filtering model")
                    return []

            all_recommendations = defaultdict(float)

            # 1. CONTENT-BASED FILTERING (40% weight)
            try:
                # Get user's recently viewed posts
                recent_views = (
                    self.db.query(UserInteraction.post_id)
                    .filter(
                        UserInteraction.username == username,
                        UserInteraction.interaction_type == "view"
                    )
                    .order_by(UserInteraction.created_at.desc())
                    .limit(5)
                    .all()
                )

                if recent_views:
                    for (post_id,) in recent_views:
                        content_recs = self.get_content_based_recommendations(post_id, num_recommendations)
                        for post_id, score in content_recs:
                            all_recommendations[post_id] += score * 0.4
                    logger.info(f"Added content-based recommendations from {len(recent_views)} recent views")
                else:
                    logger.info("No recent views found for content-based recommendations")
                    
            except Exception as e:
                logger.error(f"Error in content-based filtering: {str(e)}", exc_info=True)
                # Continue with other methods even if one fails

            # 2. COLLABORATIVE FILTERING (40% weight)
            try:
                collab_recs = self.get_collaborative_recommendations(username, num_recommendations, project_code)
                if collab_recs:
                    for post_id, score in collab_recs:
                        all_recommendations[post_id] += score * 0.4
                    logger.info(f"Added collaborative filtering recommendations: {len(collab_recs)} items")
                else:
                    logger.info("No collaborative filtering recommendations available")
                
            except Exception as e:
                logger.error(f"Error in collaborative filtering: {str(e)}", exc_info=True)
                # Continue with other methods even if one fails

            # 3. POPULARITY-BASED (20% weight)
            # If we don't have enough recommendations, fill with popular posts
            if len(all_recommendations) < num_recommendations:
                try:
                    # Calculate popularity score using a weighted sum of different metrics
                    popularity_score = (
                        (Post.view_count * 0.3) + 
                        (Post.upvote_count * 0.5) + 
                        (Post.average_rating * 0.2)
                    )
                    
                    # Get popular posts not already in recommendations
                    query = self.db.query(Post).filter(~Post.id.in_(all_recommendations.keys()))
                    if project_code:
                        query = query.filter(Post.topic.has(project_code=project_code))
                    popular_posts = (
                        query
                        .order_by(popularity_score.desc())
                        .limit(num_recommendations - len(all_recommendations))
                        .all()
                    )
                    
                    # Add popular posts to recommendations with base score
                    base_popularity_score = 0.2  # 20% weight for popularity
                    for post in popular_posts:
                        all_recommendations[post.id] = max(all_recommendations.get(post.id, 0), base_popularity_score)
                        
                    logger.info(f"Added {len(popular_posts)} popular recommendations")
                    
                except Exception as e:
                    logger.error(f"Error in popularity-based recommendations: {str(e)}", exc_info=True)

            # Apply project_code filter to accumulated recommendations if requested
            items_iter = all_recommendations.items()
            if project_code:
                allowed_ids = set(
                    pid for (pid,) in self.db.query(Post.id).filter(Post.topic.has(project_code=project_code)).all()
                )
                items_iter = ((pid, sc) for pid, sc in items_iter if pid in allowed_ids)

            # Sort recommendations by score in descending order
            sorted_recommendations = sorted(
                items_iter,
                key=lambda x: x[1],
                reverse=True
            )[:num_recommendations]
            
            # Convert to list of dicts with post_id, score, and recommendation_type
            result = [
                {"post_id": post_id, "score": float(score), "recommendation_type": "hybrid"}
                for post_id, score in sorted_recommendations
            ]
            
            logger.info(f"Generated {len(result)} hybrid recommendations for user {username}")
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in get_hybrid_recommendations: {str(e)}", exc_info=True)
            return []

    def get_collaborative_recommendations(
        self,
        username: str,
        top_n: int = 10,
        project_code: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """Recommend posts using the trained SVD model for a given user.

        Returns a list of (post_id, score) for items the user hasn't interacted with yet.
        Optionally filters by topic.project_code.
        """
        try:
            # Ensure models are ready
            if self.user_item_matrix is None:
                self.build_user_item_matrix()
            if self.user_item_matrix is None:
                return []
            if self.svd_model is None:
                ok = self.train_collaborative_filtering()
                if not ok:
                    return []

            # User must exist in mapping
            if username not in self.user_to_index:
                return []

            user_idx = self.user_to_index[username]

            # Predict scores for all items by reconstructing the user's row
            user_row = self.user_item_matrix.getrow(user_idx)
            try:
                user_embedding = self.svd_model.transform(user_row)
                reconstructed = self.svd_model.inverse_transform(user_embedding)[0]
            except Exception:
                return []

            # Exclude items the user has already interacted with
            seen_indices = set(user_row.indices.tolist())

            # Build list of candidate (post_id, score)
            candidates: List[Tuple[int, float]] = []
            for post_id, idx in self.collab_post_to_index.items():
                if idx in seen_indices:
                    continue
                # Optional project filter
                if project_code:
                    post_obj = self.db.query(Post).get(post_id)
                    if not post_obj or not post_obj.topic or post_obj.topic.project_code != project_code:
                        continue
                candidates.append((post_id, float(reconstructed[idx])))

            # Sort by predicted score
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_n]
        except Exception as e:
            logger.error(f"Error in get_collaborative_recommendations: {str(e)}", exc_info=True)
            return []
    
    def get_cold_start_recommendations(self, project_code: str = None, limit: int = 10) -> List[Dict]:
        """Get cold start recommendations based on popularity and diversity."""
        query = self.db.query(Post)
        
        if project_code:
            query = query.filter(Post.topic.has(project_code=project_code))
        
        posts = query.all()
        
        if not posts:
            return []
        
        # Calculate diverse popularity scores
        post_scores = []
        categories_shown = set()
        topics_shown = set()
        
        for post in posts:
            # Base popularity score
            popularity = (
                (post.view_count or 0) * 0.3 +
                (post.upvote_count or 0) * 0.5 +
                ((post.average_rating or 0) / 100) * 20  # Assuming rating is 0-100
            )
            
            # Diversity bonus (promote variety in categories/topics)
            diversity_bonus = 0
            if post.category and post.category.name not in categories_shown:
                diversity_bonus += 10
                categories_shown.add(post.category.name)
            
            if post.topic and post.topic.name not in topics_shown:
                diversity_bonus += 5
                topics_shown.add(post.topic.name)
            
            final_score = popularity + diversity_bonus
            post_scores.append((post, final_score))
        
        # Sort by score and return top results
        post_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [{"post_id": post.id, "score": score, "recommendation_type": "cold_start"} 
                for post, score in post_scores[:limit]]
