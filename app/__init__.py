"""
Video Recommendation Engine

A hybrid recommendation system that suggests personalized video content 
based on user preferences and engagement patterns.
"""

from .database import Base, engine, SessionLocal, get_db, User, Category, Topic, Post, UserInteraction, Tag
from .recommendation_engine import RecommendationEngine
from .schemas import UserBase, UserCreate, CategoryBase, CategoryCreate, \
    TopicBase, TopicCreate, PostBase, PostCreate, UserInteractionBase, \
    UserInteractionCreate, RecommendationResponse, FeedResponse, \
    TrainResponse, RecommendationRequest, ErrorResponse, HealthCheck

__version__ = "0.1.0"
