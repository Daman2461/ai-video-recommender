from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Table, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime
from typing import Generator
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL configuration
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recommendation.db")

# Create SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Association table for many-to-many relationship between posts and tags
post_tags = Table(
    'post_tags',
    Base.metadata,
    Column('post_id', Integer, ForeignKey('posts.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    picture_url = Column(String)
    user_type = Column(String, nullable=True)
    has_evm_wallet = Column(Boolean, default=False)
    has_solana_wallet = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    posts = relationship("Post", back_populates="owner")
    interactions = relationship("UserInteraction", back_populates="user")

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text)
    image_url = Column(String)
    count = Column(Integer, default=0)
    
    # Relationships
    posts = relationship("Post", back_populates="category")
    topics = relationship("Topic", back_populates="category")

class Topic(Base):
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(Text)
    image_url = Column(String)
    slug = Column(String, unique=True, index=True)
    is_public = Column(Boolean, default=True)
    project_code = Column(String, index=True)
    posts_count = Column(Integer, default=0)
    language = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_username = Column(String, ForeignKey("users.username"), nullable=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    
    # Relationships
    posts = relationship("Post", back_populates="topic")
    owner = relationship("User", foreign_keys=[owner_username], primaryjoin="User.username == Topic.owner_username")
    category = relationship("Category", back_populates="topics")

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    video_link = Column(String)
    thumbnail_url = Column(String)
    gif_thumbnail_url = Column(String, nullable=True)
    is_available_in_public_feed = Column(Boolean, default=True)
    is_locked = Column(Boolean, default=False)
    slug = Column(String, unique=True, index=True)
    identifier = Column(String, unique=True, index=True)
    comment_count = Column(Integer, default=0)
    upvote_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    exit_count = Column(Integer, default=0)
    rating_count = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    share_count = Column(Integer, default=0)
    bookmark_count = Column(Integer, default=0)
    contract_address = Column(String, nullable=True)
    chain_id = Column(String, nullable=True)
    chart_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign keys
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False)
    owner_username = Column(String, ForeignKey("users.username"), nullable=False)
    
    # Relationships
    category = relationship("Category", back_populates="posts")
    topic = relationship("Topic", back_populates="posts")
    owner = relationship("User", back_populates="posts")
    interactions = relationship("UserInteraction", back_populates="post")
    tags = relationship("Tag", secondary=post_tags, back_populates="posts")

class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    
    # Relationships
    posts = relationship("Post", secondary=post_tags, back_populates="tags")

class UserInteraction(Base):
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    interaction_type = Column(Enum("view", "like", "inspire", "rating", name="interaction_type"), nullable=False)
    rating = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign keys
    username = Column(String, ForeignKey("users.username"), nullable=False)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    post = relationship("Post", back_populates="interactions")

# Dependency to get DB session
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

