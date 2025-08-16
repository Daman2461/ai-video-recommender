from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Base schemas
class UserBase(BaseModel):
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    picture_url: Optional[str] = None

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: Optional[int] = None
    user_type: Optional[str] = None
    has_evm_wallet: bool = False
    has_solana_wallet: bool = False

    class Config:
        from_attributes = True

class CategoryBase(BaseModel):
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None

class CategoryCreate(CategoryBase):
    pass

class Category(CategoryBase):
    id: int
    count: int = 0

    class Config:
        from_attributes = True

class TopicBase(BaseModel):
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    project_code: Optional[str] = None
    is_public: bool = True
    language: Optional[str] = None

class TopicCreate(TopicBase):
    pass

class Topic(TopicBase):
    id: int
    slug: str
    posts_count: int = 0
    # per output-data-format.md Topic.created_at is a formatted string
    created_at: str
    # Topic owner differs from Post owner (uses profile_url, no id)
    owner: Optional["TopicOwner"] = None

    class Config:
        from_attributes = True

class TopicOwner(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    profile_url: Optional[str] = None
    user_type: Optional[str] = None
    has_evm_wallet: bool = False
    has_solana_wallet: bool = False

class PostBase(BaseModel):
    title: str
    video_link: str
    thumbnail_url: str
    is_available_in_public_feed: bool = True
    is_locked: bool = False
    slug: str
    identifier: str
    comment_count: int = 0
    upvote_count: int = 0
    view_count: int = 0
    exit_count: int = 0
    rating_count: int = 0
    average_rating: int = 0
    share_count: int = 0
    bookmark_count: int = 0
    # Client-state booleans (not persisted)
    upvoted: bool = False
    bookmarked: bool = False
    following: bool = False
    gif_thumbnail_url: Optional[str] = None
    contract_address: Optional[str] = None
    chain_id: Optional[str] = None
    chart_url: Optional[str] = None
    # milliseconds since epoch per output-data-format.md
    created_at: int
    tags: List[str] = []
    # Optional base token structure (empty by default)
    baseToken: Optional[Dict[str, Optional[str]]] = None

class PostOwner(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    picture_url: Optional[str] = None
    user_type: Optional[str] = None
    has_evm_wallet: bool = False
    has_solana_wallet: bool = False

class PostCreate(PostBase):
    category_id: int
    topic_id: int
    owner_username: str

class Post(PostBase):
    id: int
    category: Optional[Category] = None
    topic: Optional[Topic] = None
    owner: Optional[PostOwner] = None

    class Config:
        from_attributes = True

class UserInteractionBase(BaseModel):
    username: str
    post_id: int
    interaction_type: str
    rating: Optional[float] = None

class UserInteractionCreate(UserInteractionBase):
    pass

class UserInteraction(UserInteractionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Response models
class RecommendationResponse(BaseModel):
    post_id: int
    score: float
    recommendation_type: str
    post: Optional[Post] = None

class FeedResponse(BaseModel):
    status: str = "success"
    post: List[Post] = []
    items: List[Post] = []

class TrainResponse(BaseModel):
    status: str
    message: str

# Request models
class RecommendationRequest(BaseModel):
    username: str
    project_code: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)

# Error responses
class ErrorResponse(BaseModel):
    detail: str

# API status
class HealthCheck(BaseModel):
    status: str
    version: str
    database_status: str
