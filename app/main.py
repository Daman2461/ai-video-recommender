from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv
import httpx

from . import schemas
from .database import (
    SessionLocal,
    get_db,
    Post as ORMPost,
    UserInteraction as ORMUserInteraction,
    User as ORMUser,
    Category as ORMCategory,
    Topic as ORMTopic,
    Tag as ORMTag,
)
from .recommendation_engine import RecommendationEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Video Recommendation API",
    description="A hybrid recommendation system for video content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recommendation engine factory (no global caching to avoid stale DB sessions)
def get_recommendation_engine(db: Session = Depends(get_db)):
    return RecommendationEngine(db)

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup."""
    logger.info("Starting up the recommendation engine...")
    db = SessionLocal()
    try:
        engine = RecommendationEngine(db)
        # Pre-build models in the background
        engine.build_content_similarity_matrix()
        engine.build_user_item_matrix()
        engine.train_collaborative_filtering()
        logger.info("Recommendation engine initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing recommendation engine: {e}")
    finally:
        db.close()

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "Welcome to the Video Recommendation API",
        "status": "running",
        "endpoints": [
            {"path": "/docs", "description": "API documentation"},
            {"path": "/feed", "description": "Get personalized video feed"},
            {"path": "/train", "description": "Retrain recommendation models"}
        ]
    }

def _external_headers() -> dict:
    token = os.getenv("FLIC_TOKEN")
    if not token:
        logger.warning("FLIC_TOKEN is not set; external API calls may fail")
    return {"Flic-Token": token} if token else {}

def _external_base() -> str:
    base = os.getenv("API_BASE_URL", "https://api.socialverseapp.com")
    return base.rstrip("/")

async def _proxy_get(path: str, request: Request) -> dict:
    url = f"{_external_base()}{path}"
    headers = _external_headers()
    params = dict(request.query_params)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"External API error {e.response.status_code} for {url}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Error calling external API {url}: {e}")
        raise HTTPException(status_code=502, detail="Bad Gateway: external service error")


# =============================
# Local Data Management Endpoints
# =============================

@app.post("/local/users", response_model=schemas.User, tags=["Local Data"])
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(ORMUser).filter(ORMUser.username == user.username).first()
    if existing:
        return existing
    obj = ORMUser(
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        picture_url=user.picture_url,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

@app.get("/local/interactions", tags=["Local Data"])
def list_interactions(
    username: Optional[str] = Query(None, description="Filter by username"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    q = db.query(ORMUserInteraction)
    if username:
        q = q.filter(ORMUserInteraction.username == username)
    q = q.order_by(ORMUserInteraction.created_at.desc()).limit(limit)
    items = q.all()
    return [
        {
            "id": it.id,
            "username": it.username,
            "post_id": it.post_id,
            "interaction_type": it.interaction_type,
            "rating": it.rating,
            "created_at": it.created_at.isoformat() if it.created_at else None,
        }
        for it in items
    ]


@app.delete("/local/interactions", tags=["Local Data", "Admin"])
def delete_interaction(
    username: str = Query(..., description="Username"),
    post_id: int = Query(..., description="Post ID"),
    interaction_type: Optional[str] = Query(None, description="Interaction type to delete (optional)"),
    db: Session = Depends(get_db),
):
    q = db.query(ORMUserInteraction).filter(
        ORMUserInteraction.username == username,
        ORMUserInteraction.post_id == post_id,
    )
    if interaction_type:
        q = q.filter(ORMUserInteraction.interaction_type == interaction_type)
    deleted = q.delete(synchronize_session=False)
    db.commit()
    return {"status": "success", "deleted": int(deleted)}

@app.post("/local/categories", response_model=schemas.Category, tags=["Local Data"])
def create_category(cat: schemas.CategoryCreate, db: Session = Depends(get_db)):
    existing = db.query(ORMCategory).filter(ORMCategory.name == cat.name).first()
    if existing:
        return existing
    obj = ORMCategory(name=cat.name, description=cat.description, image_url=cat.image_url)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

@app.post("/local/topics", response_model=schemas.Topic, tags=["Local Data"])
def create_topic(topic: schemas.TopicCreate, db: Session = Depends(get_db)):
    # Auto-generate simple slug if not provided via name
    slug = (topic.name or "topic").lower().replace(" ", "-")
    existing = db.query(ORMTopic).filter(ORMTopic.slug == slug).first()
    if existing:
        return existing
    obj = ORMTopic(
        name=topic.name,
        description=topic.description,
        image_url=topic.image_url,
        slug=slug,
        is_public=topic.is_public,
        project_code=topic.project_code,
        language=topic.language,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

@app.post("/local/posts", response_model=schemas.Post, tags=["Local Data"])
def create_post(post: schemas.PostCreate, db: Session = Depends(get_db)):
    # Ensure owner exists
    owner = db.query(ORMUser).filter(ORMUser.username == post.owner_username).first()
    if not owner:
        raise HTTPException(status_code=400, detail="owner_username does not exist")

    # Ensure category/topic exist
    category = db.query(ORMCategory).filter(ORMCategory.id == post.category_id).first()
    topic = db.query(ORMTopic).filter(ORMTopic.id == post.topic_id).first()
    if not category:
        raise HTTPException(status_code=400, detail="category_id not found")
    if not topic:
        raise HTTPException(status_code=400, detail="topic_id not found")

    # Upsert by slug or identifier
    existing = db.query(ORMPost).filter((ORMPost.slug == post.slug) | (ORMPost.identifier == post.identifier)).first()
    if existing:
        obj = existing
    else:
        obj = ORMPost(slug=post.slug, identifier=post.identifier)
        db.add(obj)

    # Assign fields
    obj.title = post.title
    obj.video_link = post.video_link
    obj.thumbnail_url = post.thumbnail_url
    obj.gif_thumbnail_url = post.gif_thumbnail_url
    obj.is_available_in_public_feed = bool(post.is_available_in_public_feed)
    obj.is_locked = bool(post.is_locked)
    obj.comment_count = int(post.comment_count or 0)
    obj.upvote_count = int(post.upvote_count or 0)
    obj.view_count = int(post.view_count or 0)
    obj.exit_count = int(post.exit_count or 0)
    obj.rating_count = int(post.rating_count or 0)
    # average_rating in schema is int (0-100). DB expects float
    obj.average_rating = float(post.average_rating or 0)
    obj.share_count = int(post.share_count or 0)
    obj.bookmark_count = int(post.bookmark_count or 0)
    obj.contract_address = post.contract_address or ""
    obj.chain_id = post.chain_id or ""
    obj.chart_url = post.chart_url or ""
    obj.category_id = category.id
    obj.topic_id = topic.id
    obj.owner_username = post.owner_username

    # created_at expects ms in schema; ORM expects datetime. If provided, keep existing default.
    # We won't parse here to keep API simple.

    db.commit()
    db.refresh(obj)
    return obj

@app.post("/local/interactions", response_model=schemas.UserInteraction, tags=["Local Data"])
def create_interaction(inter: schemas.UserInteractionCreate, db: Session = Depends(get_db)):
    # Validate user and post
    user = db.query(ORMUser).filter(ORMUser.username == inter.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="username does not exist")
    post = db.query(ORMPost).filter(ORMPost.id == inter.post_id).first()
    if not post:
        raise HTTPException(status_code=400, detail="post_id does not exist")

    obj = ORMUserInteraction(
        username=inter.username,
        post_id=inter.post_id,
        interaction_type=inter.interaction_type,
        rating=float(inter.rating) if inter.interaction_type == "rating" and inter.rating is not None else None,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.get("/data/posts/view", tags=["Data Collection"])  # GET /posts/view
async def get_all_viewed_posts(request: Request):
    return await _proxy_get("/posts/view", request)


@app.get("/data/posts/like", tags=["Data Collection"])  # GET /posts/like
async def get_all_liked_posts(request: Request):
    return await _proxy_get("/posts/like", request)


@app.get("/data/posts/inspire", tags=["Data Collection"])  # GET /posts/inspire
async def get_all_inspired_posts(request: Request):
    return await _proxy_get("/posts/inspire", request)


@app.get("/data/posts/rating", tags=["Data Collection"])  # GET /posts/rating
async def get_all_rated_posts(request: Request):
    return await _proxy_get("/posts/rating", request)


@app.get("/data/posts/summary", tags=["Data Collection"])  # GET /posts/summary/get
async def get_all_posts_summary(request: Request):
    return await _proxy_get("/posts/summary/get", request)


@app.get("/data/users", tags=["Data Collection"])  # GET /users/get_all
async def get_all_users(request: Request):
    return await _proxy_get("/users/get_all", request)

@app.post("/admin/import_external", tags=["Admin"])
async def import_external_data(
    project_code: Optional[str] = Query(None, description="Filter by project code"),
    db: Session = Depends(get_db),
):
     
    base = _external_base()
    headers = _external_headers()
    resonance = os.getenv("RESONANCE_ALGORITHM", "")

   
    if project_code:
        try:
            gen_topic = db.query(ORMTopic).filter(ORMTopic.slug == "general").first()
            if gen_topic and (not gen_topic.project_code or gen_topic.project_code != project_code):
                gen_topic.project_code = project_code
                db.commit()
        except Exception:
            db.rollback()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch users (header required)
        users_url = f"{base}/users/get_all"
        users_resp = await client.get(users_url, params={"page": 1, "page_size": 1000}, headers=headers)
        users_resp.raise_for_status()
        users_data = users_resp.json()
        users_list = users_data.get("users") or users_data.get("data") or users_data.get("items") or []

        # Upsert users
        created_users = 0
        for u in users_list:
            username = (u.get("username") or u.get("user", {}).get("username") or "").strip()
            if not username:
                continue
            existing = db.query(ORMUser).filter(ORMUser.username == username).first()
            if not existing:
                existing = ORMUser(username=username)
                db.add(existing)
                created_users += 1
            existing.first_name = u.get("first_name") or u.get("firstName")
            existing.last_name = u.get("last_name") or u.get("lastName")
            existing.picture_url = u.get("picture_url") or u.get("profile_url") or u.get("pictureUrl")

        # Fetch posts summary (header required)
        posts_url = f"{base}/posts/summary/get"
        posts_resp = await client.get(posts_url, params={"page": 1, "page_size": 1000}, headers=headers)
        posts_resp.raise_for_status()
        posts_data = posts_resp.json()
        posts_list = posts_data.get("post") or posts_data.get("posts") or posts_data.get("data") or []

        # Helper caches
        category_cache = {}
        topic_cache = {}

        def get_or_create_category(name: str) -> ORMCategory:
            key = (name or "General").strip() or "General"
            if key in category_cache:
                return category_cache[key]
            obj = db.query(ORMCategory).filter(ORMCategory.name == key).first()
            if not obj:
                obj = ORMCategory(name=key, description=None, image_url=None)
                db.add(obj)
                db.flush()
            category_cache[key] = obj
            return obj

        def get_or_create_topic(name: str, project_code: Optional[str], category_id: Optional[int] = None, owner_username: Optional[str] = None) -> ORMTopic:
            nm = (name or "General").strip() or "General"
            slug = nm.lower().replace(" ", "-")
            key = (slug, project_code or "")
            if key in topic_cache:
                obj = topic_cache[key]
            else:
                # Try to find by slug and, if provided, matching project_code
                q = db.query(ORMTopic).filter(ORMTopic.slug == slug)
                obj = q.first()
                if not obj:
                    obj = ORMTopic(
                        name=nm,
                        description=None,
                        image_url=None,
                        slug=slug,
                        is_public=True,
                        project_code=project_code,
                        language=None,
                    )
                    db.add(obj)
                    db.flush()
                # Cache regardless
                topic_cache[key] = obj
            # Backfill fields if missing on existing topics
            changed = False
            if project_code and not getattr(obj, "project_code", None):
                obj.project_code = project_code
                changed = True
            if category_id and getattr(obj, "category_id", None) != category_id:
                obj.category_id = category_id
                changed = True
            if owner_username and not getattr(obj, "owner_username", None):
                obj.owner_username = owner_username
                changed = True
            if changed:
                db.flush()
            return obj

        # Ensure a fallback owner exists
        def ensure_user(username: str, first_name: Optional[str] = None, last_name: Optional[str] = None, picture_url: Optional[str] = None) -> ORMUser:
            u = db.query(ORMUser).filter(ORMUser.username == username).first()
            if not u:
                u = ORMUser(username=username)
                db.add(u)
                db.flush()
            # Enrich user details when available
            if first_name and not getattr(u, "first_name", None):
                u.first_name = first_name
            if last_name and not getattr(u, "last_name", None):
                u.last_name = last_name
            if picture_url and not getattr(u, "picture_url", None):
                u.picture_url = picture_url
            return u

        created_posts = 0
        upserted_posts = 0
        # Tag cache/helper
        tag_cache = {}
        def get_or_create_tag(name: str) -> ORMTag:
            key = (name or "").strip()
            if not key:
                return None
            if key in tag_cache:
                return tag_cache[key]
            obj = db.query(ORMTag).filter(ORMTag.name == key).first()
            if not obj:
                obj = ORMTag(name=key)
                db.add(obj)
                db.flush()
            tag_cache[key] = obj
            return obj

        for p in posts_list:
            ext_id = p.get("id")
            slug = p.get("slug") or (f"post-{ext_id}" if ext_id is not None else None) or os.urandom(4).hex()
            identifier = str(ext_id) if ext_id is not None else slug

            owner_username = None
            owner_obj = p.get("owner") or {}
            if isinstance(owner_obj, dict):
                owner_username = owner_obj.get("username")
                owner_first = owner_obj.get("first_name") or owner_obj.get("firstName")
                owner_last = owner_obj.get("last_name") or owner_obj.get("lastName")
                owner_pic = owner_obj.get("picture_url") or owner_obj.get("profile_url") or owner_obj.get("pictureUrl")
            if not owner_username:
                owner_username = "system"
            ensure_user(owner_username, owner_first, owner_last, owner_pic)

            cat_name = None
            cat_obj = p.get("category") or {}
            if isinstance(cat_obj, dict):
                cat_name = cat_obj.get("name")
            category = get_or_create_category(cat_name)

            topic_name = None
            topic_project_code = None
            topic_obj = p.get("topic") or {}
            if isinstance(topic_obj, dict):
                topic_name = topic_obj.get("name")
                topic_project_code = topic_obj.get("project_code") or topic_obj.get("projectCode")
                # If topic has an explicit owner, try to enrich it
                t_owner = topic_obj.get("owner") or {}
                t_owner_username = None
                if isinstance(t_owner, dict):
                    t_owner_username = t_owner.get("username")
                    if t_owner_username:
                        ensure_user(
                            t_owner_username,
                            t_owner.get("first_name") or t_owner.get("firstName"),
                            t_owner.get("last_name") or t_owner.get("lastName"),
                            t_owner.get("picture_url") or t_owner.get("profile_url") or t_owner.get("pictureUrl"),
                        )
            # If topic name is missing, use category name as topic to create segmentation by category
            if not topic_name:
                topic_name = cat_name or "General"

            # Filter by project_code if provided
            if project_code and (str(topic_project_code or "").lower() != str(project_code).lower()):
                continue

            topic = get_or_create_topic(topic_name, topic_project_code, category_id=category.id, owner_username=t_owner_username or owner_username)

            existing = db.query(ORMPost).filter((ORMPost.slug == slug) | (ORMPost.identifier == identifier)).first()
            if not existing:
                existing = ORMPost(slug=slug, identifier=identifier)
                db.add(existing)
                created_posts += 1
            else:
                upserted_posts += 1

            existing.title = p.get("title") or slug
            existing.video_link = p.get("video_link") or p.get("videoLink")
            existing.thumbnail_url = p.get("thumbnail_url") or p.get("thumbnailUrl")
            existing.gif_thumbnail_url = p.get("gif_thumbnail_url") or p.get("gifThumbnailUrl")
            existing.is_available_in_public_feed = bool(p.get("is_available_in_public_feed", True))
            existing.is_locked = bool(p.get("is_locked", False))
            existing.comment_count = int(p.get("comment_count") or 0)
            existing.upvote_count = int(p.get("upvote_count") or 0)
            existing.view_count = int(p.get("view_count") or 0)
            existing.exit_count = int(p.get("exit_count") or 0)
            existing.rating_count = int(p.get("rating_count") or 0)
            try:
                existing.average_rating = float(p.get("average_rating") or 0)
            except Exception:
                existing.average_rating = 0.0
            existing.share_count = int(p.get("share_count") or 0)
            existing.bookmark_count = int(p.get("bookmark_count") or 0)
            existing.contract_address = p.get("contract_address") or ""
            existing.chain_id = p.get("chain_id") or ""
            existing.chart_url = p.get("chart_url") or ""
            existing.category_id = category.id
            existing.topic_id = topic.id
            existing.owner_username = owner_username
            # created_at: API provides ms; set if present
            try:
                created_ms = p.get("created_at") or p.get("createdAt")
                if created_ms:
                    # Accept ms or seconds
                    ts = float(created_ms) / (1000.0 if float(created_ms) > 1e12 else 1.0)
                    existing.created_at = datetime.utcfromtimestamp(ts)
            except Exception:
                pass

            # Tags
            tags = p.get("tags") or []
            tag_objs = []
            if isinstance(tags, list):
                for t in tags:
                    # t could be string or dict with name
                    if isinstance(t, str):
                        tag_obj = get_or_create_tag(t)
                    elif isinstance(t, dict):
                        tag_obj = get_or_create_tag(t.get("name") or t.get("slug") or "")
                    else:
                        tag_obj = None
                    if tag_obj:
                        tag_objs.append(tag_obj)
            if tag_objs:
                existing.tags = tag_objs

        # Interactions (view, like, inspire, rating)
        async def fetch_interactions(path: str):
            params = {"page": 1, "page_size": 1000}
            if resonance:
                params["resonance_algorithm"] = resonance
            url = f"{base}{path}"
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
            items = data.get("data") or data.get("items") or data.get("posts") or data.get("post") or []
            return items

        inter_types = [
            ("/posts/view", "view"),
            ("/posts/like", "like"),
            ("/posts/inspire", "inspire"),
            ("/posts/rating", "rating"),
        ]

        created_interactions = 0
        for path, itype in inter_types:
            items = await fetch_interactions(path)
            for it in items:
                # Expected keys may vary; try common shapes
                username = it.get("username") or (it.get("user") or {}).get("username")
                post_id = it.get("post_id") or (it.get("post") or {}).get("id") or it.get("id")
                if not username or post_id is None:
                    continue
                ensure_user(username)
                # Map external post id to local via identifier
                p_obj = db.query(ORMPost).filter(ORMPost.identifier == str(post_id)).first()
                if not p_obj:
                    continue
                if project_code and (str(getattr(p_obj.topic, "project_code", "")).lower() != str(project_code).lower()):
                    continue
                rating_val = None
                if itype == "rating":
                    # Accept 1-5 or 0-100, store as float
                    rv = it.get("rating") or it.get("value")
                    try:
                        rating_val = float(rv) if rv is not None else None
                    except Exception:
                        rating_val = None
                db.add(ORMUserInteraction(username=username, post_id=p_obj.id, interaction_type=itype, rating=rating_val))
                created_interactions += 1

        db.commit()

    return {
        "status": "success",
        "created_users": created_users,
        "created_posts": created_posts,
        "updated_posts": upserted_posts,
        "created_interactions": created_interactions,
    }

@app.get("/feed", response_model=schemas.FeedResponse, tags=["Recommendations"])
async def get_personalized_feed(
    username: str = Query(..., description="Username to get recommendations for"),
    project_code: Optional[str] = Query(None, description="Filter by project code"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    db: Session = Depends(get_db),
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get personalized video recommendations for a user.
    
    - **username**: Username to get recommendations for
    - **project_code**: Filter recommendations by project code (optional)
    - **page**: Page number for pagination
    - **page_size**: Number of items per page (max 100)
    """
    try:
        # Calculate offset for pagination
        offset = (page - 1) * page_size
        
        # Always use hybrid recommendations
        recommendations = engine.get_hybrid_recommendations(
            username=username,
            project_code=project_code,
            num_recommendations=page_size * 2  # Get more to handle filtering
        )
        
        # Apply pagination
        paginated_recommendations = recommendations[offset:offset + page_size]

        # Fallback to cold-start if nothing found
        if not paginated_recommendations:
            fallback = engine.get_cold_start_recommendations(project_code=project_code, limit=page_size)
            paginated_recommendations = fallback

        # Fetch ORM posts
        post_ids = [rec["post_id"] for rec in paginated_recommendations]
        if not post_ids:
            return {"status": "success", "post": [], "items": []}

        posts = db.query(ORMPost).filter(ORMPost.id.in_(post_ids)).all()
        post_map = {p.id: p for p in posts}

        # Build output per output-data-format.md
        output_posts = []
        for rec in paginated_recommendations:
            p = post_map.get(rec["post_id"])
            if not p:
                continue

            topic_owner = p.topic.owner if getattr(p, "topic", None) else None

            output_posts.append({
                "id": p.id,
                "owner": {
                    "first_name": getattr(p.owner, "first_name", None),
                    "last_name": getattr(p.owner, "last_name", None),
                    "name": (f"{getattr(p.owner, 'first_name', '')} {getattr(p.owner, 'last_name', '')}").strip() if p.owner else None,
                    "username": getattr(p.owner, "username", None),
                    "picture_url": getattr(p.owner, "picture_url", None),
                    "user_type": getattr(p.owner, "user_type", None),
                    "has_evm_wallet": bool(getattr(p.owner, "has_evm_wallet", False)) if p.owner else False,
                    "has_solana_wallet": bool(getattr(p.owner, "has_solana_wallet", False)) if p.owner else False,
                } if p.owner else None,
                "category": {
                    "id": getattr(p.category, "id", None) if p.category else None,
                    "name": getattr(p.category, "name", None) if p.category else None,
                    "count": getattr(p.category, "count", 0) if p.category else 0,
                    "description": getattr(p.category, "description", None) if p.category else None,
                    "image_url": getattr(p.category, "image_url", None) if p.category else None,
                },
                "topic": {
                    "id": getattr(p.topic, "id", None) if p.topic else None,
                    "name": getattr(p.topic, "name", None) if p.topic else None,
                    "description": getattr(p.topic, "description", None) if p.topic else None,
                    "image_url": getattr(p.topic, "image_url", None) if p.topic else None,
                    "slug": getattr(p.topic, "slug", None) if p.topic else None,
                    "is_public": bool(getattr(p.topic, "is_public", True)) if p.topic else True,
                    "project_code": getattr(p.topic, "project_code", None) if p.topic else None,
                    "posts_count": getattr(p.topic, "posts_count", 0) if p.topic else 0,
                    "language": getattr(p.topic, "language", None) if p.topic else None,
                    "created_at": getattr(p.topic, "created_at").strftime("%Y-%m-%d %H:%M:%S") if getattr(p.topic, "created_at", None) else None,
                    "owner": {
                        "first_name": getattr(topic_owner, "first_name", None),
                        "last_name": getattr(topic_owner, "last_name", None),
                        "name": (f"{getattr(topic_owner, 'first_name', '')} {getattr(topic_owner, 'last_name', '')}").strip() if topic_owner else None,
                        "username": getattr(topic_owner, "username", None),
                        "profile_url": getattr(topic_owner, "picture_url", None),
                        "user_type": getattr(topic_owner, "user_type", None),
                        "has_evm_wallet": bool(getattr(topic_owner, "has_evm_wallet", False)) if topic_owner else False,
                        "has_solana_wallet": bool(getattr(topic_owner, "has_solana_wallet", False)) if topic_owner else False,
                    } if p.topic else None,
                },
                "title": p.title,
                "is_available_in_public_feed": bool(p.is_available_in_public_feed),
                "is_locked": bool(p.is_locked),
                "slug": p.slug,
                "upvoted": False,
                "bookmarked": False,
                "following": False,
                "identifier": p.identifier,
                "comment_count": int(p.comment_count or 0),
                "upvote_count": int(p.upvote_count or 0),
                "view_count": int(p.view_count or 0),
                "exit_count": int(p.exit_count or 0),
                "rating_count": int(p.rating_count or 0),
                "average_rating": int(round(float(p.average_rating or 0.0))) if p.average_rating is not None else 0,
                "share_count": int(p.share_count or 0),
                "bookmark_count": int(p.bookmark_count or 0),
                "video_link": p.video_link,
                "thumbnail_url": p.thumbnail_url,
                "gif_thumbnail_url": p.gif_thumbnail_url,
                "contract_address": p.contract_address or "",
                "chain_id": p.chain_id or "",
                "chart_url": p.chart_url or "",
                "baseToken": {"address": "", "name": "", "symbol": "", "image_url": ""},
                "created_at": int(p.created_at.timestamp() * 1000) if p.created_at else None,
                "tags": [t.name for t in p.tags] if p.tags else [],
            })

        return {"status": "success", "post": output_posts, "items": output_posts}
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

 

@app.post("/train", tags=["Admin"])
async def train_models(
    db: Session = Depends(get_db),
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Trigger retraining of the recommendation models.
    
    This endpoint rebuilds the content similarity matrix, user-item matrix,
    and retrains the collaborative filtering model.
    """
    try:
        logger.info("Retraining recommendation models...")
        
        # Rebuild all models
        engine.build_content_similarity_matrix()
        engine.build_user_item_matrix()
        engine.train_collaborative_filtering()
        
        return {
            "status": "success",
            "message": "Recommendation models retrained successfully"
        }
    except Exception as e:
        logger.error(f"Error retraining models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
