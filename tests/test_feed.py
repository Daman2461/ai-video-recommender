import os
import sys
from pathlib import Path
import pytest
import httpx

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

def assert_post_schema(post: dict):
    # Root post fields
    for key in [
        "id",
        "owner",
        "category",
        "topic",
        "title",
        "is_available_in_public_feed",
        "is_locked",
        "slug",
        "upvoted",
        "bookmarked",
        "following",
        "identifier",
        "comment_count",
        "upvote_count",
        "view_count",
        "exit_count",
        "rating_count",
        "average_rating",
        "share_count",
        "bookmark_count",
        "video_link",
        "thumbnail_url",
        "gif_thumbnail_url",
        "contract_address",
        "chain_id",
        "chart_url",
        "baseToken",
        "created_at",
        "tags",
    ]:
        assert key in post, f"missing field: {key}"

    # Types (basic checks, allow None for some nested)
    assert isinstance(post["id"], int)
    assert isinstance(post["title"], str)
    assert isinstance(post["is_available_in_public_feed"], bool)
    assert isinstance(post["is_locked"], bool)
    assert isinstance(post["upvoted"], bool)
    assert isinstance(post["bookmarked"], bool)
    assert isinstance(post["following"], bool)
    assert isinstance(post["comment_count"], int)
    assert isinstance(post["upvote_count"], int)
    assert isinstance(post["view_count"], int)
    assert isinstance(post["exit_count"], int)
    assert isinstance(post["rating_count"], int)
    assert isinstance(post["average_rating"], int)
    assert isinstance(post["share_count"], int)
    assert isinstance(post["bookmark_count"], int)
    assert isinstance(post["created_at"], int)

    # Owner object (no id per spec)
    if post["owner"] is not None:
        owner = post["owner"]
        for k in [
            "first_name",
            "last_name",
            "name",
            "username",
            "picture_url",
            "user_type",
            "has_evm_wallet",
            "has_solana_wallet",
        ]:
            assert k in owner
        assert "id" not in owner, "owner.id should not be present"

    # Topic object and topic.owner profile_url
    if post["topic"] is not None:
        topic = post["topic"]
        for k in [
            "id",
            "name",
            "description",
            "image_url",
            "slug",
            "is_public",
            "project_code",
            "posts_count",
            "language",
            "created_at",
            "owner",
        ]:
            assert k in topic
        # created_at must be a formatted string
        assert isinstance(topic["created_at"], str) or topic["created_at"] is None
        if topic["owner"] is not None:
            towner = topic["owner"]
            for k in [
                "first_name",
                "last_name",
                "name",
                "username",
                "profile_url",
                "user_type",
                "has_evm_wallet",
                "has_solana_wallet",
            ]:
                assert k in towner
            assert "id" not in towner, "topic.owner.id should not be present"


async def call_feed(client: httpx.AsyncClient, project_code: str | None = None, page_size: int = 2):
    params = {
        "username": os.getenv("TEST_USERNAME", "user1"),
        "page_size": page_size,
    }
    if project_code is not None:
        params["project_code"] = project_code
    return await client.get("/feed", params=params)

@pytest.mark.anyio
async def test_feed_hybrid_only():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await call_feed(client)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert isinstance(data["post"], list)
    if data["post"]:
        assert_post_schema(data["post"][0])

@pytest.mark.anyio
async def test_feed_empty_project_code():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Explicitly pass empty project_code to ensure it does not error
        r = await call_feed(client, project_code="")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert isinstance(data["post"], list)
    if data["post"]:
        assert_post_schema(data["post"][0])
