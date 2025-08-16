# **API Response Documentation: Post Data Format**

This document provides a detailed explanation of the JSON response format used to return post-related data in a structured manner.

---

## **Response Structure**
The response consists of a `status` field indicating the request's status and a `post` array containing post-related data.

### **1. Root-Level Fields**
| Field | Type | Description |
|--------|------|-------------|
| `status` | `string` | Status of the API response (`True` in case of a successful request or `False` in case of unsuccessful request). |
| `post` | `array` | A list of posts, each represented as an object containing detailed metadata about the post. `[]` empty in case of no post found|

---

## **2. Post Object**
Each post object contains information about the post, its owner, category, topic, and additional metadata.

### **2.1. Post-Level Fields**
| Field | Type | Description |
|--------|------|-------------|
| `id` | `integer` | Unique identifier for the post. |
| `owner` | `object` | Information about the user who created the post (explained in section 2.2). |
| `category` | `object` | Information about the category the post belongs to (explained in section 2.3). |
| `topic` | `object` | Information about the topic related to the post (explained in section 2.4). |
| `title` | `string` | Title of the post. |
| `is_available_in_public_feed` | `boolean` | Determines if the post is publicly visible in the feed. |
| `is_locked` | `boolean` | Indicates whether the post is locked (restricted from interactions). |
| `slug` | `string` | A unique identifier for the post, often used in URLs. |
| `upvoted` | `boolean` | Indicates if the requesting user has upvoted the post. |
| `bookmarked` | `boolean` | Indicates if the requesting user has bookmarked the post. |
| `following` | `boolean` | Indicates if the requesting user is following updates related to this post. |
| `identifier` | `string` | A unique reference identifier for the post. |
| `comment_count` | `integer` | Number of comments on the post. |
| `upvote_count` | `integer` | Number of upvotes received by the post. |
| `view_count` | `integer` | Total number of views on the post. |
| `exit_count` | `integer` | Number of times users exited the post without interaction. |
| `rating_count` | `integer` | Number of ratings given to the post. |
| `average_rating` | `integer` | The average rating of the post (scale may vary, e.g., 0-100). |
| `share_count` | `integer` | Number of times the post has been shared. |
| `bookmark_count` | `integer` | Number of users who have bookmarked the post. |
| `video_link` | `string` | URL of the post's video content. |
| `thumbnail_url` | `string` | URL of the video's thumbnail image. |
| `gif_thumbnail_url` | `string` | URL of the post's animated GIF thumbnail. |
| `contract_address` | `string` | Blockchain contract address associated with the post (if applicable). |
| `chain_id` | `string` | Blockchain chain ID (if applicable). |
| `chart_url` | `string` | URL for chart-related data (if applicable). |
| `baseToken` | `object` | Contains information about the base token (explained in section 2.5). |
| `created_at` | `integer` | Timestamp (in milliseconds) when the post was created. |
| `tags` | `array` | List of tags associated with the post. |

---

## **2.2. Owner Object**
Contains information about the user who created the post.

| Field | Type | Description |
|--------|------|-------------|
| `first_name` | `string` | First name of the post creator. |
| `last_name` | `string` | Last name of the post creator. |
| `name` | `string` | Full name of the post creator. |
| `username` | `string` | Unique username of the post creator. |
| `picture_url` | `string` | URL to the profile picture of the post creator. |
| `user_type` | `string` | Role of the user (`null` if not specified). |
| `has_evm_wallet` | `boolean` | Indicates if the user has an Ethereum Virtual Machine (EVM) wallet. |
| `has_solana_wallet` | `boolean` | Indicates if the user has a Solana blockchain wallet. |

---

## **2.3. Category Object**
Contains details about the category under which the post is classified.

| Field | Type | Description |
|--------|------|-------------|
| `id` | `integer` | Unique identifier of the category. |
| `name` | `string` | Name of the category. |
| `count` | `integer` | Number of posts under this category. |
| `description` | `string` | Brief description of the category. |
| `image_url` | `string` | URL of the category image. |

---

## **2.4. Topic Object**
Contains metadata about the topic associated with the post.

| Field | Type | Description |
|--------|------|-------------|
| `id` | `integer` | Unique identifier of the topic. |
| `name` | `string` | Name of the topic. |
| `description` | `string` | Description of the topic. |
| `image_url` | `string` | URL of the topic image. |
| `slug` | `string` | Unique identifier for the topic, often used in URLs. |
| `is_public` | `boolean` | Indicates if the topic is publicly accessible. |
| `project_code` | `string` | Project code associated with the topic. |
| `posts_count` | `integer` | Number of posts under this topic. |
| `language` | `string` | Language of the topic (`null` if not specified). |
| `created_at` | `string` | Timestamp when the topic was created (formatted as `YYYY-MM-DD HH:MM:SS`). |
| `owner` | `object` | Information about the topic owner (explained in section 2.4.1). |

### **2.4.1. Topic Owner Object**
Details about the user who created the topic.

| Field | Type | Description |
|--------|------|-------------|
| `first_name` | `string` | First name of the topic owner. |
| `last_name` | `string` | Last name of the topic owner. |
| `name` | `string` | Full name of the topic owner. |
| `username` | `string` | Unique username of the topic owner. |
| `profile_url` | `string` | URL of the profile image of the topic owner. |
| `user_type` | `string` | Role of the user (`null` if not specified). |
| `has_evm_wallet` | `boolean` | Indicates if the user has an Ethereum Virtual Machine (EVM) wallet. |
| `has_solana_wallet` | `boolean` | Indicates if the user has a Solana blockchain wallet. |

---

## **2.5. Base Token Object**
Contains blockchain-related token data (if applicable).

| Field | Type | Description |
|--------|------|-------------|
| `address` | `string` | Blockchain address of the base token. |
| `name` | `string` | Name of the base token. |
| `symbol` | `string` | Symbol of the base token. |
| `image_url` | `string` | URL of the token image. |

---

## **3. Example JSON Response**
```json
{
  "status": "success",
  "post": [
    {
      "id": 3104,
      "owner": {
        "first_name": "Sachin",
        "last_name": "Kinha",
        "name": "Sachin Kinha",
        "username": "sachin",
        "picture_url": "https://assets.socialverseapp.com/profile/19.png",
        "user_type": null,
        "has_evm_wallet": false,
        "has_solana_wallet": false
      },
      "category": {
        "id": 13,
        "name": "Flic",
        "count": 125,
        "description": "Where Creativity Meets Opportunity",
        "image_url": "https://socialverse-assets.s3.us-east-1.amazonaws.com/categories/NEW_COLOR.png"
      },
      "topic": {
        "id": 1,
        "name": "Social Media",
        "description": "Short form content making and editing.",
        "image_url": "https://ui-avatars.com/api/?size=300&name=Social%20Media&color=fff&background=random",
        "slug": "b9f5413f04ec58e47874",
        "is_public": true,
        "project_code": "flic",
        "posts_count": 18,
        "language": null,
        "created_at": "2025-02-15 15:02:41",
        "owner": {
          "first_name": "Shivam",
          "last_name": "Flic",
          "name": "Shivam Flic",
          "username": "random",
          "profile_url": "https://assets.socialverseapp.com/profile/random1739306567image_cropper_1739306539349.jpg.png",
          "user_type": "hirer"
        }
      },
      "title": "testing-topic",
      "is_available_in_public_feed": true,
      "is_locked": false,
      "slug": "0dcff38b97c646a37ebcfa4f039c332812aa3857",
      "upvoted": false,
      "bookmarked": false,
      "following": false,
      "identifier": "QCp8ffL",
      "comment_count": 0,
      "upvote_count": 4,
      "view_count": 235,
      "exit_count": 149,
      "rating_count": 0,
      "average_rating": 84,
      "share_count": 0,
      "bookmark_count": 0,
      "video_link": "https://video-cdn.socialverseapp.com/sachin_d323e3b5-0012-4e55-85cc-b15dbe47a470.mp4",
      "thumbnail_url": "https://video-cdn.socialverseapp.com/sachin_d323e3b5-0012-4e55-85cc-b15dbe47a470.0000002.jpg",
      "gif_thumbnail_url": "https://video-cdn.socialverseapp.com/sachin_d323e3b5-0012-4e55-85cc-b15dbe47a470.gif",
      "contract_address": "",
      "chain_id": "",
      "chart_url": "",
      "baseToken": {
        "address": "",
        "name": "",
        "symbol": "",
        "image_url": ""
      },
      "created_at": 1739791247000,
      "tags": ["testing", "editing", "social-media"]
    }
  ]
}
```

---

## **4. Notes**
- Some fields such as `contract_address`, `chain_id`, and `chart_url` may be empty (`""`) if not applicable.
- Timestamps like `created_at` in the post object are in **milliseconds** (Unix timestamp format).
- The `baseToken` object is primarily used in blockchain-related contexts and may be empty.
