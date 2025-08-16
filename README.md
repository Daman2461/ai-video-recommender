# Video Recommendation Engine

A sophisticated recommendation system that suggests personalized video content based on user preferences and engagement patterns using a hybrid approach of content-based and collaborative filtering.

## 🚀 Features

- **Hybrid Recommendation System**: Combines content-based and collaborative filtering
- **Content-Based Filtering**: Uses TF-IDF for text similarity
- **Collaborative Filtering**: Uses SVD (Singular Value Decomposition) for user-item recommendations
- **Cold Start Handling**: Uses popularity-based recommendations for new users or items
- **RESTful API**: Built with FastAPI
- **Scalable**: Designed to handle large datasets efficiently

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI
- **Database**: SQLite (can be easily switched to PostgreSQL/MySQL)
- **Machine Learning**: scikit-learn, pandas, numpy
- **API Documentation**: Swagger/OpenAPI (automatically generated)
- **Deployment**: Docker support included

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip (Python package manager)
 - SQLite (preinstalled on macOS)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   cat > .env << 'EOF'
   FLIC_TOKEN=YOUR_FLIC_TOKEN
   API_BASE_URL=https://api.socialverseapp.com
   DATABASE_URL=sqlite:///./recommendation.db
   RESONANCE_ALGORITHM=
   EOF
   ```
   Note: Never commit real tokens to source control.

5. **Initialize the database (Alembic)**
   ```bash
   alembic upgrade head
   ```

### Running the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Endpoints

### Recommendations

- `GET /feed`: Get personalized video recommendations
  - Parameters:
    - `username` (required): Username to get recommendations for
    - `project_code` (optional): Filter by project code (from the post's Topic)
    - `page` (optional): Page number (default: 1)
    - `page_size` (optional): Number of items per page (default: 10, max: 100)

- `POST /train`: Retrain the recommendation models

### Admin

- `POST /admin/import_external`: Import users, topics, posts, and interactions from the external API.
  - Query params:
    - `project_code` (optional): If provided, backfills the default `General` topic's code and filters imports to that code.

### Local Data (for manual seeding/testing)

- `POST /local/users`
- `POST /local/categories`
- `POST /local/topics`
- `POST /local/posts`
- `POST /local/interactions`
- `GET /local/interactions`
- `DELETE /local/interactions`

## 🧠 Recommendation Algorithms

### 1. Content-Based Filtering
- Uses TF-IDF to convert post features (title, category, topic, tags) into numerical vectors
- Calculates cosine similarity between posts
- Recommends similar content based on what the user has interacted with

### 2. Collaborative Filtering
- Uses SVD (Singular Value Decomposition) for matrix factorization
- Considers user interactions (views, likes, inspires, ratings)
- Weights different interaction types (e.g., inspires > likes > views)

### 3. Hybrid Approach
- Combines content-based and collaborative filtering
- Uses a weighted average of both approaches
- Falls back to popularity-based recommendations when needed

### 4. Cold Start Handling
- For new users or items with limited interactions
- Uses a combination of popularity and diversity metrics
- Ensures recommendations are still relevant and varied

## 📊 Data Model

The system uses the following main entities:

- **User**: Users who interact with the platform
- **Post**: Video content that can be recommended
- **Category**: Categories that posts belong to
- **Topic**: Topics that posts are related to
  - Note: `project_code` lives on `Topic` and is used for feed filtering. Posts inherit the code via their topic.
- **UserInteraction**: Records of user interactions (views, likes, etc.)

## 🧪 Testing

To run the test suite:

```bash
pytest tests/
```

## 📦 Data Import How-To

1. Start the API (see above).
2. Import external data:
   ```bash
   curl -X POST 'http://localhost:8000/admin/import_external'
   ```
   Optionally, set a project code for the default topic:
   ```bash
   curl -X POST 'http://localhost:8000/admin/import_external?project_code=general'
   ```
3. Inspect topics and project codes:
   ```bash
   sqlite3 recommendation.db 'SELECT id,name,slug,project_code FROM topics ORDER BY id;'
   sqlite3 recommendation.db 'SELECT DISTINCT project_code FROM topics WHERE project_code IS NOT NULL AND project_code != "" ORDER BY project_code;'
   ```
4. Fetch a feed (replace YOUR_CODE):
   ```bash
   curl 'http://localhost:8000/feed?username=daman&project_code=YOUR_CODE&page_size=10'
   ```

## 🧰 Postman Collection

An exhaustive Postman collection is provided at `postman/VideoRecommendationAPI.postman_collection.json`.

Steps:
- Import into Postman.
- Set collection variables:
  - `baseUrl` = http://localhost:8000
  - `FLIC_TOKEN` = your token
 
