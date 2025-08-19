# 🎬 Movie Recommender System

A **Movie Recommendation System** using **K-Nearest Neighbors (KNN)** for finding similar movies. The system runs on a **FastAPI** backend with a simple `preview.html` frontend. Additionally, it integrates with the **OMDb API** to fetch movie posters, ratings, and plots.

## 📂 Project Structure

```
movie_recommender/
│── app/              # FastAPI application
│── base/             # KNN logic & recommendation engine
│── movies.xlsx       # Movie dataset
│── ratings.xlsx      # Ratings dataset
│── preview.html      # Frontend preview file
│── requirements.txt  # Project dependencies
```

## ⚙️ Setup Instructions

```bash
git clone <your-repo-url>
cd movie_recommender
pip install -r requirements.txt
```

Download and place the following files in the project directory:

* `sparse_matrix.pkl`
* `knn_model.pkl`
* `movie_id_map.pkl`
* `reverse_map.pkl`

### OMDb API Setup

Sign up at [OMDb API](https://www.omdbapi.com/) and get your **API key**. Add it to your FastAPI app configuration.

## 🚀 Running the Project

```bash
uvicorn app:app --reload
```

API → `http://127.0.0.1:8000`
Frontend → Open `preview.html`

## 📌 Example Endpoint

```
GET http://127.0.0.1:8000/recommend?movie=Inception
```

Response:

```json
{
  "movie": "Inception",
  "recommendations": [
    {
      "title": "Interstellar",
      "poster": "https://...",
      "rating": "8.6",
      "plot": "A team of explorers travel through a wormhole..."
    },
    {
      "title": "The Prestige",
      "poster": "https://...",
      "rating": "8.5",
      "plot": "After a tragic accident, two magicians engage..."
    }
  ]
}
```

## 🧠 How It Works

1. `base/` contains the KNN recommender built on a sparse ratings matrix.
2. Pre-trained `.pkl` files (`sparse_matrix`, `knn_model`, `movie_id_map`, `reverse_map`) speed up lookup.
3. `app/` exposes REST endpoints via FastAPI.
4. `preview.html` acts as the simple frontend.
5. **OMDb API** enriches results with posters, IMDB ratings, and plots.

## 📌 Notes

* Ensure `.pkl` files are downloaded before running.
* Update `movies.xlsx` and `ratings.xlsx` to extend the dataset.
* You need a valid **OMDb API key** for full functionality.

👨‍💻 Author: Shivam Munday
