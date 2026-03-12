# 🎬 Movie Recommendation System

## 🌐 Live App

Try it here: https://poonam-movie-recommender.streamlit.app/

## 📌 Project Overview

This project builds a **Hybrid Movie Recommendation System** using multiple machine learning techniques to provide personalized movie suggestions.

The system combines:

* Content-Based Filtering
* Collaborative Filtering
* Unsupervised Learning (KMeans Clustering)
* Matrix Factorization (SVD)
* Reinforcement Learning (Multi-Armed Bandit)

The dataset used is the **MovieLens dataset**, which contains movie ratings from users.

---

## 🚀 Features

✔ Recommend movies based on similar genres
✔ Personalized recommendations using collaborative filtering
✔ User segmentation using clustering
✔ Dynamic learning using reinforcement learning
✔ Interactive web application built with Streamlit

---

## 🧠 Algorithms Used

### 1️⃣ Content-Based Filtering

Uses **TF-IDF vectorization** on movie genres and **cosine similarity** to recommend similar movies.

### 2️⃣ User-Based Collaborative Filtering

Finds users with similar rating patterns and recommends movies they liked.

### 3️⃣ Item-Based Collaborative Filtering

Recommends movies similar to those a user has already rated.

### 4️⃣ KMeans Clustering

Segments users into groups based on their movie preferences.

### 5️⃣ Matrix Factorization (SVD)

Discovers hidden patterns in the user-movie matrix using latent factors.

### 6️⃣ Reinforcement Learning

Implements an **Epsilon-Greedy Multi-Armed Bandit** to dynamically improve recommendations.

---

## 📂 Project Structure

Movie-Recommendation-System
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── item_based.py
│   ├── clustering.py
│   ├── hybrid_recommender.py
│   ├── evaluation.py
│   ├── rl_recommender.py
│   ├── matrix_factorization.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore

---
Recommendation Pipeline

User Input
     ↓
Data Preprocessing
     ↓
User-Movie Matrix
     ↓
Multiple Recommenders
(Content + Collaborative + Clustering)
     ↓
Hybrid Recommendation Engine
     ↓
Streamlit Web Application

## ⚙️ Installation

Clone the repository

git clone https://github.com/Poonam-Singh123/Movie-Recommendation-System.git

cd Movie-Recommendation-System

Create virtual environment

python -m venv venv

Activate environment

Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

---

## ▶️ Run the Application

streamlit run app/app.py

---

## ☁️ Deploy (Streamlit Community Cloud)

This repo may not include large `data/*.csv` files in version control.
During deployment, if `data/movies.csv` and `data/ratings.csv` are missing, the app will download the
**MovieLens Latest Small** dataset automatically on first run.

### Posters (TMDB)

To enable movie posters in the UI, set this in Streamlit Secrets:

- `TMDB_API_KEY`: your TMDB API Key (v3 auth)

---

## 📊 Dataset

Dataset used: **MovieLens Latest Small Dataset**

Download from:
https://grouplens.org/datasets/movielens/

---

## 💡 Future Improvements

* Add movie posters using TMDB API
* Improve recommendation accuracy using deep learning
* Deploy the system online

---

## 👨‍💻 Author
Poonam Singh