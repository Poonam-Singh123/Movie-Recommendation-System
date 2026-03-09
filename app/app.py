import streamlit as st
import pandas as pd

from src.preprocessing import load_data, merge_data
from src.hybrid_recommender import hybrid_recommendation
 
st.sidebar.title("About")
st.sidebar.info(
"""
Hybrid Movie Recommendation System

Algorithms Used:
- Content-Based Filtering
- Collaborative Filtering
- KMeans Clustering
- Reinforcement Learning
"""
)

st.title("🎬 Movie Recommendation System")

st.write("Select a movie and get personalized recommendations")


# Load data
movies, ratings = load_data()
df = merge_data(movies, ratings)

movie_list = sorted(df["title"].unique())


# Movie selector
selected_movie = st.selectbox(
    "Choose a movie",
    movie_list
)


# User input
user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    max_value=610,
    step=1
)


if st.button("Get Recommendations"):

    recommendations = hybrid_recommendation(
        user_id=user_id,
        movie_title=selected_movie
    )

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write(movie)