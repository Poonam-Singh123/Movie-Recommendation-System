import pandas as pd

from src.preprocessing import load_data, merge_data, create_user_movie_matrix
from src.content_based import load_movies, compute_tfidf_matrix, compute_similarity, recommend_movies
from src.item_based import compute_item_similarity, recommend_movies_item_based


def popularity_recommendations(df, top_n=10):

    movie_popularity = df.groupby("title")["rating"].count().sort_values(ascending=False)

    return movie_popularity.head(top_n).index


def hybrid_recommendation(user_id, movie_title):

    movies, ratings = load_data()
    df = merge_data(movies, ratings)

    user_movie_matrix = create_user_movie_matrix(df)

    # Content-based
    tfidf_matrix = compute_tfidf_matrix(movies)

    content_rec = recommend_movies(movie_title, movies, tfidf_matrix, top_n=5)

    # Item-based
    item_similarity = compute_item_similarity(user_movie_matrix)

    item_rec = recommend_movies_item_based(user_id, user_movie_matrix, item_similarity, top_n=5)

    # Popular movies
    pop_rec = popularity_recommendations(df, top_n=5)

    final_recommendations = list(set(content_rec) | set(item_rec) | set(pop_rec))

    return final_recommendations[:10]


if __name__ == "__main__":

    recommendations = hybrid_recommendation(
        user_id=1,
        movie_title="Toy Story (1995)"
    )

    print("Hybrid Recommendations:")
    print(recommendations)
