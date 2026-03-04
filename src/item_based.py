import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import load_data, merge_data, create_user_movie_matrix


def compute_item_similarity(user_movie_matrix):
    # Transpose matrix (Movies as rows)
    movie_matrix = user_movie_matrix.T

    item_similarity = cosine_similarity(movie_matrix)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=movie_matrix.index,
        columns=movie_matrix.index
    )

    return item_similarity_df


def recommend_movies_item_based(user_id, user_movie_matrix, item_similarity_df, top_n=5):

    user_ratings = user_movie_matrix.loc[user_id]
    liked_movies = user_ratings[user_ratings > 0]

    recommendation_scores = {}

    for movie, rating in liked_movies.items():
        similar_movies = item_similarity_df[movie].sort_values(ascending=False)[1:6]

        for similar_movie, similarity_score in similar_movies.items():
            if user_movie_matrix.loc[user_id, similar_movie] == 0:
                if similar_movie not in recommendation_scores:
                    recommendation_scores[similar_movie] = 0
                recommendation_scores[similar_movie] += similarity_score * rating

    recommended_movies = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [movie[0] for movie in recommended_movies[:top_n]]


if __name__ == "__main__":
    movies, ratings = load_data()
    df = merge_data(movies, ratings)
    user_movie_matrix = create_user_movie_matrix(df)

    item_similarity_df = compute_item_similarity(user_movie_matrix)

    print("Item-based recommendations for User 1:")
    print(recommend_movies_item_based(1, user_movie_matrix, item_similarity_df))