import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import load_data, merge_data, create_user_movie_matrix


def compute_svd(user_movie_matrix, n_components=50):

    svd = TruncatedSVD(n_components=n_components, random_state=42)

    latent_matrix = svd.fit_transform(user_movie_matrix)

    return latent_matrix


def recommend_movies_svd(user_id, user_movie_matrix, latent_matrix, top_n=5):

    similarity = cosine_similarity(latent_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]

    user_movies = user_movie_matrix.loc[user_id]

    unwatched_movies = user_movies[user_movies == 0].index

    recommendation_scores = {}

    for similar_user, score in similar_users.items():

        similar_user_movies = user_movie_matrix.loc[similar_user]

        for movie in unwatched_movies:

            if similar_user_movies[movie] > 0:

                if movie not in recommendation_scores:
                    recommendation_scores[movie] = 0

                recommendation_scores[movie] += score

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

    latent_matrix = compute_svd(user_movie_matrix)

    recommendations = recommend_movies_svd(
        user_id=1,
        user_movie_matrix=user_movie_matrix,
        latent_matrix=latent_matrix
    )

    print("SVD Recommendations for User 1:")
    print(recommendations)