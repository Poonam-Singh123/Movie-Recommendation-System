import pandas as pd
import os

def load_data():
    base_path = os.path.dirname(os.path.dirname(__file__))

    movies_path = os.path.join(base_path, "data", "movies.csv")
    ratings_path = os.path.join(base_path, "data", "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    return movies, ratings


def merge_data(movies, ratings):
    df = pd.merge(ratings, movies, on="movieId")
    return df


def create_user_movie_matrix(df):
    # Limit to top 500 movies by rating count to reduce matrix size
    top_movies = df['movieId'].value_counts().head(500).index
    df_filtered = df[df['movieId'].isin(top_movies)]
    
    user_movie_matrix = df_filtered.pivot_table(
        index="userId",
        columns="title",
        values="rating",
        aggfunc="mean"
    )

    # Replace NaN with 0
    user_movie_matrix = user_movie_matrix.fillna(0)

    return user_movie_matrix

if __name__ == "__main__":
    movies, ratings = load_data()
    df = merge_data(movies, ratings)
    user_movie_matrix = create_user_movie_matrix(df)
    user_movie_matrix.to_csv("data/user_movie_matrix.csv")
    print("Merged Data Shape:", df.shape)
    print("User-Movie Matrix Shape:", user_movie_matrix.shape)
    