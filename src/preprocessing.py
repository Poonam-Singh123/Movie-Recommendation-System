import pandas as pd

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
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
    