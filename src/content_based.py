import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies():
    movies = pd.read_csv("data/movies.csv")
    return movies


def compute_tfidf_matrix(movies):
    # Replace '|' with space in genres
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    return tfidf_matrix


def compute_similarity(tfidf_matrix):
    """(Deprecated) computes full cosine similarity matrix.

    This function can consume a huge amount of memory for large datasets
    and is no longer used by the recommendation logic. It remains for
    backward compatibility but returns None.
    """
    # NOTE: computing a full pairwise similarity matrix is expensive and
    # memory-intensive.  Instead, similarity scores are calculated on the
    # fly for the requested movie inside ``recommend_movies``.
    return None


def recommend_movies(movie_title, movies, tfidf_matrix, top_n=5):
    """Return the top N similar movie titles for the given movie.

    The similarity is computed on demand by comparing the TF-IDF vector of
    the requested movie against the full matrix. This avoids creating a
    giant pairwise matrix which can exhaust system memory.

    Parameters
    ----------
    movie_title : str
        Title of the movie to use as the query.
    movies : pd.DataFrame
        DataFrame containing at least a "title" column.
    tfidf_matrix : sparse matrix
        TF-IDF representation of movie genres.
    top_n : int, optional
        Number of recommendations to return (default is 5).
    """

    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    if movie_title not in indices:
        return "Movie not found in dataset."

    idx = indices[movie_title]

    # compute similarity for a single row against all movies
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # pair the scores with indices and sort
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # discard the first entry since it is the movie itself
    sim_scores = sim_scores[1: top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices]


if __name__ == "__main__":
    movies = load_movies()
    tfidf_matrix = compute_tfidf_matrix(movies)

    print("Recommendations for Toy Story (1995):")
    print(recommend_movies("Toy Story (1995)", movies, tfidf_matrix))