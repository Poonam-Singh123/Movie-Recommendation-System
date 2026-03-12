import pandas as pd
import os
import zipfile
import tempfile
import urllib.request
import time
from urllib.error import URLError


MOVIELENS_LATEST_SMALL_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Backup mirror: download extracted CSVs directly from GitHub raw
MOVIELENS_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/smanihwr/ml-latest-small/master"


def _download_and_extract_movielens_small(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        movies_dst = os.path.join(data_dir, "movies.csv")
        ratings_dst = os.path.join(data_dir, "ratings.csv")

        # Try official zip first
        zip_path = os.path.join(tmp, "ml-latest-small.zip")
        zip_err = None
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(
                    MOVIELENS_LATEST_SMALL_ZIP_URL,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp, open(zip_path, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        out.write(chunk)
                zip_err = None
                break
            except (URLError, TimeoutError) as e:
                zip_err = e
                time.sleep(1.5 * attempt)

        if zip_err is None:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)

            extracted_root = os.path.join(tmp, "ml-latest-small")
            movies_src = os.path.join(extracted_root, "movies.csv")
            ratings_src = os.path.join(extracted_root, "ratings.csv")

            if os.path.exists(movies_src) and os.path.exists(ratings_src):
                with open(movies_src, "rb") as r, open(movies_dst, "wb") as w:
                    w.write(r.read())
                with open(ratings_src, "rb") as r, open(ratings_dst, "wb") as w:
                    w.write(r.read())
                return

        # Fallback: GitHub raw CSVs
        raw_err = None
        for attempt in range(1, 4):
            try:
                for name, dst in [("movies.csv", movies_dst), ("ratings.csv", ratings_dst)]:
                    url = f"{MOVIELENS_GITHUB_RAW_BASE}/{name}"
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp, open(dst, "wb") as out:
                        while True:
                            chunk = resp.read(1024 * 256)
                            if not chunk:
                                break
                            out.write(chunk)
                raw_err = None
                return
            except (URLError, TimeoutError) as e:
                raw_err = e
                time.sleep(1.5 * attempt)

        raise RuntimeError(
            "Could not download MovieLens dataset automatically (zip and GitHub mirror both failed). "
            "Please download 'ml-latest-small.zip' from https://grouplens.org/datasets/movielens/ "
            "and place 'movies.csv' and 'ratings.csv' into the project's 'data/' folder."
        ) from (raw_err or zip_err)


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    movies_path = os.path.join(base_dir, "data", "movies.csv")
    ratings_path = os.path.join(base_dir, "data", "ratings.csv")

    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        _download_and_extract_movielens_small(os.path.join(base_dir, "data"))

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
    