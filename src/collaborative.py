import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity
try:
    from src.preprocessing import load_data, merge_data, create_user_movie_matrix
except ModuleNotFoundError:
    # Allow running the script directly as `python src/collaborative.py`
    # In that case the script's directory is on sys.path and `preprocessing` is importable directly.
    from preprocessing import load_data, merge_data, create_user_movie_matrix
from sklearn.neighbors import NearestNeighbors


def compute_item_similarity(user_movie_matrix):
    # Compute movie-to-movie similarity (items are columns)
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    return movie_similarity_df


def recommend_movies_item_based(user_id, user_movie_matrix, movie_similarity_df, top_n=5):
    # Ratings (or interactions) for the target user
    user_ratings = user_movie_matrix.loc[user_id]

    # Movies the user hasn't rated/watched (assumed 0)
    unwatched_movies = user_ratings[user_ratings == 0].index

    scores = {}

    # For each candidate movie, score it by similarity to movies the user liked/rated
    rated_movies = user_ratings[user_ratings > 0]
    for candidate in unwatched_movies:
        # similarity vector between candidate and all movies the user rated
        sims = movie_similarity_df.loc[candidate, rated_movies.index]
        # weighted score: sum(similarity * user_rating)
        score = (sims * rated_movies).sum()
        if score > 0:
            scores[candidate] = score

    # Return top-N movie ids with scores
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(movie, float(score)) for movie, score in ranked[:top_n]]


def get_top_k_similar_users(user_id, user_movie_matrix, k=5):
    """Return list of (neighbor_user_id, similarity) for top-k similar users (excluding self)."""
    if user_id not in user_movie_matrix.index:
        raise KeyError(f"User {user_id} not found in matrix")

    model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
    model.fit(user_movie_matrix.values)

    user_vec = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vec)

    # distances are cosine distances; convert to similarity
    sims = 1 - distances[0]
    neighbor_indices = indices[0]

    neighbors = []
    for idx, sim in zip(neighbor_indices, sims):
        neigh_id = user_movie_matrix.index[idx]
        if neigh_id == user_id:
            continue
        neighbors.append((neigh_id, float(sim)))
        if len(neighbors) >= k:
            break

    return neighbors


def recommend_movies_user_based_knn(user_id, user_movie_matrix, k=5, top_n=5):
    # get neighbors
    neighbors = get_top_k_similar_users(user_id, user_movie_matrix, k=k)

    user_ratings = user_movie_matrix.loc[user_id]
    unwatched = user_ratings[user_ratings == 0].index

    scores = {}
    for neigh_id, sim in neighbors:
        neigh_ratings = user_movie_matrix.loc[neigh_id]
        # consider only movies neighbor rated
        rated = neigh_ratings[neigh_ratings > 0]
        for movie, rating in rated.items():
            if movie in unwatched:
                scores[movie] = scores.get(movie, 0.0) + sim * float(rating)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(movie, float(score)) for movie, score in ranked[:top_n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommender (item- or user-based)")
    parser.add_argument("--mode", choices=["item", "user"], default="item", help="Recommender mode")
    parser.add_argument("--user", type=int, default=1, help="User ID to recommend for")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors for user-based mode")
    parser.add_argument("--scores", action="store_true", help="Print scores alongside movie titles")
    args = parser.parse_args()

    movies, ratings = load_data()
    df = merge_data(movies, ratings)
    user_movie_matrix = create_user_movie_matrix(df)

    if args.mode == "item":
        movie_similarity_df = compute_item_similarity(user_movie_matrix)
        recs = recommend_movies_item_based(args.user, user_movie_matrix, movie_similarity_df, top_n=args.top)
        header = f"Item-based recommendations for User {args.user}"
    else:
        recs = recommend_movies_user_based_knn(args.user, user_movie_matrix, k=args.k, top_n=args.top)
        header = f"User-based (k={args.k}) recommendations for User {args.user}"

    if args.scores:
        print(f"{header} (movie — score):")
        for movie, score in recs:
            print(f"{movie} — {score:.4f}")
    else:
        print(f"{header}:")
        print([movie for movie, _ in recs])