import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.preprocessing import load_data, merge_data, create_user_movie_matrix


def perform_clustering(user_movie_matrix, n_clusters=5):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(user_movie_matrix)

    user_clusters = pd.DataFrame({
        "userId": user_movie_matrix.index,
        "cluster": clusters
    })

    return kmeans, user_clusters


def recommend_from_cluster(user_id, user_clusters, user_movie_matrix, top_n=5):

    cluster_id = user_clusters[user_clusters["userId"] == user_id]["cluster"].values[0]

    cluster_users = user_clusters[user_clusters["cluster"] == cluster_id]["userId"]

    cluster_data = user_movie_matrix.loc[cluster_users]

    movie_scores = cluster_data.mean().sort_values(ascending=False)

    user_movies = user_movie_matrix.loc[user_id]

    recommendations = movie_scores[user_movies == 0]

    return recommendations.head(top_n).index


def visualize_clusters(user_movie_matrix, clusters):

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(user_movie_matrix)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced_data[:,0], reduced_data[:,1], c=clusters)
    plt.title("User Clusters (PCA Visualization)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


if __name__ == "__main__":

    movies, ratings = load_data()
    df = merge_data(movies, ratings)
    user_movie_matrix = create_user_movie_matrix(df)

    kmeans, user_clusters = perform_clustering(user_movie_matrix)

    print("Cluster-based recommendations for User 1:")
    print(recommend_from_cluster(1, user_clusters, user_movie_matrix))

    visualize_clusters(user_movie_matrix, user_clusters["cluster"])