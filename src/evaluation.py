import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

from src.preprocessing import load_data, merge_data


def split_dataset(df):

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    return train, test


def calculate_rmse(true_ratings, predicted_ratings):

    rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))

    return rmse


def precision_recall_at_k(recommended_movies, relevant_movies, k=5):

    recommended_k = recommended_movies[:k]

    relevant_set = set(relevant_movies)

    recommended_set = set(recommended_k)

    true_positives = len(recommended_set & relevant_set)

    precision = true_positives / k

    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0

    return precision, recall


if __name__ == "__main__":

    movies, ratings = load_data()

    df = merge_data(movies, ratings)

    train, test = split_dataset(df)

    print("Train size:", train.shape)
    print("Test size:", test.shape)

    # Example RMSE calculation
    true = np.array([4, 5, 3, 4])
    predicted = np.array([4.2, 4.8, 3.5, 3.7])

    rmse = calculate_rmse(true, predicted)

    print("RMSE:", rmse)

    # Example Precision@K
    recommended = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]
    relevant = ["Movie B", "Movie C", "Movie F"]

    precision, recall = precision_recall_at_k(recommended, relevant)

    print("Precision@5:", precision)
    print("Recall@5:", recall)