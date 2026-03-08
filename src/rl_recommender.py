import numpy as np
import pandas as pd
import random

from src.preprocessing import load_data, merge_data


class EpsilonGreedyRecommender:

    def __init__(self, movies, epsilon=0.1):
        self.movies = movies
        self.epsilon = epsilon
        self.movie_rewards = {movie: 0 for movie in movies}
        self.movie_counts = {movie: 0 for movie in movies}

    def recommend_movie(self):

        if random.random() < self.epsilon:
            # Explore
            return random.choice(self.movies)

        # Exploit
        avg_rewards = {
            movie: self.movie_rewards[movie] / self.movie_counts[movie]
            if self.movie_counts[movie] > 0 else 0
            for movie in self.movies
        }

        return max(avg_rewards, key=avg_rewards.get)

    def update_reward(self, movie, reward):

        self.movie_counts[movie] += 1
        self.movie_rewards[movie] += reward


if __name__ == "__main__":

    movies, ratings = load_data()
    df = merge_data(movies, ratings)

    movie_list = list(df["title"].unique())

    recommender = EpsilonGreedyRecommender(movie_list)

    # Simulate user feedback
    for _ in range(20):

        movie = recommender.recommend_movie()

        # Simulated reward (rating)
        reward = random.choice([1, 2, 3, 4, 5])

        recommender.update_reward(movie, reward)

        print(f"Recommended: {movie} | Reward: {reward}")