from typing import List

import pandas as pd

from .config import ANIME_INTERACTIONS, ANIME_META, MOVIE_META, MOVIE_RATINGS, RANDOM_STATE


def sample_large_csv(path, usecols: List[str], sample_frac: float, chunksize: int) -> pd.DataFrame:
    sampled_chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue
        sampled_chunks.append(chunk.sample(frac=sample_frac, random_state=RANDOM_STATE))
    return pd.concat(sampled_chunks, ignore_index=True) if sampled_chunks else pd.DataFrame(columns=usecols)


def load_movie_data(sample_frac: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = sample_large_csv(
        MOVIE_RATINGS,
        usecols=["userId", "movieId", "rating"],
        sample_frac=sample_frac,
        chunksize=1_000_000,
    )
    movies = pd.read_csv(MOVIE_META, usecols=["movieId", "title", "genres"])
    return ratings, movies


def load_anime_data(sample_frac: float = 0.003) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = sample_large_csv(
        ANIME_INTERACTIONS,
        usecols=["user_id", "anime_id", "rating"],
        sample_frac=sample_frac,
        chunksize=2_000_000,
    )
    anime_meta = pd.read_csv(
        ANIME_META,
        usecols=[
            "anime_id",
            "Name",
            "Genres",
            "Type",
            "Source",
            "Score",
            "Popularity",
            "Members",
            "Favorites",
            "Episodes",
            "Ranked",
        ],
    )
    return interactions, anime_meta
