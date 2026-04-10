import numpy as np
import pandas as pd


def prepare_unified_data(
    movie_ratings: pd.DataFrame,
    anime_interactions: pd.DataFrame,
    movie_meta: pd.DataFrame,
    anime_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    movie = movie_ratings.rename(columns={"userId": "user_id", "movieId": "item_raw_id"}).copy()
    movie["domain"] = "movie"
    movie["item_id"] = "movie_" + movie["item_raw_id"].astype(str)
    movie["user_global"] = "m_" + movie["user_id"].astype(str)

    anime = anime_interactions.rename(columns={"anime_id": "item_raw_id"}).copy()
    anime["domain"] = "anime"
    anime["item_id"] = "anime_" + anime["item_raw_id"].astype(str)
    anime["user_global"] = "a_" + anime["user_id"].astype(str)

    interactions = pd.concat(
        [
            movie[["user_global", "item_id", "rating", "domain"]],
            anime[["user_global", "item_id", "rating", "domain"]],
        ],
        ignore_index=True,
    )
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions = interactions.dropna(subset=["rating"])

    movie_catalog = movie_meta.rename(columns={"movieId": "item_raw_id", "title": "item_name", "genres": "genres"}).copy()
    movie_catalog["item_id"] = "movie_" + movie_catalog["item_raw_id"].astype(str)
    movie_catalog["domain"] = "movie"
    movie_catalog["type"] = "Movie"
    movie_catalog["source"] = "Unknown"
    movie_catalog["score"] = np.nan
    movie_catalog["popularity"] = np.nan
    movie_catalog["members"] = np.nan
    movie_catalog["favorites"] = np.nan
    movie_catalog["episodes"] = np.nan
    movie_catalog["ranked"] = np.nan

    anime_catalog = anime_meta.rename(
        columns={
            "anime_id": "item_raw_id",
            "Name": "item_name",
            "Genres": "genres",
            "Type": "type",
            "Source": "source",
            "Score": "score",
            "Popularity": "popularity",
            "Members": "members",
            "Favorites": "favorites",
            "Episodes": "episodes",
            "Ranked": "ranked",
        }
    ).copy()
    anime_catalog["item_id"] = "anime_" + anime_catalog["item_raw_id"].astype(str)
    anime_catalog["domain"] = "anime"

    catalog = pd.concat(
        [
            movie_catalog[
                [
                    "item_id",
                    "item_name",
                    "genres",
                    "domain",
                    "type",
                    "source",
                    "score",
                    "popularity",
                    "members",
                    "favorites",
                    "episodes",
                    "ranked",
                ]
            ],
            anime_catalog[
                [
                    "item_id",
                    "item_name",
                    "genres",
                    "domain",
                    "type",
                    "source",
                    "score",
                    "popularity",
                    "members",
                    "favorites",
                    "episodes",
                    "ranked",
                ]
            ],
        ],
        ignore_index=True,
    )

    catalog = catalog.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    catalog["item_name"] = catalog["item_name"].fillna("Unknown")
    catalog["genres"] = catalog["genres"].fillna("Unknown")
    catalog["type"] = catalog["type"].fillna("Unknown")
    catalog["source"] = catalog["source"].fillna("Unknown")
    catalog["content_text"] = (
        catalog["genres"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + catalog["type"].astype(str)
        + " "
        + catalog["source"].astype(str)
    )

    interactions = interactions[interactions["item_id"].isin(catalog["item_id"])].copy()
    return interactions, catalog


def filter_for_training(interactions: pd.DataFrame, min_user_interactions: int = 10, min_item_interactions: int = 10) -> pd.DataFrame:
    data = interactions.copy()
    user_counts = data["user_global"].value_counts()
    data = data[data["user_global"].isin(user_counts[user_counts >= min_user_interactions].index)]

    item_counts = data["item_id"].value_counts()
    data = data[data["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)]
    return data
