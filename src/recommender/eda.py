import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import GRAPH_DIR, RANDOM_STATE


def save_plot(name: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close()


def create_eda_graphs(
    movie_ratings: pd.DataFrame,
    anime_interactions: pd.DataFrame,
    movie_meta: pd.DataFrame,
    anime_meta: pd.DataFrame,
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.histplot(movie_ratings["rating"], bins=10, kde=True, color="#2563eb")
    plt.title("Movie Ratings Distribution")
    save_plot("01_movie_rating_distribution")

    plt.figure(figsize=(7, 4))
    sns.histplot(anime_interactions["rating"], bins=10, kde=True, color="#16a34a")
    plt.title("Anime Ratings Distribution")
    save_plot("02_anime_rating_distribution")

    movie_genres = movie_meta.assign(genres=movie_meta["genres"].fillna("Unknown").str.split("|"))
    movie_genres = movie_genres.explode("genres")
    top_movie_genres = movie_genres["genres"].value_counts().head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_movie_genres.values, y=top_movie_genres.index, color="#0891b2")
    plt.title("Top Movie Genres")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    save_plot("03_top_movie_genres")

    plt.figure(figsize=(7, 4))
    top_types = anime_meta["Type"].fillna("Unknown").value_counts().head(10)
    sns.barplot(x=top_types.index, y=top_types.values, color="#f59e0b")
    plt.xticks(rotation=35, ha="right")
    plt.title("Anime Type Distribution")
    plt.ylabel("Count")
    save_plot("04_anime_type_distribution")

    plt.figure(figsize=(7, 4))
    anime_scores = pd.to_numeric(anime_meta["Score"], errors="coerce")
    sns.histplot(anime_scores.dropna(), bins=30, kde=True, color="#db2777")
    plt.title("Anime Community Score Distribution")
    save_plot("05_anime_score_distribution")

    numeric_cols = ["Score", "Popularity", "Members", "Favorites", "Episodes", "Ranked"]
    anime_num = anime_meta[numeric_cols].apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(8, 6))
    sns.heatmap(anime_num.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Anime Numeric Feature Correlation")
    save_plot("06_anime_correlation_heatmap")

    top_movies = movie_ratings["movieId"].value_counts().head(15)
    top_movies_named = top_movies.rename_axis("movieId").reset_index(name="rating_count")
    top_movies_named = top_movies_named.merge(movie_meta[["movieId", "title"]], on="movieId", how="left")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_movies_named["rating_count"], y=top_movies_named["title"].fillna(top_movies_named["movieId"].astype(str)), color="#0d9488")
    plt.title("Top Movies by Rating Count (Sampled)")
    plt.xlabel("Ratings Count")
    plt.ylabel("Movie")
    save_plot("07_top_movies_by_count")

    anime_members = anime_meta[["Name", "Members"]].copy()
    anime_members["Members"] = pd.to_numeric(anime_members["Members"], errors="coerce")
    anime_members = anime_members.dropna().sort_values("Members", ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=anime_members["Members"], y=anime_members["Name"], color="#4f46e5")
    plt.title("Top Anime by Members")
    plt.xlabel("Members")
    plt.ylabel("Anime")
    save_plot("08_top_anime_by_members")

    movie_user_activity = movie_ratings.groupby("userId")["movieId"].count()
    plt.figure(figsize=(7, 4))
    sns.histplot(np.log1p(movie_user_activity), bins=40, color="#ea580c")
    plt.title("Movie User Activity (log)")
    save_plot("09_movie_user_activity")

    anime_user_activity = anime_interactions.groupby("user_id")["anime_id"].count()
    plt.figure(figsize=(7, 4))
    sns.histplot(np.log1p(anime_user_activity), bins=40, color="#65a30d")
    plt.title("Anime User Activity (log)")
    save_plot("10_anime_user_activity")

    movie_users = movie_ratings["userId"].nunique()
    movie_items = movie_ratings["movieId"].nunique()
    movie_density = len(movie_ratings) / max(movie_users * movie_items, 1)
    anime_users = anime_interactions["user_id"].nunique()
    anime_items = anime_interactions["anime_id"].nunique()
    anime_density = len(anime_interactions) / max(anime_users * anime_items, 1)

    dens_df = pd.DataFrame({"domain": ["movie", "anime"], "density": [movie_density, anime_density]})
    plt.figure(figsize=(6, 4))
    sns.barplot(data=dens_df, x="domain", y="density", hue="domain", legend=False)
    plt.title("Interaction Matrix Density")
    save_plot("11_sparsity_density")

    sample_catalog = catalog.sample(n=min(5000, len(catalog)), random_state=RANDOM_STATE)
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
    X = tfidf.fit_transform(sample_catalog["content_text"])
    emb2d = TruncatedSVD(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

    plot_df = sample_catalog[["domain"]].copy()
    plot_df["x"] = emb2d[:, 0]
    plot_df["y"] = emb2d[:, 1]
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="domain", alpha=0.5)
    plt.title("Content Embedding Map")
    save_plot("12_content_embedding_map")

    plt.figure(figsize=(6, 4))
    domain_rating = interactions.groupby("domain")["rating"].mean().reset_index()
    sns.barplot(data=domain_rating, x="domain", y="rating", hue="domain", legend=False)
    plt.title("Average Rating by Domain")
    save_plot("13_domain_avg_rating")

    plt.figure(figsize=(6, 4))
    domain_var = interactions.groupby("domain")["rating"].var().reset_index().fillna(0)
    sns.barplot(data=domain_var, x="domain", y="rating", hue="domain", legend=False)
    plt.title("Rating Variance by Domain")
    save_plot("14_domain_rating_variance")

    top_items = interactions["item_id"].value_counts().head(20).reset_index()
    top_items.columns = ["item_id", "cnt"]
    top_items = top_items.merge(catalog[["item_id", "item_name", "domain"]], on="item_id", how="left")
    top_items["label"] = top_items["item_name"].str.slice(0, 40)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top_items, x="cnt", y="label", hue="domain")
    plt.title("Top Items by Interaction Volume")
    save_plot("15_top_items_interactions")
