import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def sample_large_csv(
    path: Path,
    usecols: List[str],
    sample_frac: float,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    sampled_chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        if len(chunk) == 0:
            continue
        frac = min(max(sample_frac, 0.0001), 1.0)
        sampled = chunk.sample(frac=frac, random_state=RANDOM_STATE)
        sampled_chunks.append(sampled)
    if not sampled_chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(sampled_chunks, ignore_index=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movie_ratings_path = DATA_DIR / "movie" / "ratings.csv"
    movie_meta_path = DATA_DIR / "movie" / "movies.csv"

    anime_interactions_path = DATA_DIR / "Anime" / "user-filtered.csv"
    anime_meta_path = DATA_DIR / "Anime" / "anime-filtered.csv"

    print("Loading sampled interaction data...")
    movie_ratings = sample_large_csv(
        movie_ratings_path,
        usecols=["userId", "movieId", "rating"],
        sample_frac=0.01,
        chunksize=1_000_000,
    )
    anime_interactions = sample_large_csv(
        anime_interactions_path,
        usecols=["user_id", "anime_id", "rating"],
        sample_frac=0.003,
        chunksize=2_000_000,
    )

    print("Loading metadata...")
    movie_meta = pd.read_csv(movie_meta_path, usecols=["movieId", "title", "genres"])
    anime_meta = pd.read_csv(
        anime_meta_path,
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

    return movie_ratings, anime_interactions, movie_meta, anime_meta


def prepare_unified_data(
    movie_ratings: pd.DataFrame,
    anime_interactions: pd.DataFrame,
    movie_meta: pd.DataFrame,
    anime_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    movie = movie_ratings.rename(columns={"userId": "user_id", "movieId": "item_raw_id"}).copy()
    movie["domain"] = "movie"

    anime = anime_interactions.rename(columns={"anime_id": "item_raw_id"}).copy()
    anime["domain"] = "anime"

    movie["item_id"] = "movie_" + movie["item_raw_id"].astype(str)
    anime["item_id"] = "anime_" + anime["item_raw_id"].astype(str)

    movie["user_global"] = "m_" + movie["user_id"].astype(str)
    anime["user_global"] = "a_" + anime["user_id"].astype(str)

    interactions = pd.concat(
        [
            movie[["user_global", "item_id", "rating", "domain"]],
            anime[["user_global", "item_id", "rating", "domain"]],
        ],
        ignore_index=True,
    )

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
    catalog["genres"] = catalog["genres"].fillna("Unknown")
    catalog["type"] = catalog["type"].fillna("Unknown")
    catalog["source"] = catalog["source"].fillna("Unknown")
    catalog["item_name"] = catalog["item_name"].fillna("Unknown")

    catalog["content_text"] = (
        catalog["genres"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + catalog["type"].astype(str)
        + " "
        + catalog["source"].astype(str)
    )

    interactions = interactions[interactions["item_id"].isin(catalog["item_id"])].copy()
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions = interactions.dropna(subset=["rating"])

    return interactions, catalog


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
    print("Creating EDA graphs...")
    sns.set_theme(style="whitegrid")

    # 1. Movie rating distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(movie_ratings["rating"], bins=10, kde=True, color="#2563eb")
    plt.title("Movie Ratings Distribution")
    save_plot("01_movie_rating_distribution")

    # 2. Anime rating distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(anime_interactions["rating"], bins=10, kde=True, color="#16a34a")
    plt.title("Anime Ratings Distribution")
    save_plot("02_anime_rating_distribution")

    # 3. Movie top genres
    movie_genres = movie_meta.assign(genres=movie_meta["genres"].fillna("Unknown").str.split("|"))
    movie_genres = movie_genres.explode("genres")
    top_movie_genres = movie_genres["genres"].value_counts().head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_movie_genres.values, y=top_movie_genres.index, color="#0891b2")
    plt.title("Top Movie Genres")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    save_plot("03_top_movie_genres")

    # 4. Anime type distribution
    plt.figure(figsize=(7, 4))
    top_types = anime_meta["Type"].fillna("Unknown").value_counts().head(10)
    sns.barplot(x=top_types.index, y=top_types.values, color="#f59e0b")
    plt.xticks(rotation=35, ha="right")
    plt.title("Anime Type Distribution")
    plt.ylabel("Count")
    save_plot("04_anime_type_distribution")

    # 5. Anime score distribution
    plt.figure(figsize=(7, 4))
    anime_scores = pd.to_numeric(anime_meta["Score"], errors="coerce")
    sns.histplot(anime_scores.dropna(), bins=30, kde=True, color="#db2777")
    plt.title("Anime Community Score Distribution")
    save_plot("05_anime_score_distribution")

    # 6. Anime numeric correlation heatmap
    numeric_cols = ["Score", "Popularity", "Members", "Favorites", "Episodes", "Ranked"]
    anime_num = anime_meta[numeric_cols].apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(8, 6))
    sns.heatmap(anime_num.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Anime Numeric Feature Correlation")
    save_plot("06_anime_correlation_heatmap")

    # 7. Top movies by rating counts
    top_movies = movie_ratings["movieId"].value_counts().head(15)
    top_movies_named = top_movies.rename_axis("movieId").reset_index(name="rating_count")
    top_movies_named = top_movies_named.merge(movie_meta[["movieId", "title"]], on="movieId", how="left")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_movies_named["rating_count"], y=top_movies_named["title"].fillna(top_movies_named["movieId"].astype(str)), color="#0d9488")
    plt.title("Top Movies by Rating Count (Sampled Interactions)")
    plt.xlabel("Ratings Count")
    plt.ylabel("Movie")
    save_plot("07_top_movies_by_count")

    # 8. Top anime by members
    anime_members = anime_meta[["Name", "Members"]].copy()
    anime_members["Members"] = pd.to_numeric(anime_members["Members"], errors="coerce")
    anime_members = anime_members.dropna().sort_values("Members", ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=anime_members["Members"], y=anime_members["Name"], color="#4f46e5")
    plt.title("Top Anime by Members")
    plt.xlabel("Members")
    plt.ylabel("Anime")
    save_plot("08_top_anime_by_members")

    # 9. Movie user activity distribution
    movie_user_activity = movie_ratings.groupby("userId")["movieId"].count()
    plt.figure(figsize=(7, 4))
    sns.histplot(np.log1p(movie_user_activity), bins=40, color="#ea580c")
    plt.title("Movie User Activity (log(1 + ratings/user))")
    plt.xlabel("log(1 + ratings/user)")
    save_plot("09_movie_user_activity")

    # 10. Anime user activity distribution
    anime_user_activity = anime_interactions.groupby("user_id")["anime_id"].count()
    plt.figure(figsize=(7, 4))
    sns.histplot(np.log1p(anime_user_activity), bins=40, color="#65a30d")
    plt.title("Anime User Activity (log(1 + ratings/user))")
    plt.xlabel("log(1 + ratings/user)")
    save_plot("10_anime_user_activity")

    # 11. Sparsity comparison
    movie_users = movie_ratings["userId"].nunique()
    movie_items = movie_ratings["movieId"].nunique()
    movie_density = len(movie_ratings) / max(movie_users * movie_items, 1)

    anime_users = anime_interactions["user_id"].nunique()
    anime_items = anime_interactions["anime_id"].nunique()
    anime_density = len(anime_interactions) / max(anime_users * anime_items, 1)

    plt.figure(figsize=(6, 4))
    dens_df = pd.DataFrame({"domain": ["movie", "anime"], "density": [movie_density, anime_density]})
    sns.barplot(data=dens_df, x="domain", y="density", palette=["#2563eb", "#16a34a"])
    plt.title("Interaction Matrix Density")
    plt.ylabel("Density")
    save_plot("11_sparsity_density")

    # 12. Content space clustering (2D via SVD)
    sample_catalog = catalog.sample(n=min(5000, len(catalog)), random_state=RANDOM_STATE)
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
    X = tfidf.fit_transform(sample_catalog["content_text"])
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    emb2d = svd.fit_transform(X)

    plot_df = sample_catalog[["domain"]].copy()
    plot_df["x"] = emb2d[:, 0]
    plot_df["y"] = emb2d[:, 1]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="domain", alpha=0.5)
    plt.title("Content Embedding Map (Movie vs Anime)")
    save_plot("12_content_embedding_map")

    # 13. Domain-wise average rating
    plt.figure(figsize=(6, 4))
    domain_rating = interactions.groupby("domain")["rating"].mean().reset_index()
    sns.barplot(data=domain_rating, x="domain", y="rating", palette=["#1d4ed8", "#15803d"])
    plt.title("Average Rating by Domain")
    plt.ylabel("Mean Rating")
    save_plot("13_domain_avg_rating")

    # 14. Domain-wise rating variance
    plt.figure(figsize=(6, 4))
    domain_var = interactions.groupby("domain")["rating"].var().reset_index().fillna(0)
    sns.barplot(data=domain_var, x="domain", y="rating", palette=["#0ea5e9", "#22c55e"])
    plt.title("Rating Variance by Domain")
    plt.ylabel("Variance")
    save_plot("14_domain_rating_variance")

    # 15. Top combined items by interaction volume
    top_items = interactions["item_id"].value_counts().head(20).reset_index()
    top_items.columns = ["item_id", "cnt"]
    top_items = top_items.merge(catalog[["item_id", "item_name", "domain"]], on="item_id", how="left")
    top_items["label"] = top_items["item_name"].str.slice(0, 40)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=top_items, x="cnt", y="label", hue="domain")
    plt.title("Top Items by Interaction Volume")
    plt.xlabel("Interactions")
    plt.ylabel("Item")
    save_plot("15_top_items_interactions")


def filter_for_training(interactions: pd.DataFrame, min_user_interactions: int = 10, min_item_interactions: int = 10) -> pd.DataFrame:
    data = interactions.copy()
    data = data[data["rating"] >= 1]

    user_counts = data["user_global"].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    data = data[data["user_global"].isin(valid_users)]

    item_counts = data["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    data = data[data["item_id"].isin(valid_items)]

    return data


def train_test_split_leave_one(data: pd.DataFrame, max_users: int = 1200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    users = data["user_global"].unique()
    if len(users) > max_users:
        users = rng.choice(users, size=max_users, replace=False)
        data = data[data["user_global"].isin(users)].copy()

    test_idx = []
    for user, grp in data.groupby("user_global"):
        if len(grp) < 5:
            continue
        positives = grp[grp["rating"] >= 4]
        choose_from = positives if len(positives) > 0 else grp
        test_idx.append(rng.choice(choose_from.index.to_numpy(), size=1)[0])

    test = data.loc[test_idx].copy()
    train = data.drop(index=test_idx).copy()
    return train, test


def build_content_model(catalog: pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray, Dict[str, int]]:
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(catalog["content_text"])
    item_to_idx = {item: idx for idx, item in enumerate(catalog["item_id"].tolist())}
    return vectorizer, X, item_to_idx


def build_cf_similarity(train: pd.DataFrame, items: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int], csr_matrix]:
    users = sorted(train["user_global"].unique().tolist())
    item_to_idx = {item: i for i, item in enumerate(items)}
    user_to_idx = {u: i for i, u in enumerate(users)}

    row = train["user_global"].map(user_to_idx).to_numpy()
    col = train["item_id"].map(item_to_idx).to_numpy()
    val = train["rating"].to_numpy().astype(float)

    mat = csr_matrix((val, (row, col)), shape=(len(users), len(items)))
    item_user = mat.T
    sim = cosine_similarity(item_user)
    np.fill_diagonal(sim, 0)
    return sim, item_to_idx, user_to_idx, mat


def user_profile_vector(
    user_items: List[int],
    user_ratings: np.ndarray,
    content_matrix,
) -> np.ndarray:
    if len(user_items) == 0:
        return np.zeros(content_matrix.shape[1])
    user_content = content_matrix[user_items]
    weighted = user_content.multiply(user_ratings.reshape(-1, 1))
    vec = np.asarray(weighted.mean(axis=0)).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def evaluate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    active_items = sorted(set(train["item_id"].unique()).union(set(test["item_id"].unique())))
    catalog = catalog[catalog["item_id"].isin(active_items)].copy()

    item_popularity = train.groupby("item_id")["rating"].mean().to_dict()

    _, content_X, content_item_idx = build_content_model(catalog)
    cf_sim, cf_item_idx, cf_user_idx, train_mat = build_cf_similarity(train, catalog["item_id"].tolist())

    users_train = train.groupby("user_global")

    metrics = {
        "Popularity": {"hit": [], "mrr": [], "ndcg": []},
        "Content": {"hit": [], "mrr": [], "ndcg": []},
        "Collaborative": {"hit": [], "mrr": [], "ndcg": []},
        "Hybrid": {"hit": [], "mrr": [], "ndcg": []},
    }

    all_items = catalog["item_id"].tolist()
    rng = np.random.default_rng(RANDOM_STATE)

    for _, row in test.iterrows():
        user = row["user_global"]
        true_item = row["item_id"]

        if user not in users_train.groups:
            continue

        user_hist = users_train.get_group(user)
        seen_items = set(user_hist["item_id"].tolist())

        candidates_pool = [i for i in all_items if i not in seen_items and i != true_item]
        if len(candidates_pool) < 200:
            continue

        negatives = rng.choice(candidates_pool, size=200, replace=False).tolist()
        candidates = negatives + [true_item]

        # Popularity scores
        pop_scores = {c: float(item_popularity.get(c, 0.0)) for c in candidates}

        # Content scores
        hist_items = [content_item_idx[i] for i in seen_items if i in content_item_idx]
        hist_ratings = user_hist[user_hist["item_id"].isin(seen_items)]["rating"].to_numpy()
        hist_ratings = hist_ratings[: len(hist_items)] if len(hist_items) > 0 else np.array([])

        if len(hist_items) > 0 and len(hist_ratings) > 0:
            u_vec = user_profile_vector(hist_items, hist_ratings, content_X)
            cand_idx = [content_item_idx[c] for c in candidates if c in content_item_idx]
            if len(cand_idx) > 0:
                c_mat = content_X[cand_idx]
                content_scores_arr = cosine_similarity(c_mat, u_vec.reshape(1, -1)).ravel()
                content_scores = {
                    candidates[i]: float(content_scores_arr[j])
                    for j, i in enumerate([k for k, c in enumerate(candidates) if c in content_item_idx])
                }
            else:
                content_scores = {c: 0.0 for c in candidates}
        else:
            content_scores = {c: 0.0 for c in candidates}

        # Collaborative scores
        if user in cf_user_idx:
            uidx = cf_user_idx[user]
            user_vector = train_mat[uidx].toarray().ravel()
            seen_idx = np.where(user_vector > 0)[0]
            collab_scores = {}
            for c in candidates:
                if c not in cf_item_idx:
                    collab_scores[c] = 0.0
                    continue
                c_idx = cf_item_idx[c]
                sims = cf_sim[c_idx, seen_idx]
                ratings = user_vector[seen_idx]
                denom = np.sum(np.abs(sims)) + 1e-9
                collab_scores[c] = float(np.dot(sims, ratings) / denom) if len(seen_idx) > 0 else 0.0
        else:
            collab_scores = {c: 0.0 for c in candidates}

        # Hybrid score = weighted normalized mix
        def normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
            vals = np.array(list(d.values()), dtype=float)
            if np.allclose(vals.max(), vals.min()):
                return {k: 0.0 for k in d.keys()}
            vals_n = (vals - vals.min()) / (vals.max() - vals.min())
            return {k: float(v) for k, v in zip(d.keys(), vals_n)}

        n_pop = normalize_dict(pop_scores)
        n_con = normalize_dict(content_scores)
        n_col = normalize_dict(collab_scores)

        hybrid_scores = {
            c: (0.50 * n_col.get(c, 0.0)) + (0.35 * n_con.get(c, 0.0)) + (0.15 * n_pop.get(c, 0.0))
            for c in candidates
        }

        model_scores = {
            "Popularity": pop_scores,
            "Content": content_scores,
            "Collaborative": collab_scores,
            "Hybrid": hybrid_scores,
        }

        for model_name, score_dict in model_scores.items():
            ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            ranked_items = [x[0] for x in ranked]
            if true_item in ranked_items:
                rank = ranked_items.index(true_item) + 1
                hit = 1.0
                mrr = 1.0 / rank
                ndcg = 1.0 / np.log2(rank + 1)
            else:
                hit = 0.0
                mrr = 0.0
                ndcg = 0.0

            metrics[model_name]["hit"].append(hit)
            metrics[model_name]["mrr"].append(mrr)
            metrics[model_name]["ndcg"].append(ndcg)

    rows = []
    for model_name, vals in metrics.items():
        rows.append(
            {
                "model": model_name,
                "hit_rate_at_10": float(np.mean(vals["hit"])) if vals["hit"] else 0.0,
                "mrr_at_10": float(np.mean(vals["mrr"])) if vals["mrr"] else 0.0,
                "ndcg_at_10": float(np.mean(vals["ndcg"])) if vals["ndcg"] else 0.0,
                "users_evaluated": int(len(vals["hit"])),
            }
        )

    result = pd.DataFrame(rows).sort_values("ndcg_at_10", ascending=False)
    return result


def plot_model_comparison(results: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="model", y="hit_rate_at_10", color="#0284c7")
    plt.title("Model Comparison: Hit Rate@10")
    plt.ylabel("Hit Rate@10")
    save_plot("16_model_hit_rate")

    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="model", y="ndcg_at_10", color="#7c3aed")
    plt.title("Model Comparison: NDCG@10")
    plt.ylabel("NDCG@10")
    save_plot("17_model_ndcg")

    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="model", y="mrr_at_10", color="#059669")
    plt.title("Model Comparison: MRR@10")
    plt.ylabel("MRR@10")
    save_plot("18_model_mrr")


def write_presentation_notes(results: pd.DataFrame) -> None:
    best_model = results.iloc[0]["model"] if len(results) else "Hybrid"

    notes = f"""# Project Talking Points (For Viva/Presentation)

## 1. Novelty You Can Claim
- Cross-domain hybrid recommendation: one framework for both MovieLens movies and anime catalog.
- Weighted hybrid strategy combines three signals:
  - collaborative preference signal
  - content similarity signal
  - popularity prior
- Practical cold-start handling: content + popularity helps when collaborative history is weak.

## 2. Why Hybrid Is Best
- Collaborative alone: strong personalization, but sparse users/items suffer.
- Content alone: stable for cold-start, but may over-specialize.
- Popularity baseline: robust but not personalized.
- Hybrid balances accuracy and robustness; in this run, best model is: {best_model}.

## 3. Variable Importance and Relations to Explain
- Rating behavior: `rating`, user activity count, item interaction count.
- Content descriptors: `genres`, `type`, `source`.
- Popularity descriptors (anime): `Members`, `Favorites`, `Popularity`, `Ranked`, `Score`.
- Key relation examples to discuss:
  - `Members` vs `Favorites` (community engagement relationship)
  - `Popularity` vs `Score` (mainstream vs quality trade-off)
  - user activity skew and matrix sparsity impact on CF performance

## 4. Evaluation Metrics Used
- Hit Rate@10
- MRR@10
- NDCG@10

## 5. Current Model Comparison Table
{results.to_string(index=False)}
"""

    (OUTPUT_DIR / "presentation_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    movie_ratings, anime_interactions, movie_meta, anime_meta = load_data()
    interactions, catalog = prepare_unified_data(movie_ratings, anime_interactions, movie_meta, anime_meta)

    create_eda_graphs(movie_ratings, anime_interactions, movie_meta, anime_meta, interactions, catalog)

    print("Preparing training data...")
    trainable = filter_for_training(interactions, min_user_interactions=10, min_item_interactions=10)
    train, test = train_test_split_leave_one(trainable, max_users=1200)

    print(f"Train size: {train.shape}, Test size: {test.shape}")

    print("Evaluating models...")
    results = evaluate_models(train, test, catalog)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plot_model_comparison(results)
    write_presentation_notes(results)

    print("Done. Files generated:")
    print(f"- Graphs: {GRAPH_DIR}")
    print(f"- Metrics: {OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"- Notes: {OUTPUT_DIR / 'presentation_notes.md'}")


if __name__ == "__main__":
    main()
