from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _best_match_domain(query: str, catalog: pd.DataFrame) -> tuple[str, str]:
    q = query.lower().strip()
    anime = catalog[catalog["domain"] == "anime"]["item_name"].dropna().astype(str).head(5000)
    movie = catalog[catalog["domain"] == "movie"]["item_name"].dropna().astype(str).head(5000)

    def score_series(series: pd.Series) -> tuple[float, str]:
        best_score, best_name = 0.0, ""
        for name in series:
            name_l = name.lower()
            if q in name_l:
                s = 1.0
            else:
                s = SequenceMatcher(None, q, name_l).ratio()
            if s > best_score:
                best_score, best_name = s, name
        return best_score, best_name

    anime_s, anime_name = score_series(anime)
    movie_s, movie_name = score_series(movie)

    if anime_s >= movie_s:
        return "anime", anime_name
    return "movie", movie_name


def build_content_index(catalog: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=7000, stop_words="english")
    x = vectorizer.fit_transform(catalog["content_text"].fillna(""))
    item_to_idx = {item: idx for idx, item in enumerate(catalog["item_id"].tolist())}
    return vectorizer, x, item_to_idx


def recommend_mixed_by_query(query: str, catalog: pd.DataFrame, top_n: int = 10) -> tuple[str, str, pd.DataFrame]:
    query_domain, matched_name = _best_match_domain(query, catalog)

    row = catalog[catalog["item_name"] == matched_name].head(1)
    if row.empty:
        return query_domain, matched_name, pd.DataFrame(columns=["item_name", "domain", "genres", "score", "sim_score"])

    vectorizer, x, item_to_idx = build_content_index(catalog)
    anchor_item_id = row.iloc[0]["item_id"]
    anchor_idx = item_to_idx[anchor_item_id]

    sim = cosine_similarity(x[anchor_idx], x).ravel()
    candidates = catalog.copy()
    candidates["sim_score"] = sim
    candidates = candidates[candidates["item_id"] != anchor_item_id]

    primary = "anime" if query_domain == "anime" else "movie"
    secondary = "movie" if primary == "anime" else "anime"

    primary_n = int(np.ceil(top_n * 0.8))
    secondary_n = max(top_n - primary_n, 1)

    rec_primary = candidates[candidates["domain"] == primary].nlargest(primary_n, "sim_score")
    rec_secondary = candidates[candidates["domain"] == secondary].nlargest(secondary_n, "sim_score")

    recs = pd.concat([rec_primary, rec_secondary], ignore_index=True)
    recs = recs.nlargest(top_n, "sim_score")
    recs = recs[["item_name", "domain", "genres", "score", "sim_score"]].reset_index(drop=True)
    return query_domain, matched_name, recs
