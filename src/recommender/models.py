from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import RANDOM_STATE


def train_test_split_leave_one(data: pd.DataFrame, max_users: int = 1200) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    users = data["user_global"].unique()
    if len(users) > max_users:
        users = rng.choice(users, size=max_users, replace=False)
        data = data[data["user_global"].isin(users)].copy()

    test_idx = []
    for _, grp in data.groupby("user_global"):
        if len(grp) < 5:
            continue
        positives = grp[grp["rating"] >= 4]
        choose_from = positives if len(positives) > 0 else grp
        test_idx.append(rng.choice(choose_from.index.to_numpy(), size=1)[0])

    test = data.loc[test_idx].copy()
    train = data.drop(index=test_idx).copy()
    return train, test


def build_content_matrix(catalog: pd.DataFrame) -> tuple[TfidfVectorizer, any, Dict[str, int]]:
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    x = vec.fit_transform(catalog["content_text"])
    idx = {item: i for i, item in enumerate(catalog["item_id"].tolist())}
    return vec, x, idx


def build_cf_similarity(train: pd.DataFrame, items: List[str]) -> tuple[np.ndarray, Dict[str, int], Dict[str, int], csr_matrix]:
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


def build_user_cf_similarity(train_mat: csr_matrix) -> np.ndarray:
    sim = cosine_similarity(train_mat)
    np.fill_diagonal(sim, 0)
    return sim


def build_matrix_factorization(train_mat: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    n_users, n_items = train_mat.shape
    n_comp = max(2, min(40, n_users - 1, n_items - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(train_mat)
    item_factors = svd.components_.T
    return user_factors, item_factors


def _user_profile_vector(user_items: List[int], user_ratings: np.ndarray, content_matrix) -> np.ndarray:
    if not user_items:
        return np.zeros(content_matrix.shape[1])
    user_content = content_matrix[user_items]
    weighted = user_content.multiply(user_ratings.reshape(-1, 1))
    vec = np.asarray(weighted.mean(axis=0)).ravel()
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def _norm_dict(scores: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(scores.values()), dtype=float)
    if len(vals) == 0 or np.allclose(vals.max(), vals.min()):
        return {k: 0.0 for k in scores.keys()}
    vals_n = (vals - vals.min()) / (vals.max() - vals.min())
    return {k: float(v) for k, v in zip(scores.keys(), vals_n)}


def evaluate_models(train: pd.DataFrame, test: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    active_items = sorted(set(train["item_id"].unique()).union(set(test["item_id"].unique())))
    catalog = catalog[catalog["item_id"].isin(active_items)].copy()
    all_items = catalog["item_id"].tolist()

    item_popularity = train.groupby("item_id")["rating"].mean().to_dict()
    _, content_x, content_idx = build_content_matrix(catalog)
    cf_sim, cf_item_idx, cf_user_idx, train_mat = build_cf_similarity(train, all_items)
    user_cf_sim = build_user_cf_similarity(train_mat)
    mf_user_factors, mf_item_factors = build_matrix_factorization(train_mat)
    users_train = train.groupby("user_global")

    rng = np.random.default_rng(RANDOM_STATE)
    metrics = {
        k: {"hit": [], "mrr": [], "ndcg": []}
        for k in [
            "Popularity",
            "Content",
            "Item-CF",
            "User-CF",
            "Matrix-Factorization",
            "Hybrid",
        ]
    }

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

        pop_scores = {c: float(item_popularity.get(c, 0.0)) for c in candidates}

        hist_items = [content_idx[i] for i in seen_items if i in content_idx]
        hist_r = user_hist[user_hist["item_id"].isin(seen_items)]["rating"].to_numpy()
        hist_r = hist_r[: len(hist_items)] if hist_items else np.array([])

        if hist_items and len(hist_r) > 0:
            u_vec = _user_profile_vector(hist_items, hist_r, content_x)
            cand_pos = [k for k, c in enumerate(candidates) if c in content_idx]
            cand_idx = [content_idx[candidates[k]] for k in cand_pos]
            c_mat = content_x[cand_idx]
            c_scores = cosine_similarity(c_mat, u_vec.reshape(1, -1)).ravel()
            content_scores = {candidates[k]: float(c_scores[j]) for j, k in enumerate(cand_pos)}
            for c in candidates:
                content_scores.setdefault(c, 0.0)
        else:
            content_scores = {c: 0.0 for c in candidates}

        if user in cf_user_idx:
            uidx = cf_user_idx[user]
            user_vector = train_mat[uidx].toarray().ravel()
            seen_idx = np.where(user_vector > 0)[0]
            item_cf_scores = {}
            for c in candidates:
                if c not in cf_item_idx:
                    item_cf_scores[c] = 0.0
                    continue
                c_idx = cf_item_idx[c]
                sims = cf_sim[c_idx, seen_idx]
                ratings = user_vector[seen_idx]
                denom = np.sum(np.abs(sims)) + 1e-9
                item_cf_scores[c] = float(np.dot(sims, ratings) / denom) if len(seen_idx) > 0 else 0.0

            # User-based collaborative filtering score.
            user_sim_vec = user_cf_sim[uidx]
            user_cf_scores = {}
            for c in candidates:
                if c not in cf_item_idx:
                    user_cf_scores[c] = 0.0
                    continue
                c_idx = cf_item_idx[c]
                item_ratings_by_users = train_mat[:, c_idx].toarray().ravel()
                rated_mask = item_ratings_by_users > 0
                if not np.any(rated_mask):
                    user_cf_scores[c] = 0.0
                    continue
                sims = user_sim_vec[rated_mask]
                vals = item_ratings_by_users[rated_mask]
                denom = np.sum(np.abs(sims)) + 1e-9
                user_cf_scores[c] = float(np.dot(sims, vals) / denom)

            # Matrix factorization score.
            mf_scores = {}
            u_vec = mf_user_factors[uidx]
            for c in candidates:
                if c not in cf_item_idx:
                    mf_scores[c] = 0.0
                    continue
                c_idx = cf_item_idx[c]
                mf_scores[c] = float(np.dot(u_vec, mf_item_factors[c_idx]))
        else:
            item_cf_scores = {c: 0.0 for c in candidates}
            user_cf_scores = {c: 0.0 for c in candidates}
            mf_scores = {c: 0.0 for c in candidates}

        npop = _norm_dict(pop_scores)
        ncon = _norm_dict(content_scores)
        nitm = _norm_dict(item_cf_scores)
        nusr = _norm_dict(user_cf_scores)
        nmf = _norm_dict(mf_scores)
        hybrid_scores = {
            c: 0.30 * nitm[c] + 0.20 * nusr[c] + 0.20 * nmf[c] + 0.20 * ncon[c] + 0.10 * npop[c]
            for c in candidates
        }

        model_scores = {
            "Popularity": pop_scores,
            "Content": content_scores,
            "Item-CF": item_cf_scores,
            "User-CF": user_cf_scores,
            "Matrix-Factorization": mf_scores,
            "Hybrid": hybrid_scores,
        }

        for model_name, score_map in model_scores.items():
            ranked_items = [x[0] for x in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:10]]
            if true_item in ranked_items:
                rank = ranked_items.index(true_item) + 1
                hit = 1.0
                mrr = 1.0 / rank
                ndcg = 1.0 / np.log2(rank + 1)
            else:
                hit = mrr = ndcg = 0.0
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
    return pd.DataFrame(rows).sort_values("ndcg_at_10", ascending=False)
