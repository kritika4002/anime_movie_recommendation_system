import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import GRAPH_DIR, OUTPUT_DIR


def save_metric_plots(results: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="model", y="hit_rate_at_10", color="#0284c7")
    plt.title("Model Comparison: Hit Rate@10")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "16_model_hit_rate.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="model", y="ndcg_at_10", color="#7c3aed")
    plt.title("Model Comparison: NDCG@10")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "17_model_ndcg.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="model", y="mrr_at_10", color="#059669")
    plt.title("Model Comparison: MRR@10")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "18_model_mrr.png", dpi=160)
    plt.close()


def write_notes(results: pd.DataFrame) -> None:
    best_model = results.iloc[0]["model"] if not results.empty else "Hybrid"
    text = f"""# Presentation Notes

## Novelty
- Cross-domain hybrid recommendation using both anime and movie datasets.
- Weighted hybrid signal: item-CF + user-CF + matrix factorization + content + popularity.

## ML Model Comparison (Core Project Objective)
- Popularity Baseline
- Content-Based (TF-IDF + cosine)
- Item-Based Collaborative Filtering
- User-Based Collaborative Filtering
- Matrix Factorization (TruncatedSVD)
- Hybrid Ensemble Model

## Why This Model
- Item/User CF: personalized but sparse for cold users/items.
- Matrix Factorization: captures latent interaction patterns.
- Content: stable for cold-start but can over-specialize.
- Hybrid: balances personalization and robustness.
- Best model in current run: {best_model}

## Metrics Table
{results.to_string(index=False)}
"""
    (OUTPUT_DIR / "presentation_notes.md").write_text(text, encoding="utf-8")
