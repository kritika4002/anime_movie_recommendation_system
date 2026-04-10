# Presentation Notes

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
- Best model in current run: User-CF

## Metrics Table
               model  hit_rate_at_10  mrr_at_10  ndcg_at_10  users_evaluated
             User-CF        0.081272   0.027717    0.040189             1132
Matrix-Factorization        0.086572   0.026104    0.040010             1132
              Hybrid        0.075088   0.025524    0.037080             1132
          Popularity        0.076855   0.016230    0.029988             1132
             Item-CF        0.056537   0.015149    0.024496             1132
             Content        0.044170   0.012827    0.019944             1132
