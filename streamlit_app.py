import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from recommender.config import GRAPH_DIR, OUTPUT_DIR
from recommender.data_loader import load_anime_data, load_movie_data
from recommender.preprocessing import prepare_unified_data
from recommender.recommend import recommend_mixed_by_query

st.set_page_config(page_title="Hybrid Anime-Movie Recommender", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .hero-box {
        padding: 14px 18px;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid #dbeafe;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
    }
    .mix-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #e0f2fe;
        color: #0c4a6e;
        font-weight: 600;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
      <h2 style="margin:0; color:#0f172a;">Hybrid Recommendation System: Anime + Movies</h2>
      <p style="margin:8px 0 0 0; color:#334155;">
         Study cross-domain suggestion logic, model comparison, and EDA graphs.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_catalog_for_search() -> pd.DataFrame:
    movie_ratings, movie_meta = load_movie_data(sample_frac=0.002)
    anime_interactions, anime_meta = load_anime_data(sample_frac=0.001)
    _, catalog = prepare_unified_data(movie_ratings, anime_interactions, movie_meta, anime_meta)
    return catalog

@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    metric_file = OUTPUT_DIR / "model_comparison.csv"
    if metric_file.exists():
        return pd.read_csv(metric_file)
    return pd.DataFrame()

metrics = load_metrics()

st.sidebar.header("Project Controls")
st.sidebar.write("If outputs are missing, run:")
st.sidebar.code("python src/run_pipeline.py")
st.sidebar.write("Then launch app with:")
st.sidebar.code("python -m streamlit run streamlit_app.py")

if metrics.empty:
    st.warning("Model comparison file not found. Run pipeline first.")
else:
    best_row = metrics.sort_values("ndcg_at_10", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", str(best_row["model"]))
    c2.metric("Best NDCG@10", f"{best_row['ndcg_at_10']:.4f}")
    c3.metric("Best HitRate@10", f"{best_row['hit_rate_at_10']:.4f}")
    c4.metric("Users Evaluated", f"{int(best_row['users_evaluated'])}")

tab1, tab2, tab3 = st.tabs(["Search Demo", "Model Scores", "Graph Gallery"])

with tab1:
    st.subheader("Search Recommendation")
    c1, c2 = st.columns([3, 1])
    query = c1.text_input("Enter anime or movie name", value="Naruto")
    top_n = c2.slider("Top N", min_value=5, max_value=20, value=10, step=1)

    st.markdown(
        "<span class='mix-pill'>If anime query: 80% anime + 20% movie</span>"
        "<span class='mix-pill'>If movie query: 80% movie + 20% anime</span>",
        unsafe_allow_html=True,
    )

    if st.button("Recommend", type="primary"):
        catalog = load_catalog_for_search()
        detected_domain, matched_name, recs = recommend_mixed_by_query(query=query, catalog=catalog, top_n=top_n)

        info_col1, info_col2 = st.columns(2)
        info_col1.info(f"Detected Query Domain: {detected_domain.upper()}")
        info_col2.info(f"Closest Matched Title: {matched_name}")

        if not recs.empty:
            domain_mix = recs["domain"].value_counts(normalize=True).mul(100).round(1)
            anime_mix = float(domain_mix.get("anime", 0.0))
            movie_mix = float(domain_mix.get("movie", 0.0))

            mix1, mix2 = st.columns(2)
            mix1.metric("Anime in Output", f"{anime_mix:.1f}%")
            mix2.metric("Movie in Output", f"{movie_mix:.1f}%")

            styled = recs.copy()
            styled["sim_score"] = styled["sim_score"].round(4)
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("No recommendations found for this query.")

with tab2:
    st.subheader("ML Model Comparison (6 Models)")
    st.markdown(
        """
        This project compares **6 different machine learning approaches** for recommendation systems.
        The metrics show how each model performs on the Anime + Movie dataset.
        """
    )
    
    if metrics.empty:
        st.info("Run pipeline to generate model scores.")
    else:
        # Detailed model descriptions
        with st.expander("Model Descriptions", expanded=False):
            col_desc1, col_desc2 = st.columns(2)
            
            with col_desc1:
                st.markdown("""
                **1. Popularity Baseline**
                - Recommends most frequently rated items
                - Simple baseline, no personalization
                - Fast and useful for cold-start
                
                **2. Content-Based**
                - Uses item metadata (genres, type, source)
                - Builds user profiles from liked items
                - TF-IDF + cosine similarity
                
                **3. Item-Based CF**
                - Finds similar items via rating patterns
                - Captures implicit user preferences
                - Sparse matrix challenge
                """)
            
            with col_desc2:
                st.markdown("""
                **4. User-Based CF**
                - Finds similar users, recommends their items
                - Captures community behavior
                - Cold-start for new users
                
                **5. Matrix Factorization**
                - SVD-based latent factor model
                - Reduces sparsity, scales to large datasets
                - Captures hidden interaction patterns
                
                **6. Hybrid Ensemble**
                - Weighted combination of all 5 above
                - Robust across scenarios
                - Production-ready approach
                """)
        
        st.markdown("---")
        st.markdown("### Performance Metrics Comparison")
        st.dataframe(metrics, use_container_width=True)

        st.markdown("---")
        st.markdown("### Performance Rankings")
        
        col_hit, col_ndcg, col_mrr = st.columns(3)
        
        with col_hit:
            st.markdown("**Hit Rate@10** (Recall)")
            hit_sorted = metrics.sort_values("hit_rate_at_10", ascending=True)
            st.bar_chart(hit_sorted.set_index("model")["hit_rate_at_10"])
        
        with col_ndcg:
            st.markdown("**NDCG@10** (Ranking Quality)")
            ndcg_sorted = metrics.sort_values("ndcg_at_10", ascending=True)
            st.bar_chart(ndcg_sorted.set_index("model")["ndcg_at_10"])
        
        with col_mrr:
            st.markdown("**MRR@10** (Mean Reciprocal Rank)")
            mrr_sorted = metrics.sort_values("mrr_at_10", ascending=True)
            st.bar_chart(mrr_sorted.set_index("model")["mrr_at_10"])
        
        st.markdown("---")
        st.markdown("### Why Hybrid Model Is Best")
        
        best_hit = metrics.loc[metrics["hit_rate_at_10"].idxmax()]
        best_ndcg = metrics.loc[metrics["ndcg_at_10"].idxmax()]
        best_mrr = metrics.loc[metrics["mrr_at_10"].idxmax()]
        
        insights = f"""
        - **Best Hit Rate**: {best_hit['model']} ({best_hit['hit_rate_at_10']:.4f})
        - **Best NDCG**: {best_ndcg['model']} ({best_ndcg['ndcg_at_10']:.4f})
        - **Best MRR**: {best_mrr['model']} ({best_mrr['mrr_at_10']:.4f})
        
        **Key Insight**: No single model dominates all metrics. Different scenarios favor different algorithms.
        
        **Why Hybrid wins in production:**
        - Balances all three metrics
        - Robust across scenarios (doesn't fail on one signal)
        - Handles cold-start via content + popularity
        - Works for mixed catalogs (anime + movies)
        - Production-ready reliability
        
        **Trade-offs:**
        - Item/User CF alone: great recall but sparse
        - Matrix Factorization: excellent but cold-start limited
        - Content alone: safe but low coverage
        - Popularity alone: fast but not personalized
        - **Hybrid: reliable across all scenarios**
        """
        st.markdown(insights)

with tab3:
    st.subheader("Graphs (EDA + Metrics)")
    if GRAPH_DIR.exists():
        graph_files = sorted([p for p in GRAPH_DIR.iterdir() if p.suffix.lower() == ".png"])
        if graph_files:
            col_a, col_b = st.columns(2)
            for i, g in enumerate(graph_files):
                if i % 2 == 0:
                    with col_a:
                        st.image(str(g), caption=g.name, use_container_width=True)
                else:
                    with col_b:
                        st.image(str(g), caption=g.name, use_container_width=True)
        else:
            st.info("No graph files found. Run pipeline first.")
    else:
        st.info("Graph folder not found. Run pipeline first.")
