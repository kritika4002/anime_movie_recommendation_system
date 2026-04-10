# Hybrid Anime & Movie Recommendation System

A machine learning project that builds and compares 6 different recommendation algorithms across two domains (anime and movies), demonstrating when each model excels and how hybrid approaches improve robustness.

## Project Overview

This project implements a **cross-domain hybrid recommendation system** that combines collaborative filtering, content-based filtering, matrix factorization, and popularity-based approaches. The core objective is to:

- Compare 6 distinct ML models for recommendation accuracy
- Demonstrate understanding of when each model performs best
- Show how hybrid ensembles mitigate individual model weaknesses
- Handle real-world challenges (sparsity, cold-start problem)
- Provide interactive recommendations via Streamlit

## Key Features

- **Multi-Domain Recommendation**: Combined anime and movie catalogs with weighted domain preferences
  - Anime queries → 80% anime + 20% movie recommendations
  - Movie queries → 80% movie + 20% anime recommendations
- **6 ML Models Evaluated**:
  1. **Popularity Baseline** - Most rated/highest-rated items (cold-start baseline)
  2. **Content-Based** - TF-IDF vectorization + cosine similarity matching
  3. **Item-Based Collaborative Filtering** - Similarity based on user rating patterns
  4. **User-Based Collaborative Filtering** - Find similar users and recommend their liked items
  5. **Matrix Factorization** - TruncatedSVD for latent factor discovery
  6. **Hybrid Ensemble** - Weighted combination of all approaches

- **Comprehensive Evaluation**:
  - Hit Rate@10 (recall metric)
  - Mean Reciprocal Rank@10 (ranking quality)
  - NDCG@10 (normalized discounted cumulative gain)

- **Interactive Streamlit App** - Get personalized recommendations by domain

## Project Structure

```
anime_movie_recommendation_system/
├── data/                    # Dataset files
│   ├── Anime/              # Anime ratings and metadata
│   └── movie/              # Movie ratings, metadata, and links
├── src/                    # Source code
│   ├── recommender/        # Core recommendation engine
│   │   ├── config.py       # Configuration (paths, constants)
│   │   ├── data_loader.py  # Data loading utilities
│   │   ├── eda.py          # Exploratory data analysis
│   │   ├── models.py       # ML model implementations
│   │   ├── preprocessing.py # Data preprocessing
│   │   ├── recommend.py    # Recommendation logic
│   │   └── reporting.py    # Results reporting
│   └── run_pipeline.py     # ML pipeline execution
├── notebooks/              # Jupyter notebooks for analysis
├── outputs/                # Generated reports and visualizations
├── streamlit_app.py        # Interactive recommendation app
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd anime_movie_recommendation_system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the ML Pipeline

Execute the full model training, evaluation, and comparison:

```bash
python src/run_pipeline.py
```

This will:
- Load and preprocess anime + movie data
- Generate exploratory data analysis visualizations
- Train and evaluate all 6 models
- Generate comparison metrics and reports in `outputs/`

### Launch Interactive App

Start the Streamlit app for real-time recommendations:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Run Jupyter Notebook

Explore the analysis workflow interactively:

```bash
jupyter notebook notebooks/student_project_workflow.ipynb
```

## Model Performance

### Benchmark Results
| Model | Hit Rate@10 | MRR@10 | NDCG@10 | Best For |
|-------|-------------|--------|---------|----------|
| User-CF | 0.0813 | 0.0277 | 0.0402 | Established communities |
| Matrix-Factorization | 0.0866 | 0.0261 | 0.0400 | Large-scale systems |
| Hybrid | 0.0795 | 0.0248 | 0.0376 | Production robustness |
| Popularity | 0.0769 | 0.0162 | 0.0300 | Cold-start scenarios |
| Item-CF | 0.0567 | 0.0151 | 0.0245 | Dense item catalogs |
| Content | 0.0504 | 0.0147 | 0.0227 | Rich metadata scenarios |

### Key Insights
- **No single winner**: Different models excel in different metrics
- **Hybrid advantage**: Competitive across all metrics, handles edge cases well
- **Cold-start**: Popularity baseline provides essential baseline for new users/items
- **Trade-offs**: Item-CF excels with established items, but suffers from sparse data

## Datasets

### Movie Data
- **movies.csv**: Movie metadata (id, title, genres)
- **ratings.csv**: User ratings (userId, movieId, rating, timestamp)
- **links.csv**: Links to IMDb and TMDb

### Anime Data
- **anime-filtered.csv**: Anime metadata (anime_id, name, genre, type, episodes, etc.)
- **user-filtered.csv**: User anime ratings (user_id, anime_id, rating)

## Output Files

Generated in `outputs/` directory:

- **model_comparison.csv**: Performance metrics for all models
- **graphs/**: EDA visualizations and model performance charts
- **presentation_notes.md**: Summary findings for presentation

## Technologies Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, scipy
- **Visualization**: matplotlib, seaborn
- **Web App**: streamlit
- **Development**: Python 3.8+

## Author

Developed as a VIT MCA Machine Learning project

## License

This project is for educational purposes.
