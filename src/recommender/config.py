from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"

MOVIE_RATINGS = DATA_DIR / "movie" / "ratings.csv"
MOVIE_META = DATA_DIR / "movie" / "movies.csv"
ANIME_INTERACTIONS = DATA_DIR / "Anime" / "user-filtered.csv"
ANIME_META = DATA_DIR / "Anime" / "anime-filtered.csv"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
