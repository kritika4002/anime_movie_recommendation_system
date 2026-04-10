from recommender.config import OUTPUT_DIR, ensure_output_dirs
from recommender.data_loader import load_anime_data, load_movie_data
from recommender.eda import create_eda_graphs
from recommender.models import evaluate_models, train_test_split_leave_one
from recommender.preprocessing import filter_for_training, prepare_unified_data
from recommender.reporting import save_metric_plots, write_notes


def main() -> None:
    ensure_output_dirs()

    movie_ratings, movie_meta = load_movie_data(sample_frac=0.01)
    anime_interactions, anime_meta = load_anime_data(sample_frac=0.003)

    interactions, catalog = prepare_unified_data(movie_ratings, anime_interactions, movie_meta, anime_meta)

    create_eda_graphs(movie_ratings, anime_interactions, movie_meta, anime_meta, interactions, catalog)

    trainable = filter_for_training(interactions, min_user_interactions=10, min_item_interactions=10)
    train, test = train_test_split_leave_one(trainable, max_users=1200)

    results = evaluate_models(train, test, catalog)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    save_metric_plots(results)
    write_notes(results)

    print("Pipeline completed.")
    print(f"Saved results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
