from article_outputs.build_allocation_tables import build_allocation_tables
from article_outputs.build_feature_importance_figures import build_feature_importance_figures
from article_outputs.build_majority_vote_figures import build_majority_vote_figures
from article_outputs.build_prediction_figures import build_prediction_figures


def main():
    build_majority_vote_figures()
    build_prediction_figures()
    build_feature_importance_figures()
    build_allocation_tables()


if __name__ == "__main__":
    main()
