import os
from src.train_base import train_base_decision_tree
from src.train_bagging import train_bagging_decision_tree
from src.train_boosting import train_adaboost, train_gradient_boosting
from src.utils import save_metrics_to_csv, plot_comparison_chart


def main():
    """ Main function """
    os.makedirs("outputs", exist_ok=True)

    metrics_list = []

    print("\n--- Training Base Decision Tree ---")
    _, m1 = train_base_decision_tree()
    metrics_list.append(m1)

    print("\n--- Training Bagging ---")
    _, m2 = train_bagging_decision_tree()
    metrics_list.append(m2)

    print("\n--- Training AdaBoost ---")
    _, m3 = train_adaboost()
    metrics_list.append(m3)

    print("\n--- Training Gradient Boosting ---")
    _, m4 = train_gradient_boosting()
    metrics_list.append(m4)

    save_metrics_to_csv(metrics_list)
    plot_comparison_chart(metrics_list)


if __name__ == "__main__":
    main()