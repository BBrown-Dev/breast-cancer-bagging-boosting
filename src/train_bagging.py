from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from src.utils import evaluate_model
from src.preprocess import preprocess_data


def train_bagging_decision_tree(data_path: str = "data/data.csv"):
    """ Train a Bagging Classifier with Decision Trees on the given dataset """
    X_train, X_test, y_train, y_test, scaler, df = preprocess_data(data_path)

    # Create a Bagging Classifier with Decision Trees
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    metrics = evaluate_model("Bagging (Decision Tree)", y_test, y_pred)
    return model, metrics