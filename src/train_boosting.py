from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from src.utils import evaluate_model
from src.preprocess import preprocess_data


def train_adaboost(data_path: str = "data/data.csv"):
    """ Train an AdaBoost Classifier on the given dataset """
    X_train, X_test, y_train, y_test, scaler, df = preprocess_data(data_path)

    # Create an AdaBoost classifier with a Decision Tree as the base estimator
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )

    # Fit the model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    metrics = evaluate_model("AdaBoost", y_test, y_pred)
    return model, metrics


def train_gradient_boosting(data_path: str = "data/data.csv"):
    """ Train a Gradient Boosting Classifier on the given dataset """
    X_train, X_test, y_train, y_test, scaler, df = preprocess_data(data_path)

    # Create a Gradient Boosting classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    # Fit the model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    metrics = evaluate_model("Gradient Boosting", y_test, y_pred)
    return model, metrics