from sklearn.tree import DecisionTreeClassifier
from src.utils import evaluate_model
from src.preprocess import preprocess_data


def train_base_decision_tree(data_path: str = "data/data.csv"):
    """ Train the Decision Tree model on training data """
    X_train, X_test, y_train, y_test, scaler, df = preprocess_data(data_path)

    # Train Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = evaluate_model("Decision Tree", y_test, y_pred)

    return model, metrics