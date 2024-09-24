import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score
from joblib import dump


def load_data():
    data = pd.read_csv("../data/feature_engineering.csv")
    x = data.drop("target", axis=1)
    y = data["target"]

    # split data in train and test splits
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42
    )

    return x_train, x_test, y_train, y_test


def main():
    x_train, x_test, y_train, y_test = load_data()

    # train model
    model = RandomForestClassifier(random_state=42, class_weight={0: 7, 1: 3})
    model.fit(x_train, y_train)

    # evaluate model
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Cross validation score: {cross_val_score(model, x_train, y_train, cv=5).mean():.2f}")
    print(f"Cross validation recall score: {cross_val_score(model, x_train, y_train, cv=5, scoring='recall').mean():.2f}")

    # save model
    dump(model, "final_model.pkl")

if __name__ == "__main__":
    main()