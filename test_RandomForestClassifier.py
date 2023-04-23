import pandas as pd
from rpart.RandomForestClassifier import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    adult = pd.read_csv("data/adult.csv").sample(1000)
    X = adult.drop("income", axis=1)
    y = adult["income"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    start_time = time.time()

    # Create and train the DecisionTreeClassifier
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=3, n_workers=10, metric="gini"
    )
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("--- %s seconds ---" % (time.time() - start_time))
