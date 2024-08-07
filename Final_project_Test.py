from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import pandas as pd
from main import Classifier  # Import from main.py

def compute_final_score(classifier_f1):
    classifier_f1 = round(100 * classifier_f1)
    if classifier_f1 >= 90:
        return classifier_f1 + 4
    elif classifier_f1 >= 80:
        return classifier_f1 + 3
    elif classifier_f1 >= 70:
        return classifier_f1 + 2
    return classifier_f1

if __name__ == '__main__':
    # Read the data file
    print("Loading data...")
    df = pd.read_csv("data.csv")  # Use 'data.csv' directly
    print("Data loaded successfully.")

    # Split the dataset
    print("Splitting data into training and test sets...")
    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("Data split completed.")

    # Initialize classifier
    print("Initializing the classifier...")
    classifier = Classifier()

    # Preprocess the data
    print("Preprocessing data...")
    X_train = Classifier.preprocess(X_train)
    X_test = Classifier.preprocess(X_test)

    # Train the model
    print("Training the model...")
    classifier.fit(X_train, y_train)

    # Predict on the test set
    print("Making predictions...")
    classifier_predictions = classifier.predict(X_test)

    # Calculate F1 score
    print("Calculating F1 score...")
    classifier_f1 = f1_score(y_test, classifier_predictions, average='weighted')
    print(f"Classifier F1: {classifier_f1:.2f}")

    # Compute final score
    final_score = compute_final_score(classifier_f1)
    print("Final project grade = " + str(final_score))

    # Optional additional information
    # Uncomment if you want more insights
    # classifier_accuracy = accuracy_score(y_test, classifier_predictions)
    # print(f"Classifier Accuracy: {classifier_accuracy:.2f}")
    # print(classification_report(y_test, classifier_predictions))
