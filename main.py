import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

class Classifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', gamma='scale', random_state=42)
        self.scaler = StandardScaler()

    @staticmethod
    def preprocess(X):
        """
        Preprocess the input features by scaling them.
        """
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def fit(self, X_train, y_train):
        """
        Train the SVM model using the provided training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict labels for the test data.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using F1 score and print the confusion matrix.
        """
        y_pred = self.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        return f1

# Example usage
if __name__ == "__main__":
    # Load data
    data_path = 'data.csv'  # Ensure this is the correct path
    data = pd.read_csv(data_path)
    X = data.drop('label', axis=1)
    y = data['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create and train classifier
    classifier = Classifier()
    X_train_scaled = Classifier.preprocess(X_train)
    classifier.fit(X_train_scaled, y_train)

    # Evaluate model
    X_test_scaled = Classifier.preprocess(X_test)
    f1 = classifier.evaluate(X_test_scaled, y_test)
    print(f"F1 Score: {f1:.2f}")
