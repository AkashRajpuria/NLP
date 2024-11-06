from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

from data_preprocess import load_training_data

def train_and_evaluate_models(model_save_path='outputs'):
    """
    Train and evaluate multiple models on the training dataset and save the best performing model.

    Args:
        model_save_path (str): Directory path where the trained model and metrics will be saved.

    Outputs:
        Saves the best model and training metrics in the specified output directory.
    """
    try:
        # Load training data
        train_x, train_y = load_training_data()

        # Initialize models
        models = {
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Dictionary to store model accuracies
        model_accuracies = {}

        # Train and evaluate each model
        for model_name, model in models.items():
            model.fit(train_x, train_y)
            y_pred = model.predict(train_x)
            accuracy = accuracy_score(train_y, y_pred)
            model_accuracies[model_name] = accuracy
            print(f"{model_name} accuracy on Train dataset: {accuracy}")

        # Select the best model
        best_model_name = max(model_accuracies, key=model_accuracies.get)
        best_model = models[best_model_name]
        best_accuracy = model_accuracies[best_model_name]

        # Save the best model
        os.makedirs(model_save_path, exist_ok=True)
        joblib.dump(best_model, os.path.join(model_save_path, 'model.pkl'))
        
        # Save metrics to a file
        metrics_path = os.path.join(model_save_path, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Accuracy: {best_accuracy}\n")
            for model_name, accuracy in model_accuracies.items():
                f.write(f"{model_name} Accuracy: {accuracy}\n")
        
        print(f"Best model ({best_model_name}) saved with accuracy: {best_accuracy}")
        print(f"Metrics saved to {metrics_path}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    train_and_evaluate_models('outputs')
