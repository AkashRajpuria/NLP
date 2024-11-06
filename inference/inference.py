from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import training.data_preprocess as data_preprocess

def evaluate_model_performance(model_path):
    """
    Load the trained model and test data, then evaluate the model's performance on the test set.

    Args:
        model_path (str): Path to the saved model file.

    Outputs:
        Prints and saves the accuracy, precision, recall, and F1 score to outputs/metrics.txt.
    """
    try:
        # Load test data
        test_x, test_y = data_preprocess.load_test_data()
        
        # Load the model
        model = joblib.load(model_path)
        
        # Perform predictions and evaluate
        y_pred = model.predict(test_x)
        
        # Calculate metrics
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='binary')
        recall = recall_score(test_y, y_pred, average='binary')
        f1 = f1_score(test_y, y_pred, average='binary')
        
        # Print metrics to console
        print(f"Model accuracy on Test dataset: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        
        # Save metrics to a file
        metrics_path = 'outputs/metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
        
        print(f"Metrics saved to {metrics_path}")

    except FileNotFoundError as e:
        print(f"Error loading model or data: {e}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate_model_performance('outputs/model.pkl')
