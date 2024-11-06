
# Sentiment Analysis Project

This project implements a binary sentiment classification on movie reviews, showcasing a complete Data Science and Machine Learning Engineering pipeline. The project includes exploratory data analysis, text preprocessing, model training, evaluation, and deployment in a Docker environment.

## Project Structure

The project is organized into the following directories:

- `data/`: Contains the training and inference datasets.
- `training/`: Holds the training script and Dockerfile for model training.
- `inference/`: Contains the inference script and Dockerfile for making predictions.
- `.gitignore`: Configures files to be ignored in version control (e.g., temporary files and model artifacts).
- `requirements.txt`: Lists all dependencies required to run the project.
- `ds_part.ipynb`: Jupyter notebook covering the Data Science workflow, including data analysis, preprocessing, and modeling.

## How to Run

### Data Science Part

1. Open and run `ds_part.ipynb` for data exploration, preprocessing, model training, and evaluation.
   - This notebook includes all the steps needed to preprocess data, explore insights, build models, and select the best-performing model.

### Machine Learning Engineering Part

The MLE part includes Dockerized scripts for both training and inference, allowing for seamless deployment and reproducibility.

1. **Model Training**:
   - Build the Docker image:
     ```bash
     docker build -t sentiment_train -f training/Dockerfile .
     ```
   - Run the Docker container for training:
     ```bash
     docker run --name train-container sentiment_train
     ```
   - After training, copy the model to your local machine:
     ```bash
     docker cp train-container:/app/outputs/model.pkl ./outputs/model.pkl
     ```

2. **Model Inference**:
   - Build the Docker image for inference:
     ```bash
     docker build -t sentiment_inference -f inference/Dockerfile .
     ```
   - Run the Docker container to make predictions on new data:
     ```bash
     docker run --name test-container sentiment_inference
     ```


