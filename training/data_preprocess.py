import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load GloVe embeddings
def load_glove_embeddings(filepath='data/glove.6B.50d.txt'):
    """
    Load GloVe word embeddings from a specified file.

    Args:
        filepath (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary mapping words to embedding vectors.
    """
    embeddings_index = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError:
                    print("Skipping line due to invalid format")
    except FileNotFoundError:
        print(f"Embedding file not found at {filepath}")
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

embeddings_index = load_glove_embeddings()

def get_average_embedding(tokens, embedding_dim=50):
    """
    Compute the average embedding for a list of tokens.

    Args:
        tokens (list of str): List of tokens from the preprocessed text.
        embedding_dim (int): Dimension of the embeddings.

    Returns:
        np.array: Average embedding vector of specified dimension.
    """
    valid_embeddings = [embeddings_index[word] for word in tokens if word in embeddings_index]
    if not valid_embeddings:
        return np.zeros(embedding_dim)
    return np.mean(valid_embeddings, axis=0)

def preprocess_text(text):
    """
    Preprocess the input text by cleaning, tokenizing, removing stop words, and lemmatizing.

    Args:
        text (str): The input text.

    Returns:
        list: List of processed tokens.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def preprocess_dataframe(df):
    """
    Preprocesses an input DataFrame containing reviews by tokenizing, embedding, and labeling.

    Args:
        df (pd.DataFrame): DataFrame with columns 'review' and 'sentiment'.

    Returns:
        tuple: Feature matrix (train_x) and label array (train_y).
    """
    # Process each review
    df['tokens'] = df['review'].apply(preprocess_text)
    df['embedding'] = df['tokens'].apply(get_average_embedding)
    train_x = np.vstack(df['embedding'].values)
    train_y = df['sentiment'].values
    return train_x, train_y

def load_training_data(filepath="data/train.csv"):
    """
    Load and preprocess training data.

    Args:
        filepath (str): Path to the training CSV file.

    Returns:
        tuple: Preprocessed feature matrix and labels.
    """
    df = pd.read_csv(filepath)
    return preprocess_dataframe(df)

def load_testing_data(filepath="data/test.csv"):
    """
    Load and preprocess test data.

    Args:
        filepath (str): Path to the test CSV file.

    Returns:
        tuple: Preprocessed feature matrix and labels.
    """
    df = pd.read_csv(filepath)
    return preprocess_dataframe(df)
