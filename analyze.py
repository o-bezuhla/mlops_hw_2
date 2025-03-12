from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


import json

CLASS_FILE = "classes.json"

def load_classes():
    try:
        with open(CLASS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return ["Work", "Sports", "Food"]  # Default classes

def save_classes(classes):
    with open(CLASS_FILE, "w") as f:
        json.dump(classes, f)

EMAIL_CLASSES = load_classes()


def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response

def compute_embeddings():
    classes = load_classes()  # Reload the updated classes
    embeddings = model.encode(classes)
    return zip(classes, embeddings)

def classify_email(text):
    # Encode the input text
    text_embedding = model.encode([text])[0]
    
    # Get embeddings for all classes
    class_embeddings = compute_embeddings()
    
    # Calculate distances and return results
    results = []
    for class_name, class_embedding in class_embeddings:
        # Compute cosine similarity between text and class embedding
        similarity = np.dot(text_embedding, class_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding))
        results.append({
            "class": class_name,
            "similarity": float(similarity)  # Convert tensor to float for JSON serialization
        })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results