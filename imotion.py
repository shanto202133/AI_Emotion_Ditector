import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('punkt')

# Load the dataset (assume a CSV file with 'text' and 'emotion' columns)
data = pd.read_csv('\python\mydata.csv')

# Preprocess the text data
X = data['text']
y = data['emotion']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF Vectorizer and Naive Bayes Classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_val, predictions))

# Test the model with a sample text
sample_text = ["I am feeling so happy today!"]
print("Prediction:", model.predict(sample_text))
