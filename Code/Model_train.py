import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
dataset = pd.read_csv("/Users/digreelal/Docs/Fake Job Posting Project/Dataset/fake_job_postings 2.csv")

# Fill missing values
dataset.fillna('', inplace=True)

# Combine text columns
dataset['text'] = dataset['title'] + " " + dataset['company_profile'] + " " + dataset['description'] + " " + dataset['requirements'] + " " + dataset['benefits']

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

dataset['text'] = dataset['text'].apply(clean_text)

# Features and target
X = dataset['text']
y = dataset['fraudulent']

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_features = tfidf.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_features, y)

# Save model
pickle.dump(model, open("fake_job_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model saved successfully")