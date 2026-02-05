import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('synthetic_social_media_data.csv')

# 2. Data Cleaning
print("Cleaning data...")
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

df['cleaned_text'] = df['Post Content'].apply(clean_text)

# 3. Feature Extraction
print("Extracting features...")
X = df['cleaned_text']
y = df['Sentiment Label']
# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# TF-IDF Vectorization
# Using ngram_range=(1, 3) to capture unigrams, bigrams, and trigrams for better context
print("Extracting features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# 4. Train-Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# 5. Save Processed Data
print("Saving artifacts...")
with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Preprocessing complete. Files saved.")
