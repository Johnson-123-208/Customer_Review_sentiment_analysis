import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
import kagglehub
import os

path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
csv_path = os.path.join(path, "Reviews.csv")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna(subset=['ProfileName', 'Summary'])

df['sentiment'] = df['Score'].apply(lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive'))

print("\nSentiment distribution:")
print(df['sentiment'].value_counts())
print(df['sentiment'].value_counts(normalize=True))

def preprocess_text(text):
    # Converting to lowercase
    text = text.lower()
    
    # Removing special characters, punctuation, and numbers
    text = re.sub(r"\b(not|no|never|n't)\s+(\w+)", r"\1_\2", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenizing the text
    tokens = word_tokenize(text)
    
    # Removing stopwords
    stop_words = set(stopwords.words('english')) - {'not', "n't", 'no', 'never'}
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Joining tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

df['processed_text'] = df['Text'].apply(preprocess_text)

sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_code'] = df['sentiment'].map(sentiment_map)

#Splitting Data into test and train
X = df['processed_text']
y = df['sentiment_code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)

#Evaluating the Model
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['negative', 'neutral', 'positive']))

#Training Linear SVM model
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train_tfidf, y_train)

y_pred_svm = svm_model.predict(X_test_tfidf)
#Evaluating the model
print("\nLinear SVM Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['negative', 'neutral', 'positive']))

#Saving Vocabulary and models
import joblib
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(lr_model, 'sentiment_lr_model.joblib')
joblib.dump(svm_model, 'sentiment_svm_model.joblib')

# Function to analyze sentiment of new reviews
def analyze_sentiment(review_text, model=lr_model):
    """
    Analyze the sentiment of a product review
    
    Parameters:
    review_text (str): The product review text
    model: The sentiment analysis model to use
    
    Returns:
    str: Sentiment prediction ('negative', 'neutral', or 'positive')
    """
    # Preprocessing the review
    processed_review = preprocess_text(review_text)
    
    # Converting to TF-IDF features
    review_tfidf = tfidf_vectorizer.transform([processed_review])
    
    # Predicting sentiment
    sentiment_code = model.predict(review_tfidf)[0]
    
    # Mapping back to sentiment label
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_labels[sentiment_code]
    
    return sentiment


print("\nExample of sentiment analysis for new reviews:")
example_reviews = [
    "This product is amazing! It works perfectly and exceeded all my expectations.",
    "It's okay but not worth the price. There are better alternatives available.",
    "Terrible product. Broke after two days of use. Don't waste your money."
]


for review in example_reviews:
    sentiment = analyze_sentiment(review)
    print(f"\nReview: {review}")
    print(f"Predicted sentiment: {sentiment}")

# Plotting the most common words for each sentiment
def plot_most_common_words(df, sentiment, top_n=20):
    # Getting reviews for the specified sentiment
    sentiment_reviews = df[df['sentiment'] == sentiment]['processed_text']
    
    # Combining all reviews into a single string
    all_words = ' '.join(sentiment_reviews).split()
    
    # Counting word frequencies
    word_freq = pd.Series(all_words).value_counts().head(top_n)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x=word_freq.values, y=word_freq.index)
    plt.title(f'Top {top_n} Most Common Words in {sentiment.capitalize()} Reviews')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.savefig(f'{sentiment}_common_words.png')
    plt.close()



plot_most_common_words(df, 'positive')
plot_most_common_words(df, 'neutral')
plot_most_common_words(df, 'negative')