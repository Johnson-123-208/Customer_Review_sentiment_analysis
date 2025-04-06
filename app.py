from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('sentiment_lr_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\b(not|no|never|n't)\s+(\w+)", r"\1_\2", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) - {'not', "n't", 'no', 'never'}
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

@app.route('/')

def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])
    pred = model.predict(review_vector)[0]
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = labels.get(pred, 'Unknown')
    return render_template('index.html', review=review, prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
