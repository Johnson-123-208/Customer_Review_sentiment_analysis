# 📦 Amazon Product Reviews Sentiment Analysis

This project performs sentiment analysis on Amazon product reviews using both traditional machine learning models (Logistic Regression, Linear SVM) and a fine-tuned BERT model (optional). The goal is to classify reviews into three sentiment categories: **Positive**, **Neutral**, and **Negative**.

---

## 🧠 Project Highlights

- 📊 **Dataset**: [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
- 🧹 **Text Preprocessing**: Lowercasing, punctuation removal, stopword removal, lemmatization
- 🔍 **Feature Extraction**: TF-IDF Vectorization
- ⚙️ **Models Used**:
  - Logistic Regression
  - Linear Support Vector Machine (SVM)
  - BERT (`bert-base-uncased`) *(optional)*
- 📈 **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report
- 📉 **Visualizations**: Most common words in each sentiment category
- 💾 **Model Saving**: Joblib for traditional ML models, PyTorch for BERT

---

## 🗂️ Project Structure

```
├── sentiment_analysis.py            # Main script
├── tfidf_vectorizer.joblib          # Saved TF-IDF vectorizer
├── sentiment_lr_model.joblib        # Logistic Regression model
├── sentiment_svm_model.joblib       # Linear SVM model
├── positive_common_words.png        # Visualization for positive reviews
├── neutral_common_words.png         # Visualization for neutral reviews
├── negative_common_words.png        # Visualization for negative reviews
├── README.md                        # Project description
```

---

## 📥 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK resources**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

4. **Run the analysis**
```bash
python sentiment_analysis.py
```

---

## 🔎 Sentiment Classes

| Rating | Sentiment |
|--------|-----------|
| 1-2    | Negative  |
| 3      | Neutral   |
| 4-5    | Positive  |

---

## 📊 Model Evaluation (Example)

### ✅ Logistic Regression
- Accuracy: ~85%
- Confusion Matrix & Classification Report shown in console

### ✅ Linear SVM
- Accuracy: ~86%
- Slightly better performance than Logistic Regression


---

## 💬 Example Output

```text
Review: This product is amazing! It works perfectly and exceeded all my expectations.
Predicted sentiment: positive

Review: It's okay but not worth the price. There are better alternatives available.
Predicted sentiment: neutral

Review: Terrible product. Broke after two days of use. Don't waste your money.
Predicted sentiment: negative
```

---

## 📈 Visualizations

Bar plots for the most common words in each sentiment class are saved as PNG images:
- `positive_common_words.png`
- `neutral_common_words.png`
- `negative_common_words.png`

These visualizations help understand what words are dominant in each review type.

---

## 🚀 Future Improvements

- Deploy as a web app using Flask or Streamlit
- Add interactive review prediction UI
- Train BERT on larger subsets or full dataset using GPU
- Use cross-validation and hyperparameter tuning

---

## 📚 References

- Dataset: [Kaggle - Amazon Product Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- Transformers: [Hugging Face](https://huggingface.co/transformers/)

---

## 🧑‍💻 Author

**Your Name**  
GitHub: ([Johnson](https://github.com/Johnson-123-208))  
LinkedIn: ([Johnson](https://www.linkedin.com/in/johnson-obhalloju-8747a6320/))

---
