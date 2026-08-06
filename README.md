# 📰 Veritas: Leakage-Free Fake News NLP Classifier
An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline that classifies news articles as **Real** or **Fake**. Built with custom text-cleaning pipelines to prevent dataset leakage, benchmarked across multiple vectorizers and classifiers, and deployed live as an interactive vintage-styled Streamlit web application.
🚀 **Live Web Application:** [https://fake-news-detector-nlp-practice.streamlit.app/](https://fake-news-detector-nlp-practice.streamlit.app/)
---
## 📌 Project Overview
Standard news classification models often achieve artificial ~99% accuracy by relying on **dataset leakage artifacts**—such as news agency attributions (`reuters`), temporal anchors (`tuesday`, `wednesday`), or specific political figure names. 
This project specifically isolates and removes structural shortcuts and topic-bias tokens to force the underlying linear models to evaluate **genuine stylistic, linguistic, and structural patterns**. 
* **Final Generalized Accuracy:** ~95.17% (Linear SVC + TF-IDF)
* **Dataset Artifact Mitigation:** Multi-stage custom stop-word expansion to strip leakage tokens.
* **Interface:** Styled Streamlit dashboard mimicking a classic vintage newspaper front page.
---
## 🛠️ Tech Stack
* **Core Programming**: Python
* **Machine Learning & NLP**: Scikit-Learn (v1.6.1), NLTK, XGBoost
* **Data Manipulation & Analysis**: Pandas, NumPy
* **Model Serialization & Deployment**: Joblib, Streamlit Community Cloud
* **Frontend UI Styling**: Custom CSS / Streamlit Markdown Injection
---
## 📊 Model Benchmarking & Methodology
The pipeline evaluates combination pairings of feature extraction techniques (**Bag-of-Words** vs. **TF-IDF**) against multiple supervised learning classifiers:
1. **Text Preprocessing**: Lowercasing, punctuation regex filtering, standard NLTK stop-word removal, and custom domain-specific leakage token removal.
2. **Feature Vectorization**: 
   * `CountVectorizer` (`ngram_range=(1,1)`, `max_features=5000`, `min_df=5`)
   * `TfidfVectorizer` (`ngram_range=(1,1)`, `max_features=5000`, `min_df=5`)
3. **Model Evaluation**:
   * Logistic Regression
   * Support Vector Classifier (Linear Kernel)
   * Multinomial Naive Bayes
   * XGBoost Classifier
### Performance Summary

| Feature Extraction | Model | Test Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **TF-IDF** | **SVC (Linear)** | **95.17%** | **Winner** |
| TF-IDF | XGBClassifier | 93.83% | Benchmark |
| TF-IDF | Logistic Regression | 93.42% | Benchmark |
| Bag of Words | XGBClassifier | 93.75% | Benchmark |
| Bag of Words | Logistic Regression | 93.17% | Benchmark |
| Bag of Words | SVC (Linear) | 90.92% | Benchmark |

---
## 📁 Repository Structure
```text
├── app.py                  # Streamlit application with vintage newspaper UI
├── tfidf_vectorizer.pkl    # Exported TF-IDF Vectorizer
├── fake_news_model.pkl     # Exported trained Linear SVC model
├── requirements.txt        # Production dependencies (scikit-learn==1.6.1)
└── README.md               # Project documentation
```

## 📞 Contact Information:-
* **Email:-**[englandengland271@gmail.com]
* **Linkedin:-**[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share_via&utm_content=profile&utm_medium=member_android]
* **GitHub:-**[https://github.com/M-Nafay-Ali]
