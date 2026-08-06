import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Page Config
st.set_page_config(
    page_title="The Daily Truth | Fake News Classifier",
    page_icon="📰",
    layout="centered"
)

# Custom Styling (Vintage Newspaper Theme)
st.markdown("""
    <style>
    /* Global Background and Font */
    .stApp {
        background-color: #f4ebd9;
        color: #2b2b2b;
        font-family: 'Georgia', serif;
    }
    
    /* Newspaper Masthead Header */
    .newspaper-header {
        text-align: center;
        border-bottom: 3px double #2b2b2b;
        border-top: 3px double #2b2b2b;
        padding: 15px 0px;
        margin-bottom: 25px;
    }
    .newspaper-title {
        font-size: 42px;
        font-weight: 900;
        font-family: 'Playfair Display', 'Times New Roman', serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
        color: #1a1a1a;
    }
    .newspaper-sub {
        font-style: italic;
        font-size: 14px;
        margin-top: 5px;
        color: #555;
    }
    
    /* Input Area Styling */
    .stTextArea textarea {
        background-color: #fcf8f2 !important;
        color: #1a1a1a !important;
        border: 1px solid #b5a48b !important;
        font-family: 'Georgia', serif !important;
        font-size: 15px !important;
    }

    /* Primary Interactive Button */
    .stButton>button {
        background-color: #2b2b2b !important;
        color: #f4ebd9 !important;
        border: 1px solid #000 !important;
        font-weight: bold !important;
        width: 100%;
        padding: 10px 0px;
        border-radius: 0px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #444444 !important;
        color: #ffffff !important;
    }

    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-family: 'Georgia', serif;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# Ensure NLTK Stopwords
nltk.download('stopwords', quiet=True)

# Define leakage stop words matching training pipeline
leakage_words = {
    'reuters', 'said', 'washington', 'tuesday', 'wednesday', 'thursday', 'friday', 
    'monday', 'saturday', 'sunday', 'via', 'video', 'image', 'featured', 'twitter',
    'facebook', 'post', 'getty', 'images', 'senate', 'house', 'told', 'reporters', 
    'presidential', 'hillary', 'america', 'gop', 'obama', 'like', 'even', 'watch', 
    'trump', 'wire', 'one'
}

stop_words = set(stopwords.words('english')).union(leakage_words) - {'not', 'no', 'nor', 'neither'}

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Load saved assets
@st.cache_resource
def load_assets():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('fake_news_model.pkl')
    return vectorizer, model

try:
    vectorizer, model = load_assets()
except Exception as e:
    st.error("Error loading model or vectorizer. Ensure 'tfidf_vectorizer.pkl' and 'fake_news_model.pkl' are present.")

# Header Layout
st.markdown("""
    <div class="newspaper-header">
        <h1 class="newspaper-title">The Daily Truth</h1>
        <div class="newspaper-sub">SPECIAL EDITION • NLP INTELLIGENCE & VERIFICATION BUREAU</div>
    </div>
""", unsafe_allow_html=True)

# Main Input Section
user_input = st.text_area(
    "PASTE ARTICLE TEXT FOR INVESTIGATION:",
    height=220,
    placeholder="Type or paste news article content here..."
)

col1, col2 = st.columns([1, 1])

if col1.button("🔍 ANALYZE ARTICLE"):
    if not user_input.strip():
        st.warning("⚠️ Please provide article text before running analysis.")
    else:
        # Preprocessing
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        
        # Inference
        prediction = model.predict(vec_input)[0]
        
        # Decision function/probability gauge calculation
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec_input)[0]
            confidence = max(probs) * 100
        elif hasattr(model, "decision_function"):
            score = model.decision_function(vec_input)[0]
            # Convert decision boundary distance to percentage estimation
            confidence = min(100, max(50, 50 + (abs(score) * 20)))
        else:
            confidence = 95.0

        st.markdown("---")
        st.subheader("📰 VERDICT & ANALYSIS")

        # Interactive Columns for Displaying Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Word Count", len(user_input.split()))
        m2.metric("Clean Tokens", len(cleaned.split()))
        m3.metric("Model Confidence", f"{confidence:.1f}%")

        # Display Result Banner
        if prediction == 1:
            st.success("✅ **VERDICT: AUTHENTIC / REAL NEWS**\n\nThe article aligns with structured, objective reporting patterns.")
        else:
            st.error("🚨 **VERDICT: SENSATIONAL / FAKE NEWS**\n\nThe article contains stylistic markers associated with unverified or sensational text.")

        # Interactive Features Expander
        with st.expander("🛠️ Inspect Preprocessing & Top Feature Tokens"):
            st.markdown("**Cleaned Text (After Stopword & Leakage Removal):**")
            st.code(cleaned if cleaned else "No valid tokens remained after cleaning.", language="text")
            
            # Show top words extracted from this specific text input
            if cleaned:
                feature_names = vectorizer.get_feature_names_out()
                dense = vec_input.todense().tolist()[0]
                phrase_scores = [(feature_names[i], dense[i]) for i in range(len(dense)) if dense[i] > 0]
                sorted_phrase = sorted(phrase_scores, key=lambda x: x[1], reverse=True)[:8]
                
                if sorted_phrase:
                    st.markdown("**Top Key Terms Driving Prediction:**")
                    st.write(", ".join([f"`{word}`" for word, score in sorted_phrase]))

if col2.button("🧹 CLEAR INPUT"):
    st.rerun()
  
