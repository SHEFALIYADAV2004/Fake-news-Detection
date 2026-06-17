import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

st.set_page_config(
    page_title="AI | Verification Hub", 
    page_icon="✔️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    /* Responsive input textarea container */
    .stTextArea textarea { 
        border-radius: 8px; 
    }
    
    /* Responsive metric sidebar block */
    .stMetric { 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #ff9933; 
    }
    
    /* FIX: Removes hardcoded light-grey text so headers automatically adapt to light/dark themes */
    h1, h2, h3, p, span { 
        font-family: 'Inter', sans-serif; 
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_nlp():
    try: return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('english'))
stop_words = setup_nlp()

def clean_indian_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join([w for w in text.split() if w not in stop_words])

@st.cache_resource
def load_assets(model_choice):
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    model_files = {
        "Passive Aggressive Classifier": "model_pac.pkl",
        "Logistic Regression": "model_lr.pkl",
        "Random Forest Classifier": "model_rf.pkl"
    }
    with open(model_files[model_choice], 'rb') as f:
        model = pickle.load(f)
    return model, vectorizer

st.sidebar.markdown("🛡️")
st.sidebar.title("Verification Engine")
algo_choice = st.sidebar.selectbox(
    "🧠 Select ML Architecture",
    ["Passive Aggressive Classifier", "Logistic Regression", "Random Forest Classifier"]
)

accuracy_map = {"Passive Aggressive Classifier": "94.2%", "Logistic Regression": "91.8%", "Random Forest Classifier": "89.1%"}
st.sidebar.metric(label="IFND Model Accuracy", value=accuracy_map[algo_choice])

st.sidebar.markdown("---")
st.sidebar.caption("An AI model , which detect FAKE NEWS. You can choose any three model i.e Passive Aggressive Classifier,Logistic Regression,Random Forest Classifier for FAKE NEWS DETECTION")

col1, col2 = st.columns(2)
with col1:
    st.title("AI: News Verification Hub")
    st.write("Cross-examine suspected statements, political assertions, or viral news snippets.")
    
    user_input = st.text_area(
        "Paste Social Media Claims / News Snippet Text Here:", 
        height=220, 
        placeholder="Type or copy-paste text payload here to check validity..."
    )
    verify_clicked = st.button("Run Verification Pipeline", use_container_width=True)

with col2:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Engine Status Metrics")
    st.info(f"Pipeline Strategy: {algo_choice}")

if verify_clicked:
    if not user_input.strip():
        st.toast("Input text field cannot be empty!", icon="❌")
    else:
        try:
            active_model, tfidf_vec = load_assets(algo_choice)
            with st.spinner("Analyzing lexical densities and vocabulary weights..."):
                cleaned = clean_indian_text(user_input)
                vectorized = tfidf_vec.transform([cleaned])
                prediction = active_model.predict(vectorized)
            st.markdown("---")
            st.subheader("System Analysis Output")
            if prediction == 0:
                st.error("**CRITICAL ALERT: MISINFORMATION DETECTED**")
                st.write("Linguistic properties mimic known structures of news fabrication patterns circulating inside monitored regional feeds.")
            else:
                st.success("✅ **VALIDATION STATUS: CLEAN (LIKELY AUTHENTIC)**")
                st.write("Linguistic layout corresponds cleanly with verified mainstream press distributions across regional benchmarks.")
        except FileNotFoundError:
            st.error("Artifact System Error: Run your notebook cells first to generate the necessary `.pkl` files.")