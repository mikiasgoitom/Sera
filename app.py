import streamlit as st
import pandas as pd
import re
import string
import nltk
import PyPDF2
import pdfplumber
import joblib
import os

st.set_page_config(page_title="Resume Screening App", layout="wide")

def setup_nltk_local_path_priority_streamlit():
    local_nltk_path = os.path.join(os.getcwd(), 'nltk_data_local')
    
    if not os.path.exists(local_nltk_path):
        st.error(f"CRITICAL ERROR: Local NLTK data directory '{local_nltk_path}' not found. App cannot function.")
        st.error("Please ensure you have run the `data_analysis.py` script or manually created and populated this folder.")
        return False

    if local_nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_path)
        print(f"Streamlit App: Prioritized local NLTK data path: {local_nltk_path}")
    else:
        print(f"Streamlit App: Local NLTK data path already in search paths: {local_nltk_path}")
    
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        print("Streamlit App: Essential NLTK resources (wordnet, omw-1.4, stopwords, punkt) found.")
        return True
    except LookupError as e:
        resource_name = str(e).split('/')[-1].replace("'", "").split('.')[0]
        st.error(f"Streamlit App: CRITICAL NLTK resource '{resource_name}' missing after path setup: {e}. Lemmatization will fail.")
        st.error(f"Please ensure '{resource_name}' is correctly unzipped in the directory: {os.path.join(local_nltk_path, 'corpora')}")
        return False

if '_streamlit_nltk_setup_complete' not in st.session_state:
    print("Streamlit App: Initializing NLTK path setup...")
    if setup_nltk_local_path_priority_streamlit():
        st.session_state['_streamlit_nltk_setup_complete'] = True
    else:
        st.session_state['_streamlit_nltk_setup_complete'] = False 
    print("Streamlit App: NLTK path setup process complete.")

if st.session_state.get('_streamlit_nltk_setup_complete', False):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    stop_words_english = set(stopwords.words('english') + ['``', "''"])
else:
    print("Streamlit App: NLTK setup failed, using dummy NLTK components.")
    stopwords = nltk.corpus.LazyCorpusLoader('stopwords', nltk.corpus.reader.WordListCorpusReader, r'.*\.txt')
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'): return word 
    lemmatizer = DummyLemmatizer()
    stop_words_english = set()

MODEL_PATH = 'best_LogisticRegression_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
LABELENCODER_PATH = 'label_encoder.pkl'

def clean_resume_text(resumeText):
    if not st.session_state.get('_streamlit_nltk_setup_complete', False):
        st.warning("NLTK not initialized properly, text cleaning might be incomplete.")
    
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)
    resumeText = re.sub(r'RT|cc', ' ', resumeText)
    resumeText = re.sub(r'#\S+', '', resumeText)
    resumeText = re.sub(r'@\S+', ' ', resumeText)
    resumeText = resumeText.lower()
    resumeText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)
    
    if st.session_state.get('_streamlit_nltk_setup_complete', False):
        words = nltk.word_tokenize(resumeText)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_english and len(word) > 2]
        resumeText = " ".join(words)
    return resumeText.strip()

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        if text.strip(): 
            return text
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}. Trying PyPDF2...")
    try:
        pdf_file.seek(0) 
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception as e:
        st.error(f"PyPDF2 also failed to read PDF: {e}")
        return None 
    return text

@st.cache_resource 
def load_model_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABELENCODER_PATH)
        if model and vectorizer and label_encoder:
            st.success("Model artifacts loaded successfully!")
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        st.error(
            f"Model artifacts not found. Expected '{MODEL_PATH}', '{VECTORIZER_PATH}', and '{LABELENCODER_PATH}'."
            "Please run the training script (`data_analysis.py`) first."
        )
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

def predict_resume_category(resume_text, model, vectorizer, label_encoder):
    if not resume_text or not resume_text.strip():
        return "Error: Empty resume text provided."
    
    cleaned_text = clean_resume_text(resume_text)
    if not cleaned_text.strip():
        return "Error: Resume text became empty after cleaning."
        
    try:
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction_encoded = model.predict(vectorized_text)
        predicted_category = label_encoder.inverse_transform(prediction_encoded)
        return predicted_category[0]
    except Exception as e:
        return f"Error during prediction: {e}"

st.title("📄 Resume Screening Application")
st.markdown("Upload a candidate's resume (PDF or text) to predict its category and match against a job posting.")

clf, word_vectorizer, le = load_model_artifacts()

if clf and word_vectorizer and le and st.session_state.get('_streamlit_nltk_setup_complete', False):
    st.sidebar.header("🎯 Job Posting Details")
    available_categories = sorted(list(le.classes_))
    job_category = st.sidebar.selectbox("Select the Job Category", options=available_categories)
    st.sidebar.write("Selected Job Category:", f"**{job_category}**")

    st.header("👤 Candidate Resume Input")
    upload_option = st.radio("How would you like to provide the resume?",
                             ("Upload PDF File", "Enter Text Manually"), horizontal=True)
    candidate_resume_text = None
    if upload_option == "Upload PDF File":
        uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_pdf is not None:
            with st.spinner("Extracting text from PDF..."):
                candidate_resume_text = extract_text_from_pdf(uploaded_pdf)
            if candidate_resume_text:
                st.subheader("Extracted Resume Text (First 2000 characters):")
                st.text_area("Resume Content", candidate_resume_text[:2000] + "...", height=500, disabled=True)
            else:
                st.error("Could not extract text from the PDF.")
    else: 
        candidate_resume_text_input = st.text_area("Paste Candidate Resume Text Here:", height=250)
        if candidate_resume_text_input:
            candidate_resume_text = candidate_resume_text_input

    if candidate_resume_text:
        if st.button("Screen Candidate's Resume", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume..."):
                predicted_cat = predict_resume_category(candidate_resume_text, clf, word_vectorizer, le)
            st.subheader("Screening Result:")
            if "Error:" in str(predicted_cat):
                st.error(predicted_cat)
            else:
                st.write(f"**Predicted Resume Category:** `{predicted_cat}`")
                if predicted_cat == job_category:
                    st.success("✅ Candidate's resume category **MATCHES** the selected job posting!")
                else:
                    st.warning(f"⚠️ Candidate's resume category **DOES NOT MATCH** the selected job posting. (Expected: `{job_category}`)")
    elif st.button("Screen Candidate's Resume", use_container_width=True, disabled=True):
        pass

    st.markdown("---")
    st.info(
        "**Note:** This application uses a Machine Learning model to categorize resumes. "
        "The accuracy depends on the training data and model complexity."
    )
elif not (clf and word_vectorizer and le):
    st.error("Application could not start: Model or associated files are missing or failed to load. Please ensure the training script (`data_analysis.py`) has been run successfully and generated the .pkl files.")
elif not st.session_state.get('_streamlit_nltk_setup_complete', False):
    st.error("Application could not start: Essential NLTK resources (like WordNet for lemmatization) could not be initialized. Please check console logs and ensure NLTK data is correctly placed in 'nltk_data_local'.")