# Intelligent Resume Screening System 📄🤖

This project presents an **Intelligent Resume Screening System** that leverages Natural Language Processing (NLP) and Machine Learning (ML) to automate the classification of resumes into predefined job categories. It features an end-to-end pipeline from data preprocessing and model training to a user-friendly web application for real-time screening.

## ✨ Key Features

*   **Automated Resume Ingestion:** Extracts text from uploaded PDF files (using `pdfplumber` with `PyPDF2` as fallback) or direct text input.
*   **Robust Text Preprocessing:** Includes cleaning (URLs, special characters, etc.), lowercasing, stop-word removal (NLTK), and lemmatization (NLTK WordNet).
*   **Optimized Feature Engineering:** Converts resume text into meaningful numerical vectors using TF-IDF with tuned parameters (e.g., `max_features=600`, `min_df=3`, `ngram_range=(1,2)`).
*   **Machine Learning Classification:** Employs a tuned **Logistic Regression** model (`class_weight='balanced'`, L1 penalty) for robust multi-class categorization.
*   **Interactive Web Application:** Built with Streamlit, providing an intuitive UI for:
    *   Selecting a target job category.
    *   Uploading resumes (PDF) or pasting text.
    *   Viewing the model's predicted category and assessing its match with the target job.
*   **Comprehensive Model Evaluation:** Performance detailed with metrics like F1-score (macro and weighted averages, per-class), accuracy, precision, recall, and a confusion matrix.
*   **Insightful Data Visualizations:** Includes plots for category distribution, resume length analysis, word clouds, and per-class F1-scores vs. support.
*   **Reproducible Environment:** Utilizes `venv` for dependency management and provides clear setup instructions.

## 🚀 Project Goal

To develop an automated, data-driven solution for initial resume categorization, addressing the inefficiencies and potential biases in traditional manual screening. This enables recruiters to focus their efforts more effectively on the most promising candidates.

## 🛠️ Technologies & Libraries

*   **Python 3.x**
*   **Core ML & Data Science:**
    *   `scikit-learn`: For TF-IDF, Logistic Regression, model evaluation, `train_test_split`, `GridSearchCV`, `StratifiedKFold`.
    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical operations.
*   **Natural Language Processing (NLP):**
    *   `nltk`: For tokenization, stop-word removal, and WordNet lemmatization.
*   **PDF Processing:**
    *   `pdfplumber`: Primary library for robust PDF text extraction.
    *   `PyPDF2`: Fallback library for PDF text extraction.
*   **Web Application Framework:**
    *   `streamlit`: For building the interactive user interface.
*   **Data Visualization:**
    *   `matplotlib`: For foundational plotting.
    *   `seaborn`: For enhanced statistical visualizations.
    *   `wordcloud`: For generating word cloud images.
*   **Model & Object Persistence:**
    *   `joblib`: For saving and loading trained models, vectorizers, and encoders.
*   **Development Environment:**
    *   `venv`: For Python virtual environment management.

## 📊 Dataset

The system was trained and evaluated on `resume_dataset.csv`, a publicly available dataset containing **169 resumes across 25 distinct job categories**. A key characteristic of this dataset is its **significant class imbalance** (e.g., 'Java Developer' with 14 samples vs. 'PMO' with 3 samples). This imbalance heavily influenced modeling choices (e.g., using `class_weight='balanced'` in Logistic Regression, `StratifiedKFold` for cross-validation) and evaluation strategies (focusing on `f1_macro` score).

## ⚙️ Methodology

1.  **NLTK Setup:** Prioritized a local `nltk_data_local` directory for robust access to essential NLTK resources (WordNet, OMW-1.4, Punkt, Stopwords).
2.  **Data Loading & Exploratory Data Analysis (EDA):** Loaded the dataset and performed initial analysis to understand resume content and category distribution. Visualized class imbalance using bar and pie charts, and analyzed resume lengths pre/post-cleaning.
3.  **Text Preprocessing:** Implemented a custom pipeline to clean resume text by:
    *   Removing URLs, RTs, mentions, hashtags, punctuation, and non-ASCII characters.
    *   Converting text to lowercase.
    *   Tokenizing text into words (NLTK `word_tokenize`).
    *   Filtering out English stop words (NLTK).
    *   Lemmatizing words to their base form (NLTK `WordNetLemmatizer`).
4.  **Feature Engineering (TF-IDF):**
    *   Utilized `TfidfVectorizer` to convert cleaned text into numerical feature vectors.
    *   Parameters were iteratively tuned, with a representative configuration being: `max_features=600`, `min_df=3`, `ngram_range=(1,2)`, `max_df=0.90`, and `sublinear_tf=True`.
5.  **Model Selection & Training:**
    *   Experimented with K-Nearest Neighbors (KNN), Multinomial Naive Bayes, and Logistic Regression.
    *   **Logistic Regression** with `class_weight='balanced'` and L1 penalty demonstrated the most promising balance of performance and interpretability for this dataset.
    *   Hyperparameters (specifically `C` for L1 regularization) were tuned using `GridSearchCV` with `StratifiedKFold` (typically 2-fold due to small class sizes) and `f1_macro` as the scoring metric.
6.  **Model Evaluation:** Assessed the best model using:
    *   Accuracy (training and test).
    *   F1-score (per-class, macro average, weighted average for both training and test).
    *   Detailed classification report (precision, recall, F1-score, support per class).
    *   Confusion matrix.
    *   A custom plot showing per-class F1-scores against their support in the test set.
7.  **Application Development:** A Streamlit web application (`app.py`) was developed to deploy the trained model, allowing users to upload resumes or paste text for interactive category prediction and matching against a selected job role.

## 📈 Performance Insights (Example: Logistic Regression, L1, C=5.0, 600 features, min_df=3)

*(Note: Performance metrics can vary slightly based on exact TF-IDF parameters and CV splits. The following represents a typical good run.)*

*   **Best Cross-Validation (f1_macro) Score:** ~0.70 - 0.74
*   **Training Set F1-Macro Score:** ~0.96 - 0.98 (Indicates some overfitting due to small dataset size and complexity)
*   **Test Set F1-Macro Score:** ~0.72 - 0.74
*   **Test Set Accuracy:** ~76% - 80%

**Key Observation:** While the model achieves reasonable overall F1-macro scores, performance on individual categories with very low sample counts (e.g., "PMO", "Sales", "Testing" often with < 3 samples in the test set) can be highly variable and frequently results in F1-scores of 0.00. This highlights the critical impact of data scarcity and imbalance on model reliability for minority classes. The L1 penalty in Logistic Regression helps by performing implicit feature selection, making the model somewhat more robust against the high dimensionality of text features.

## 🔧 Setup and Installation

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and Activate a Python Virtual Environment:**
    ```bash
    # For Python 3
    python -m venv venv
    ```
    Activate:
    ```bash
    # On Windows (Git Bash or similar)
    source venv/Scripts/activate
    # On Windows (Command Prompt/PowerShell)
    # venv\Scripts\activate.bat  OR  venv\Scripts\Activate.ps1
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Ensure you have a `requirements.txt` file. If not, create one from your working environment: `pip freeze > requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **NLTK Resource Setup (Crucial for First Run):**
    *   Create a folder named `nltk_data_local` in the root of your project directory.
    *   Run the provided `download_nltk_manually.py` script:
        ```bash
        python download_nltk_manually.py
        ```
        This script will attempt to download essential NLTK packages (`wordnet`, `omw-1.4`, `punkt`, `stopwords`) into the `nltk_data_local` folder. NLTK often downloads these as `.zip` files.
    *   **Verification:** Ensure that after running the script, you have subdirectories like `nltk_data_local/corpora/wordnet/`, `nltk_data_local/corpora/omw-1.4/` etc., containing the actual data files (not just the zips, though NLTK can sometimes work with zips directly in its path). If you only see zips, NLTK *should* unzip them on first use by the main `data_analysis.py` script. The `data_analysis.py` script will confirm if it can find these resources.

## 🚀 Running the Project

### 1. Model Training, EDA, and Artifact Generation
This script performs the complete data processing and model training pipeline.
```bash
python data_analysis.py
```
**Output Artifacts from `data_analysis.py`:**
*   `best_LogisticRegression_model.pkl`: The trained classification model.
*   `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer.
*   `label_encoder.pkl`: The fitted label encoder for categories.
*   Various `.png` image files for EDA and evaluation plots (e.g., `resume_category_counts.png`, `confusion_matrix_LogisticRegression.png`).

### 2. Launch the Streamlit Web Application
This script starts the interactive resume screening application. **Ensure `data_analysis.py` has been run successfully at least once to generate the required `.pkl` model artifacts.**
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (usually `http://localhost:8501`) in a web browser.


## 📁 Project File Structure (Illustrative)

Resume-Screening-Project/
├── app.py # Streamlit web application script
├── data_analysis.py # Main script for EDA, preprocessing, training, evaluation
├── download_nltk_manually.py # Helper script for NLTK setup
├── resume_dataset.csv # The input dataset
├── requirements.txt # Python package dependencies
├── nltk_data_local/ # (Created by NLTK setup) Local NLTK data
│ ├── corpora/
│ │ ├── wordnet/ # (Contains data files, not just wordnet.zip)
│ │ ├── omw-1.4/ # (Contains data files, not just omw-1.4.zip)
│ │ └── stopwords/
│ └── tokenizers/
│ └── punkt/
├── best_LogisticRegression_model.pkl # (Generated) Saved trained model
├── tfidf_vectorizer.pkl # (Generated) Saved TF-IDF vectorizer
├── label_encoder.pkl # (Generated) Saved label encoder
├── *.png # (Generated) Various EDA and evaluation plots
└── README.md # This file


## 🎯 Key Challenges & Learnings

*   **Data Limitations:** The primary challenge was the **small dataset size (169 samples)** and **significant class imbalance** across 25 categories. This led to:
    *   Model overfitting (high training scores, lower generalization scores).
    *   Difficulty in reliably classifying underrepresented categories (often resulting in F1-scores of 0.00).
    *   The necessity for 2-fold cross-validation, which is less stable than higher-fold CV.
*   **NLTK Resource Management:** Ensuring consistent and reliable access to NLTK data (WordNet, OMW-1.4 for lemmatization; Punkt for tokenization) required careful local path management.
*   **Iterative Model & Feature Refinement:** The selection of Logistic Regression with L1 penalty and specific TF-IDF parameters (`max_features`, `min_df`) was an iterative process of experimentation to balance model complexity and performance on limited data.
*   **Importance of Robust Evaluation:** This project underscored the need for metrics like F1-macro, weighted F1, and detailed per-class scores (over simple accuracy) to accurately assess model performance on imbalanced datasets.

## 🔮 Future Work & Potential Enhancements

*   **Dataset Augmentation & Expansion:**
    *   Prioritize acquiring a significantly larger, more diverse, and balanced dataset.
    *   Explore text data augmentation techniques (e.g., back-translation, synonym replacement, EDA) cautiously, given the risk of introducing noise with very small original samples.
*   **Advanced Text Representation:**
    *   Experiment with pre-trained word embeddings (Word2Vec, GloVe, FastText).
    *   Utilize contextual embeddings from Transformer models (e.g., Sentence-BERT) for richer semantic features, which may require more data and computational resources.
*   **Alternative Modeling Approaches:**
    *   With more data, explore Deep Learning models (CNNs, RNNs/LSTMs, or fine-tuning pre-trained Transformers).
    *   Investigate ensemble methods (e.g., Random Forest, Gradient Boosting adapted for text).
*   **Explainable AI (XAI):** Integrate techniques like LIME or SHAP to provide insights into why the model makes certain predictions, increasing transparency and trust.
*   **Enhanced Application Functionality:**
    *   Develop functionality to rank resumes against specific job descriptions using semantic similarity (beyond just category matching).
    *   Implement Named Entity Recognition (NER) to extract specific skills, years of experience, and education levels.
*   **Bias Auditing & Mitigation:** Conduct a more formal audit for potential biases in the data and model predictions, and explore mitigation techniques.
*   **OCR Integration:** Add Optical Character Recognition (OCR) capabilities to process image-based PDFs and extract text.