import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import string
import nltk
import joblib
import os

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def setup_nltk_local_path_priority():
    local_nltk_path = os.path.join(os.getcwd(), 'nltk_data_local')
    
    if not os.path.exists(local_nltk_path):
        print(f"INFO: Local NLTK path '{local_nltk_path}' does not exist. NLTK will use default paths.")
        print("If lemmatization fails, consider creating this folder and manually downloading 'wordnet' and 'omw-1.4' into it using a separate script (e.g., download_nltk_manually.py).")
        return False

    if local_nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_path)
        print(f"Prioritized local NLTK data path: {local_nltk_path}")
    else:
        print(f"Local NLTK data path already in search paths (and prioritized): {local_nltk_path}")
    
    try:
        nltk.data.find('corpora/wordnet.zip') 
        nltk.data.find('corpora/omw-1.4.zip')
        print("Essential NLTK resources for lemmatization (wordnet, omw-1.4) appear to be findable.")
        return True
    except LookupError as e:
        print(f"WARNING: Cannot find an NLTK resource needed for lemmatization: {e}")
        print(f"Ensure 'wordnet.zip' and 'omw-1.4.zip' (or their unzipped contents) are in '{local_nltk_path}/corpora/' or a default NLTK path.")
        return False

print("Setting up NLTK resource paths...")
NLTK_SETUP_SUCCESS = setup_nltk_local_path_priority()
if not NLTK_SETUP_SUCCESS:
    print("NLTK local path setup might be incomplete. Lemmatization could be affected if resources are not in default NLTK locations.")
print("NLTK resource path setup attempt complete.\n")


TFIDF_MAX_FEATURES = 600
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.90       
SELECTED_MODEL = 'LogisticRegression'
TEST_SET_SIZE = 0.20      
RANDOM_STATE = 42         

print("Loading dataset...")
try:
    resumeDataSet = pd.read_csv('resume_dataset.csv', encoding='utf-8')
except FileNotFoundError:
    print("ERROR: 'resume_dataset.csv' not found. Please place it in the script's directory.")
    exit()
resumeDataSet['cleaned_resume'] = ''
print(f"Dataset loaded successfully. Shape: {resumeDataSet.shape}")


print("\n--- Starting Exploratory Data Analysis ---")

category_counts = resumeDataSet['Category'].value_counts()
print("\nCategory counts (number of resumes per category):\n", category_counts)

min_class_count_overall = category_counts.min()
smallest_class_name = category_counts.idxmin()
tentative_cv_folds = 3
if np.floor(min_class_count_overall * (1 - TEST_SET_SIZE)) < tentative_cv_folds :
    cv_folds = 2
    print(f"INFO: Smallest class '{smallest_class_name}' has {min_class_count_overall} samples. With test_size={TEST_SET_SIZE}, training set for this class will have < {tentative_cv_folds} samples. Using {cv_folds}-fold CV.")
else:
    cv_folds = tentative_cv_folds
    print(f"INFO: Smallest class '{smallest_class_name}' has {min_class_count_overall} samples. Using {cv_folds}-fold CV.")

def save_and_show_plot(filename, title="Plot"):
    plt.tight_layout()
    plt.savefig(filename); print(f"Saved {filename}")
    plt.close()

print("\nPlotting initial EDA graphs...")
plt.figure(figsize=(12, 10))
sns.countplot(y="Category", data=resumeDataSet, order=category_counts.index, palette="viridis")
plt.title("Resume Category Counts (Class Imbalance)", fontsize=16)
plt.xlabel("Number of Resumes", fontsize=14); plt.ylabel("Job Category", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=10)
save_and_show_plot("resume_category_counts.png", "Category Counts")

plt.figure(figsize=(12, 12))
cmap = plt.get_cmap('coolwarm'); colors = [cmap(i) for i in np.linspace(0, 1, len(category_counts))]
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, pctdistance=0.85)
plt.axis('equal'); plt.title('CATEGORY DISTRIBUTION (Illustrating Imbalance)', fontsize=18, pad=20)
save_and_show_plot("resume_category_distribution_pie.png", "Category Distribution")


print("\n--- Starting Text Preprocessing ---")
print("Initializing lemmatizer and stop words...")
lemmatizer = WordNetLemmatizer()
stop_words_english = set(stopwords.words('english') + ['``', "''"])

def cleanResume(resumeText, idx="Unknown"):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)
    resumeText = re.sub(r'RT|cc', ' ', resumeText)
    resumeText = re.sub(r'#\S+', '', resumeText)
    resumeText = re.sub(r'@\S+', ' ', resumeText)
    resumeText = resumeText.lower()
    resumeText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)
    try:
        words = nltk.word_tokenize(resumeText)
        lemmatized_words = []
        for word_idx, word in enumerate(words):
            if word not in stop_words_english and len(word) > 2:
                try:
                    lemmatized_word = lemmatizer.lemmatize(word)
                    lemmatized_words.append(lemmatized_word)
                except Exception: 
                    if idx == 0 and word_idx < 5: 
                        print(f"WARNING: Lemmatization failed for a word in resume {idx}. Keeping original word. Ensure NLTK 'wordnet' & 'omw-1.4' are correctly set up.")
                    lemmatized_words.append(word)
        words = lemmatized_words
    except Exception: 
        if idx == 0:
            print(f"WARNING: Tokenization failed for resume {idx}. Using simple split. Ensure NLTK 'punkt' is correctly set up.")
        words = [word for word in resumeText.split() if word not in stop_words_english and len(word) > 2]
    return " ".join(words).strip()

print("Cleaning resumes (this may take a moment)...")
resumeDataSet['cleaned_resume'] = [cleanResume(text, idx) for idx, text in enumerate(resumeDataSet['Resume'])]

print("\nPlotting text-based EDA graphs (after cleaning)...")
resumeDataSet['cleaned_resume_word_count'] = resumeDataSet['cleaned_resume'].apply(lambda x: len(x.split()))
plt.figure(figsize=(12, 6))
sns.histplot(resumeDataSet['cleaned_resume_word_count'], bins=30, kde=True)
plt.title('Distribution of Resume Lengths (Word Count after Cleaning)', fontsize=16)
plt.xlabel('Word Count', fontsize=14); plt.ylabel('Frequency', fontsize=14)
save_and_show_plot("resume_length_distribution.png", "Resume Length Distribution")

plt.figure(figsize=(12, 10))
sns.boxplot(x='cleaned_resume_word_count', y='Category', data=resumeDataSet, order=category_counts.index, palette="coolwarm")
plt.title('Resume Lengths (Word Count) by Category', fontsize=16)
plt.xlabel('Word Count', fontsize=14); plt.ylabel('Job Category', fontsize=14)
plt.xticks(fontsize=10); plt.yticks(fontsize=10)
save_and_show_plot("resume_length_by_category_boxplot.png", "Resume Length by Category")

print("\nGenerating word cloud...")
all_cleaned_text = " ".join(resume for resume in resumeDataSet['cleaned_resume'] if pd.notnull(resume) and resume.strip())
if all_cleaned_text:
    wc = WordCloud(width=1200, height=600, background_color='white', stopwords=stop_words_english, collocations=False).generate(all_cleaned_text)
    plt.figure(figsize=(15, 7.5)); plt.imshow(wc, interpolation='bilinear'); plt.axis("off")
    plt.title("Word Cloud of Cleaned Resume Texts", fontsize=16)
    save_and_show_plot("resume_wordcloud.png", "Word Cloud")
else:
    print("Not enough content for word cloud after cleaning.")


print("\n--- Preparing Labels and Features ---")
print("Encoding labels...")
le = LabelEncoder()
resumeDataSet['Category_encoded'] = le.fit_transform(resumeDataSet['Category'])
joblib.dump(le, 'label_encoder.pkl')
print("Label Encoder saved. Encoded classes:", le.classes_)

print("\nExtracting features using TF-IDF...")
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category_encoded'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True, stop_words='english', max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2), min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF
)
word_vectorizer.fit(requiredText); WordFeatures = word_vectorizer.transform(requiredText)
joblib.dump(word_vectorizer, 'tfidf_vectorizer.pkl')
print(f"TF-IDF Vectorizer created and saved. Feature matrix shape: {WordFeatures.shape}")


print(f"\nSplitting data into training and testing sets (Test size: {TEST_SET_SIZE}, CV folds: {cv_folds})...")
X_train, X_test, y_train, y_test = train_test_split(
    WordFeatures, requiredTarget, test_size=TEST_SET_SIZE,
    random_state=RANDOM_STATE, stratify=requiredTarget
)
print(f"Training set shape: Features {X_train.shape}, Labels {y_train.shape}")
print(f"Test set shape: Features {X_test.shape}, Labels {y_test.shape}")

train_class_counts = pd.Series(y_train).value_counts()
min_class_in_train = train_class_counts.min()
if cv_folds > min_class_in_train and min_class_in_train >=2 :
    print(f"WARNING: Smallest class in training set has {min_class_in_train} samples. Reducing cv_folds from {cv_folds} to {min_class_in_train} for GridSearchCV.")
    cv_folds = min_class_in_train
elif min_class_in_train < 2:
    print(f"ERROR: Smallest class in training set has {min_class_in_train} samples. Cannot perform CV with < 2 samples. Exiting.")
    exit()


print(f"\n--- Model Training and Hyperparameter Tuning: {SELECTED_MODEL} ---")
model_to_tune = None; param_grid = {}
stratified_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

if SELECTED_MODEL == 'LogisticRegression':
    model_to_tune = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE,
                                     max_iter=500, multi_class='ovr',
                                     class_weight='balanced',
                                     penalty='l1')
    param_grid = {
        'C': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0],
    }
elif SELECTED_MODEL == 'NaiveBayes':
    model_to_tune = MultinomialNB()
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
else:
    print(f"ERROR: Model '{SELECTED_MODEL}' is not configured. Exiting.")
    exit()

num_param_values = [len(v) for v in param_grid.values()]
num_candidates = np.prod(num_param_values) if num_param_values else 0
total_fits = num_candidates * cv_folds
print(f"GridSearchCV: Fitting {cv_folds} folds for each of {num_candidates} candidates, totalling {total_fits} fits...")

grid_search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid,
                           cv=stratified_kfold, scoring='f1_macro',
                           verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train) 

best_clf = grid_search.best_estimator_
print("\nBest hyperparameters found by GridSearchCV:", grid_search.best_params_)
print(f"Best cross-validation (f1_macro) score: {grid_search.best_score_:.4f}")
final_model_name = f'best_{SELECTED_MODEL}_model.pkl'
joblib.dump(best_clf, final_model_name); print(f"Best {SELECTED_MODEL} model saved as '{final_model_name}'.")


print("\n--- Model Evaluation ---")
y_train_pred = best_clf.predict(X_train) 
train_f1_macro = metrics.f1_score(y_train, y_train_pred, average='macro', zero_division=0)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nPerformance on Training Set:\n  F1 Macro Score: {train_f1_macro:.4f}\n  Accuracy: {train_accuracy:.4f}")

prediction = best_clf.predict(X_test)
test_f1_macro = metrics.f1_score(y_test, prediction, average='macro', zero_division=0)
test_accuracy = accuracy_score(y_test, prediction)
print(f"\nPerformance on Test Set:\n  F1 Macro Score: {test_f1_macro:.4f}\n  Accuracy: {test_accuracy:.4f}")

print(f"\nClassification Report for {SELECTED_MODEL} on Test Set:\n")
report_dict = classification_report(y_test, prediction, target_names=le.classes_, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print(report_df.to_string(float_format="%.4f"))

macro_avg_f1_report = report_df.loc['macro avg', 'f1-score']
weighted_avg_f1_report = report_df.loc['weighted avg', 'f1-score']
accuracy_from_report = report_df.loc['accuracy', 'support']
print(f"\nSummary from Classification Report (Test Set):")
print(f"  Macro Avg F1-Score: {macro_avg_f1_report:.4f}")
print(f"  Weighted Avg F1-Score: {weighted_avg_f1_report:.4f}")
print(f"  Overall Test Accuracy (from report): {accuracy_from_report:.4f}")

print("\nGenerating evaluation plots...")
cm = confusion_matrix(y_test, prediction, labels=np.arange(len(le.classes_)))
plt.figure(figsize=(max(12, len(le.classes_)*0.6), max(10, len(le.classes_)*0.5)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, annot_kws={"size": 8})
plt.title(f'Confusion Matrix for {SELECTED_MODEL} (Test Set)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14); plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=65, ha="right", fontsize=9); plt.yticks(rotation=0, fontsize=9)
save_and_show_plot(f"confusion_matrix_{SELECTED_MODEL}.png", "Confusion Matrix")

class_f1_scores = report_df['f1-score'][:-3].copy()
class_support = report_df['support'][:-3].copy().astype(int)
fig, ax1 = plt.subplots(figsize=(14, 8))
color = 'tab:blue'; ax1.set_xlabel('Job Category', fontsize=12)
ax1.set_ylabel('F1-Score (Test Set)', color=color, fontsize=12)
bars = ax1.bar(class_f1_scores.index, class_f1_scores.values, color=color, alpha=0.7, label='F1-Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(len(class_f1_scores.index)))
ax1.set_xticklabels(class_f1_scores.index, rotation=70, ha="right", fontsize=9)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
for bar_idx, bar in enumerate(bars):
    yval = bar.get_height()
    if yval > 0.001:
      ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
ax1.set_ylim(0, 1.1)
ax2 = ax1.twinx(); color = 'tab:red'
ax2.set_ylabel('Support (Test Set Sample Count)', color=color, fontsize=12)
ax2.plot(class_support.index, class_support.values, color=color, marker='o', linestyle='--', label='Support Count')
ax2.tick_params(axis='y', labelcolor=color)
max_support = class_support.max()
ax2.set_ylim(0, max_support * 1.15 if max_support > 0 else 10)
fig.legend(loc="upper right", bbox_to_anchor=(0.98,0.98), bbox_transform=ax1.transAxes)
plt.title(f'Per-Class F1-Scores and Support for {SELECTED_MODEL} (Test Set)', fontsize=16, pad=20)
save_and_show_plot(f"f1_scores_vs_support_{SELECTED_MODEL}.png", "F1 vs Support")

print("\nClosing all plot figures to free memory...")
plt.close('all')

print(f"\n--- Training and Analysis Script Finished ({SELECTED_MODEL}) ---")
print(f"Configuration used: TFIDF_MAX_FEATURES={TFIDF_MAX_FEATURES}, TFIDF_MIN_DF={TFIDF_MIN_DF}, TFIDF_MAX_DF={TFIDF_MAX_DF}, TEST_SET_SIZE={TEST_SET_SIZE}")