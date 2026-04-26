import os
import re
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def extract_stylometric_features(texts):
    features = pd.DataFrame()
    
    # Text length (character count)
    features['text_len'] = texts.apply(lambda x: len(str(x)))
    
    # Word count
    features['word_count'] = texts.apply(lambda x: len(str(x).split()))
    
    # Average word length (avoid division by zero)
    features['avg_word_len'] = features['text_len'] / (features['word_count'] + 1)
    
    # Count of exclamation and question marks
    features['exclamation_count'] = texts.apply(lambda x: str(x).count('!'))
    features['question_count'] = texts.apply(lambda x: str(x).count('?'))
    
    # Ratio of ALL CAPS words
    features['caps_ratio'] = texts.apply(lambda x: sum(1 for word in str(x).split() if word.isupper()) / (len(str(x).split()) + 1))
    
    return features.values

def main():
    print("Loading data...")
    # Using absolute paths so it never crashes no matter where you run it from!
    DIR = os.path.dirname(os.path.abspath(__file__))
    fake_df = pd.read_csv(os.path.join(DIR, 'Fake.csv'))
    true_df = pd.read_csv(os.path.join(DIR, 'True.csv'))

    # Assign labels (1 for FAKE, 0 for REAL)
    fake_df['label'] = 1
    true_df['label'] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Cleaning text (Removing sneaky (Reuters) tags to prevent cheating)...")
    df['text'] = df['text'].apply(lambda x: re.sub(r'^.*?\(Reuters\) - ', '', str(x)))

    X = df['text']
    y = df['label']

    print("Vectorizing text with N-Grams (1 to 3 words)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Upgraded TFIDF with n-grams
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 3), max_features=15000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Save the vectorizer using absolute path
    joblib.dump(tfidf, os.path.join(DIR, 'tfidf_vectorizer.pkl'))

    print("Extracting stylometric features...")
    X_train_stylo = extract_stylometric_features(X_train)
    X_test_stylo = extract_stylometric_features(X_test)
    
    # Scale stylometric features
    scaler = StandardScaler()
    X_train_stylo_scaled = scaler.fit_transform(X_train_stylo)
    X_test_stylo_scaled = scaler.transform(X_test_stylo)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(DIR, 'scaler.pkl'))
    
    # Combine TF-IDF and stylometric features
    X_train_combined = hstack([X_train_tfidf, X_train_stylo_scaled])
    X_test_combined = hstack([X_test_tfidf, X_test_stylo_scaled])

    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Passive Aggressive': PassiveAggressiveClassifier(max_iter=50),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_f1 = 0
    best_model = None
    best_model_name = ""

    print("Training models...")
    for name, model in models.items():
        model.fit(X_train_combined, y_train)
        preds = model.predict(X_test_combined)
        f1 = f1_score(y_test, preds)
        print(f"[{name}] F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")
    
    # Save the best model
    joblib.dump(best_model, os.path.join(DIR, 'best_model.pkl'))
    print("Saved 'best_model.pkl', 'tfidf_vectorizer.pkl', and 'scaler.pkl'")

if __name__ == "__main__":
    main()
