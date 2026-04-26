import os
import pandas as pd
import joblib
from scipy.sparse import hstack

# Ensure absolute paths so the ensemble can import this cleanly from anywhere
DIR = os.path.dirname(os.path.abspath(__file__))

# Load artifacts once when this file is imported
try:
    vectorizer = joblib.load(os.path.join(DIR, 'tfidf_vectorizer.pkl'))
    model = joblib.load(os.path.join(DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(DIR, 'scaler.pkl'))
except FileNotFoundError:
    vectorizer = None
    model = None
    scaler = None
    print("Warning: Model files not found. Please navigate to layer1_pattern and run train.py first.")

def extract_stylometric_features(texts):
    features = pd.DataFrame()
    features['text_len'] = texts.apply(lambda x: len(str(x)))
    features['word_count'] = texts.apply(lambda x: len(str(x).split()))
    features['avg_word_len'] = features['text_len'] / (features['word_count'] + 1)
    features['exclamation_count'] = texts.apply(lambda x: str(x).count('!'))
    features['question_count'] = texts.apply(lambda x: str(x).count('?'))
    features['caps_ratio'] = texts.apply(lambda x: sum(1 for word in str(x).split() if word.isupper()) / (len(str(x).split()) + 1))
    return features.values

def predict_layer1(article_text: str) -> dict:
    """
    Takes an article_text and returns the expected structured dictionary.
    """
    if not model or not vectorizer or not scaler:
        return {"label": "ERROR", "confidence": 0.0, "reason": "Model not trained yet."}
    
    # Vectorize input text
    text_tfidf = vectorizer.transform([article_text])
    
    # Extract stylometric features
    text_series = pd.Series([article_text])
    stylo_features = extract_stylometric_features(text_series)
    stylo_scaled = scaler.transform(stylo_features)
    
    # Combine features
    text_combined = hstack([text_tfidf, stylo_scaled])
    
    # Get prediction and confidence
    prediction = model.predict(text_combined)[0]  # 1 (Fake) or 0 (Real)
    
    # Calculate Confidence 
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_combined)[0]
        confidence = float(max(probabilities))
    elif hasattr(model, "decision_function"):
        dec_val = abs(model.decision_function(text_combined)[0])
        confidence = float(min(0.5 + (dec_val / 2.0), 0.99))
    else:
        confidence = 0.85
        
    label_str = "FAKE" if prediction == 1 else "REAL"
    reason = "Lexical patterns and N-Grams strongly match known text characteristics"
    
    return {
        "label": label_str,
        "confidence": round(confidence, 4),
        "reason": f"{reason} for {label_str} news."
    }

# Quick local test if you run predict.py directly
if __name__ == "__main__":
    sample_text = "Scientists have just discovered a massive planet entirely made of chocolate!"
    print(predict_layer1(sample_text))
