# 📰 Multi-Layer Fake News Detection System

## Overview

The rapid spread of misinformation across digital platforms has made it increasingly difficult to distinguish between genuine and fake news. This project presents a **Multi-Layer Fake News Detection System** that combines traditional machine learning, semantic understanding, and reasoning-based verification to accurately classify news articles as **REAL** or **FAKE**.

Unlike conventional fake news detectors that rely on a single model, our approach uses a **stacking ensemble architecture** where multiple specialized layers analyze different aspects of a news article before a final decision is made. This improves robustness, interpretability, and overall reliability.

---

## Key Features

- Multi-layer hybrid AI architecture
- Pattern-based fake news detection
- Semantic understanding using transformer embeddings
- Reasoning-based verification using Natural Language Inference (NLI)
- Stacking ensemble with meta-classifier
- Stylometric feature analysis
- Batch-optimized inference pipeline
- Explainable layer-wise predictions
- Scalable and modular design

---

# System Architecture

```text
News Article
     │
     ▼
┌─────────────────────┐
│ Data Preprocessing  │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Layer 1             │
│ Pattern Detection   │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Layer 2             │
│ Semantic Analysis   │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Layer 3             │
│ NLI Verification    │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Stacking Ensemble   │
│ Meta Classifier     │
└─────────────────────┘
     │
     ▼
REAL / FAKE
```

---

# Dataset

The system is trained and evaluated using globally separated datasets:

- `global_train.csv`
- `global_val.csv`

The dataset contains labeled news articles categorized as:

- REAL (0)
- FAKE (1)

A global train-validation split ensures consistent evaluation and prevents train-test contamination.

---

# Layer 1: Pattern-Based Detection

## Objective

The first layer focuses on identifying linguistic and statistical patterns commonly found in fake and real news articles.

## Features Used

### TF-IDF Features

Term Frequency–Inverse Document Frequency (TF-IDF) is used to convert textual content into numerical feature vectors by measuring the importance of words relative to the entire corpus.

### Stylometric Features

Writing style indicators are extracted to identify suspicious linguistic patterns:

- Text Length
- Word Count
- Average Word Length
- Exclamation Mark Count
- Question Mark Count
- Capitalization Ratio

## Model

The extracted features are passed through the best-performing machine learning classifier trained during experimentation.

### Output

```python
Probability(Fake)
Probability(Real)
```

### Strengths

- Fast inference
- Effective at detecting linguistic manipulation
- Captures writing patterns often associated with misinformation

---

# Layer 2: Semantic Analysis

## Objective

While Layer 1 identifies patterns, it cannot fully understand meaning. Layer 2 focuses on semantic understanding and contextual relationships within the article.

## Model

Sentence Transformer:

```text
all-MiniLM-L6-v2
```

The model converts articles into dense semantic embeddings that capture contextual meaning.

## Methodology

The article is transformed into semantic vectors and compared against learned representations of fake and real news.

This layer captures:

- Contextual meaning
- Paraphrased misinformation
- Semantic similarity
- Meaning beyond keywords

### Output

```python
Semantic Probability(Fake)
```

### Strengths

- Understands contextual relationships
- Handles paraphrased content
- More robust than keyword-based methods

---

# Layer 3: NLI Verification

## Objective

Some news articles may appear linguistically and semantically valid while still containing misleading or contradictory information. Layer 3 introduces reasoning-based verification.

## Model

```text
facebook/bart-large-mnli
```

Natural Language Inference (NLI) is used to determine whether a statement:

- Entails known facts
- Contradicts known facts
- Is neutral

## Verification Process

The article is compared against a curated knowledge base of trusted factual statements.

Example facts include:

- Vaccines are safe and effective.
- Climate change is scientifically proven.
- The Moon orbits the Earth.
- Antibiotics do not treat viral infections.

The model computes contradiction and entailment scores to estimate the likelihood of misinformation.

### Output

```python
NLI Verification Score
```

### Strengths

- Adds logical reasoning capability
- Detects contradictions
- Helps validate claims beyond surface-level patterns

---

# Stacking Ensemble Layer

## Why an Ensemble?

Each layer specializes in a different aspect of fake news detection:

| Layer | Focus |
|---------|---------|
| Layer 1 | Linguistic patterns |
| Layer 2 | Semantic understanding |
| Layer 3 | Logical verification |

Rather than manually assigning importance to each layer, we employ a stacking ensemble strategy.

## Meta-Classifier

Model Used:

```text
Logistic Regression
```

### Input Features

```python
[
 Layer1_Probability,
 Layer2_Probability,
 Layer3_Probability
]
```

The meta-classifier learns how much importance should be assigned to each layer and generates the final prediction.

### Benefits

- Improved robustness
- Better generalization
- Reduced dependency on a single model
- Adaptive weighting of layer outputs

---

# Data Preprocessing Pipeline

1. Load news article
2. Clean text
3. Tokenization
4. TF-IDF transformation
5. Stylometric feature extraction
6. Semantic embedding generation
7. NLI verification
8. Layer output aggregation
9. Meta-classifier prediction
10. Final REAL/FAKE classification

---

# Performance Optimization

During development, several optimizations were introduced:

### Batch Processing

- Batch TF-IDF transformation
- Batch embedding generation
- Reduced repeated computations

### Conditional Reasoning

- Expensive NLI verification can be selectively executed when deeper analysis is required

### Feature Caching

- Intermediate outputs reused to reduce processing overhead

### Global Dataset Split

- Consistent train-validation separation
- Eliminates evaluation inconsistencies
- Improves experimental reliability

---

# Technologies Used

## Machine Learning

- Scikit-Learn
- Logistic Regression
- Support Vector Machines (SVM)

## Natural Language Processing

- Sentence Transformers
- Hugging Face Transformers
- BART-MNLI

## Data Processing

- Pandas
- NumPy
- SciPy

## Visualization

- Matplotlib
- Seaborn

---

# Project Structure

```text
project/
│
├── layer1_pattern/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── scaler.pkl
│
├── layer2_semantic/
│
├── layer3_nli/
│
├── global_train.csv
├── global_val.csv
│
├── ensemble_meta_model.pkl
│
└── notebooks/
```

---

# Future Enhancements

- Real-time fact verification using external knowledge bases
- Multi-language fake news detection
- Explainable AI dashboard
- Knowledge graph integration
- Dynamic confidence-based reasoning
- Real-time news stream processing

---

# Conclusion

This project demonstrates a robust multi-layer approach to fake news detection by combining traditional machine learning, semantic understanding, and reasoning-based verification. The stacking ensemble architecture enables the system to leverage the strengths of each layer while reducing the limitations of individual models, resulting in a scalable and intelligent misinformation detection framework.

---

## Authors

Developed as a multi-layer AI-driven fake news detection system for reliable misinformation analysis and classification.
