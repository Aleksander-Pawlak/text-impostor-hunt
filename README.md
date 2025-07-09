# Text Impostor Hunt 🕵️‍♂️

A machine learning project for comparing AI-generated texts to determine which one more closely resembles a human-written reference, using linguistic and statistical features.

## Overview

This project analyzes pairs of AI-generated texts alongside a human-written reference to identify which generated text is more similar to the original. It uses various text analysis techniques including language detection, character encoding analysis, readability metrics, and linguistic features to train an XGBoost classifier.


## Features

### Text Analysis Functions
- **Language Detection**: Calculates the ratio of English text using chunk-based language detection
- **Character Encoding**: Analyzes the ratio of Latin characters in text
- **Readability Metrics**: Computes Flesch-Kincaid grade level and reading ease scores
- **Linguistic Features**: Extracts word count, sentence structure, punctuation usage, and more

### Machine Learning Pipeline
- **Feature Engineering**: Extracts 19 different text features including:
  - Word and character counts
  - Stopword ratios
  - Sentence structure metrics
  - Readability scores
  - Punctuation and formatting patterns
- **XGBoost Classifier**: Uses gradient boosting with standard scaling for classification
- **Model Evaluation**: Provides accuracy, classification reports, and confusion matrices

### Data Visualization
- Scatter plots comparing real vs fake text ratios
- Distribution histograms for different text features
- Feature importance visualization
- Summary statistics for all metrics

## Dependencies

```python
numpy
pandas
matplotlib
seaborn
nltk
textstat
langdetect
scikit-learn
xgboost
```

## Usage

1. **Data Structure**: The script expects training data in folders with `file_1.txt` and `file_2.txt` pairs
2. **Labels**: A CSV file with `id` and `real_text_id` columns indicating which file contains real text
3. **Execution**: Run the script and choose whether to visualize the data when prompted

## Key Insights

The model identifies distinguishing features between real and fake text, with feature importance analysis showing which linguistic patterns are most predictive of authenticity.

## Output

- Processed feature dataset saved as `output_check.csv`
- Validation accuracy and performance metrics
- Feature importance rankings
- Optional data visualizations

Perfect for researchers, data scientists, or anyone interested in text authenticity detection! 📊✨
