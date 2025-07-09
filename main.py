import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from nltk.corpus import stopwords # natural language processing
import unicodedata
import string
import textstat  # for readability measurements
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 69

def english_ratio(text, n = 10):
    """
    Calculates the ratio of English text in a given text.
    Parameters:
    text (str): The input text to analyze.
    n (int): The number of words to consider in each chunk for language detection.
    Returns:
    float: The ratio of English text in the input text.
    0 if no chunks are detected.
    """
    text = str(text.lower()).translate(str.maketrans('', '', string.punctuation + '\n'))
    words = text.split()
    if not words:
        return 0
    
    chunks = [' '.join(words[i:i+n]) for i in range(0, len(words), n)]
    if not chunks:
        return 0
    
    english_count = 0
    for chunk in chunks:
        try:
            if detect(chunk) == 'en':
                english_count += 1
        except LangDetectException:
            continue
    
    return english_count / len(chunks)

def latin_ratio(text):
    """
    Calculates the ratio of Latin characters in a given text.
    
    Parameters:
    text (str): The input text to analyze.
    
    Returns:
    float: The ratio of Latin characters in the input text.
    """
    text = str(text).translate(str.maketrans('', '', string.punctuation + '\n'))
    non_space_chars = [c for c in text if c != ' ']
    if not non_space_chars:
        return 0
    latin_count = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]
    return len(latin_count) / len(non_space_chars)

def read_data(file_path):
    """
    Reads a txt file and returns a DataFrame.
    
    Parameters:
    file_path (str): The path to the txt file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the txt file.
    """
    data = []
    for folder_name in sorted(os.listdir(file_path)):
        folder_path = os.path.join(file_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding = 'utf-8') as file:
                    text_1 = file.read().strip()
                with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding = 'utf-8') as file:
                    text_2 = file.read().strip()
                # Extract the index from the folder name
                index = int(folder_name.split('_')[-1])
                data.append([index, text_1, text_2])
            except Exception as e:
                print(f"Error reading files in {folder_path}: {e}")
    # Create a DataFrame from the collected data
    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])

train_path = 'C:\\Users\\olopa\\Downloads\\fake-or-real-the-impostor-hunt\\data\\train'
test_path = 'C:\\Users\\olopa\\Downloads\\fake-or-real-the-impostor-hunt\\data\\test'
label_path = 'C:\\Users\\olopa\\Downloads\\fake-or-real-the-impostor-hunt\\data\\train.csv'

dataframe_train = read_data(train_path)
dataframe_test = read_data(test_path)
dataframe_labels = pd.read_csv(label_path, index_col='id')
# Merge the train DataFrame with labels
dataframe_train = dataframe_train.merge(dataframe_labels, on='id', how='left')

def split_real_fake(row):
    if row['real_text_id'] == 1:
        return pd.Series([row['file_1'], row['file_2']])
    else:
        return pd.Series([row['file_2'], row['file_1']])

merged_dataframe_train = dataframe_train.copy()
merged_dataframe_train[['real_text', 'fake_text']] = dataframe_train.apply(split_real_fake, axis=1)
merged_dataframe_train = merged_dataframe_train.drop(columns=['file_1', 'file_2', 'real_text_id'])
merged_dataframe_train = merged_dataframe_train.dropna(subset=['real_text', 'fake_text'])

merged_dataframe_train['latin_real_text_ratio'] = merged_dataframe_train['real_text'].apply(latin_ratio)
merged_dataframe_train['latin_fake_text_ratio'] = merged_dataframe_train['fake_text'].apply(latin_ratio)
merged_dataframe_train['english_real_text_ratio'] = merged_dataframe_train['real_text'].apply(english_ratio)
merged_dataframe_train['english_fake_text_ratio'] = merged_dataframe_train['fake_text'].apply(english_ratio)

dataframe_output_check = merged_dataframe_train[['id', 'real_text', 'fake_text', 'latin_real_text_ratio', 'latin_fake_text_ratio', 'english_real_text_ratio', 'english_fake_text_ratio']]
dataframe_output_check.to_csv('C:\\Users\\olopa\\Downloads\\fake-or-real-the-impostor-hunt\\data\\output_check.csv', index=False)
print("Data processing complete. Output saved to 'output_check.csv'.")

merged_dataframe_train['english_real_text_ratio_diff'] = merged_dataframe_train['english_real_text_ratio'] - merged_dataframe_train['english_fake_text_ratio']
merged_dataframe_train['latin_real_text_ratio_diff'] = merged_dataframe_train['latin_real_text_ratio'] - merged_dataframe_train['latin_fake_text_ratio']

# Ask the user if they want to visualize the data
visualize_data = input("Do you want to visualize the data? (yes/no): ").strip().lower()
if visualize_data == 'yes':
    plt.figure(figsize=(12, 6))
    plt.scatter(merged_dataframe_train['english_real_text_ratio'], merged_dataframe_train['english_fake_text_ratio'], alpha=0.5)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('English Text Ratio Comparison')
    plt.xlabel('English Real Text Ratio')
    plt.ylabel('English Fake Text Ratio')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(merged_dataframe_train['latin_real_text_ratio'], merged_dataframe_train['latin_fake_text_ratio'], alpha=0.5)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('Latin Text Ratio Comparison')
    plt.xlabel('Latin Real Text Ratio')
    plt.ylabel('Latin Fake Text Ratio')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(merged_dataframe_train['english_real_text_ratio_diff'], bins=50, alpha=0.5, label='English Ratio Diff (Real - Fake)', color='blue')
    plt.hist(merged_dataframe_train['latin_real_text_ratio_diff'], bins=50, alpha=0.5, label='Latin Ratio Diff (Real - Fake)', color='red')
    plt.title('Distribution of Text Ratio Differences')
    plt.xlabel('Ratio Difference')
    plt.axvline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

    melted = pd.melt(merged_dataframe_train, id_vars='id', value_vars=['english_real_text_ratio', 'english_fake_text_ratio'],
                        var_name='text_type', value_name='english_ratio')
    melted['label'] = melted['text_type'].str.extract('(real|fake)')

    plt.figure(figsize=(12, 6))
    sns.histplot(data=melted, x='english_ratio', hue='label', multiple='stack', bins=50, kde=True, element='step')
    plt.title('Distribution of English Text Ratios')
    plt.xlabel('English Text Ratio')
    plt.axvline(0, color='black', linestyle='--')
    plt.show()

    melted = pd.melt(merged_dataframe_train, id_vars='id', value_vars=['latin_real_text_ratio', 'latin_fake_text_ratio'],
                        var_name='text_type', value_name='latin_ratio')
    melted['label'] = melted['text_type'].str.extract('(real|fake)')

    plt.figure(figsize=(12, 6))
    sns.histplot(data=melted, x='latin_ratio', hue='label', multiple='stack', bins=50, kde=True, element='step')
    plt.title('Distribution of Latin Text Ratios')
    plt.xlabel('Latin Text Ratio')
    plt.axvline(0, color='black', linestyle='--')
    plt.show()

    #summary statistics
    summary_stats = {
        'english_real_text_ratio': {
            'mean': merged_dataframe_train['english_real_text_ratio'].mean(),
            'std': merged_dataframe_train['english_real_text_ratio'].std(),
            'min': merged_dataframe_train['english_real_text_ratio'].min(),
            'max': merged_dataframe_train['english_real_text_ratio'].max()
        },
        'latin_real_text_ratio': {
            'mean': merged_dataframe_train['latin_real_text_ratio'].mean(),
            'std': merged_dataframe_train['latin_real_text_ratio'].std(),
            'min': merged_dataframe_train['latin_real_text_ratio'].min(),
            'max': merged_dataframe_train['latin_real_text_ratio'].max()
        },
        'english_fake_text_ratio': {
            'mean': merged_dataframe_train['english_fake_text_ratio'].mean(),
            'std': merged_dataframe_train['english_fake_text_ratio'].std(),
            'min': merged_dataframe_train['english_fake_text_ratio'].min(),
            'max': merged_dataframe_train['english_fake_text_ratio'].max()
        },
        'latin_fake_text_ratio': {
            'mean': merged_dataframe_train['latin_fake_text_ratio'].mean(),
            'std': merged_dataframe_train['latin_fake_text_ratio'].std(),
            'min': merged_dataframe_train['latin_fake_text_ratio'].min(),
            'max': merged_dataframe_train['latin_fake_text_ratio'].max()
        }
    }

    for key, stats in summary_stats.items():
        print(f"\n{key}:")
        for stat, stat_value in stats.items():
            print(f"  {stat}: {stat_value:.4f}")

stopwords_list = set(stopwords.words('english'))

def extract_text_features(text):
    if pd.isna(text) or not str(text).strip():
        return {
            'is_empty': True,
            'word_count': 0,
            'char_count': 0,
            'stopword_count': 0,
            'unique_word_count': 0,
            'avg_sentence_length': 0.0,
            'avg_word_length': 0.0,
            'repeat_word_count': 0,
            'repeat_word_ratio': 0,
            'english_ratio': 0.0,
            'latin_ratio': 0.0,
            'stopword_ratio': 0.0,
            'punctuation_ratio': 0.0,
            'space_ratio': 0.0,
            'sentence_count': 0,
            'count_double_hash': 0,
            'count_double_star': 0,
            'flesch_kincaid_grade': 0.0,
            'flesch_reading_ease': 0.0
        }
    text = str(text)
    tokens = text.split()
    words = [word.strip(string.punctuation) for word in tokens if word.strip(string.punctuation)]
    word_count = len(words)
    char_count = sum(len(word) for word in words)
    stopword_count = sum(1 for word in words if word.lower() in stopwords_list)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    repeat_word_count = len(words) - len(set(words))

    # Calculate readability metrics
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text) if text.strip() else 0.0
    flesch_reading_ease = textstat.flesch_reading_ease(text) if text.strip() else 0.0

    return {
        'is_empty': False,
        'word_count': word_count,
        'char_count': char_count,
        'stopword_count': stopword_count,
        'unique_word_count': len(set(words)),
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0.0,
        'avg_word_length': char_count / word_count if word_count > 0 else 0.0,
        'repeat_word_count': repeat_word_count,
        'repeat_word_ratio': repeat_word_count / word_count if word_count > 0 else 0.0,
        'english_ratio': english_ratio(text),
        'latin_ratio': latin_ratio(text),
        'stopword_ratio': stopword_count / word_count if word_count > 0 else 0.0,
        'punctuation_ratio': sum(1 for char in text if char in string.punctuation) / char_count if char_count > 0 else 0.0,
        'space_ratio': text.count(' ') / char_count if char_count > 0 else 0.0,
        'sentence_count': sentence_count,
        'count_double_hash': text.count('##'),
        'count_double_star': text.count('**'),
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'flesch_reading_ease': flesch_reading_ease
    }

rows = []
for _, row in merged_dataframe_train.iterrows():
    real_text_features = extract_text_features(row['real_text'])
    real_text_features['label'] = 1
    fake_text_features = extract_text_features(row['fake_text'])
    fake_text_features['label'] = 0
    rows.extend([real_text_features, fake_text_features])

processed_dataframe_train = pd.DataFrame(rows)
X_train = processed_dataframe_train.drop(columns=['label'])
y_train = processed_dataframe_train['label']


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb_model)
])
pipeline.fit(X_train_split, y_train_split)
y_pred = pipeline.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report: \n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Feature Importances:")

plt.figure(figsize=(12, 6))
feature_importances = pipeline.named_steps['xgb'].feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
top_n = min(20, len(X_train.columns))  # Ensure we don't exceed available columns

visualize_data_feat_importance = input("Do you want to visualize the feature importances? (yes/no): ").strip().lower()
if visualize_data_feat_importance == 'yes':
    print("Visualizing feature importances...")
    plt.bar(range(top_n), feature_importances[sorted_indices][:top_n], align='center')
    plt.xticks(range(top_n), [X_train.columns[i] for i in sorted_indices[:top_n]], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()

# Ensemble Model with Multiple Random Seeds
print("\n\nBuilding Ensemble Model (9 XGBoost models)")

from collections import Counter

# Define different random seeds for ensemble
ensemble_seeds = [42, 123, 456, 789, 741, 258, 963, 735, 846]
ensemble_models = []
ensemble_predictions = []

# Train 9 models with different random seeds
for i, seed in enumerate(ensemble_seeds):
    print(f"Training model {i+1}/9 with random seed {seed}...")
    
    # Create model with different random seed
    xgb_ensemble = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=seed
    )
    
    # Create pipeline
    ensemble_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb_ensemble)
    ])
    
    # Fit the model
    ensemble_pipeline.fit(X_train_split, y_train_split)
    
    # Make predictions on validation set
    y_pred_ensemble = ensemble_pipeline.predict(X_val)
    ensemble_predictions.append(y_pred_ensemble)
    ensemble_models.append(ensemble_pipeline)
    
    # Print individual model accuracy
    individual_accuracy = accuracy_score(y_val, y_pred_ensemble)
    print(f"  Model {i+1} accuracy: {individual_accuracy:.4f}")

# Majority voting for final predictions
print("\nPerforming majority voting...")
final_predictions = []

for i in range(len(y_val)):
    # Get predictions from all 9 models for this sample
    votes = [pred[i] for pred in ensemble_predictions]
    
    # Count votes (0 = fake, 1 = real)
    vote_counts = Counter(votes)
    
    # Choose majority vote (in case of tie, choose 0)
    final_pred = vote_counts.most_common(1)[0][0]
    final_predictions.append(final_pred)

final_predictions = np.array(final_predictions)

# Evaluate ensemble performance
ensemble_accuracy = accuracy_score(y_val, final_predictions)
print(f"\nEnsemble Results:")
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"Improvement over single model: {ensemble_accuracy - accuracy_score(y_val, y_pred):.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(y_val, final_predictions))
print("\nEnsemble Confusion Matrix:")
print(confusion_matrix(y_val, final_predictions))

# Visualize ensemble vs individual model performance
plt.figure(figsize=(12, 6))
individual_accuracies = [accuracy_score(y_val, pred) for pred in ensemble_predictions]
model_names = [f'Model {i+1}' for i in range(9)] + ['Ensemble']
accuracies = individual_accuracies + [ensemble_accuracy]

plt.bar(model_names, accuracies, color=['lightblue']*9 + ['darkblue'])
plt.axhline(y=accuracy_score(y_val, y_pred), color='red', linestyle='--', label='Original Single Model')
plt.title('Individual Model vs Ensemble Performance')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nEnsemble Summary:")
print(f"Number of models: {len(ensemble_models)}")
print(f"Best individual model accuracy: {max(individual_accuracies):.4f}")
print(f"Worst individual model accuracy: {min(individual_accuracies):.4f}")
print(f"Average individual model accuracy: {np.mean(individual_accuracies):.4f}")
print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
print(f"Standard deviation of individual models: {np.std(individual_accuracies):.4f}")

# Analysis: Better Approaches for Small Datasets
print("\n" + "="*60)
print("SMALL DATASET ANALYSIS & RECOMMENDATIONS")
print("="*60)

print(f"Dataset Size Analysis:")
print(f"Total training samples: {len(X_train)}")
print(f"Training split size: {len(X_train_split)}")
print(f"Validation split size: {len(X_val)}")
print(f"Features per sample: {len(X_train.columns)}")

# Calculate the ratio of features to samples (curse of dimensionality indicator)
feature_to_sample_ratio = len(X_train.columns) / len(X_train_split)
print(f"Feature-to-sample ratio: {feature_to_sample_ratio:.2f}")

if feature_to_sample_ratio > 0.1:
    print("⚠️  WARNING: High feature-to-sample ratio suggests potential overfitting!")

if len(X_val) < 20:
    print("⚠️  WARNING: Validation set too small for reliable accuracy estimates!")

print(f"\nRecommendations for small datasets:")
print("1. Use Cross-Validation instead of train/val split")
print("2. Apply regularization (L1/L2) to prevent overfitting")
print("3. Reduce feature dimensions (PCA, feature selection)")
print("4. Use simpler models (fewer parameters)")
print("5. Bootstrap sampling for confidence intervals")

# Demonstrate Cross-Validation approach
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer

print(f"\n" + "="*50)
print("CROSS-VALIDATION APPROACH (RECOMMENDED)")
print("="*50)

# Use the full training data for cross-validation
cv_folds = min(5, len(y_train) // 2)  # Ensure at least 2 samples per fold
print(f"Using {cv_folds}-fold cross-validation...")

cv_scores = cross_val_score(
    pipeline, X_train, y_train, 
    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"This gives us a more reliable estimate with confidence intervals!")

# Feature importance with regularization
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

print(f"\n" + "="*50)
print("SIMPLIFIED MODEL WITH FEATURE SELECTION")
print("="*50)

# Select top 10 most important features
selector = SelectKBest(score_func=f_classif, k=min(10, len(X_train.columns)))
X_train_selected = selector.fit_transform(X_train, y_train)

# Use simpler, regularized model
simple_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(C=0.1, random_state=42))  # High regularization
])

# Cross-validate the simpler model
simple_cv_scores = cross_val_score(
    simple_model, X_train_selected, y_train,
    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"Simplified model CV scores: {simple_cv_scores}")
print(f"Simplified model mean accuracy: {simple_cv_scores.mean():.4f} (+/- {simple_cv_scores.std() * 2:.4f})")

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("For small datasets like yours:")
print("✅ Cross-validation > Train/val split")
print("✅ Simpler models > Complex ensembles") 
print("✅ Feature selection > All features")
print("✅ Regularization > No regularization")
print("✅ Confidence intervals > Point estimates")
print("\nEnsemble methods work best with larger datasets (1000+ samples)")
print("where individual models can learn diverse patterns.")