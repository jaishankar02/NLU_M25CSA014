
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import fetch_20newsgroups
import re

# Load data again here or import from data_loader
# For simplicity, let's redefine loading to keep this script self-contained or import if preferred.
# We'll import to avoid code duplication if I had put it in a shared module, 
# but since data_loader was a script, I'll just copy the relevant loading logic or refactor.
# Let's Refactor data_loader to be a module.

categories = [
    'rec.sport.baseball', 'rec.sport.hockey',
    'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'
]

def load_data():
    print("Loading 20 Newsgroups dataset...")
    train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    return train_data, test_data

def preprocess_text(text):
    """
    Basic text cleaning: lowercase, remove special characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_binary_labels(target, target_names):
    """
    Maps targets to 0 (Sport) and 1 (Politics).
    'rec.sport.*' -> 0
    'talk.politics.*' -> 1
    """
    # Identify indices for sport and politics
    sport_indices = [i for i, name in enumerate(target_names) if 'sport' in name]
    politics_indices = [i for i, name in enumerate(target_names) if 'politics' in name]
    
    # Create a mapping
    mapping = {}
    for i in sport_indices: mapping[i] = 0 # Sport
    for i in politics_indices: mapping[i] = 1 # Politics
    
    # Apply mapping
    binary_target = np.array([mapping[t] for t in target])
    return binary_target


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name}...")
    predictions = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Sport', 'Politics'], output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, predictions, target_names=['Sport', 'Politics']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sport', 'Politics'], yticklabels=['Sport', 'Politics'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'output/cm_{model_name.replace(" ", "_")}.png')
    plt.close()
    
    return {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1': f1}

def plot_accuracy_comparison(results):
    names = list(results.keys())
    accuracies = [metrics['Accuracy'] for metrics in results.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color='skyblue')
    plt.xlabel('Model / Technique')
    plt.ylabel('Accuracy')
    plt.title('accuracy Comparison (Zoomed In)')
    plt.xticks(rotation=45)
    plt.ylim(0.95, 1.0)  # Zoom in to show differences
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig('output/model_accuracy_comparison.png')
    print("Accuracy comparison plot saved.")

def plot_multi_metric_comparison(results):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    names = list(results.keys())
    
    x = np.arange(len(names))
    width = 0.2
    
    plt.figure(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in names]
        plt.bar(x + i*width, values, width, label=metric)
        
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Multi-Metric Comparison by Model')
    plt.xticks(x + width*1.5, names, rotation=45)
    plt.ylim(0.95, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('output/multi_metric_comparison.png')
    print("Multi-metric comparison plot saved.")

def run_experiments():
    train, test = load_data()
    
    y_train = get_binary_labels(train.target, train.target_names)
    y_test = get_binary_labels(test.target, test.target_names)
    
    pipelines = {
        'NB_BoW': Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('clf', MultinomialNB()),
        ]),
        'NB_TFIDF': Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ]),
        'NB_Ngrams': Pipeline([
            ('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ]),
        'LR_TFIDF': Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(max_iter=1000)),
        ]),
        'SVM_TFIDF': Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(max_iter=1000)),
        ]),
        'RF_TFIDF': Pipeline([
             ('vect', CountVectorizer(stop_words='english')),
             ('tfidf', TfidfTransformer()),
             ('clf', RandomForestClassifier(n_estimators=100)),
        ])
    }
    
    results = {}
    
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(train.data, y_train)
        metrics = evaluate_model(pipeline, test.data, y_test, name)
        results[name] = metrics
        
    print("\nSummary of Results:")
    print("-" * 30)
    for name, metrics in results.items():
        print(f"{name}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}")

    # Generate Plots
    plot_accuracy_comparison(results)
    plot_multi_metric_comparison(results)
    
    # Additional Visualizations
    plot_class_distribution(train.target, train.target_names)
    
    # Feature Importance (Keywords) for NB_BoW
    # We need to access the pipeline step
    nb_model = pipelines['NB_BoW'].named_steps['clf']
    vectorizer = pipelines['NB_BoW'].named_steps['vect']
    plot_top_keywords(nb_model, vectorizer)

def plot_class_distribution(target, target_names):
    # Map targets to binary class names
    binary_names = ['Sport' if 'sport' in name else 'Politics' for name in target_names]
    # Count occurrences
    sport_count = sum(1 for t in target if 'sport' in target_names[t])
    politics_count = len(target) - sport_count
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Sport', 'Politics'], [sport_count, politics_count], color=['#3498db', '#e74c3c'])
    plt.title('Class Distribution in Training Set')
    plt.ylabel('Number of Documents')
    plt.tight_layout()
    plt.savefig('output/class_distribution.png')
    print("Class distribution plot saved.")

def plot_top_keywords(model, vectorizer, n=20):
    """
    Plots the top N keywords for each class using Log Probabilites from Naive Bayes.
    """
    feature_names = vectorizer.get_feature_names_out()
    
    # Top words for Class 0 (Sport)
    top0 = argsort(model.feature_log_prob_[0])[-n:]
    top0_words = [feature_names[i] for i in top0][::-1]
    top0_scores = model.feature_log_prob_[0][top0][::-1]
    
    # Top words for Class 1 (Politics)
    top1 = argsort(model.feature_log_prob_[1])[-n:]
    top1_words = [feature_names[i] for i in top1][::-1]
    top1_scores = model.feature_log_prob_[1][top1][::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(x=top0_scores, y=top0_words, ax=axes[0], color='#3498db')
    axes[0].set_title('Top Keywords for Sport')
    axes[0].set_xlabel('Log Probability')
    
    sns.barplot(x=top1_scores, y=top1_words, ax=axes[1], color='#e74c3c')
    axes[1].set_title('Top Keywords for Politics')
    axes[1].set_xlabel('Log Probability')
    
    plt.tight_layout()
    plt.savefig('output/top_keywords.png')
    print("Top keywords plot saved.")

if __name__ == "__main__":
    from numpy import argsort # Import argsort locally or at top
    run_experiments()
