
import numpy as np
from sklearn.datasets import fetch_20newsgroups

def load_data():
    """
    Loads the 20 Newsgroups dataset, filtering for Sport and Politics categories.
    Returns train and test datasets.
    """
    categories = [
        'rec.sport.baseball', 'rec.sport.hockey',
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'
    ]
    
    print("Loading 20 Newsgroups dataset...")
    # Load training data
    train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    # Load test data
    test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    
    print(f"Training samples: {len(train_data.data)}")
    print(f"Test samples: {len(test_data.data)}")
    
    # Map targets to binary: 0 for Sport, 1 for Politics
    # Check target names to ensure correct mapping
    print("Target names:", train_data.target_names)
    
    # Sport categories start with 'rec.sport', Politics with 'talk.politics'
    # We'll rely on the string names to create binary labels if needed, 
    # but for now let's just inspect the distribution.
    
    return train_data, test_data

if __name__ == "__main__":
    train, test = load_data()
    
    print("\nSample Data (First 200 chars):")
    print("-" * 30)
    print(train.data[0][:200])
    print("-" * 30)
    print(f"Label: {train.target_names[train.target[0]]}")
