import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_sms_dataset(path):
    df = pd.read_csv(path, encoding='latin-1')

    # If dataset uses v1 (label) and v2 (text), rename
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})

    # Keep only required columns
    df = df[['label', 'text']]

    # Convert labels to numeric
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df

def preprocess_sms(df):
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    return X_vec, y, vectorizer

if __name__ == "__main__":
    df = load_sms_dataset("data/sms_spam.csv")

    X, y, vectorizer = preprocess_sms(df)
    print("Preprocessing done. Shape:", X.shape)

