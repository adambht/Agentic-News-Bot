import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib  # or pickle if you prefer
import os
import sys
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved objects
tfidf = joblib.load("src/models/tfidf_vectorizer.pkl")   # your fitted TF-IDF
scaler = joblib.load("src/models/minmax_scaler.pkl")     # your fitted scaler

# Initialize text processing
stopwords_En = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def remove_punct(text):
    return "".join([char for char in text if char not in string.punctuation])

def count_punct_words(text):
    words = text.split()
    if len(words) == 0:
        return 0
    punct_count = sum(1 for char in text if char in string.punctuation)
    return round(punct_count / len(words), 3) * 100

def count_cap_words(text):
    words = text.split()
    if len(words) == 0:
        return 0
    cap_count = sum(1 for char in text if char.isupper())
    return round(cap_count / len(words), 3) * 100

def preprocess_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess new data exactly like the original training pipeline.
    df must contain a 'text' column.
    Returns a DataFrame ready for prediction.
    """
    # 1️⃣ Clean text
    df['text_clean'] = df['text'].apply(lambda x: remove_punct(str(x).lower()))
    
    # 2️⃣ Tokenize
    df['text_tokens'] = df['text_clean'].apply(word_tokenize)
    
    # 3️⃣ Remove stopwords
    df['text_tokens'] = df['text_tokens'].apply(lambda tokens: [w for w in tokens if w not in stopwords_En])
    
    # 4️⃣ Lemmatize
    df['text_tokens'] = df['text_tokens'].apply(lambda tokens: [wn.lemmatize(w) for w in tokens])
    
    # 5️⃣ Join tokens back for TF-IDF
    df['text_final'] = df['text_tokens'].apply(lambda tokens: ' '.join(tokens))
    
    # 6️⃣ Transform using fitted TF-IDF
    text_tfidf = tfidf.transform(df['text_final']).toarray()
    
    # 7️⃣ Extra numeric features
    df['body_len'] = df['text'].apply(lambda x: len(x) - x.count(' '))
    df['punct_per_word%'] = df['text'].apply(count_punct_words)
    df['cap_per_word%'] = df['text'].apply(count_cap_words)
    
    # 8️⃣ Combine features
    features_df = pd.DataFrame(text_tfidf, columns=tfidf.get_feature_names_out())
    features_df['body_len'] = df['body_len']
    features_df['punct_per_word%'] = df['punct_per_word%']
    features_df['cap_per_word%'] = df['cap_per_word%']
    
    # 9️⃣ Scale using fitted scaler
    features_scaled = scaler.transform(features_df)
    
    return pd.DataFrame(features_scaled, columns=features_df.columns)
