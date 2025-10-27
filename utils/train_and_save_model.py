# utils/train_and_save_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report, accuracy_score

from src.embeddings.embed_model import embed_model, preprocess_and_embed  # your local embedding model

# Load data
fake_df = pd.read_csv("src/data/News_dataset/Fake.csv")
true_df = pd.read_csv("src/data/News_dataset/True.csv")
fake_df["label"] = 0
true_df["label"] = 1
merged_news = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Embed
X = preprocess_and_embed(merged_news, text_column="text")  # returns numpy array
y = merged_news["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save everything
# 1️⃣ Save classifier
joblib.dump(clf, "src/models/logisticRegressor.pkl")

# 2️⃣ Save embeddings model for later use
embed_model.save("src/models/embedding_model")  # sentence-transformers model
