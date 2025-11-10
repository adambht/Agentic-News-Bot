import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import joblib
from src.embeddings.embed_model import preprocess_and_embed  # use wrapper for saved or HF model
from utils.simulation_helpers import generate_single_news_structured_llm
from utils.data_validation import NewsItem

# Load your saved LogisticRegression model
model = joblib.load("src/models/logisticRegressor.pkl")

# Generate one structured news article
news_item = generate_single_news_structured_llm()
news_dict = news_item.model_dump()
ground_truth = news_dict.pop("label", None)

# Convert to DataFrame
temp_df = pd.DataFrame([news_dict])

# Generate embeddings for new data
X_new = preprocess_and_embed(temp_df, text_column='text')

# Predict class and probabilities
y_pred = model.predict(X_new)
y_prob = model.predict_proba(X_new)[0]

# Map numeric label to readable form
label_map = {0: "Fake News", 1: "True News"}
prediction = label_map[y_pred[0]]
confidence = y_prob[y_pred[0]] * 100  # percentage of the predicted class

# Print results
print("\n🗞️ === Generated News ===")
print(f"Title: {news_item.title}")
print(f"Subject: {news_item.subject}")
print(f"Date: {news_item.date}")
print(f"\nBody:\n{news_item.text}")

print("\n=== Prediction Result ===")
print({
    "Prediction": prediction,
    "Confidence": f"{confidence:.2f}%",
})
if ground_truth is not None:
    print("Ground Truth:", "True News" if ground_truth == 1 else "Fake News")
