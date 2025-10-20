import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import joblib
from utils.data_preprocessing import preprocess_new_data
from utils.simulation_helpers import generate_single_news_structured_llm  # returns a dict including ground_truth
from utils.data_validation import NewsItem  # import pydantic model


# Load your saved model
model = joblib.load("src/models/best_model.pkl")

# Generate one structured news article (returns a NewsItem instance)
news_item = generate_single_news_structured_llm()

news_dict = news_item.dict()        # convert to dict
ground_truth = news_dict.pop("label", None)  # remove and save label
temp_df = pd.DataFrame([news_dict])  # only title, text, subject, date


# Preprocess
X_new = preprocess_new_data(temp_df)

# Predict
y_pred = model.predict(X_new)

# Map numeric label to readable form
label_map = {0: "Fake News", 1: "True News"}
prediction = label_map[y_pred[0]]

# Print results
print("\n🗞️ === Generated News ===")
print(f"Title: {news_item.title}")
print(f"Subject: {news_item.subject}")
print(f"Date: {news_item.date}")
print(f"\nBody:\n{news_item.text}")

print("\n=== Prediction Result ===")
print({"Prediction": prediction})
if ground_truth is not None:
    print("Ground Truth:", "True News" if ground_truth == 1 else "Fake News")
