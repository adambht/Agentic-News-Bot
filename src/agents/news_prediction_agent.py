import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.data_validation import NewsItem, VerificationResult
import joblib
from src.embeddings.embed_model import preprocess_and_embed  # use wrapper for saved or HF model
import pandas as pd 
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json

class NewsPredictionAgent:

    def __init__(self, model_path: str, openai_model: str = "gpt-4.1-mini", temperature: float = 0.7):
        self.model = joblib.load(model_path)
        self.label_map = {0: "Fake News", 1: "True News"}
        self.chat = ChatOpenAI(model_name=openai_model, temperature=temperature)


    def predict_news(self, news_item: NewsItem) -> dict:
            news_dict = news_item.model_dump()
            ground_truth = news_dict.pop("label", None)
            temp_df = pd.DataFrame([news_dict])
            X_new = preprocess_and_embed(temp_df, text_column='text')
            y_pred = self.model.predict(X_new)
            y_prob = self.model.predict_proba(X_new)[0]
            prediction = self.label_map[y_pred[0]]
            confidence = y_prob[y_pred[0]] * 100
            return {
                "Prediction": prediction,
                "Confidence": f"{confidence:.2f}%",
                "Ground Truth": "True News" if ground_truth == 1 else "Fake News" if ground_truth == 0 else None
            }

    def verify_news_with_websearch(self, news_item: NewsItem) -> VerificationResult:
        system_prompt = (
            "You are a news verification assistant. "
            "You are given a news article with title, subject, and text. "
            "Check if the news is credible using online sources. "
            "Return ONLY a JSON object with keys 'verdict' (1 or 0) and 'url' (empty string if not credible)."
        )

        user_prompt = f"""
        News item to verify:

    Title: {news_item.title}
    Subject: {news_item.subject}
    Date: {news_item.date}
    Body:
    \"\"\"{news_item.text}\"\"\"

    Return ONLY a JSON object, like:
    {{"verdict": 1, "url": "https://example.com"}} or {{"verdict": 0, "url": "Not found"}}
    """

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.chat.invoke(messages, tools=[{"type": "web_search"}])

        # Safely extract text
        if isinstance(response.content, list):
            text_output = response.content[0]['text']
        else:
            text_output = response.content

        text_output = text_output.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text_output)
            return VerificationResult(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            print("⚠️ LLM returned unexpected format, returning default:", text_output)
            return VerificationResult(verdict=0, url="")
        
        
    def decide_final_result(self, prediction: dict, verification: VerificationResult) -> dict:
            """
            Combine model prediction and web verification to produce a final verdict.
            """
            if verification.verdict == 1:
                return {
                    "Final Verdict": "True News",
                    "Source": verification.url
                }
            else:
                return {
                    "Final Verdict": prediction["Prediction"],
                    "Confidence": prediction["Confidence"],
                    "Source": None
                }