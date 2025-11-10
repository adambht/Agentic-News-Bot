import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.data_validation import NewsItem, VerificationResult
import joblib
from src.embeddings.embed_model import preprocess_and_embed
import pandas as pd 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

class NewsPredictionAgent:

    def __init__(self, model_path: str, openai_model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = joblib.load(model_path)
        self.label_map = {0: "Fake News", 1: "True News"}
        self.chat = ChatOpenAI(model=openai_model, temperature=temperature)

    def predict_news(self, news_item: NewsItem) -> dict:
        news_dict = news_item.model_dump()
        news_dict.pop("label", None)
        temp_df = pd.DataFrame([news_dict])
        X_new = preprocess_and_embed(temp_df, text_column='text')
        y_pred = self.model.predict(X_new)
        y_prob = self.model.predict_proba(X_new)[0]
        prediction = self.label_map[y_pred[0]]
        confidence = y_prob[y_pred[0]] * 100
        return {
            "Prediction": prediction,
            "Confidence": f"{confidence:.2f}%"
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

        # ✅ Tool calling is now handled with `tools=[{"type": "web_search"}]`
        response = self.chat.invoke(messages)

        if isinstance(response.content, list):
            text_output = response.content[0].get('text', '')
        else:
            text_output = response.content

        text_output = text_output.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text_output)
            return VerificationResult(**data)
        except Exception:
            print("⚠️ LLM returned unexpected format:", text_output)
            return VerificationResult(verdict=0, url="")

    def decide_final_result(self, prediction: dict, verification: VerificationResult) -> dict:
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
