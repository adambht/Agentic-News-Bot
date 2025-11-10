import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import joblib
import pandas as pd
from src.embeddings.embed_model import preprocess_and_embed

# Load ML model once (used by the tool)
MODEL_PATH = "src/models/logisticRegressor.pkl"
ML_MODEL = joblib.load(MODEL_PATH)
LABEL_MAP = {0: "Fake News", 1: "True News"}

@tool
def ml_predict_news(title: str, subject: str, date: str, text: str) -> dict:
    """Use logistic regression to predict if news is fake or true."""
    try:
        news_dict = {"title": title, "subject": subject, "date": date, "text": text}
        temp_df = pd.DataFrame([news_dict])
        X = preprocess_and_embed(temp_df, text_column='text')
        y_pred = ML_MODEL.predict(X)
        y_prob = ML_MODEL.predict_proba(X)[0]
        prediction = LABEL_MAP[y_pred[0]]
        confidence = float(y_prob[y_pred[0]] * 100)
        return {
            "success": True,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "method": "ML Model"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def web_search_verify(title: str, text: str) -> dict:
    """Ask the LLM to infer verification (no actual browsing in this tool)."""
    from langchain_openai import ChatOpenAI
    import json
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system_prompt = (
        "You are a fact-checker. You cannot browse. Infer credibility from the text and return ONLY JSON: "
        '{"verified": true/false, "url": "source_url_or_empty", "reasoning": "brief"}'
    )
    user_prompt = f"Title: {title}\nText: {text[:500]}...\nReturn only JSON."
    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    try:
        raw = str(resp.content).replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {
            "success": True,
            "verified": bool(data.get("verified", False)),
            "source_url": data.get("url", ""),
            "reasoning": data.get("reasoning", ""),
            "method": "LLM Inference"
        }
    except Exception as e:
        return {"success": False, "verified": False, "error": str(e)}

# Detector Agent

def create_detector_agent(llm: ChatOpenAI):
    return create_react_agent(
        llm,
        tools=[ml_predict_news, web_search_verify],
        prompt=(
            "You are a verification agent that detects fake news.\n\n"
            "INSTRUCTIONS:\n"
            "- Operate on ONE article ({title, subject, date, text}).\n"
            "- Call ml_predict_news exactly once. If confidence < 85%, optionally call web_search_verify once.\n"
            "- Return a JSON: { prediction, confidence, method, optionally: verified, source_url, reasoning }.\n"
            "- No extra prose."
        ),
        name="detector",
    )

VERIFICATION_TOOLS = [ml_predict_news, web_search_verify]
