"""
Tools for Multi-Agent System

Organized into three categories:
- Content Tools: News generation
- Analysis Tools: Summarization and sentiment analysis
- Verification Tools: Fake news detection (ML + Web Search)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import joblib
import pandas as pd
import json

from utils.data_validation import NewsItem
from src.embeddings.embed_model import preprocess_and_embed
from utils.simulation_helpers import generate_single_news_structured_llm


# ============================================================================
# CONTENT GENERATION TOOLS
# ============================================================================

@tool
def generate_news_article(topic: str = "", subject: str = "") -> dict:
    """
    Generate ONE news article (can be real or fake).
    Optional args:
    - topic: what the article should be about (e.g., "AI chips").
    - subject: category (e.g., "US_News", "worldnews").
    Returns JSON with title, text, subject, date, label.
    """
    try:
        t = topic.strip() or None
        s = subject.strip() or None
        news_item = generate_single_news_structured_llm(topic=t, subject=s)
        return {
            "success": True,
            "title": news_item.title,
            "text": news_item.text,
            "subject": news_item.subject,
            "date": news_item.date,
            "label": "real" if news_item.label == 1 else "fake"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ANALYSIS TOOLS
# ============================================================================

@tool
def summarize_text(title: str, text: str) -> dict:
    """
    Summarize a news article into 2-3 concise sentences.
    
    Args:
        title: The news article title
        text: The full news article text
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt = f"""Summarize this news article in 2-3 clear sentences:

Title: {title}
Article: {text}

Provide only the summary, no extra text."""
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {
            "success": True,
            "summary": response.content.strip()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def analyze_sentiment(title: str, text: str) -> dict:
    """
    Analyze the sentiment and tone of a news article.
    
    Args:
        title: The news article title
        text: The full news article text
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt = f"""Analyze this news article:

Title: {title}
Article: {text}

Provide:
1. Sentiment: (positive/negative/neutral)
2. Tone: (objective/sensational/biased)
3. Key topics: (2-3 main themes)

Format as brief bullet points."""
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {
            "success": True,
            "analysis": response.content.strip()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# VERIFICATION TOOLS (FAKE NEWS DETECTION)
# ============================================================================

# Load ML model once at module level
ML_MODEL = joblib.load("src/models/logisticRegressor.pkl")
LABEL_MAP = {0: "Fake News", 1: "True News"}


@tool
def ml_predict_news(title: str, subject: str, date: str, text: str) -> dict:
    """
    Use machine learning to predict if news is fake or true.
    
    Args:
        title: News article title
        subject: News category/subject
        date: Publication date
        text: Full article text
    """
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
    """
    Verify news credibility by searching the web for sources.
    Use this to check if the news appears on credible websites.
    
    Args:
        title: News article title
        text: Article text (first 500 chars will be used)
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        system_prompt = """You are a fact-checker with web search capabilities.
Search for credible sources about this news.
Return ONLY valid JSON: {"verified": true/false, "url": "source_url_or_empty", "reasoning": "brief_explanation"}"""
        
        user_prompt = f"""Verify this news:

Title: {title}
Text: {text[:500]}...

Search the web and return ONLY JSON format."""
        
        response = llm.invoke(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            tools=[{"type": "web_search"}]
        )
        
        # Parse response
        text_output = str(response.content).replace("```json", "").replace("```", "").strip()
        data = json.loads(text_output)
        
        return {
            "success": True,
            "verified": data.get("verified", False),
            "source_url": data.get("url", ""),
            "reasoning": data.get("reasoning", ""),
            "method": "Web Search"
        }
    except Exception as e:
        return {
            "success": False,
            "verified": False,
            "error": str(e)
        }


# ============================================================================
# TOOL GROUPS (for easy import)
# ============================================================================

CONTENT_TOOLS = [generate_news_article]
ANALYSIS_TOOLS = [summarize_text, analyze_sentiment]
VERIFICATION_TOOLS = [ml_predict_news, web_search_verify]
ALL_TOOLS = CONTENT_TOOLS + ANALYSIS_TOOLS + VERIFICATION_TOOLS