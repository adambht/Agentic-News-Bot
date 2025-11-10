import numpy as np
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
import random
from dotenv import load_dotenv
from utils.data_validation import NewsItem  # Pydantic model
import os

load_dotenv()

# Initialize LangChain OpenAI chat model (aligned with rest of codebase)
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def generate_single_news_structured_llm(topic: str | None = None, subject: str | None = None) -> NewsItem:
    """
    Generate one news article using LLM in a structured way (title + body + subject + date).
    - If `topic` is provided, write about that topic.
    - If `subject` is provided, use it; otherwise pick a plausible subject.
    Returns a validated NewsItem.
    """
    # Pick type, subject, and date
    news_type = random.choice(["real", "fake"])
    subjects = [
        "politicsNews", "worldnews", "News", "politics",
        "left-news", "Government News", "US_News", "Middle-east"
    ]
    final_subject = subject if subject else random.choice(subjects)
    random_days = np.random.randint(0, 5 * 365)
    date = (datetime.now() - timedelta(days=random_days)).strftime("%Y-%m-%d")

    about_text = topic if topic else final_subject

    # Build prompt
    human_prompt = f"""
You are a professional news writer. Generate a realistic {news_type} news article about {about_text}.
Return your answer in the following exact format:

Title: <a complete, engaging title>
Body: <2-4 complete sentences for the article body>
"""

    # Generate text directly via messages
    response = chat.invoke([
        {"role": "user", "content": human_prompt}
    ])
    text = response.content.strip()

    # Parse title and body
    title = ""
    body = ""
    lines = text.split("\n")
    for line in lines:
        if line.lower().startswith("title:"):
            title = line[len("Title:"):].strip()
        elif line.lower().startswith("body:"):
            body = line[len("Body:"):].strip()

    # Fallbacks
    if not title:
        title = body.split(".")[0].strip()
    if not body:
        body = text.strip()

    label = 1 if news_type == "real" else 0

    # Return Pydantic NewsItem
    return NewsItem(title=title, text=body, subject=final_subject, date=date, label=label)
