import numpy as np
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import random
from dotenv import load_dotenv
from utils.data_validation import NewsItem  # Pydantic model
import os

load_dotenv()

# Initialize LangChain OpenAI chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def generate_single_news_structured_llm() -> NewsItem:
    """
    Generate one news article using LLM in a structured way (title + body + subject + date).
    Returns a validated NewsItem.
    """
    # Pick type, subject, and date
    news_type = random.choice(["real", "fake"])
    subjects = [
        "politicsNews", "worldnews", "News", "politics",
        "left-news", "Government News", "US_News", "Middle-east"
    ]
    subject = random.choice(subjects)
    random_days = np.random.randint(0, 5 * 365)
    date = (datetime.now() - timedelta(days=random_days)).strftime("%Y-%m-%d")

    # Build prompt
    human_prompt = f"""
You are a professional news writer. Generate a realistic {news_type} news article about {subject}.
Return your answer in the following exact format:

Title: <a complete, engaging title>
Body: <2-4 complete sentences for the article body>
"""

    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{text}")
    ])
    formatted_prompt = prompt.format_messages(text=human_prompt)

    # Generate text
    response = chat.invoke(formatted_prompt)
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
    return NewsItem(title=title, text=body, subject=subject, date=date, label=label)
