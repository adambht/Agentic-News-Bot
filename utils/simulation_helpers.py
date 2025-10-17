import numpy as np
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import random
from dotenv import load_dotenv
from utils.data_validation import NewsItem  # import Pydantic model

import os

load_dotenv()

# Initialize LangChain OpenAI chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def generate_single_news_structured_llm() -> NewsItem:
    """
    Generate one news article using LLM in a structured way (with subject).
    Returns a validated NewsItem.
    """
    news_type = random.choice(["real", "fake"])
    subjects = [
        "politicsNews", "worldnews", "News", "politics",
        "left-news", "Government News", "US_News", "Middle-east"
    ]
    subject = random.choice(subjects)

    random_days = np.random.randint(0, 5 * 365)
    date = (datetime.now() - timedelta(days=random_days)).strftime("%Y-%m-%d")

    # Build prompt
    human_prompt = f"Write a realistic {news_type} news article in 2-3 sentences about {subject}."
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{text}")
    ])
    formatted_prompt = prompt.format_messages(text=human_prompt)

    # Generate text
    response = chat.invoke(formatted_prompt)
    text = response.content.strip()

    # Title from first sentence (up to 80 chars)
    title = text.split(".")[0][:80]
    label = 1 if news_type == "real" else 0


    # Return Pydantic NewsItem
    return NewsItem(title=title, text=text, subject=subject, date=date, label=label)
