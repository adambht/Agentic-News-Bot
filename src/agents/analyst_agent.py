import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Analysis tools
@tool
def summarize_text(title: str, text: str) -> dict:
    """Summarize a news article in 2-3 sentences."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = f"""Summarize this news article in 2-3 clear sentences:

Title: {title}
Article: {text}

Provide only the summary."""
    resp = llm.invoke([{"role": "user", "content": prompt}])
    return {"success": True, "summary": resp.content.strip()}

@tool
def analyze_sentiment(title: str, text: str) -> dict:
    """Return sentiment and tone as brief bullets."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = f"""Analyze this news article:

Title: {title}
Article: {text}

Provide:
1. Sentiment: (positive/negative/neutral)
2. Tone: (objective/sensational/biased)
3. Key topics: (2-3 main themes)

Format as brief bullet points."""
    resp = llm.invoke([{"role": "user", "content": prompt}])
    return {"success": True, "analysis": resp.content.strip()}

# Analyst Agent
def create_analyst_agent(llm: ChatOpenAI):
    return create_react_agent(
        llm,
        tools=[summarize_text, analyze_sentiment],
        prompt=(
            "You are an analysis agent for summarization and sentiment.\n\n"
            "INSTRUCTIONS:\n"
            "- You MUST use tools, not free-form replies.\n"
            "- Perform EXACTLY the requested operation and nothing else.\n"
            "  * If instructed to summarize, call summarize_text once.\n"
            "  * If instructed to analyze sentiment, call analyze_sentiment once.\n"
            "  * Only call both if explicitly asked for both in the same turn.\n"
            "- Return only the tool result(s) as compact JSON. No extra prose."
        ),
        name="analyst",
    )

ANALYSIS_TOOLS = [summarize_text, analyze_sentiment]
