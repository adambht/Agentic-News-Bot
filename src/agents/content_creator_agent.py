import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from utils.data_validation import NewsItem


class GeneratedNews(BaseModel):
    title: str = Field(description="A complete, engaging title")
    text: str = Field(description="2-4 sentences article body")
    subject: str = Field(description="News subject/category")
    date: str = Field(description="Publication date in YYYY-MM-DD")
    label: str = Field(description="Either 'real' or 'fake'")


# Content generation tool
@tool
def generate_news_article(subject: str, date: str) -> dict:
    """
    Generate ONE news article using an LC chain that parses into JSON.
    Required args:
    - subject: category/topic to write about (e.g., "US_News", "worldnews", "AI chips").
    - date: publication date (YYYY-MM-DD).
    Returns JSON with title, text, subject, date, label ("real"|"fake").
    """
    try:
        parser = PydanticOutputParser(pydantic_object=GeneratedNews)
        format_instructions = parser.get_format_instructions()

        system = (
            "You are a professional news writer. Write realistic short news. "
            "Choose the label randomly as 'real' or 'fake' with roughly 50/50 frequency."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            (
                "human",
                "Write one news article about subject: {subject} dated {date}.\n"
                "Return ONLY valid JSON that follows these instructions:\n{format_instructions}"
            ),
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        chain = prompt | llm | parser
        parsed: GeneratedNews = chain.invoke({
            "subject": subject,
            "date": date,
            "format_instructions": format_instructions,
        })

        # Validate via NewsItem for consistency, then return dict
        label_int = 1 if parsed.label.lower() == "real" else 0
        news_item = NewsItem(title=parsed.title, text=parsed.text, subject=parsed.subject, date=parsed.date, label=label_int)
        return {
            "success": True,
            **news_item.model_dump(),
            "label": "real" if news_item.label == 1 else "fake"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Content Creator Agent
def create_content_creator_agent(llm: ChatOpenAI):
    return create_react_agent(
        llm,
        tools=[generate_news_article],
        prompt=(
            "You are a content creation agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Generate EXACTLY ONE article by calling generate_news_article(subject, date) ONCE.\n"
            "- Use the subject and date present in the user's latest message.\n"
            "- Do NOT ask follow-ups, do NOT call multiple times, do NOT perform analysis or verification.\n"
            "- After you're done, respond to the supervisor directly with ONLY the tool JSON."
        ),
        name="content_creator",
    )

# Export tools for standalone use
CONTENT_TOOLS = [generate_news_article]
