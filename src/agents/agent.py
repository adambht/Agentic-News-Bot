import os
import sys

# Ensure project root is on sys.path for imports like `utils.*`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
try:
    # Prefer user's import style if available
    from langgraph_supervisor import create_supervisor  # type: ignore
except ImportError:  # fallback for environments without `langgraph_supervisor`
    from langgraph.prebuilt import create_supervisor  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver

# Import per-agent factories
from src.agents.content_creator_agent import create_content_creator_agent
from src.agents.analyst_agent import create_analyst_agent
from src.agents.detector_agent import create_detector_agent

# Shared LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Instantiate agents
content_creator_agent = create_content_creator_agent(llm)
analysis_agent = create_analyst_agent(llm)
detector_agent = create_detector_agent(llm)

# Supervisor graph
checkpointer = InMemorySaver()

def build_supervisor(thread_id: str = "1", user_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    supervisor = create_supervisor(
        model=llm,
        agents=[content_creator_agent, analysis_agent, detector_agent],
        prompt=(
            "You are a supervisor orchestrating three agents. Follow these rules strictly:\n"
            "1) Obey the user's explicit intent. Do NOT proactively chain tasks.\n"
            "   - If the user asks to GENERATE only, call content_creator once and STOP.\n"
            "   - If the user asks to SUMMARIZE or SENTIMENT, operate on the most recently generated article; do not regenerate.\n"
            "   - If the user asks to VERIFY, operate on the most recently generated article; do not regenerate.\n"
            "2) Never call more than ONE agent per user message. No loops, no repeats.\n"
            "3) Reuse prior results in this thread. If an article already exists, do not call content_creator again unless the user requests a new article.\n"
            "4) If the user's intent is 'summarize', hand off to analyst with instruction to ONLY summarize (no sentiment).\n"
            "   If the intent is 'sentiment', hand off to analyst with instruction to ONLY analyze sentiment (no summary).\n"
            "   Only request both if the user explicitly asks for both.\n"
            "5) Assign work to one agent at a time; do NOT call agents in parallel. Do not do any work yourself.\n"
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer)

    return supervisor, config

# Default, ready-to-use compiled supervisor and a default config
supervisor, default_config = build_supervisor()
