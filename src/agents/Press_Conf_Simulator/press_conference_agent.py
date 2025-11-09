# src/agents/Press_Conf_Simulator/press_conference_agent.py
"""
Press Conference Agent Graph
----------------------------

This module defines the LangGraph pipeline that runs one full
Press Conference turn:
    build_prompt → mistral_query → explainability → END

All model inference and explainability are executed remotely
on Kaggle (via ngrok). The local graph only orchestrates requests
and maintains conversation state.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import requests, json
from src.agents.Press_Conf_Simulator.journalist_nodes import build_prompt_node
from utils.Press_Simulator.api_endpoints import KAGGLE_GENERATE_API, KAGGLE_EXPLAIN_API
from utils.Press_Simulator.logger import log_info, log_error, log_warning


# ===============================================================
# State Schema
# ===============================================================
class AgentState(TypedDict, total=False):
    persona: str
    topic: str
    role: str
    speech: str
    history: list
    messages: list
    journalist_question: str
    explanation: str


# ===============================================================
# 1️⃣ Query the Kaggle backend for journalist question
# ===============================================================
def _extract_question(text: str) -> str:
    """
    Extracts the journalist's question between the LAST <QUESTION> and <eoa> tags.
    If <eoa> is missing, returns everything after the last <QUESTION>.
    Returns '[Empty model output]' if nothing valid is found.
    """
    import re
    if not text:
        return "[Empty model output]"

    # --- 1️⃣ Try to find complete <QUESTION> ... <eoa> blocks ---
    matches = re.findall(r"<QUESTION>(.*?)<eoa>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # --- 2️⃣ Handle the case where <QUESTION> exists but no <eoa> follows ---
    start_match = re.search(r"<QUESTION>(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if start_match:
        # Take everything after the last <QUESTION> tag
        body = start_match.group(1).strip()
        return body

    # --- 3️⃣ Last resort: return a fallback preview of text ---
    return text.strip()[:500]





def mistral_query_node(state: AgentState) -> AgentState:
    """Sends the prepared messages to the Kaggle backend for generation."""
    messages = state.get("messages", [])
    if not messages:
        log_error("No messages to send to Kaggle backend.")
        state["journalist_question"] = "[Prompt missing]"
        return state

    try:
        log_info("🚀 Sending prompt to Kaggle backend...")
        res = requests.post(KAGGLE_GENERATE_API, json={"messages": messages}, timeout=2000)
        log_info(f"🌐 Status: {res.status_code}")

        data = res.json()
        raw = data.get("response", "")
        question = _extract_question(raw)
        state["journalist_question"] = question
        log_info(f"🗞️ Journalist question: {question}")

    except Exception as e:
        log_error(f"❌ Error contacting Kaggle backend: {e}")
        state["journalist_question"] = f"[Backend error: {e}]"

    return state



# ===============================================================
# 2️⃣ Call Kaggle explainability endpoint
# ===============================================================
def explainability_api_node(state: AgentState) -> AgentState:
    """Calls Kaggle backend to compute SHAP/semantic/attention explanations."""
    question = state.get("journalist_question", "")
    speech = state.get("speech", "")
    if not question or not speech:
        log_warning("Insufficient data for explainability.")
        state["explanation"] = "No data for explainability."
        return state

    log_info("🧩 Running explainability modes on Kaggle backend...")

    try:
        payload = {"speech": speech, "question": question, "mode": "shap"}
        res = requests.post(KAGGLE_EXPLAIN_API, json=payload, timeout=2000)
        data = res.json()
        explanation = data.get("explanation", "No explanation returned.")
        state["explanation"] = explanation
        log_info(f"✅ Explainability (SHAP): {explanation[:120]}...")

    except Exception as e:
        log_error(f"❌ Error during explainability: {e}")
        state["explanation"] = f"[Explainability error: {e}]"

    return state




# ===============================================================
# 3️⃣ Build LangGraph pipeline
# ===============================================================
def press_conference_agent():
    """Compiles and returns the Press Conference LangGraph pipeline."""
    g = StateGraph(AgentState)

    g.add_node("build_prompt", build_prompt_node)
    g.add_node("mistral_query", mistral_query_node)
    g.add_node("explain", explainability_api_node)

    g.set_entry_point("build_prompt")
    g.add_edge("build_prompt", "mistral_query")
    g.add_edge("mistral_query", "explain")
    g.add_edge("explain", END)

    log_info("🧱 Press Conference Graph compiled successfully.")
    return g.compile()
