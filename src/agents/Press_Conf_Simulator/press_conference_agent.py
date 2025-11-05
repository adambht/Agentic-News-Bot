# src/agents/Press_Conf_Simulator/press_conference_agent.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from src.agents.Press_Conf_Simulator.journalist_nodes import build_prompt_node
import requests, os, json

# --- Define the state schema ---
class AgentState(TypedDict, total=False):
    persona: str
    topic: str
    role: str
    speech: str
    prompt: str
    journalist_question: str
    explanation: str


# --- Kaggle endpoints (edit with your live ngrok URLs) ---
KAGGLE_GENERATE_API = os.getenv(
    "KAGGLE_GENERATE_API",
    "https://unbevelled-articularly-linn.ngrok-free.dev/generate"  # <-- update this
)
KAGGLE_EXPLAIN_API = os.getenv(
    "KAGGLE_EXPLAIN_API",
    "https://unbevelled-articularly-linn.ngrok-free.dev/explain"   # <-- update this
)

# --- 1️⃣ Query the Kaggle backend for question generation ---
def mistral_query_node(state: AgentState) -> AgentState:
    prompt = state.get("prompt", "")
    print("=" * 80)
    print("🚀 Sending prompt to Kaggle backend:")
    print(prompt[:400])
    print("=" * 80)

    try:
        # Send the prompt to the Kaggle backend
        res = requests.post(KAGGLE_GENERATE_API, json={"prompt": prompt}, timeout=90)
        print("🌐 Backend status:", res.status_code)
        print("🌐 Raw Kaggle response:", res.text[:300])

        try:
            # Parse JSON response
            data = res.json()
            raw = data.get("response", "")

            # 🧹 Clean the output and extract the question
            if "?" in raw:
                # Try to isolate the question after structured markers
                question = raw.split("### OUTPUT")[-1].split("\n")[-1].strip('" ')
                if not question.strip():
                    question = raw.split("?")[0] + "?"
            else:
                question = raw.strip()

            state["journalist_question"] = question

        except Exception as e:
            state["journalist_question"] = f"[Error decoding Kaggle response: {e}]"

    except Exception as e:
        print("❌ ERROR contacting Kaggle backend:", e)
        state["journalist_question"] = f"[Error contacting backend: {e}]"

    return state



# --- 2️⃣ Call Kaggle explainability endpoint ---
def explainability_api_node(state: AgentState) -> AgentState:
    question = state.get("journalist_question", "")
    speech = state.get("speech", "")

    modes = ["semantic", "attention", "shap", "lime"]  # run all modes sequentially
    print("=" * 80)
    print(f"🧩 Running explainability for all modes on Kaggle")
    print(f"Speech: {speech[:100]}...")
    print(f"Question: {question}")
    print("=" * 80)

    explanations = {}

    for mode in modes:
        try:
            payload = {"speech": speech, "question": question, "mode": mode}
            res = requests.post(KAGGLE_EXPLAIN_API, json=payload, timeout=120)
            data = res.json()
            explanations[mode] = data.get("explanation", "No output")
            print(f"[{mode.upper()}] {explanations[mode]}")
        except Exception as e:
            explanations[mode] = f"Error: {e}"
            print(f"[{mode.upper()}] ❌ Error calling explainability API: {e}")

    # Only keep SHAP (or nothing) visible to frontend if you wish
    state["explanation"] = explanations.get("shap", "")
    return state


# --- 3️⃣ Full pipeline graph ---
def press_conference_agent():
    g = StateGraph(AgentState)

    g.add_node("build_prompt", build_prompt_node)
    g.add_node("mistral_query", mistral_query_node)
    g.add_node("explain", explainability_api_node)

    g.set_entry_point("build_prompt")
    g.add_edge("build_prompt", "mistral_query")
    g.add_edge("mistral_query", "explain")
    g.add_edge("explain", END)

    app = g.compile()
    return app
