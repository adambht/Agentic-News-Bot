# src/agents/Press_Conf_Simulator/journalist_nodes.py
"""
Journalist Prompt Node for the Press Conference Simulator
----------------------------------------------------------

This node prepares the full Mistral input messages based on:
- persona (journalist style)
- topic and guest role
- opening speech and dialogue history

Output:
    state["messages"] = [
        {"role": "system", "content": ...},
        {"role": "user", "content": ...}
    ]
"""

from typing import Dict, Any
from src.agents.Press_Conf_Simulator.prompts.system_prompts import get_system_prompt
from src.agents.Press_Conf_Simulator.prompts.prompt_utils import summarize_history, build_user_prompt
from utils.Press_Simulator.logger import log_info


def build_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the complete input messages for Mistral based on the agent's state.
    """
    persona = state.get("persona", "investigative_hawk")
    topic = state.get("topic", "")
    role = state.get("role", "CEO")
    speech = state.get("speech", "")
    history = state.get("history", [])

    log_info(f"🧠 Building prompt for persona='{persona}', topic='{topic}'")

    # --- Summarize previous conversation ---
    history_summary = summarize_history(history)

    # --- Construct prompts ---
    system_prompt = get_system_prompt(persona, topic, role)
    user_prompt = build_user_prompt(topic, role, speech, history_summary)

    # --- Prepare chat-style messages ---
    state["messages"] = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    # --- (Optional) Keep raw text for debugging ---
    state["prompt_preview"] = f"{system_prompt}\n\n{user_prompt[:400]}..."

    return state


# ===============================================================
# Example usage (debug)
# ===============================================================
if __name__ == "__main__":
    state = {
        "persona": "tech_policy",
        "topic": "AI in healthcare",
        "role": "CEO",
        "speech": "Our new AI model improves diagnostic accuracy.",
        "history": [
            {"role": "journalist", "content": "Is this compliant with EU medical standards?"},
            {"role": "guest", "content": "We are currently working on certification."}
        ]
    }

    updated_state = build_prompt_node(state)
    print(updated_state["messages"][0]["content"][:300])
    print("\n---\n")
    print(updated_state["messages"][1]["content"][:300])
