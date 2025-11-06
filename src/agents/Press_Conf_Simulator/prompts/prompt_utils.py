# src/agents/Press_Conf_Simulator/prompts/prompt_utils.py
"""
Prompt Utility Functions for the Press Conference Simulator
------------------------------------------------------------

Handles dynamic prompt creation:
- Summarizes the dialogue history to keep context compact.
- Builds the user-side prompt for Mistral (speech + recent turns).
- Maintains coherence and persona alignment across turns.

Usage:
    from src.agents.Press_Conf_Simulator.prompts.prompt_utils import (
        summarize_history, build_user_prompt
    )
"""

from typing import List, Dict


# ===============================================================
# Dialogue Summarization
# ===============================================================
def summarize_history(turns: List[Dict[str, str]], max_chars: int = 800) -> str:
    """
    Compresses the dialogue history to a short, readable summary.

    Args:
        turns: A list of {role, content} dictionaries.
        max_chars: Maximum character length to include.

    Returns:
        A short bullet-point style text summarizing the latest exchanges.
    """
    if not turns:
        return "Aucun échange précédent."

    # Keep last 6 exchanges max for brevity
    recent_turns = turns[-6:]
    bullets = []
    for turn in recent_turns:
        role = "Journalist" if turn["role"] == "journalist" else "Guest"
        bullets.append(f"- {role}: {turn['content']}")

    text = "\n".join(bullets)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


# ===============================================================
# Dynamic User Prompt Construction
# ===============================================================
def build_user_prompt(topic: str, role: str, opening_speech: str, history_summary: str) -> str:
    """
    Builds the 'user' message that Mistral receives each turn.

    Args:
        topic: Main press conference topic.
        role: Role of the guest (e.g., CEO, Minister).
        opening_speech: The initial speech or statement.
        history_summary: Condensed dialogue summary from summarize_history().

    Returns:
        A formatted string representing the user input context.
    """
    return f"""
Contexte de la conférence :
- Sujet : {topic}
- Interlocuteur (guest) : {role}

Discours d'ouverture (référence constante) :
\"\"\"{opening_speech}\"\"\"

Résumé des derniers échanges :
{history_summary}

Tâche :
1. Identifier les points encore ambigus ou peu explorés.
2. Décider s’il faut relancer sur le discours initial ou sur la réponse du guest.
3. Poser UNE nouvelle question utile et cohérente.
4. La question doit etre OBLIGATOIREMENT entre : <QUESTION> .... <eoa>
"""


# ===============================================================
# Example usage (debug)
# ===============================================================
if __name__ == "__main__":
    history = [
        {"role": "journalist", "content": "Pouvez-vous préciser les tests réalisés ?"},
        {"role": "guest", "content": "Nous collaborons avec l'OMS pour les validations."},
    ]
    summary = summarize_history(history)
    prompt = build_user_prompt("IA en santé", "CEO", "Nous avons lancé un nouveau modèle d'IA.", summary)
    print(summary)
    print(prompt)
