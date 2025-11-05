# src/agents/Press_Conf_Simulator/journalist_nodes.py
from src.agents.Press_Conf_Simulator import PERSONAS

def build_prompt_node(state):
    """
    Builds a high-quality, instruction-tuned prompt for the journalist agent.
    Combines role prompting, contextual grounding, and explicit constraints.
    """
    # Extract state variables
    persona = PERSONAS[state.get("persona", "investigative_hawk")]
    topic = state.get("topic", "")
    role = state.get("role", "guest")
    speech = state.get("speech", "")
    history = state.get("history", [])

    # --- Format conversation history ---
    conversation = ""
    for turn in history:
        who = "Guest" if turn["role"] == "guest" else persona["name"]
        conversation += f"{who}: {turn['content']}\n"

    # --- Build prompt using best-practice structure ---
    prompt = f"""
    ### ROLE PROMPTING
    You are **{persona['name']}**, a professional journalist.
    Your characterization: {persona['description']}
    
    ### CONTEXT
    Press conference topic: **{topic}**
    Guest type: **{role}**
    Opening statement:
    \"\"\"{speech}\"\"\"
    
    Ongoing dialogue transcript:
    \n{conversation.strip()}\n
    
    ### CONSTRAINTS
    - Ask **only one** concise, sharp, and contextually relevant question.
    - Stay aligned with your persona’s tone.
    - Do **not** answer on behalf of the guest.
    - Avoid repetition or generic questions.
    - Base your question strictly on the dialogue and factual context above.
    
    ### THINKING (internal, invisible)
    Before producing your question, briefly reason about:
    1. What new, valuable angle has not been covered?
    2. What factual or ethical gap can be exposed?
    3. How to phrase the question to elicit meaningful detail?
    Then output **only** the final question.
    
    ### OUTPUT
    Produce the next question:
    
    """

    state["prompt"] = prompt.strip()
    return state
