# src/agents/Press_Conf_Simulator/app.py

from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from src.agents.Press_Conf_Simulator.press_conference_agent import press_conference_agent

# --------------------------------------------------------------------
# 1️⃣ App setup
# --------------------------------------------------------------------
app = Flask(__name__, template_folder="templates/Press_Conf_Simulator")
app.secret_key = "super-secret-session-key"  # Use a strong random string in prod
CORS(app)

# Build once; reuse across sessions
graph = press_conference_agent()

# --------------------------------------------------------------------
# 2️⃣ Routes
# --------------------------------------------------------------------

@app.route('/')
def home():
    """Serve the Press Conference Simulator frontend."""
    return render_template('index_Press_Conf_Simulator.html')


@app.route('/start', methods=['POST'])
def start():
    """Initialize a new interview session with persona and topic."""
    data = request.get_json(force=True)
    print("=" * 80)
    print("🎬 Starting new press conference session")
    print(f"Persona: {data.get('persona')}, Topic: {data.get('topic')}, Role: {data.get('role')}")
    print("=" * 80)

    # Initialize the conversation state
    state = {
        "persona": data.get("persona", "investigative_hawk"),
        "topic": data.get("topic", ""),
        "role": data.get("role", "guest"),
        "speech": data.get("speech", ""),
        "history": []
    }

    # Run through the LangGraph pipeline
    result = graph.invoke(state)
    question = result.get("journalist_question", "[No question generated]")

    # Store conversation history in session
    state["history"].append({"role": "journalist", "content": question})
    session["state"] = state

    print(f"🗞️ First question: {question}")
    return jsonify({"question": question})


@app.route('/reply', methods=['POST'])
def reply():
    """Handle user (guest) responses and trigger next journalist question."""
    user_answer = request.get_json(force=True).get("answer", "")
    state = session.get("state", {})

    # Log and update conversation
    print("=" * 80)
    print("🗣️ Guest reply:", user_answer)
    print("=" * 80)

    state.setdefault("history", [])
    state["history"].append({"role": "guest", "content": user_answer})

    # Run next iteration
    result = graph.invoke(state)
    question = result.get("journalist_question", "[No question generated]")

    # Append journalist response and save state
    state["history"].append({"role": "journalist", "content": question})
    session["state"] = state

    print(f"🎤 Journalist asks: {question}")
    return jsonify({"question": question})


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the entire conversation session."""
    session.clear()
    print("🔄 Conversation reset.")
    return jsonify({"message": "Session reset."})


# --------------------------------------------------------------------
# 3️⃣ Run server
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
