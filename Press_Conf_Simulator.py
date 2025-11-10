# Press_Conf_Simulator.py
"""
Flask Application - Press Conference Simulator
----------------------------------------------

This server connects the HTML front-end with the local LangGraph agent
that orchestrates communication with the Kaggle backend via ngrok.

Routes:
    /               -> Frontend (HTML interface)
    /start          -> Starts a new interview (first journalist question)
    /reply          -> Handles user (guest) response and generates next question
    /reset          -> Clears the current session

All heavy inference runs on Kaggle — this Flask app only coordinates state.
"""

from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from src.agents.Press_Conf_Simulator.press_conference_agent import press_conference_agent
from utils.Press_Simulator.logger import log_info, log_warning


# ===============================================================
# 1️⃣ App setup
# ===============================================================
app = Flask(__name__, template_folder="templates/Press_Conf_Simulator")
app.secret_key = "super-secret-session-key"  # Replace for prod
CORS(app)

# Initialize LangGraph pipeline once
graph = press_conference_agent()
log_info("✅ LangGraph pipeline initialized.")


# ===============================================================
# 2️⃣ Routes
# ===============================================================
@app.route("/")
def home():
    """Serve the Press Conference Simulator frontend."""
    return render_template("index_Press_Conf_Simulator.html")


@app.route("/start", methods=["POST"])
def start():
    """Initialize a new interview session."""
    data = request.get_json(force=True)
    persona = data.get("persona", "investigative_hawk")
    topic = data.get("topic", "")
    role = data.get("role", "CEO")
    speech = data.get("speech", "")

    log_info("🎬 Starting new Press Conference Session")
    log_info(f"Persona={persona}, Topic={topic}, Role={role}")

    # Initialize agent state
    state = {
        "persona": persona,
        "topic": topic,
        "role": role,
        "speech": speech,
        "history": [],
    }

    # Run LangGraph pipeline (1st journalist question)
    result = graph.invoke(state)
    question = result.get("journalist_question", "[No question generated]")
    explanation = result.get("explanation", "")

    # Save conversation in session
    state["history"].append({"role": "journalist", "content": question})
    session["state"] = state

    log_info(f"🗞️ First question: {question}")
    return jsonify({"question": question, "explanation": explanation})


@app.route("/reply", methods=["POST"])
def reply():
    """Handle the guest's response and trigger the next question."""
    user_answer = request.get_json(force=True).get("answer", "").strip()
    if not user_answer:
        return jsonify({"error": "Empty response"}), 400

    state = session.get("state", {})
    if not state:
        log_warning("⚠️ No active session found.")
        return jsonify({"error": "No active session"}), 400

    # Log and update conversation
    log_info(f"🗣️ Guest reply: {user_answer}")
    state.setdefault("history", [])
    state["history"].append({"role": "guest", "content": user_answer})

    # Run next journalist question
    result = graph.invoke(state)
    question = result.get("journalist_question", "[No question generated]")
    explanation = result.get("explanation", "")

    # Update session
    state["history"].append({"role": "journalist", "content": question})
    session["state"] = state

    log_info(f"🎤 Journalist asks: {question}")
    return jsonify({"question": question, "explanation": explanation})


@app.route("/stop", methods=["POST"])
def stop():
    """Analyze the whole conversation when the user stops."""
    state = session.get("state", {})
    if not state:
        return jsonify({"error": "No active session"}), 400

    from src.agents.Press_Conf_Simulator.press_conference_agent import analysis_api_node
    result = analysis_api_node(state)
    analysis = result.get("analysis", {})

    session.clear()
    return jsonify({"analysis": analysis})


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the current interview session."""
    session.clear()
    log_info("🔄 Session reset by user.")
    return jsonify({"message": "Session reset."})


# ===============================================================
# 3️⃣ Run server
# ===============================================================
if __name__ == "__main__":
    log_info("🚀 Press Conference Simulator running on http://127.0.0.1:7860")
    app.run(host="0.0.0.0", port=7860, debug=True)
