import os
import sys
import uuid
from datetime import date as _date

import streamlit as st

# Ensure project root on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.agents.agent import supervisor  # noqa: E402

st.set_page_config(page_title="Agentic News Bot", page_icon="🗞️", layout="centered")
st.title("🗞️ Agentic News Bot — Chat")
st.caption("Conversational multi-agent UI. Generate first, then ask to summarize, analyze sentiment, or verify. See routing in the trace.")

# Session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {role, content}

with st.sidebar:
    st.header("Generation parameters")
    subject = st.text_input("Subject (required for generation)", value="AI chips")
    sel_date = st.date_input("Date (required for generation)", value=_date.today())
    st.markdown("- Tip: first say something like ‘generate an article’.\n- Then you can say ‘summarize’, ‘sentiment’, or ‘verify’.")
    if st.button("Reset conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat = []
        st.experimental_rerun()

# Render history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_msg = st.chat_input("Type a message (e.g., 'generate an article', then 'summarize')")

def _compose_user_content(raw: str) -> str:
    text = raw.strip()
    lower = text.lower()
    # If user intends generation and provided subject/date, append structured hints
    if any(k in lower for k in ["generate", "create"]) and any(k in lower for k in ["article", "news"]):
        if subject and sel_date:
            date_str = sel_date.strftime("%Y-%m-%d")
            text += f"\n\nSubject: {subject}\nDate: {date_str}"
    return text

if user_msg:
    # Add user message to history
    composed = _compose_user_content(user_msg)
    st.session_state.chat.append({"role": "user", "content": composed})

    # Stream through supervisor with a persistent thread
    cfg = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": "web"}}
    with st.chat_message("assistant"):
        placeholder = st.empty()
        trace = st.expander("Routing trace", expanded=False)
        final = None
        run_lines: list[str] = []  # reset per user turn
        try:
            for chunk in supervisor.stream({"messages": st.session_state.chat}, config=cfg):
                # update trace live only for this turn
                for node, payload in chunk.items():
                    if node.startswith("__") or not isinstance(payload, dict):
                        continue
                    msgs = payload.get("messages")
                    if not msgs:
                        continue
                    last = msgs[-1]
                    role = getattr(last, "type", getattr(last, "role", ""))
                    content = getattr(last, "content", str(last))
                    run_lines.append(f"[{node}] {role}: {content}")
                trace.write("\n\n".join(run_lines))
                if "__end__" in chunk:
                    final = chunk["__end__"]
            # Render latest assistant content
            if final is not None:
                last = final["messages"][-1]
                assistant_text = getattr(last, "content", str(last))
                placeholder.markdown(assistant_text)
                st.session_state.chat.append({"role": "assistant", "content": assistant_text})
        except Exception as e:
            placeholder.error(f"Error: {e}")
            st.stop()
