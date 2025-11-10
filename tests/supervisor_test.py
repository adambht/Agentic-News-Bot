import os
import sys
from datetime import date as _date
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agent import supervisor, default_config

# Simple pretty printer similar to course_helper_functions.pretty_print_messages
def pretty_print_messages(chunk):
    for node, payload in chunk.items():
        if node.startswith("__"):
            continue
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not messages:
            continue
        last = messages[-1]
        role = getattr(last, "type", getattr(last, "role", ""))
        content = getattr(last, "content", str(last))
        print(f"\n[{node}] {role}: {content}")

# Provide both subject and date for the content creator tool
_today = _date.today().strftime("%Y-%m-%d")
prompt = (
    f"Generate one article with subject 'AI chips' and date {_today}. "
    "Then summarize it, analyze sentiment, and verify authenticity."
)

print("=== Streaming run (step-by-step) ===")
final = None
for chunk in supervisor.stream({"messages": [{"role": "user", "content": prompt}]}, config=default_config):
    pretty_print_messages(chunk)
    if "__end__" in chunk:
        final = chunk["__end__"]

# Fall back to non-streaming if needed
if final is None:
    final = supervisor.invoke({"messages": [("user", prompt)]}, config=default_config)

print("\n=== Final result ===")
last_msg = final["messages"][-1]
print(getattr(last_msg, "content", last_msg))
