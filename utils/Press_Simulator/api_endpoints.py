# src/utils/Press_Simulator/api_endpoints.py
"""
API Endpoints Configuration for the Press Conference Simulator
---------------------------------------------------------------

This file centralizes the remote endpoints used by the local Flask app
to communicate with the Kaggle backend (hosted via ngrok).

All model-heavy operations (generation, explainability, SHAP, etc.)
run on the Kaggle side — this module only defines lightweight URLs
for the local agent pipeline.

Usage:
    from src.utils.Press_Simulator.api_endpoints import (
        KAGGLE_GENERATE_API,
        KAGGLE_EXPLAIN_API,
        check_endpoints
    )

Notes:
    - If the environment variables are not set, fallback URLs will be used.
    - Update your ngrok tunnel link manually or via environment variables
      when it changes after restarting Kaggle.
"""

import os
from utils.Press_Simulator.logger import log_info


# ===============================================================
# Default fallback URLs (update if ngrok restarts)
# ===============================================================
_DEFAULT_BASE = "https://unbevelled-articularly-linn.ngrok-free.dev"

KAGGLE_GENERATE_API = os.getenv(
    "KAGGLE_GENERATE_API",
    f"{_DEFAULT_BASE}/generate"
)

KAGGLE_ANALYZE_API = os.getenv(
    "KAGGLE_ANALYZE_API",
    f"{_DEFAULT_BASE}/analyze"
)


KAGGLE_EXPLAIN_API = os.getenv(
    "KAGGLE_EXPLAIN_API",
    f"{_DEFAULT_BASE}/explain"
)


# ===============================================================
# Helper Function
# ===============================================================
def check_endpoints() -> None:
    """
    Prints the current Kaggle backend endpoints to verify connectivity.
    Useful to call once when launching the Flask app.
    """
    log_info("🔗 Kaggle backend endpoints in use:")
    log_info(f"   • GENERATE: {KAGGLE_GENERATE_API}")
    log_info(f"   • EXPLAIN:  {KAGGLE_EXPLAIN_API}")


# ===============================================================
# Optional: run quick test
# ===============================================================
if __name__ == "__main__":
    check_endpoints()
