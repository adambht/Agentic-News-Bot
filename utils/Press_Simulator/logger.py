# src/utils/Press_Simulator/logger.py
"""
Lightweight Logger Utility for Press Conference Simulator
---------------------------------------------------------

This logger provides consistent, colored console output across
the local Flask app and Kaggle backend requests.

Usage:
    from src.utils.Press_Simulator.logger import (
        log_info, log_warning, log_error
    )

Example:
    log_info("Starting Flask server...")
    log_warning("Response took longer than expected.")
    log_error("Failed to connect to Kaggle backend.")
"""

import sys

# ANSI color codes for readability (auto-disabled if output not a terminal)
_COLOR = sys.stdout.isatty()

def _colorize(text: str, color_code: str) -> str:
    if not _COLOR:
        return text
    return f"\033[{color_code}m{text}\033[0m"

# ===============================================================
# Core logging helpers
# ===============================================================
def log_info(message: str) -> None:
    """Prints informational messages (blue color)."""
    print(_colorize(f"[INFO] {message}", "34"))

def log_warning(message: str) -> None:
    """Prints warning messages (yellow color)."""
    print(_colorize(f"[WARN] {message}", "33"))

def log_error(message: str) -> None:
    """Prints error messages (red color)."""
    print(_colorize(f"[ERROR] {message}", "31"), file=sys.stderr)

def log_success(message: str) -> None:
    """Optional: green highlight for success logs."""
    print(_colorize(f"[OK] {message}", "32"))

# ===============================================================
# Example usage (debug)
# ===============================================================
if __name__ == "__main__":
    log_info("Logger initialized successfully.")
    log_success("Connected to backend.")
    log_warning("Using fallback ngrok URLs.")
    log_error("Example error: could not fetch /generate.")
